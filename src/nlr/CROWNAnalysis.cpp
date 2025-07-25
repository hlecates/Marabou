#include "CROWNAnalysis.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "InfeasibleQueryException.h"
#include "MStringf.h"
#include "NLRError.h"
#include "TimeUtils.h"

#include <boost/thread.hpp>

namespace NLR {

CROWNAnalysis::CROWNAnalysis( TorchModel *torchModel )
    : _torchModel( torchModel )
    , _workLowerBounds( NULL )
    , _workUpperBounds( NULL )
    , _workLinearWeights( NULL )
    , _workLinearBias( NULL )
    , _lowerBias( torch::Tensor() )
    , _upperBias( torch::Tensor() )
{
    // Get the bounded modules from the torch model
    const Vector<std::shared_ptr<ITorchModuleBounded>>& boundedModules = _torchModel->getBoundedModules();
    
    // Get the element to bounded module mapping from the torch model
    const Map<unsigned, unsigned>& elementToBoundedModuleIndex = _torchModel->getElementToBoundedModuleIndex();
    
    // Initialize bounded elements map - map network node indices to bounded modules
    log(Stringf("CROWN: Initializing bounded elements from %u mappings", elementToBoundedModuleIndex.size()));
    for ( const auto& pair : elementToBoundedModuleIndex )
    {
        unsigned networkNodeIndex = pair.first;
        unsigned boundedModuleIndex = pair.second;
        
        log(Stringf("CROWN: Mapping network node %u to bounded module %u", networkNodeIndex, boundedModuleIndex));
        
        if ( boundedModuleIndex < boundedModules.size() )
        {
            _boundedElements[networkNodeIndex] = boundedModules[boundedModuleIndex];
            log(Stringf("CROWN: Successfully mapped network node %u to bounded module", networkNodeIndex));
        }
        else
        {
            log(Stringf("CROWN: Warning: bounded module index %u out of range (max: %u)", boundedModuleIndex, boundedModules.size()));
        }
    }
    log(Stringf("CROWN: Created %u bounded elements", _boundedElements.size()));

    // Build dependency graph
    buildDependencyGraph();

    allocateMemory();
}

CROWNAnalysis::~CROWNAnalysis()
{
    freeMemoryIfNeeded();
}

void CROWNAnalysis::freeMemoryIfNeeded()
{
    if ( _workLowerBounds )
    {
        delete _workLowerBounds;
        _workLowerBounds = NULL;
    }
    if ( _workUpperBounds )
    {
        delete _workUpperBounds;
        _workUpperBounds = NULL;
    }
    if ( _workLinearWeights )
    {
        delete _workLinearWeights;
        _workLinearWeights = NULL;
    }
    if ( _workLinearBias )
    {
        delete _workLinearBias;
        _workLinearBias = NULL;
    }
}

void CROWNAnalysis::allocateMemory()
{
    freeMemoryIfNeeded();
    
    _workLowerBounds = new torch::Tensor();
    _workUpperBounds = new torch::Tensor();
    _workLinearWeights = new torch::Tensor();
    _workLinearBias = new torch::Tensor();
}

void CROWNAnalysis::run()
{
    printf("CROWNAnalysis::run() - Starting\n");
    try {
        // Forward Interval Bound propagation
        printf("CROWNAnalysis::run() - Starting IBP bounds computation...\n");
        computeIBPBounds();
        printf("CROWNAnalysis::run() - IBP bounds computation completed.\n");
        
        // Backward linear relaxations following auto-LiRPA's approach
        printf("CROWNAnalysis::run() - Starting CROWN backward propagation...\n");
        computeCrownBackwardPropagation();
        printf("CROWNAnalysis::run() - CROWN backward propagation completed.\n");
        
        // Concretize symbolic bounds
        printf("CROWNAnalysis::run() - Starting bound concretization...\n");
        concretizeBounds();
        printf("CROWNAnalysis::run() - Bound concretization completed.\n");
        
    } catch (const std::exception& e) {
        printf("CROWNAnalysis::run() - Exception caught: %s\n", e.what());
        throw;
    }
    printf("CROWNAnalysis::run() - Completed successfully\n");
}


void CROWNAnalysis::buildDependencyGraph()
{
    // clear any existing dependency structures
    _dependencies.clear();
    _dependents.clear();
    _degreeOut.clear();
    _degreeIn.clear();
    _processed.clear();

    // Get the dependencies from TorchModel
    const Map<unsigned, Vector<unsigned>>& modelDependencies = _torchModel->getDependencies();

    // Build the dependency graph for ALL nodes in the model, not just bounded elements
    // This includes input nodes that are not bounded modules
    for (const auto& pair : modelDependencies)
    {
        unsigned elementIndex = pair.first;

        // Create the initial structures for the current layer
        _dependencies[elementIndex] = Vector<unsigned>();
        _dependents[elementIndex] = Vector<unsigned>();
        _degreeOut[elementIndex] = 0;
        _degreeIn[elementIndex] = 0;
        _processed[elementIndex] = false;

        // Copy the layer's dependencies directly
        _dependencies[elementIndex] = modelDependencies[elementIndex];

        // Compute the dependents ie which layers rely on the current layer
        // For every input into the current layer (ie a dependency of the current layer), 
        // the current layer is a dependent of the previous input layer
        // So build them iteratively in reverse
        for (unsigned inputIndex : modelDependencies[elementIndex]) 
        {
            _dependents[inputIndex].append(elementIndex);
            _degreeOut[inputIndex]++;
            // For every input, the degree of inputs for the elementIndex increases
            _degreeIn[elementIndex]++;
        }
    }
    
    // Also add any nodes that are not in modelDependencies but are in boundedElements
    // (this handles cases where a node has no dependencies)
    for (const auto& pair : _boundedElements)
    {
        unsigned elementIndex = pair.first;
        if (!_dependencies.exists(elementIndex))
        {
            _dependencies[elementIndex] = Vector<unsigned>();
            _dependents[elementIndex] = Vector<unsigned>();
            _degreeOut[elementIndex] = 0;
            _degreeIn[elementIndex] = 0;
            _processed[elementIndex] = false;
        }
    }
}


Vector<unsigned> CROWNAnalysis::topologicalSort()
{
    Vector<unsigned> sortedOrder;
    Queue<unsigned> queue;
    // Copy the in-degree map for tracking/editing in the method
    Map<unsigned, unsigned> degreeIn = _degreeIn;

    // Create initial queue for nodes that have no incoming edges
    // Include ALL nodes in the dependency graph, not just bounded elements
    for (const auto& pair : _dependencies)
    {
        unsigned elementIndex = pair.first;
        if (degreeIn.exists(elementIndex)) {
            if (degreeIn[elementIndex] == 0) {
                queue.push(elementIndex);
            }
        } else {
            // If not in degreeIn, it has no incoming edges
            queue.push(elementIndex);
        }
    }

    // Traverse nodes in topo order
    while (!queue.empty())
    {
        unsigned current = queue.peak();
        queue.pop();
        sortedOrder.append(current);

        // Update the degrees for the dependent nodes, if the node then has no dependents add to the queue
        if (_dependents.exists(current))
        {
            for (unsigned dependent : _dependents[current])
            {
                if (degreeIn.exists(dependent)) {
                    degreeIn[dependent]--;
                    if (degreeIn[dependent] == 0)
                    {
                        queue.push(dependent);
                    }
                }
            }
        }
    }

    return sortedOrder;
}


void CROWNAnalysis::computeIBPBounds()
{
    resetProcessingState();

    // Use the existing topological sort method
    Vector<unsigned> forwardOrder = topologicalSort(); 

    log(Stringf("IBP: Processing %u elements in forward order", forwardOrder.size()));
    
    // Compute IBP bounds for all elements in forward order
    for (unsigned elementIndex : forwardOrder)
    {
        log(Stringf("IBP: Processing element %u", elementIndex));
        
        if (isProcessed(elementIndex)) {
            log(Stringf("IBP: Element %u already processed, skipping", elementIndex));
            continue;
        }

        markProcessed(elementIndex);
        log(Stringf("IBP: Computing bounds for element %u", elementIndex));

        if (_boundedElements.exists(elementIndex)) {
            // This is a bounded module, compute bounds using the module
            auto& boundedElement = _boundedElements[elementIndex];
            
            // Get input bounds for this element (convert to old format for IBP)
            Vector<std::pair<torch::Tensor, torch::Tensor>> inputBounds;
            Vector<BoundedTensor<torch::Tensor>> boundedInputBounds = getInputBoundsForElement(elementIndex);
            
            // Convert BoundedTensor to std::pair for IBP
            for (const auto& boundedBound : boundedInputBounds) {
                inputBounds.append(std::make_pair(boundedBound.lower(), boundedBound.upper()));
            }
            
            // Compute IBP bounds for this element
            auto [lowerBound, upperBound] = boundedElement->computeIntervalBoundPropagation(inputBounds);
            
            // Store the computed bounds
            _ibpBounds[elementIndex] = std::make_pair(lowerBound, upperBound);
            log(Stringf("IBP: Stored bounds for element %u (bounded module)", elementIndex));
        } else {
            // This is not a bounded module (e.g., input node), get bounds from TorchModel
            const Map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& inputBounds = _torchModel->getInputBounds();
            if (inputBounds.exists(elementIndex)) {
                _ibpBounds[elementIndex] = inputBounds[elementIndex];
                log(Stringf("IBP: Stored bounds for element %u (input node)", elementIndex));
            } else {
                log(Stringf("IBP: Warning: No bounds found for element %u", elementIndex));
            }
        }
    }
}

void CROWNAnalysis::computeCrownBackwardPropagation()
{
    log("Starting CROWN backward propagation following auto-LiRPA's approach...");
    
    // Get output index
    // Right now this is assumed to be the bounded element with the highest index
    // TODO: Within the input parsing and torchmodel creation, add parsing of the actual index for output and add to the torch model --> Would skip an iteration through the layers
    unsigned outputIndex = getOutputIndex();
    
    log(Stringf("Output index determined: %u", outputIndex));
    
    if (!_boundedElements.exists(outputIndex)) {
        log(Stringf("Warning: Output index %u not found in bounded elements. Skipping CROWN analysis.", outputIndex));
        return;
    }
    
    log("Initializing A matrices following auto-LiRPA's approach...");
    
    // Initialize with identity matrices for the output
    // For CROWN, A matrices should have shape (1, output_features) for single constraint verification
    unsigned outputSize = _boundedElements[outputIndex]->getOutputSize();
    
    log(Stringf("Output size: %u", outputSize));
    
    // Use preprocessC to establish consistent format
    torch::Tensor C = torch::Tensor(); // Empty tensor for identity matrix
    torch::Tensor identityMatrix = preprocessC(C, outputSize); // Shape (1, output_size)
    
    // Initialize A matrices at the starting node
    // A matrices represent the linear transformation from output to input
    _lA[outputIndex] = identityMatrix;
    _uA[outputIndex] = identityMatrix;
    
    // Initialize bias terms with zeros
    // Bias terms should have shape (1, output_size) for single constraint
    _lowerBias = torch::zeros({1, outputSize}); // Shape (1, output_size)
    _upperBias = torch::zeros({1, outputSize}); // Shape (1, output_size)

    resetProcessingState();
    
    log("Starting queue-based processing following auto-LiRPA's approach...");
    
    // Queue-based processing 
    Queue<unsigned> queue;
    queue.push(outputIndex);
    
    while (!queue.empty())
    {
        unsigned current = queue.peak();
        queue.pop();
        
        log(Stringf("Processing element %u", current));
        
        if (isProcessed(current))
            continue;
            
        markProcessed(current);
        
        if (!_boundedElements.exists(current))
            continue;
            
        auto& boundedElement = _boundedElements[current];

        // Check if this element has A matrices to process
        if (!_lA.exists(current) && !_uA.exists(current))
        {
            log(Stringf("No A matrices for element %u, skipping", current));
            continue;
        }

        log(Stringf("Computing CROWN backward propagation for element %u using bounded module", current));

        // Get the IBP bounds of this element's actual inputs as BoundedTensor
        Vector<BoundedTensor<torch::Tensor>> inputIBPBounds = getInputBoundsForElement(current);

        // Get current A matrices
        torch::Tensor currentLowerAlpha = _lA.exists(current) ? _lA[current] : torch::Tensor();
        torch::Tensor currentUpperAlpha = _uA.exists(current) ? _uA[current] : torch::Tensor();

        // Compute the CROWN backward relaxarions 
        // Unpacks the return tuple
        auto [A_matrices, lbias, ubias] = boundedElement->boundBackward(currentLowerAlpha, currentUpperAlpha, inputIBPBounds);

        log(Stringf("Linear bounds computed for element %u by bounded module", current));

        // Propagate A matrices to input layers 
        if (_dependencies.exists(current))
        {
            for (unsigned i = 0; i < _dependencies[current].size() && i < A_matrices.size(); ++i)
            {
                unsigned inputIndex = _dependencies[current][i];
                log(Stringf("Propagating A matrices to input %u", inputIndex));
                
                // Get A matrices from the bounded module's result
                torch::Tensor new_lA = A_matrices[i].first;
                torch::Tensor new_uA = A_matrices[i].second;
                
                // Proper A matrix accumulation 
                addBound(inputIndex, new_lA, new_uA);
                
                // Add input to queue for processing
                queue.push(inputIndex);
            }
        }
        
        // Accumulate bias terms
        if (lbias.defined() && lbias.numel() > 0)
        {
            if (_lowerBias.numel() == 0) {
                _lowerBias = lbias;
            } else {
                _lowerBias = _lowerBias + lbias;
            }
        }
        
        if (ubias.defined() && ubias.numel() > 0)
        {
            if (_upperBias.numel() == 0) {
                _upperBias = ubias;
            } else {
                _upperBias = _upperBias + ubias;
            }
        }
        
        _lA.erase(current);
        _uA.erase(current);
    }
    
    log("CROWN backward propagation completed successfully following auto-LiRPA's approach.");
}

// Helper function for establishing consistent tensor format (following auto-LiRPA's _preprocess_C)
torch::Tensor CROWNAnalysis::preprocessC(const torch::Tensor& C, unsigned outputSize) {
    // auto-LiRPA uses consistent (spec, batch, ...) format 
    // User provides (batch, spec) but internally converts to (spec, batch), 
    // where batch is the number of constraints being verified, and spec is the number of outputs
    // For Marabou, we  are assuming single constraint verification, so batch_size = 1
    
    if (C.numel() == 0) {
        // Create identity matrix for single constraint verification
        return torch::eye(1, outputSize); // Shape (1, output_size)
    }
    
    // If C is provided, ensure it has the correct format
    if (C.dim() == 2) {
        // C has shape (batch, spec) -> convert to (spec, batch)
        return C.transpose(0, 1); // Shape (spec, batch)
    } else if (C.dim() == 1) {
        // C has shape (spec) -> add batch dimension
        return C.unsqueeze(1); // Shape (spec, 1)
    }
    
    // Default: return identity matrix
    return torch::eye(1, outputSize);
}

void CROWNAnalysis::concretizeBounds()
{
    // Get input bounds from TorchModel or from the stored IBP bounds or the bound manager
    // Currently just use a getter from the torch model which assumes default bounds
    const Map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& inputBounds = _torchModel->getInputBounds();
    
    // Process each element that has A matrices (not just bounded elements)
    for ( const auto& pair : _lA ) 
    {
        unsigned elementIndex = pair.first;
        
        // Skip elements without A matrices
        if ( !_lA.exists(elementIndex) && !_uA.exists(elementIndex) ) 
        {
            continue; 
        }
        
        // Get A matrices for this element
        torch::Tensor lA = _lA.exists(elementIndex) ? _lA[elementIndex] : torch::Tensor();
        torch::Tensor uA = _uA.exists(elementIndex) ? _uA[elementIndex] : torch::Tensor();
        
        // Get bias terms
        torch::Tensor lbias = _lowerBias;
        torch::Tensor ubias = _upperBias;
        
        // Initialize concrete bounds
        torch::Tensor concreteLower, concreteUpper;

        // For each input element, compute concrete bounds
        for ( const auto& inputPair : inputBounds ) 
        {
            const auto& [inputLower, inputUpper] = inputPair.second;
            
            // Compute center and eps following auto-LiRPA's approach
            torch::Tensor center = (inputUpper + inputLower) / 2.0;
            torch::Tensor eps = (inputUpper - inputLower) / 2.0;
            
            // Reshape for matrix operations
            center = center.unsqueeze(0); // Add batch dimension
            eps = eps.unsqueeze(0);

            torch::Tensor elementLowerBound, elementUpperBound;
            computeConcreteBounds(lA, uA, lbias, ubias, center, eps, elementLowerBound, elementUpperBound);

            // Accumulate bounds across all inputs
            // Accounts for multiple inputs, skips, residuals etc
            if ( elementLowerBound.defined() )
            {
                if ( !concreteLower.defined() ) 
                {
                    concreteLower = elementLowerBound;
                }
                else
                {
                    concreteLower = torch::min(concreteLower, elementLowerBound);
                }
            } 
            
            if ( elementUpperBound.defined() )
            {
                if ( !concreteUpper.defined() ) 
                {
                    concreteUpper = elementUpperBound;
                }
                else
                {
                    concreteUpper = torch::max(concreteUpper, elementUpperBound);
                }
            }  
        }

        // Store the concrete bounds
        if ( concreteLower.defined() || concreteUpper.defined() )
        {
            _concreteBounds[elementIndex] = std::make_pair(concreteLower, concreteUpper);
        }
        
        // Also compute and store the linear bounds for CROWN
        // Linear bounds are the symbolic linear relationships (A matrices)
        if ( lA.defined() || uA.defined() )
        {
            LinearBound linearBound;
            linearBound.lw = lA;
            linearBound.uw = uA;
            linearBound.lb = lbias;
            linearBound.ub = ubias;
            _linearBounds[elementIndex] = linearBound;
        }

    }
}

torch::Tensor CROWNAnalysis::computeConcreteLowerBound(const torch::Tensor& lA, const torch::Tensor& lBias,
                                                       const torch::Tensor& center, const torch::Tensor& eps)
{
    if (!lA.defined()) {
        return torch::Tensor();
    }

    // Debug tensor shapes
    std::string lA_shape = "[" + std::to_string(lA.size(0));
    for (int i = 1; i < lA.dim(); i++) {
        lA_shape += ", " + std::to_string(lA.size(i));
    }
    lA_shape += "]";
    
    std::string lBias_shape = "[" + std::to_string(lBias.size(0));
    for (int i = 1; i < lBias.dim(); i++) {
        lBias_shape += ", " + std::to_string(lBias.size(i));
    }
    lBias_shape += "]";
    
    log(Stringf("computeConcreteLowerBound: lA shape: %s, lBias shape: %s", lA_shape.c_str(), lBias_shape.c_str()));

    // Handle different tensor shapes more robustly
    torch::Tensor lAReshaped, lBiasReshaped, centerReshaped, epsReshaped;
    
    if (lA.dim() == 2) {
        // lA is already (spec_dim, input_dim)
        lAReshaped = lA.unsqueeze(0); // (1, spec_dim, input_dim)
    } else if (lA.dim() == 1) {
        // lA is (input_dim), reshape to (1, 1, input_dim)
        lAReshaped = lA.unsqueeze(0).unsqueeze(0);
    } else if (lA.dim() == 3) {
        // lA is already (batch, spec_dim, input_dim)
        lAReshaped = lA;
    } else {
        // For any other case, try to make it 3D
        lAReshaped = lA.unsqueeze(0);
        if (lAReshaped.dim() < 3) {
            lAReshaped = lAReshaped.unsqueeze(0);
        }
    }
    
    if (lBias.dim() == 1) {
        lBiasReshaped = lBias.unsqueeze(0).unsqueeze(-1); // (1, spec_dim, 1)
    } else if (lBias.dim() == 2) {
        lBiasReshaped = lBias.unsqueeze(-1); // (spec_dim, 1)
    } else {
        lBiasReshaped = lBias.unsqueeze(-1);
    }
    
    if (center.dim() == 1) {
        centerReshaped = center.unsqueeze(0).unsqueeze(0); // (1, 1, input_dim)
    } else if (center.dim() == 2) {
        centerReshaped = center.unsqueeze(0); // (1, batch, input_dim)
    } else {
        centerReshaped = center;
    }
    
    if (eps.dim() == 1) {
        epsReshaped = eps.unsqueeze(0).unsqueeze(0); // (1, 1, input_dim)
    } else if (eps.dim() == 2) {
        epsReshaped = eps.unsqueeze(0); // (1, batch, input_dim)
    } else {
        epsReshaped = eps;
    }

    // Following auto-LiRPA's concretize_bounds formula for lower bound:
    // ret = lA.bmm(x_hat) - lA.abs().bmm(x_eps) + lbias
    // (1, spec_dim, 1)
    torch::Tensor lowerTerm = lAReshaped.bmm(centerReshaped.transpose(0, 1)); 
    // (1, spec_dim, 1)
    torch::Tensor absTerm = lAReshaped.abs().bmm(epsReshaped.transpose(0, 1)); 
    torch::Tensor concreteLower = lowerTerm - absTerm + lBiasReshaped;
    
    // Remove batch dims and return
    return concreteLower.squeeze(-1).squeeze(0); 
}


}

torch::Tensor NLR::CROWNAnalysis::computeConcreteUpperBound(const torch::Tensor& uA, const torch::Tensor& uBias,
                                                       const torch::Tensor& center, const torch::Tensor& eps)
{
    if (!uA.defined()) {
        return torch::Tensor();
    }

    // Debug tensor shapes
    std::string uA_shape = "[" + std::to_string(uA.size(0));
    for (int i = 1; i < uA.dim(); i++) {
        uA_shape += ", " + std::to_string(uA.size(i));
    }
    uA_shape += "]";
    
    std::string uBias_shape = "[" + std::to_string(uBias.size(0));
    for (int i = 1; i < uBias.dim(); i++) {
        uBias_shape += ", " + std::to_string(uBias.size(i));
    }
    uBias_shape += "]";
    
    log(Stringf("computeConcreteUpperBound: uA shape: %s, uBias shape: %s", uA_shape.c_str(), uBias_shape.c_str()));

    // Handle different tensor shapes more robustly
    torch::Tensor uAReshaped, uBiasReshaped, centerReshaped, epsReshaped;
    
    if (uA.dim() == 2) {
        // uA is already (spec_dim, input_dim)
        uAReshaped = uA.unsqueeze(0); // (1, spec_dim, input_dim)
    } else if (uA.dim() == 1) {
        // uA is (input_dim), reshape to (1, 1, input_dim)
        uAReshaped = uA.unsqueeze(0).unsqueeze(0);
    } else if (uA.dim() == 3) {
        // uA is already (batch, spec_dim, input_dim)
        uAReshaped = uA;
    } else {
        // For any other case, try to make it 3D
        uAReshaped = uA.unsqueeze(0);
        if (uAReshaped.dim() < 3) {
            uAReshaped = uAReshaped.unsqueeze(0);
        }
    }
    
    if (uBias.dim() == 1) {
        uBiasReshaped = uBias.unsqueeze(0).unsqueeze(-1); // (1, spec_dim, 1)
    } else if (uBias.dim() == 2) {
        uBiasReshaped = uBias.unsqueeze(-1); // (spec_dim, 1)
    } else {
        uBiasReshaped = uBias.unsqueeze(-1);
    }
    
    if (center.dim() == 1) {
        centerReshaped = center.unsqueeze(0).unsqueeze(0); // (1, 1, input_dim)
    } else if (center.dim() == 2) {
        centerReshaped = center.unsqueeze(0); // (1, batch, input_dim)
    } else {
        centerReshaped = center;
    }
    
    if (eps.dim() == 1) {
        epsReshaped = eps.unsqueeze(0).unsqueeze(0); // (1, 1, input_dim)
    } else if (eps.dim() == 2) {
        epsReshaped = eps.unsqueeze(0); // (1, batch, input_dim)
    } else {
        epsReshaped = eps;
    }

    // Following auto-LiRPA's concretize_bounds formula for upper bound:
    // ret = uA.bmm(x_hat) + uA.abs().bmm(x_eps) + ubias
    // (1, spec_dim, 1)
    torch::Tensor upperTerm = uAReshaped.bmm(centerReshaped.transpose(0, 1)); 
    // (1, spec_dim, 1)
    torch::Tensor absTerm = uAReshaped.abs().bmm(epsReshaped.transpose(0, 1)); 
    torch::Tensor concreteUpper = upperTerm + absTerm + uBiasReshaped;
    
    // Remove batch dims and return
    return concreteUpper.squeeze(-1).squeeze(0); 
}

void NLR::CROWNAnalysis::computeConcreteBounds(const torch::Tensor& lA, const torch::Tensor& uA,
                                         const torch::Tensor& lBias, const torch::Tensor& uBias,
                                         const torch::Tensor& center, const torch::Tensor& eps,
                                         torch::Tensor& concreteLower, torch::Tensor& concreteUpper)
{
    // Compute lower bound
    concreteLower = computeConcreteLowerBound(lA, lBias, center, eps);
    
    // Compute upper bound
    concreteUpper = computeConcreteUpperBound(uA, uBias, center, eps);
}

Vector<BoundedTensor<torch::Tensor>> NLR::CROWNAnalysis::getInputBoundsForElement(unsigned elementIndex) {
    Vector<BoundedTensor<torch::Tensor>> inputBounds;

    // Get the bounded element
    if (!_boundedElements.exists(elementIndex)) {
        throw std::runtime_error("Element index not found: " + std::to_string(elementIndex));
    }

    auto& boundedElement = _boundedElements[elementIndex];

    // Get the actual bounds from dependencies
    if (_dependencies.exists(elementIndex) && !_dependencies[elementIndex].empty())
    {
        for (unsigned inputIndex : _dependencies[elementIndex])
        {
            if(_ibpBounds.exists(inputIndex)){
                // Convert std::pair to BoundedTensor
                auto [lower, upper] = _ibpBounds[inputIndex];
                inputBounds.append(BoundedTensor<torch::Tensor>(lower, upper));
            }
            else 
            {
                // No IBP bounds available for this dependency - create reasonable defaults
                // Use the input size of the current element to create appropriate bounds
                unsigned inputSize = boundedElement->getInputSize();
                torch::Tensor placeholderLower = torch::zeros({inputSize});
                torch::Tensor placeholderUpper = torch::ones({inputSize});
                
                // If we have some information about the input, use it
                if (_boundedElements.exists(inputIndex)) {
                    auto& inputElement = _boundedElements[inputIndex];
                    unsigned actualInputSize = inputElement->getOutputSize();
                    if (actualInputSize != inputSize) {
                        // Resize to match the actual input size
                        placeholderLower = torch::zeros({actualInputSize});
                        placeholderUpper = torch::ones({actualInputSize});
                    }
                }
                
                inputBounds.append(BoundedTensor<torch::Tensor>(placeholderLower, placeholderUpper));
            }
        }
    } 
    else
    {
        // No dependencies -> this is an input element
        // Use the input bounds from the TorchModel
        unsigned inputSize = boundedElement->getInputSize();
        torch::Tensor inputLower = torch::zeros({inputSize});
        torch::Tensor inputUpper = torch::ones({inputSize});
        
        // TODO: Get actual input bounds from TorchModel if available, via a check
        // For now, use default 
        inputBounds.append(BoundedTensor<torch::Tensor>(inputLower, inputUpper));
    }

    return inputBounds;
}

unsigned NLR::CROWNAnalysis::getOutputIndex() const {
    // Find the element with the highest index (assuming it's the output)
    unsigned outputIndex = 0;
    for (const auto& pair : _boundedElements) {
        if (pair.first > outputIndex) {
            outputIndex = pair.first;
        }
    }
    return outputIndex;
}

// Add helper function for A matrix addition
torch::Tensor NLR::CROWNAnalysis::addA(const torch::Tensor& A1, const torch::Tensor& A2) {
    // Simple tensor addition following auto-LiRPA's addA pattern
    if (A1.numel() == 0) {
        return A2;
    }
    if (A2.numel() == 0) {
        return A1;
    }
    return A1 + A2;
}

// Add helper function for proper A matrix accumulation (following auto-LiRPA's add_bound pattern)
void NLR::CROWNAnalysis::addBound(unsigned elementIndex, const torch::Tensor& lA, const torch::Tensor& uA) {
    // Proper A matrix accumulation following auto-LiRPA's add_bound pattern
    if (lA.numel() > 0) {
        if (!_lA.exists(elementIndex) || _lA[elementIndex].numel() == 0) {
            // First A added to this element
            _lA[elementIndex] = lA;
        } else {
            // Accumulate A matrices using addition
            _lA[elementIndex] = addA(_lA[elementIndex], lA);
        }
    }
    
    if (uA.numel() > 0) {
        if (!_uA.exists(elementIndex) || _uA[elementIndex].numel() == 0) {
            // First A added to this element
            _uA[elementIndex] = uA;
        } else {
            // Accumulate A matrices using addition
            _uA[elementIndex] = addA(_uA[elementIndex], uA);
        }
    }
}

bool NLR::CROWNAnalysis::isProcessed(unsigned elementIndex) const
{
    return _processed.exists(elementIndex) && _processed[elementIndex];
}

void NLR::CROWNAnalysis::resetProcessingState() 
{
    // Reset processed state
    for ( auto& pair : _boundedElements )
    {
        _processed[pair.first] = false;
    }
}

void NLR::CROWNAnalysis::markProcessed(unsigned elementIndex)
{
    _processed[elementIndex] = true;
}

void NLR::CROWNAnalysis::log( const String &message )
{
    if ( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
    {
        printf( "CROWNAnalysis: %s\n", message.ascii() );
    }
}


// --------------------------------------
// Public Get mothods for the unit testing -> shouldn't be needed in the actual CROWN analysis (maybe for creating tightenings to update Marabou engine)


torch::Tensor NLR::CROWNAnalysis::getIBPLowerBound(unsigned elementIndex)
{
    if (_ibpBounds.exists(elementIndex)) {
        return _ibpBounds[elementIndex].first;
    }
    return torch::Tensor();
}

torch::Tensor NLR::CROWNAnalysis::getIBPUpperBound(unsigned elementIndex)
{
    if (_ibpBounds.exists(elementIndex)) {
        return _ibpBounds[elementIndex].second;
    }
    return torch::Tensor();
}

torch::Tensor NLR::CROWNAnalysis::getCrownLowerBound(unsigned elementIndex)
{
    if (_linearBounds.exists(elementIndex)) {
        return _linearBounds[elementIndex].lw;
    }
    return torch::Tensor();
}

torch::Tensor NLR::CROWNAnalysis::getCrownUpperBound(unsigned elementIndex)
{
    if (_linearBounds.exists(elementIndex)) {
        return _linearBounds[elementIndex].uw;
    }
    return torch::Tensor();
}

bool NLR::CROWNAnalysis::hasIBPBounds(unsigned elementIndex)
{
    return _ibpBounds.exists(elementIndex);
}

bool NLR::CROWNAnalysis::hasCrownBounds(unsigned elementIndex)
{
    return _linearBounds.exists(elementIndex);
}

unsigned NLR::CROWNAnalysis::getNumElements() const
{
    return _boundedElements.size();
}

// Concrete bound access methods
torch::Tensor NLR::CROWNAnalysis::getConcreteLowerBound(unsigned elementIndex)
{
    if (_concreteBounds.exists(elementIndex)) {
        return _concreteBounds[elementIndex].first;
    }
    return torch::Tensor();
}

torch::Tensor NLR::CROWNAnalysis::getConcreteUpperBound(unsigned elementIndex)
{
    if (_concreteBounds.exists(elementIndex)) {
        return _concreteBounds[elementIndex].second;
    }
    return torch::Tensor();
}

bool NLR::CROWNAnalysis::hasConcreteBounds(unsigned elementIndex)
{
    return _concreteBounds.exists(elementIndex);
}