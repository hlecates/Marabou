#include "CROWNAnalysis.h"
#include "BoundedConstantNode.h"

#include "Debug.h"
#include "FloatUtils.h"
#include "InfeasibleQueryException.h"
#include "MStringf.h"
#include "NLRError.h"
#include "TimeUtils.h"

namespace NLR {

CROWNAnalysis::CROWNAnalysis( TorchModel *torchModel )
    : _torchModel( torchModel )
{
    // Get all nodes from the torch model
    const Vector<std::shared_ptr<BoundedTorchNode>>& nodes = _torchModel->getNodes();
    
    // Initialize nodes map - map network node indices to bounded nodes
    log(Stringf("CROWN: Initializing nodes from %u nodes", nodes.size()));
    for ( unsigned i = 0; i < nodes.size(); ++i ) 
    {
        _nodes[i] = nodes[i];
        log(Stringf("CROWN: Mapped network node %u to %s node", i, nodeTypeToString(nodes[i]->getNodeType()).c_str()));
    }
    log(Stringf("CROWN: Created %u nodes", _nodes.size()));
}

CROWNAnalysis::~CROWNAnalysis()
{

}

void CROWNAnalysis::run()
{
    log("CROWNAnalysis::run() - Starting");
    try {
        
        //_torchModel->obtainCurrentBoundsFromNLR();

        computeIBPBounds();

        // After computeIBPBounds();
        std::cout << "\n=== IBP Bounds ===" << std::endl;
        for (unsigned i = 0; i < _nodes.size(); ++i) {
            if (_ibpBounds.exists(i)) {
                auto bounds = _ibpBounds[i];
                std::cout << "Node " << i << " IBP: lower=" << bounds.lower() << ", upper=" << bounds.upper() << std::endl;
            }
        }

        computeForwardPassValues();

        std::cout << "\n=== Forward Values ===" << std::endl;
        for (unsigned i = 0; i < _nodes.size(); ++i) {
            if (_forwardPassValues.exists(i)) {
                auto value = _forwardPassValues[i];
                std::cout << "Node " << i << " ForwardPass value =" << value << std::endl;
            }
        }

        computeCrownBackwardPropagation();

        concretizeBounds();

        // _torchModel->updateNLRWithTighterBounds();  
        
    } catch (const std::exception& e) {
        log(Stringf("CROWNAnalysis::run() - Exception caught: %s", e.what()));
        throw;
    }
    log("CROWNAnalysis::run() - Completed successfully");
    
    // Should add a clearing of the temp bound storage (for when mutliple iterations of CROWN are done in either Marabou process or through alpha CROWN)

}


void CROWNAnalysis::computeForwardPassValues()
{
    log("[DEBUG] computeForwardPassValues() - Start");

    // Compute the center of input bounds, this will be forward pass input
    torch::Tensor inputCenter;
    if( _torchModel->hasInputBounds() ) 
    {
        torch::Tensor inputLower = _torchModel->getInputLowerBounds();
        torch::Tensor inputUpper = _torchModel->getInputUpperBounds();
        inputCenter = (inputLower + inputUpper) / 2.0;
        log("[DEBUG] computeForwardPassValues() - Computed input center from bounds"); 
    } 
    else 
    {
        // Default to a center of [0,1] ie 0.5
        unsigned inputSize = _torchModel->getInputSize();
        inputCenter = torch::full({(long)inputSize}, 0.5, torch::kFloat32);
        log("[DEBUG] computeForwardPassValues() - Using DEFAULT input center");
    }

    _forwardPassValues = _torchModel->forwardAndStoreActivations(inputCenter);

    log(Stringf("[DEBUG] computeForwardPassValues() - Stored forward pass values for %u nodes", _forwardPassValues.size()));
}


void CROWNAnalysis::computeIBPBounds()
{
    resetProcessingState();
    Vector<unsigned> forwardOrder = _torchModel->topologicalSort(); 

    log(Stringf("IBP: Processing %u nodes in forward order", forwardOrder.size()));
    
    for (unsigned nodeIndex : forwardOrder) {

        if (isProcessed(nodeIndex)) continue;
        markProcessed(nodeIndex);

        auto& node = _nodes[nodeIndex];
        NodeType nodeType = node->getNodeType();
        
        log(Stringf("IBP: Computing bounds for node %u (%s)", nodeIndex, nodeTypeToString(nodeType).c_str()));

        // Get input bounds for this node
        Vector<BoundedTensor<torch::Tensor>> inputBounds = getInputBoundsForNode(nodeIndex);
        
        // ADD DEBUGGING FOR INPUT BOUNDS
        std::cout << "[DEBUG] IBP: Node " << nodeIndex << " input bounds:" << std::endl;
        for (unsigned i = 0; i < inputBounds.size(); i++) {
            std::cout << "[DEBUG] IBP: Input " << i << " lower: " << inputBounds[i].lower() << std::endl;
            std::cout << "[DEBUG] IBP: Input " << i << " upper: " << inputBounds[i].upper() << std::endl;
        }
        
        // In computeIBPBounds(), before computing IBP bounds:
        if (node->getNodeType() == NodeType::INPUT) {
            if (_torchModel->hasInputBounds()) {
                torch::Tensor inputLower = _torchModel->getInputLowerBounds();
                torch::Tensor inputUpper = _torchModel->getInputUpperBounds();
                _ibpBounds[nodeIndex] = BoundedTensor<torch::Tensor>(inputLower, inputUpper);
                log(Stringf("IBP: Set input node %u bounds from model", nodeIndex));
                continue; // Skip the normal IBP computation
            }
        }

        // Compute IBP bounds
        BoundedTensor<torch::Tensor> ibpBounds = node->computeIntervalBoundPropagation(inputBounds);
        
        // ADD DEBUGGING FOR COMPUTED IBP BOUNDS
        std::cout << "[DEBUG] IBP: Node " << nodeIndex << " computed bounds:" << std::endl;
        std::cout << "[DEBUG] IBP: Lower: " << ibpBounds.lower() << std::endl;
        std::cout << "[DEBUG] IBP: Upper: " << ibpBounds.upper() << std::endl;
        
        // Store IBP bounds
        _ibpBounds[nodeIndex] = ibpBounds;
        
        log(Stringf("IBP: Node %u bounds computed", nodeIndex));
    }
}

    
void CROWNAnalysis::computeCrownBackwardPropagation()
{
    log("Starting CROWN backward propagation following auto-LiRPA's approach...");
    
    // Get output index
    unsigned outputIndex = getOutputIndex();
    log(Stringf("Output index determined: %u", outputIndex));
    
    if ( !_nodes.exists(outputIndex) ) 
    {
        log(Stringf("Warning: Output index %u not found in nodes. Skipping CROWN analysis.", outputIndex));
        return;
    }
    
    // Initialize with identity matrices for the output
    auto& outputNode = _nodes[outputIndex];
    unsigned outputSize = outputNode->getOutputSize();
    log(Stringf("Output size: %u", outputSize));
    
    // Use preprocessC to establish consistent format
    torch::Tensor identityMatrix = preprocessC(torch::Tensor(), outputSize);
    
    // Initialize A matrices at the starting node
    _lA[outputIndex] = identityMatrix;
    _uA[outputIndex] = identityMatrix;
    
    // Initialize bias terms with zeros
    // Initialize bias maps for the output node
    _lowerBias[outputIndex] = torch::zeros({outputSize}, torch::kFloat32); // Shape (output_size)
    _upperBias[outputIndex] = torch::zeros({outputSize}, torch::kFloat32); // Shape (output_size)

    resetProcessingState();
    
    log("Starting queue-based processing.");
    
    // Queue-based processing 
    Queue<unsigned> queue;
    queue.push(outputIndex);
    
    while (!queue.empty())
    {
        unsigned current = queue.peak();
        queue.pop();
        
        if ( isProcessed(current) ) continue;
        markProcessed(current);
        
        auto& node = _nodes[current];
        NodeType nodetype = node->getNodeType();

        log(Stringf("Processing node %u (%s)", current, nodeTypeToString(nodetype).c_str()));

        // Check if this element has A matrices to process
        if (!_lA.exists(current) && !_uA.exists(current))
        {
            log(Stringf("No A matrices for element %u, skipping", current));
            continue;
        }

        // Get the IBP bounds of this element's actual inputs as BoundedTensor
        Vector<BoundedTensor<torch::Tensor>> inputIBPBounds = getInputBoundsForNode(current);
        
        // ADD DEBUGGING FOR IBP BOUNDS DURING CROWN BACKWARD
        std::cout << "[DEBUG] CROWN Backward: Node " << current << " IBP bounds:" << std::endl;
        if (_ibpBounds.exists(current)) {
            std::cout << "[DEBUG] CROWN Backward: Node " << current << " own IBP bounds: lower=" << _ibpBounds[current].lower() << ", upper=" << _ibpBounds[current].upper() << std::endl;
        } else {
            std::cout << "[DEBUG] CROWN Backward: Node " << current << " has no IBP bounds" << std::endl;
        }

        // Get current A matrices
        torch::Tensor currentLowerAlpha = _lA.exists(current) ? _lA[current] : torch::Tensor();
        torch::Tensor currentUpperAlpha = _uA.exists(current) ? _uA[current] : torch::Tensor();

        // Compute the CROWN backward relaxations using the node's method
        Vector<Pair<torch::Tensor, torch::Tensor>> A_matrices;
        torch::Tensor lbias, ubias;
        
        node->boundBackward(currentLowerAlpha, currentUpperAlpha, inputIBPBounds, 
                           A_matrices, lbias, ubias);

        log(Stringf("Linear bounds computed for node %u (%s)", current, nodeTypeToString(nodetype).c_str()));

        // Propagate A matrices to input layers 
        if (_torchModel->getDependenciesMap().exists(current))
        {
            for (unsigned i = 0; i < _torchModel->getDependencies(current).size() && i < A_matrices.size(); ++i)
            {
                unsigned inputIndex = _torchModel->getDependencies(current)[i];
                log(Stringf("Propagating A matrices to input %u", inputIndex));
                
                // Get A matrices from the bounded module's result
                // These are already computed correctly by the bounded nodes
                torch::Tensor new_lA = A_matrices[i].first();
                torch::Tensor new_uA = A_matrices[i].second();
                
                // FIXED: No additional multiplication needed - bounded nodes already computed A matrices
                addBound(inputIndex, new_lA, new_uA);
                
                torch::Tensor propagated_lbias = lbias.defined() ? lbias.clone() : torch::zeros_like(_lowerBias[current]);
                torch::Tensor propagated_ubias = ubias.defined() ? ubias.clone() : torch::zeros_like(_upperBias[current]);

                if (_lowerBias.exists(current)) propagated_lbias = propagated_lbias + _lowerBias[current];
                if (_upperBias.exists(current)) propagated_ubias = propagated_ubias + _upperBias[current];

                addBias(inputIndex, propagated_lbias, propagated_ubias);
                
                // Add input to queue for processing
                queue.push(inputIndex);
            }
        }
    
        /*
        if (lbias.defined() && lbias.numel() > 0)
        {
            _lowerBias = _lowerBias + lbias;
        }
        if (ubias.defined() && ubias.numel() > 0)
        {
            _upperBias = _upperBias + ubias;
        }
        */
        
    }
    
    log("CROWN backward propagation completed.");
}



// Helper function for establishing consistent tensor format (following auto-LiRPA's _preprocess_C)
torch::Tensor CROWNAnalysis::preprocessC(const torch::Tensor& C, unsigned outputSize) {
    // auto-LiRPA uses consistent (spec, batch, ...) format 
    // User provides (batch, spec) but internally converts to (spec, batch), 
    // where batch is the number of constraints being verified, and spec is the number of outputs
    // For Marabou, we are assuming single constraint verification, so batch_size = 1
    
    // Ensure outputSize is valid
    if (outputSize == 0) {
        throw std::runtime_error("CROWNAnalysis: outputSize cannot be zero");
    }
    
    if (C.numel() == 0) {
        // Create identity matrix for single constraint verification
        // Shape should be [batch_size, output_size, output_size] for proper 3D operations
        // Following auto-LiRPA's approach: torch.eye(dim).unsqueeze(0).expand(batch_size, -1, -1)
        return torch::eye(outputSize, torch::kFloat32).unsqueeze(0); // Shape (1, output_size, output_size)
    }
    
    // If C is provided, ensure it has the correct format
    if (C.dim() == 2) {
        // C has shape (batch, spec) -> keep as is for proper matrix multiplication
        return C; // Shape (batch, spec)
    } else if (C.dim() == 1) {
        // C has shape (spec) -> add batch dimension
        return C.unsqueeze(0); // Shape (1, spec)
    }
    
    // Default: return identity matrix with proper 3D shape - SPECIFY torch::kFloat32
    // This creates [1, output_size, output_size] following auto-LiRPA's pattern
    return torch::eye(outputSize, torch::kFloat32).unsqueeze(0); // Shape (1, output_size, output_size)
}

void CROWNAnalysis::concretizeBounds()
{
    log("[DEBUG] concretizeBounds() - Starting (output-only mode)");

    // Determine output node
    unsigned outputIndex = getOutputIndex();
    if (!_nodes.exists(outputIndex)) {
        log(Stringf("[DEBUG] concretizeBounds() - Output index %u not found", outputIndex));
        return;
    }

    // Find the (single) input node index
    int inputIndex = -1;
    for (const auto &p : _nodes) {
        if (p.second->getNodeType() == NodeType::INPUT) {
            inputIndex = static_cast<int>(p.first);
            break;
        }
    }

    if (inputIndex < 0) {
        log("[DEBUG] concretizeBounds() - No input node found; falling back to IBP at output");
        if (_ibpBounds.exists(outputIndex)) {
            _concreteBounds[outputIndex] = _ibpBounds[outputIndex];
            _torchModel->setConcreteBounds(outputIndex, _ibpBounds[outputIndex]);
        }
        log("[DEBUG] concretizeBounds() - Completed (fallback)");
        return;
    }

    // Get input bounds
    torch::Tensor inputLower, inputUpper;
    if (_torchModel->hasInputBounds()) {
        inputLower = _torchModel->getInputLowerBounds();
        inputUpper = _torchModel->getInputUpperBounds();
        std::cout << "[DEBUG] concretizeBounds() - Using provided input bounds: lower=" << inputLower
                  << ", upper=" << inputUpper << std::endl;
    } else {
        unsigned inputSize = _torchModel->getInputSize();
        inputLower = torch::zeros({(long)inputSize}, torch::kFloat32);
        inputUpper = torch::ones({(long)inputSize}, torch::kFloat32);
        std::cout << "[DEBUG] concretizeBounds() - Using default input bounds: lower=" << inputLower
                  << ", upper=" << inputUpper << std::endl;
    }

    inputLower = inputLower.to(torch::kFloat32);
    inputUpper = inputUpper.to(torch::kFloat32);

    // Retrieve final A matrices (w.r.t. inputs) and biases at the input node
    if (!_lA.exists(inputIndex) && !_uA.exists(inputIndex)) {
        log(Stringf("[DEBUG] concretizeBounds() - No A matrices at input node %d; fallback to IBP output", inputIndex));
        if (_ibpBounds.exists(outputIndex)) {
            _concreteBounds[outputIndex] = _ibpBounds[outputIndex];
            _torchModel->setConcreteBounds(outputIndex, _ibpBounds[outputIndex]);
        }
        log("[DEBUG] concretizeBounds() - Completed (fallback: no input A)");
        return;
    }

    torch::Tensor lA = _lA.exists(inputIndex) ? _lA[inputIndex] : torch::Tensor();
    torch::Tensor uA = _uA.exists(inputIndex) ? _uA[inputIndex] : torch::Tensor();
    torch::Tensor lBias = _lowerBias.exists(inputIndex) ? _lowerBias[inputIndex] : torch::Tensor();
    torch::Tensor uBias = _upperBias.exists(inputIndex) ? _upperBias[inputIndex] : torch::Tensor();

    std::cout << "[DEBUG] concretizeBounds() - Using inputIndex=" << inputIndex << " for final A matrices" << std::endl;

    // Sanity: if shapes don't align, fallback to IBP
    if (lA.defined() && lA.dim() >= 2) {
        int nodeDim = inputLower.size(0);
        int expectedNodeDim = lA.size(-1);
        if (nodeDim != expectedNodeDim) {
            log(Stringf("[DEBUG] concretizeBounds() - Dim mismatch: input=%d, expected=%d; fallback to IBP", nodeDim, expectedNodeDim));
            if (_ibpBounds.exists(outputIndex)) {
                _concreteBounds[outputIndex] = _ibpBounds[outputIndex];
                _torchModel->setConcreteBounds(outputIndex, _ibpBounds[outputIndex]);
            }
            log("[DEBUG] concretizeBounds() - Completed (fallback: dim mismatch)");
            return;
        }
    }

    // Compute concrete bounds at the output using only the final A matrices and input bounds
    torch::Tensor concreteLower, concreteUpper;
    computeConcreteBounds(lA, uA, lBias, uBias, inputLower, inputUpper, concreteLower, concreteUpper);

    if (concreteLower.defined() && concreteUpper.defined()) {
        std::cout << "[DEBUG] Output node " << outputIndex << " Concrete bounds: lower="
                  << concreteLower << ", upper=" << concreteUpper << std::endl;
        // Store only for the output node
        BoundedTensor<torch::Tensor> concreteBounds(concreteLower, concreteUpper);
        _concreteBounds[outputIndex] = concreteBounds;
        _torchModel->setConcreteBounds(outputIndex, concreteBounds);
    } else {
        log("[DEBUG] concretizeBounds() - Concrete bounds undefined; fallback to IBP if available");
        if (_ibpBounds.exists(outputIndex)) {
            _concreteBounds[outputIndex] = _ibpBounds[outputIndex];
            _torchModel->setConcreteBounds(outputIndex, _ibpBounds[outputIndex]);
        }
    }

    log("[DEBUG] concretizeBounds() - Completed (output-only mode)");
}

// Helpers to coerce shapes to (1, spec, n) and (1, n, 1)
static inline torch::Tensor ensure3A(const torch::Tensor& A) {
    if (!A.defined()) return A;
    if (A.dim() == 3) return A;
    if (A.dim() == 2) return A.unsqueeze(0);        // (1, spec, n)
    if (A.dim() == 1) return A.unsqueeze(0).unsqueeze(0);
    return A.unsqueeze(0); // best effort
}
static inline torch::Tensor ensure3x(const torch::Tensor& x) {
    // x is (n,) -> (1, n, 1); (b,n)->(b,n,1)
    if (!x.defined()) return x;
    if (x.dim() == 1) return x.unsqueeze(0).unsqueeze(-1);
    if (x.dim() == 2) return x.unsqueeze(-1);
    return x;
}
static inline torch::Tensor ensure3b(const torch::Tensor& b) {
    // b is (spec,) -> (1, spec, 1)
    if (!b.defined()) return b;
    if (b.dim() == 1) return b.unsqueeze(0).unsqueeze(-1);
    if (b.dim() == 2) return b.unsqueeze(-1);
    return b;
}


torch::Tensor CROWNAnalysis::computeConcreteLowerBound(
    const torch::Tensor& lA, const torch::Tensor& lBias,
    const torch::Tensor& xLower, const torch::Tensor& xUpper)
{
    if (!lA.defined()) return torch::Tensor();

    torch::Tensor AL = ensure3A(lA.to(torch::kFloat32));         // (1,spec,n)
    torch::Tensor xL = ensure3x(xLower.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor xU = ensure3x(xUpper.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor bL = ensure3b(lBias.to(torch::kFloat32));      // (1,spec,1)

    torch::Tensor Apos = torch::clamp_min(AL, 0);
    torch::Tensor Aneg = torch::clamp_max(AL, 0);

    // LB = βL + Apos * xL + Aneg * xU
    torch::Tensor term = Apos.bmm(xL) + Aneg.bmm(xU);            // (1,spec,1)
    torch::Tensor out  = term + bL;                              // (1,spec,1)
    return out.squeeze(-1).squeeze(0);                           // (spec,)
}


torch::Tensor CROWNAnalysis::computeConcreteUpperBound(
    const torch::Tensor& uA, const torch::Tensor& uBias,
    const torch::Tensor& xLower, const torch::Tensor& xUpper)
{
    if (!uA.defined()) return torch::Tensor();

    torch::Tensor AU = ensure3A(uA.to(torch::kFloat32));         // (1,spec,n)
    torch::Tensor xL = ensure3x(xLower.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor xU = ensure3x(xUpper.to(torch::kFloat32));     // (1,n,1)
    torch::Tensor bU = ensure3b(uBias.to(torch::kFloat32));      // (1,spec,1)

    torch::Tensor Apos = torch::clamp_min(AU, 0);
    torch::Tensor Aneg = torch::clamp_max(AU, 0);

    // UB = βU + Apos * xU + Aneg * xL
    torch::Tensor term = Apos.bmm(xU) + Aneg.bmm(xL);            // (1,spec,1)
    torch::Tensor out  = term + bU;                              // (1,spec,1)
    return out.squeeze(-1).squeeze(0);                           // (spec,)
}


void CROWNAnalysis::computeConcreteBounds(
    const torch::Tensor& lA, const torch::Tensor& uA,
    const torch::Tensor& lBias, const torch::Tensor& uBias,
    const torch::Tensor& nodeLower, const torch::Tensor& nodeUpper,
    torch::Tensor& concreteLower, torch::Tensor& concreteUpper)
{
    concreteLower = computeConcreteLowerBound(lA, lBias, nodeLower, nodeUpper);
    concreteUpper = computeConcreteUpperBound(uA, uBias, nodeLower, nodeUpper);
}

Vector<BoundedTensor<torch::Tensor>> CROWNAnalysis::getInputBoundsForNode(unsigned nodeIndex) {
    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    std::cout << "[DEBUG] getInputBoundsForNode() - Called for node " << nodeIndex << std::endl;
    
    if (_torchModel->getDependenciesMap().exists(nodeIndex) && !_torchModel->getDependencies(nodeIndex).empty()) {
        std::cout << "[DEBUG] getInputBoundsForNode() - Node " << nodeIndex << " has " << _torchModel->getDependencies(nodeIndex).size() << " dependencies" << std::endl;
        
        for (unsigned i = 0; i < _torchModel->getDependencies(nodeIndex).size(); ++i) {
            unsigned inputIndex = _torchModel->getDependencies(nodeIndex)[i];
            std::cout << "[DEBUG] getInputBoundsForNode() - Input " << i << " is node " << inputIndex << std::endl;
            
            auto node = _nodes[inputIndex];
            unsigned outputSize = node->getOutputSize();
            torch::Tensor lower, upper;

            if (node->getNodeType() == NodeType::INPUT) {
                std::cout << "[DEBUG] getInputBoundsForNode() - Input node " << inputIndex << " is INPUT type" << std::endl;
                // Use standalone input bounds instead of Marabou bounds
                if (_torchModel->hasInputBounds()) {
                    lower = _torchModel->getInputLowerBounds();
                    upper = _torchModel->getInputUpperBounds();
                    std::cout << "[DEBUG] getInputBoundsForNode() - Using input bounds: lower=" << lower << ", upper=" << upper << std::endl;
                } else {
                    // Default bounds [0, 1] for all inputs
                    lower = torch::zeros({(long)outputSize}, torch::kFloat32);
                    upper = torch::ones({(long)outputSize}, torch::kFloat32);
                    std::cout << "[DEBUG] getInputBoundsForNode() - Using default bounds: lower=" << lower << ", upper=" << upper << std::endl;
                }
            } else {
                std::cout << "[DEBUG] getInputBoundsForNode() - Input node " << inputIndex << " is " << nodeTypeToString(node->getNodeType()) << " type" << std::endl;
                // Use IBP bounds for non-input nodes
                if (_ibpBounds.exists(inputIndex)) {
                    lower = _ibpBounds[inputIndex].lower();
                    upper = _ibpBounds[inputIndex].upper();
                    std::cout << "[DEBUG] getInputBoundsForNode() - Using IBP bounds for node " << inputIndex << ": lower=" << lower << ", upper=" << upper << std::endl;
                } else {
                    std::cout << "[DEBUG] getInputBoundsForNode() - No IBP bounds for node " << inputIndex << ", using default" << std::endl;
                    lower = torch::zeros({(long)outputSize}, torch::kFloat32);
                    upper = torch::ones({(long)outputSize}, torch::kFloat32);
                }
            }
            inputBounds.append(BoundedTensor<torch::Tensor>(lower, upper));
        }
    } else {
        std::cout << "[DEBUG] getInputBoundsForNode() - Node " << nodeIndex << " has no dependencies" << std::endl;
    }
    return inputBounds;
}


unsigned CROWNAnalysis::getOutputIndex() const {
    // Find the node with the highest index (assuming it's the output)
    unsigned outputIndex = 0;
    for (const auto& pair : _nodes) {
        if (pair.first > outputIndex) {
            outputIndex = pair.first;
        }
    }
    return outputIndex;
}

// Add helper function for A matrix addition
torch::Tensor CROWNAnalysis::addA(const torch::Tensor& A1, const torch::Tensor& A2) {
    // Handle different tensor shapes correctly
    if (A1.numel() == 0) {
        return A2;
    }
    if (A2.numel() == 0) {
        return A1;
    }
    
    // Check if tensors have compatible shapes for addition
    if (A1.sizes() == A2.sizes()) {
        return A1 + A2;
    }
    
    return A2;
}

// Add helper function for proper A matrix accumulation (following auto-LiRPA's add_bound pattern)
void CROWNAnalysis::addBound(unsigned nodeIndex, const torch::Tensor& lA, const torch::Tensor& uA) {
    log(Stringf("[DEBUG] addBound() - Called for node %d", nodeIndex));
    
    // ADD DEBUGGING FOR A MATRIX ACCUMULATION
    std::cout << "[DEBUG] addBound() - Adding A matrices for node " << nodeIndex << std::endl;
    if (lA.defined()) {
        std::cout << "[DEBUG] addBound() - New lA: " << lA << std::endl;
    }
    if (uA.defined()) {
        std::cout << "[DEBUG] addBound() - New uA: " << uA << std::endl;
    }
    
    if (_lA.exists(nodeIndex)) {
        std::cout << "[DEBUG] addBound() - Existing lA: " << _lA[nodeIndex] << std::endl;
        _lA[nodeIndex] = addA(_lA[nodeIndex], lA);
        std::cout << "[DEBUG] addBound() - Accumulated lA: " << _lA[nodeIndex] << std::endl;
    } else {
        _lA[nodeIndex] = lA;
        std::cout << "[DEBUG] addBound() - Set new lA: " << _lA[nodeIndex] << std::endl;
    }
    
    if (_uA.exists(nodeIndex)) {
        std::cout << "[DEBUG] addBound() - Existing uA: " << _uA[nodeIndex] << std::endl;
        _uA[nodeIndex] = addA(_uA[nodeIndex], uA);
        std::cout << "[DEBUG] addBound() - Accumulated uA: " << _uA[nodeIndex] << std::endl;
    } else {
        _uA[nodeIndex] = uA;
        std::cout << "[DEBUG] addBound() - Set new uA: " << _uA[nodeIndex] << std::endl;
    }
}

void CROWNAnalysis::addBias(unsigned nodeIndex, const torch::Tensor& lBias, const torch::Tensor& uBias) 
{
    log(Stringf("[DEBUG] addBias() - Called for node %d", nodeIndex));
    
    if (lBias.defined() && lBias.numel() > 0) {
        std::cout << "[DEBUG] addBias() - Adding lBias: " << lBias << std::endl;
        if (_lowerBias.exists(nodeIndex)) {
            std::cout << "[DEBUG] addBias() - Existing lowerBias: " << _lowerBias[nodeIndex] << std::endl;
            _lowerBias[nodeIndex] = _lowerBias[nodeIndex] + lBias;
        } else {
            _lowerBias[nodeIndex] = lBias;
        }
        std::cout << "[DEBUG] addBias() - New lowerBias: " << _lowerBias[nodeIndex] << std::endl;
    }
    if (uBias.defined() && uBias.numel() > 0) {
        std::cout << "[DEBUG] addBias() - Adding uBias: " << uBias << std::endl;
        if (_upperBias.exists(nodeIndex)) {
            std::cout << "[DEBUG] addBias() - Existing upperBias: " << _lowerBias[nodeIndex] << std::endl;
            _upperBias[nodeIndex] = _upperBias[nodeIndex] + uBias;
        } else {
            _upperBias[nodeIndex] = uBias;
        }
        std::cout << "[DEBUG] addBias() - New upperBias: " << _upperBias[nodeIndex] << std::endl;
    }
    
    
}

bool CROWNAnalysis::isProcessed(unsigned nodeIndex) const
{
    return _torchModel->isProcessed(nodeIndex);
}

void CROWNAnalysis::resetProcessingState() 
{
    _torchModel->resetProcessingState();
}

void CROWNAnalysis::markProcessed(unsigned nodeIndex)
{
    _torchModel->markProcessed(nodeIndex);
}

void CROWNAnalysis::log( const String &message )
{
    if ( GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
    {
        printf( "CROWNAnalysis: %s\n", message.ascii() );
    }
}


// --------------------------------------
// Public Get mothods for the unit testing -> shouldn't be needed in the actual CROWN analysis (maybe for creating tightenings to update Marabou engine)


torch::Tensor CROWNAnalysis::getIBPLowerBound(unsigned nodeIndex)
{
    if (_ibpBounds.exists(nodeIndex)) {
        return _ibpBounds[nodeIndex].lower();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getIBPUpperBound(unsigned nodeIndex)
{
    if (_ibpBounds.exists(nodeIndex)) {
        return _ibpBounds[nodeIndex].upper();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getCrownLowerBound(unsigned nodeIndex)
{
    if (_lA.exists(nodeIndex)) {
        return _lA[nodeIndex];
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getCrownUpperBound(unsigned nodeIndex)
{
    if (_uA.exists(nodeIndex)) {
        return _uA[nodeIndex];
    }
    return torch::Tensor();
}

bool CROWNAnalysis::hasIBPBounds(unsigned nodeIndex)
{
    return _ibpBounds.exists(nodeIndex);
}

bool CROWNAnalysis::hasCrownBounds(unsigned nodeIndex)
{
    return _lA.exists(nodeIndex) || _uA.exists(nodeIndex);
}

unsigned CROWNAnalysis::getNumNodes() const
{
    return _nodes.size();
}

std::shared_ptr<BoundedTorchNode> CROWNAnalysis::getNode(unsigned index) const
{
    if (_nodes.exists(index)) {
        return _nodes[index];
    }
    return nullptr;
}

unsigned CROWNAnalysis::getInputSize() const
{
    return _torchModel->getInputSize();
}

unsigned CROWNAnalysis::getOutputSize() const
{
    return _torchModel->getOutputSize();
}

// Concrete bound access methods
torch::Tensor CROWNAnalysis::getConcreteLowerBound(unsigned nodeIndex)
{
    if (_concreteBounds.exists(nodeIndex)) {
        return _concreteBounds[nodeIndex].lower();
    }
    return torch::Tensor();
}

torch::Tensor CROWNAnalysis::getConcreteUpperBound(unsigned nodeIndex)
{
    if (_concreteBounds.exists(nodeIndex)) {
        return _concreteBounds[nodeIndex].upper();
    }
    return torch::Tensor();
}

bool CROWNAnalysis::hasConcreteBounds(unsigned nodeIndex)
{
    return _concreteBounds.exists(nodeIndex);
}

// Output bound access methods
BoundedTensor<torch::Tensor> CROWNAnalysis::getOutputBounds() const 
{
    unsigned outputIndex = getOutputIndex();
    if (_concreteBounds.exists(outputIndex)) {
        return _concreteBounds[outputIndex];
    }
    return BoundedTensor<torch::Tensor>(torch::Tensor(), torch::Tensor());
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getOutputIBPBounds() const 
{
    unsigned outputIndex = getOutputIndex();
    if (_ibpBounds.exists(outputIndex)) {
        return _ibpBounds[outputIndex];
    }
    return BoundedTensor<torch::Tensor>(torch::Tensor(), torch::Tensor());
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeIBPBounds(unsigned nodeIndex) const {
    if (_ibpBounds.exists(nodeIndex)) {
        return _ibpBounds[nodeIndex];
    }
    return BoundedTensor<torch::Tensor>();
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeCrownBounds(unsigned nodeIndex) const {
    if (_lA.exists(nodeIndex) || _uA.exists(nodeIndex)) {
        torch::Tensor lA = _lA.exists(nodeIndex) ? _lA[nodeIndex] : torch::Tensor();
        torch::Tensor uA = _uA.exists(nodeIndex) ? _uA[nodeIndex] : torch::Tensor();
        return BoundedTensor<torch::Tensor>(lA, uA);
    }
    return BoundedTensor<torch::Tensor>();
}

BoundedTensor<torch::Tensor> CROWNAnalysis::getNodeConcreteBounds(unsigned nodeIndex) const {
    if (_concreteBounds.exists(nodeIndex)) {
        return _concreteBounds[nodeIndex];
    }
    return BoundedTensor<torch::Tensor>();
}

} // namespace NLR