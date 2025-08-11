#include "TorchModel.h"
#include "CROWNAnalysis.h"
#include "MarabouError.h"
#include "GlobalConfiguration.h"
#include <iostream>

namespace NLR {


// CONSTRUCTOR AND INITIALIZATION


TorchModel::TorchModel(const Vector<std::shared_ptr<BoundedTorchNode>>& nodes, 
                       const Vector<Vector<Variable>>& marabouVars,
                       const Vector<unsigned>& inputIndices,
                       unsigned outputIndex,
                       const Map<unsigned, Vector<Variable>>& neuronToMarabouMap,
                       const Map<unsigned, Vector<unsigned>>& dependencies)
    : _nodes(nodes), _marabouVars(marabouVars), _inputIndices(inputIndices), 
      _outputIndex(outputIndex), _neuronToMarabouMap(neuronToMarabouMap), 
      _dependencies(dependencies), _input_size(0), _output_size(0) {
    
    log(Stringf("[TorchModel] Constructor called with %u nodes", nodes.size()));
    
    // Set input size from the first input node
    if (!inputIndices.empty() && inputIndices[0] < nodes.size()) {
        auto inputNode = nodes[inputIndices[0]];
        if (inputNode) {
            _input_size = inputNode->getOutputSize();
            log(Stringf("[TorchModel] Set input size to %u from input node %u", _input_size, inputIndices[0]));
        }
    }
    
    // Set output size from the output node
    if (outputIndex < nodes.size()) {
        auto outputNode = nodes[outputIndex];
        if (outputNode) {
            _output_size = outputNode->getOutputSize();
            log(Stringf("[TorchModel] Set output size to %u from output node %u", _output_size, outputIndex));
        }
    }
    
    // Build dependency graph
    buildDependencyGraph();
}


// LOGGING


void TorchModel::log(const String& message) const {
   if (GlobalConfiguration::NETWORK_LEVEL_REASONER_LOGGING) {
       printf("TorchModel: %s\n", message.ascii());
   }
}


// GRAPH MANAGEMENT AND TRAVERSAL


void TorchModel::buildDependencyGraph() {
   // Clear existing graph structures
   _dependents.clear();
   _degreeOut.clear();
   _degreeIn.clear();
   _processed.clear();
   
   // Initialize structures for all nodes
   for (const auto& pair : _dependencies) {
       unsigned nodeIndex = pair.first;
       _dependents[nodeIndex] = Vector<unsigned>();
       _degreeOut[nodeIndex] = 0;
       _degreeIn[nodeIndex] = 0;
       _processed[nodeIndex] = false;
   }
   
   // Also initialize nodes that might not be in dependencies
   for (unsigned i = 0; i < _nodes.size(); ++i) {
       if (!_dependents.exists(i)) {
           _dependents[i] = Vector<unsigned>();
           _degreeOut[i] = 0;
           _degreeIn[i] = 0;
           _processed[i] = false;
       }
   }
   
   // Build dependents and compute degrees
   buildDependents();
   computeDegrees();
}

void TorchModel::buildDependents() {
   // For each node, build its dependents (reverse dependencies)
   for (const auto& pair : _dependencies) {
       unsigned nodeIndex = pair.first;
       
       // For each dependency of this node
       for (unsigned inputIndex : _dependencies[nodeIndex]) {
           // This node is a dependent of its input
           _dependents[inputIndex].append(nodeIndex);
       }
   }
}

void TorchModel::computeDegrees() {
   // Compute outgoing degrees (number of dependents)
   for (const auto& pair : _dependents) {
       unsigned nodeIndex = pair.first;
       _degreeOut[nodeIndex] = _dependents[nodeIndex].size();
   }
   
   // Compute incoming degrees (number of dependencies)
   for (const auto& pair : _dependencies) {
       unsigned nodeIndex = pair.first;
       _degreeIn[nodeIndex] = _dependencies[nodeIndex].size();
   }
}

void TorchModel::resetProcessingState() {
   for (auto& pair : _processed) {
       pair.second = false;
   }
}

Vector<unsigned> TorchModel::topologicalSort() const {
   Vector<unsigned> sortedOrder;
   Queue<unsigned> queue;
   
   // Copy degree map for tracking
   Map<unsigned, unsigned> degreeIn = _degreeIn;
   
   // Initialize queue with nodes that have no incoming edges
   // Check all nodes, not just those in dependencies
   for (unsigned i = 0; i < _nodes.size(); ++i) {
       if (degreeIn.exists(i) && degreeIn[i] == 0) {
           queue.push(i);
       }
   }
   
   // Process nodes in topological order
   while (!queue.empty()) {
       unsigned current = queue.peak();
       queue.pop();
       sortedOrder.append(current);
       
       // Update degrees for dependents
       if (_dependents.exists(current)) {
           for (unsigned dependent : _dependents[current]) {
               if (degreeIn.exists(dependent)) {
                   degreeIn[dependent]--;
                   if (degreeIn[dependent] == 0) {
                       queue.push(dependent);
                   }
               }
           }
       }
   }
   
   return sortedOrder;
}

Vector<unsigned> TorchModel::getRoots() const {
   Vector<unsigned> roots;
   for (const auto& pair : _degreeIn) {
       if (pair.second == 0) {
           roots.append(pair.first);
       }
   }
   return roots;
}

Vector<unsigned> TorchModel::getLeaves() const {
   Vector<unsigned> leaves;
   for (const auto& pair : _degreeOut) {
       if (pair.second == 0) {
           leaves.append(pair.first);
       }
   }
   return leaves;
}

Vector<unsigned> TorchModel::getDependents(unsigned nodeIndex) const {
   if (_dependents.exists(nodeIndex)) {
       return _dependents[nodeIndex];
   }
   return Vector<unsigned>();
}

Vector<unsigned> TorchModel::getDependencies(unsigned nodeIndex) const {
   if (_dependencies.exists(nodeIndex)) {
       return _dependencies[nodeIndex];
   }
   return Vector<unsigned>();
}

unsigned TorchModel::getDegreeOut(unsigned nodeIndex) const {
   if (_degreeOut.exists(nodeIndex)) {
       return _degreeOut[nodeIndex];
   }
   return 0;
}

unsigned TorchModel::getDegreeIn(unsigned nodeIndex) const {
   if (_degreeIn.exists(nodeIndex)) {
       return _degreeIn[nodeIndex];
   }
   return 0;
}

bool TorchModel::isProcessed(unsigned nodeIndex) const {
   if (_processed.exists(nodeIndex)) {
       return _processed[nodeIndex];
   }
   return false;
}

void TorchModel::markProcessed(unsigned nodeIndex) {
   if (_processed.exists(nodeIndex)) {
       _processed[nodeIndex] = true;
   }
}


// NODE ACCESS AND INFORMATION


std::shared_ptr<BoundedTorchNode> TorchModel::getNode(unsigned index) const {
    if ( index < _nodes.size() ) 
    {
        return _nodes[index];
    }
    return nullptr;
}

Vector<unsigned> TorchModel::getAllNodeIndices() const {
    Vector<unsigned> indices;
    for ( unsigned i = 0; i < _nodes.size(); ++i ) 
    {
        indices.append(i);
    }
    return indices;
}

Vector<unsigned> TorchModel::getNodesByType(NodeType type) const {
    Vector<unsigned> indices;
    for ( unsigned i = 0; i < _nodes.size(); ++i ) 
    {
        if ( _nodes[i]->getNodeType() == type ) 
        {
            indices.append(i);
        }
    }
    return indices;
}


// FORWARD PASS AND COMPUTATION


torch::Tensor TorchModel::forward(unsigned nodeIndex, Map<unsigned, torch::Tensor>& activations, 
    const Map<unsigned, torch::Tensor>& inputs) {
    // Return cached result if available
    if ( activations.exists(nodeIndex) ) 
    {
        return activations[nodeIndex];
    }

    if ( nodeIndex >= _nodes.size() ) 
    {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, (String("Node index not found: ") + std::to_string(nodeIndex)).ascii());
    }

    auto& node = _nodes[nodeIndex];
    NodeType nodeType = node->getNodeType();

    // Handle based on node type
    switch (nodeType) 
    {
        case NodeType::INPUT: 
        {
        // Input nodes get their value from the inputs map
        unsigned inputIndex = nodeIndex; // Assuming input indices match node indices
        if ( !inputs.exists(inputIndex) ) 
        {
            throw MarabouError(MarabouError::TORCH_MODEL_ERROR, (String("Input index not found: ") + std::to_string(inputIndex)).ascii());
        }
        activations[nodeIndex] = inputs[inputIndex];
        return inputs[inputIndex];
        }
        
        case NodeType::CONSTANT: 
        {
            // Constants return their fixed value
            torch::Tensor result = node->forward(torch::Tensor());
            activations[nodeIndex] = result;
            return result;
        }

        case NodeType::LINEAR:
        case NodeType::RELU:
        case NodeType::RESHAPE:
        case NodeType::IDENTITY: {
            // Module nodes need to compute their inputs first
            if ( !_dependencies.exists(nodeIndex) ) 
            {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, (String("No dependencies found for node at index: ") + std::to_string(nodeIndex)).ascii());
            }
            
            Vector<unsigned> deps = _dependencies[nodeIndex];
            
            // Recursively compute all input activations
            std::vector<torch::Tensor> inputTensors;
            for ( unsigned dep : deps ) 
            {
                inputTensors.push_back(forward(dep, activations, inputs));
            }

            if ( inputTensors.empty() ) 
            {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, (String("No input tensors for node at index: ") + std::to_string(nodeIndex)).ascii());
            }

            torch::Tensor result;
            if ( inputTensors.size() == 1 ) 
            {
                result = node->forward(inputTensors[0]);
            } 
            else 
            {
                result = node->forward(inputTensors);
            }

            activations[nodeIndex] = result;
            return result;
        }
    }
    
    // This should never be reached
    throw MarabouError(MarabouError::TORCH_MODEL_ERROR, (String("Unknown node type for node index: ") + std::to_string(nodeIndex)).ascii());
}

/*
torch::Tensor TorchModel::forward(const torch::Tensor& input) {
    // Forward pass through the entire model
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> inputs_map;
    
    // Set input for input nodes
    for (unsigned inputIndex : _inputIndices) {
        inputs_map[inputIndex] = input;
    }
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs_map);
    }
    
    // Return output from the output index
    return activations[_outputIndex];
}

torch::Tensor TorchModel::forward(const Map<unsigned, torch::Tensor>& inputs) {
    // Forward pass with pre-defined inputs
    Map<unsigned, torch::Tensor> activations;
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs);
    }
    
    // Return output from the output index
    return activations[_outputIndex];
}
*/


// NEW: Forward pass that stores and returns activations for all nodes
Map<unsigned, torch::Tensor> TorchModel::forwardAndStoreActivations(const torch::Tensor& input) {
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> inputs_map;
    
    // Set input for input nodes
    for (unsigned inputIndex : _inputIndices) {
        inputs_map[inputIndex] = input;
    }
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs_map);
    }
    
    return activations;
}

Map<unsigned, torch::Tensor> TorchModel::forwardAndStoreActivations(const Map<unsigned, torch::Tensor>& inputs) {
    Map<unsigned, torch::Tensor> activations;
    
    // Forward through all nodes
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        forward(i, activations, inputs);
    }
    
    return activations;
}

// CROWN ANALYSIS INTEGRATION


/*
void TorchModel::updateConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds) {
    _concreteBounds[nodeIndex] = concreteBounds;
    log(Stringf("[TorchModel] Updated concrete bounds for node %u", nodeIndex));
}
*/




// BOUND MANAGEMENT INTERFACE

void TorchModel::setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds) {
    log(Stringf("[TorchModel] Setting input bounds"));
    
    // Store the input bounds in the TorchModel
    _inputBounds = inputBounds;
    
    log(Stringf("[TorchModel] Input bounds set with shape: %s", 
                inputBounds.lower().sizes().vec().data()));
}

// MARABOU/NLR INTEGRATION


/*
void TorchModel::obtainCurrentBoundsFromNLR() {
    if (!_layerOwner) {
        log("[TorchModel] No LayerOwner set, cannot obtain bounds from NLR");
        return;
    }
    
    log("[TorchModel] Obtaining current bounds from NLR");
    
    // Get bounds from NLR for each node
    for (unsigned i = 0; i < _nodes.size(); ++i) {
        auto node = _nodes[i];
        if (!node) continue;
        
        if (_neuronToMarabouMap.exists(i)) {
            const Vector<Variable>& variables = _neuronToMarabouMap[i];
            unsigned outputSize = node->getOutputSize();
            
            // Create tensors for NLR bounds
            torch::Tensor nlrLowerBounds = torch::zeros({(long)outputSize}, torch::kFloat64);
            torch::Tensor nlrUpperBounds = torch::zeros({(long)outputSize}, torch::kFloat64);
            
            for (unsigned j = 0; j < outputSize && j < variables.size(); ++j) {
                try {
                    // Get bounds from NLR's tableau
                    double lb = _layerOwner->getTableau()->getLowerBound(variables[j]);
                    double ub = _layerOwner->getTableau()->getUpperBound(variables[j]);
                    nlrLowerBounds[j] = lb;
                    nlrUpperBounds[j] = ub;
                } catch (const std::exception& e) {
                    log(Stringf("[TorchModel] Exception getting bounds for variable %u: %s", variables[j], e.what()));
                }
            }
            
            // Store NLR bounds for comparison
            _nlrBounds[i] = BoundedTensor<torch::Tensor>(nlrLowerBounds, nlrUpperBounds);
        }
    }
    
    log("[TorchModel] NLR bounds obtained and communicated to nodes");
}
*/

/*
void TorchModel::updateNLRWithTighterBounds() {
    if (!_layerOwner) {
        log("[TorchModel] No LayerOwner set, cannot update NLR");
        return;
    }
    
    log("[TorchModel] Checking for tighter bounds to communicate to NLR");
    
    // Check each node for tighter concrete bounds
    for (const auto& pair : _concreteBounds) {
        unsigned nodeIndex = pair.first;
        const BoundedTensor<torch::Tensor>& concreteBounds = pair.second;
        
        if (!_nlrBounds.exists(nodeIndex)) continue;
        
        const BoundedTensor<torch::Tensor>& nlrBounds = _nlrBounds[nodeIndex];
        
        // Compare concrete bounds with NLR bounds
        for (unsigned j = 0; j < concreteBounds.lower().size(0); ++j) {
            double concreteLb = concreteBounds.lower()[j].item<double>();
            double concreteUb = concreteBounds.upper()[j].item<double>();
            double nlrLb = nlrBounds.lower()[j].item<double>();
            double nlrUb = nlrBounds.upper()[j].item<double>();
            
            // If concrete bounds are tighter, communicate to NLR
            if (concreteLb > nlrLb) {
                log(Stringf("[TorchModel] Tighter lower bound for node %u, neuron %u: %f > %f",
                           nodeIndex, j, concreteLb, nlrLb));
                communicateTighterBound(nodeIndex, j, concreteLb, Tightening::LB);
            }
            
            if (concreteUb < nlrUb) {
                log(Stringf("[TorchModel] Tighter upper bound for node %u, neuron %u: %f < %f",
                           nodeIndex, j, concreteUb, nlrUb));
                communicateTighterBound(nodeIndex, j, concreteUb, Tightening::UB);
            }
        }
    }
    
    log("[TorchModel] NLR update completed");
}

void TorchModel::communicateTighterBound(unsigned nodeIndex, unsigned neuronIndex, 
                                        double bound, Tightening::BoundType type) {
    if (!_layerOwner || !_neuronToMarabouMap.exists(nodeIndex)) return;
    
    const Vector<Variable>& variables = _neuronToMarabouMap[nodeIndex];
    if (neuronIndex >= variables.size()) return;
    
    Variable variable = variables[neuronIndex];
    Tightening tightening(variable, bound, type);
    
    log(Stringf("[TorchModel] Communicating tighter bound to NLR: variable %u, bound %f, type %s",
                variable, bound, (type == Tightening::LB ? "LB" : "UB")));
    
    _layerOwner->receiveTighterBound(tightening);
}

bool TorchModel::hasTighterBounds() const {
    // Check if any concrete bounds are tighter than NLR bounds
    for (const auto& pair : _concreteBounds) {
        unsigned nodeIndex = pair.first;
        const BoundedTensor<torch::Tensor>& concreteBounds = pair.second;
        
        if (!_nlrBounds.exists(nodeIndex)) continue;
        
        const BoundedTensor<torch::Tensor>& nlrBounds = _nlrBounds[nodeIndex];
        
        for (unsigned j = 0; j < concreteBounds.lower().size(0); ++j) {
            double concreteLb = concreteBounds.lower()[j].item<double>();
            double concreteUb = concreteBounds.upper()[j].item<double>();
            double nlrLb = nlrBounds.lower()[j].item<double>();
            double nlrUb = nlrBounds.upper()[j].item<double>();
            
            if (concreteLb > nlrLb || concreteUb < nlrUb) {
                return true;
            }
        }
    }
    return false;
}
*/

// CONCRETE BOUND STORAGE METHODS

void TorchModel::setConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds) {
    validateNodeIndex(nodeIndex);
    
    _concreteBounds[nodeIndex] = concreteBounds;
    log(Stringf("[TorchModel] Set concrete bounds for node %u", nodeIndex));
}

BoundedTensor<torch::Tensor> TorchModel::getConcreteBounds(unsigned nodeIndex) const {
    validateNodeIndex(nodeIndex);
    
    if (!hasConcreteBounds(nodeIndex)) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                          Stringf("Concrete bounds not computed for node %u", nodeIndex).ascii());
    }
    
    return _concreteBounds[nodeIndex];
}

bool TorchModel::hasConcreteBounds(unsigned nodeIndex) const {
    validateNodeIndex(nodeIndex);
    return _concreteBounds.exists(nodeIndex);
}

// INPUT BOUND ACCESS METHODS

BoundedTensor<torch::Tensor> TorchModel::getInputBounds() const {
    if (!hasInputBounds()) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, "No input bounds set");
    }
    return _inputBounds;
}

bool TorchModel::hasInputBounds() const {
    return _inputBounds.lower().defined() && _inputBounds.upper().defined();
}

torch::Tensor TorchModel::getInputLowerBounds() const {
    if (!hasInputBounds()) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, "No input bounds set");
    }
    return _inputBounds.lower();
}

torch::Tensor TorchModel::getInputUpperBounds() const {
    if (!hasInputBounds()) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, "No input bounds set");
    }
    return _inputBounds.upper();
}

// ROBUST ERROR CHECKING
void TorchModel::validateNodeIndex(unsigned nodeIndex) const {
    if (nodeIndex >= _nodes.size()) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                          Stringf("Node index %u out of bounds for model with %u nodes", nodeIndex, _nodes.size()).ascii());
    }
}

} // namespace NLR


