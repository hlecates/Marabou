#include "BoundedReshapeNode.h"

namespace NLR {

BoundedReshapeNode::BoundedReshapeNode(const Operations::ReshapeWrapper& reshape_module) 
    : _reshape_module(reshape_module) {
    _nodeName = "reshape";  // Set default name
    _nodeIndex = 0;
    _input_size = 0;  // Will be set dynamically
    _output_size = 0; // Will be set dynamically
    std::cout << "[BoundedReshapeNode] Constructor called with name: " << _nodeName << std::endl;
}

// Standard PyTorch forward pass
torch::Tensor BoundedReshapeNode::forward(const torch::Tensor& input) {
    std::cout << "[BoundedReshapeNode] Forward called with input shape: " << input.sizes() << std::endl;
    
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Reshape preserves total number of elements
    }
    
    // Use the reshape module's forward method
    torch::Tensor output = _reshape_module.forward(input);
    
    std::cout << "[BoundedReshapeNode] Forward output shape: " << output.sizes() << std::endl;
    return output;
}

// Auto-LiRPA style boundBackward method (NEW)
// Reshape operations don't change the linear relationships, just pass through A matrices
void BoundedReshapeNode::boundBackward(
    const torch::Tensor& last_lA, 
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    std::cout << "[BoundedReshapeNode::boundBackward] Starting boundBackward" << std::endl;
    std::cout << "[BoundedReshapeNode::boundBackward] last_lA defined: " << last_lA.defined() << std::endl;
    std::cout << "[BoundedReshapeNode::boundBackward] last_uA defined: " << last_uA.defined() << std::endl;
    std::cout << "[BoundedReshapeNode::boundBackward] lbias defined: " << lbias.defined() << std::endl;
    std::cout << "[BoundedReshapeNode::boundBackward] ubias defined: " << ubias.defined() << std::endl;
    std::cout << "[BoundedReshapeNode::boundBackward] inputBounds size: " << inputBounds.size() << std::endl;
    
    if (last_lA.defined()) {
        std::cout << "[BoundedReshapeNode::boundBackward] last_lA shape: " << last_lA.sizes() << std::endl;
    }
    if (last_uA.defined()) {
        std::cout << "[BoundedReshapeNode::boundBackward] last_uA shape: " << last_uA.sizes() << std::endl;
    }
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReshapeNode expects at least one input");
    }
    
    // Reshape operations don't change the linear relationships
    // Simply pass through the A matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(last_lA, last_uA));
    std::cout << "[BoundedReshapeNode::boundBackward] Added A matrices to output" << std::endl;
    
    // Reshape operations don't add bias - initialize to zeros with correct size
    if (last_lA.defined()) {
        // Get the output size from the A matrix
        int output_size = last_lA.size(1); // Second dimension is output size
        std::cout << "[BoundedReshapeNode::boundBackward] Output size from A matrix: " << output_size << std::endl;
        
        if (!lbias.defined()) {
            lbias = torch::zeros({output_size});
            std::cout << "[BoundedReshapeNode::boundBackward] Initialized lbias to zeros with size: " << output_size << std::endl;
        }
    } else {
        if (!lbias.defined()) {
            lbias = torch::zeros({1});
            std::cout << "[BoundedReshapeNode::boundBackward] Initialized lbias to zeros (fallback)" << std::endl;
        }
    }
    
    if (last_uA.defined()) {
        // Get the output size from the A matrix
        int output_size = last_uA.size(1); // Second dimension is output size
        std::cout << "[BoundedReshapeNode::boundBackward] Output size from A matrix: " << output_size << std::endl;
        
        if (!ubias.defined()) {
            ubias = torch::zeros({output_size});
            std::cout << "[BoundedReshapeNode::boundBackward] Initialized ubias to zeros with size: " << output_size << std::endl;
        }
    } else {
        if (!ubias.defined()) {
            ubias = torch::zeros({1});
            std::cout << "[BoundedReshapeNode::boundBackward] Initialized ubias to zeros (fallback)" << std::endl;
        }
    }
    
    std::cout << "[BoundedReshapeNode::boundBackward] Final lbias shape: " << lbias.sizes() << std::endl;
    std::cout << "[BoundedReshapeNode::boundBackward] Final ubias shape: " << ubias.sizes() << std::endl;
}



// IBP (Interval Bound Propagation): Fast interval-based bound computation for Reshape
BoundedTensor<torch::Tensor> BoundedReshapeNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    std::cout << "[BoundedReshapeNode::computeIntervalBoundPropagation] Starting IBP computation" << std::endl;
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Reshape module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    std::cout << "[BoundedReshapeNode::computeIntervalBoundPropagation] Input lower shape: " << inputLowerBound.sizes() << std::endl;
    std::cout << "[BoundedReshapeNode::computeIntervalBoundPropagation] Input upper shape: " << inputUpperBound.sizes() << std::endl;
    
    // Apply reshape to both lower and upper bounds
    torch::Tensor reshapedLower = _reshape_module.forward(inputLowerBound);
    torch::Tensor reshapedUpper = _reshape_module.forward(inputUpperBound);
    
    std::cout << "[BoundedReshapeNode::computeIntervalBoundPropagation] Reshaped lower shape: " << reshapedLower.sizes() << std::endl;
    std::cout << "[BoundedReshapeNode::computeIntervalBoundPropagation] Reshaped upper shape: " << reshapedUpper.sizes() << std::endl;
    
    return BoundedTensor<torch::Tensor>(reshapedLower, reshapedUpper);
}

// Variable mapping
void BoundedReshapeNode::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& BoundedReshapeNode::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

void NLR::BoundedReshapeNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedReshapeNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR 