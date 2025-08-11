// Marabou/src/nlr/BoundedConstantNode.cpp
#include "BoundedConstantNode.h"

NLR::BoundedConstantNode::BoundedConstantNode(const torch::Tensor& constantValue, const String& name)
    : _constantValue(constantValue) {
    _nodeName = name;
    _nodeIndex = 0;
    std::cout << "[BoundedConstantNode] Constructor called with constant value: " << constantValue << std::endl;
}

torch::Tensor NLR::BoundedConstantNode::forward(const torch::Tensor& input) {
    (void)input; // Suppress unused parameter warning
    std::cout << "[BoundedConstantNode] Forward called, returning constant: " << _constantValue << std::endl;
    return _constantValue;
}

void NLR::BoundedConstantNode::boundBackward(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias
) {
    std::cout << "[BoundedConstantNode::boundBackward] Starting boundBackward" << std::endl;
    std::cout << "[BoundedConstantNode::boundBackward] last_lA defined: " << last_lA.defined() << std::endl;
    std::cout << "[BoundedConstantNode::boundBackward] last_uA defined: " << last_uA.defined() << std::endl;
    std::cout << "[BoundedConstantNode::boundBackward] lbias defined: " << lbias.defined() << std::endl;
    std::cout << "[BoundedConstantNode::boundBackward] ubias defined: " << ubias.defined() << std::endl;
    
    if (last_lA.defined()) {
        std::cout << "[BoundedConstantNode::boundBackward] last_lA shape: " << last_lA.sizes() << std::endl;
    }
    if (last_uA.defined()) {
        std::cout << "[BoundedConstantNode::boundBackward] last_uA shape: " << last_uA.sizes() << std::endl;
    }
    std::cout << "[BoundedConstantNode::boundBackward] _constantValue shape: " << _constantValue.sizes() << std::endl;
    
    (void)inputBounds; // Suppress unused parameter warning
    
    // Constants add to bias, zero A matrices
    // Initialize bias tensors if they're not defined with correct size
    int output_size = _constantValue.size(0); // Use constant value size as output size
    std::cout << "[BoundedConstantNode::boundBackward] Output size from constant: " << output_size << std::endl;
    
    if (!lbias.defined()) {
        lbias = torch::zeros({output_size});
        std::cout << "[BoundedConstantNode::boundBackward] Initialized lbias to zeros with size: " << output_size << std::endl;
    }
    if (!ubias.defined()) {
        ubias = torch::zeros({output_size});
        std::cout << "[BoundedConstantNode::boundBackward] Initialized ubias to zeros with size: " << output_size << std::endl;
    }
    
    if (last_lA.defined()) {
        // Compute bias contribution: A * constant
        torch::Tensor constant_flat = _constantValue.flatten();
        std::cout << "[BoundedConstantNode::boundBackward] constant_flat shape: " << constant_flat.sizes() << std::endl;
        
        // Sum over all dimensions except the first two (batch and output dimensions)
        torch::Tensor bias_contribution = last_lA;
        for (int dim = 2; dim < last_lA.dim(); ++dim) {
            bias_contribution = bias_contribution.sum(dim, true);
        }
        bias_contribution = bias_contribution.squeeze();
        
        // Multiply by constant and sum
        torch::Tensor new_lbias = (bias_contribution * constant_flat).sum();
        std::cout << "[BoundedConstantNode::boundBackward] new_lbias: " << new_lbias << std::endl;
        
        // Add to existing bias
        lbias = lbias + new_lbias;
        std::cout << "[BoundedConstantNode::boundBackward] Updated lbias: " << lbias << std::endl;
    }
    
    if (last_uA.defined()) {
        // Compute bias contribution: A * constant
        torch::Tensor constant_flat = _constantValue.flatten();
        
        // Sum over all dimensions except the first two (batch and output dimensions)
        torch::Tensor bias_contribution = last_uA;
        for (int dim = 2; dim < last_uA.dim(); ++dim) {
            bias_contribution = bias_contribution.sum(dim, true);
        }
        bias_contribution = bias_contribution.squeeze();
        
        // Multiply by constant and sum
        torch::Tensor new_ubias = (bias_contribution * constant_flat).sum();
        std::cout << "[BoundedConstantNode::boundBackward] new_ubias: " << new_ubias << std::endl;
        
        // Add to existing bias
        ubias = ubias + new_ubias;
        std::cout << "[BoundedConstantNode::boundBackward] Updated ubias: " << ubias << std::endl;
    }
    
    // Zero A matrices for inputs (constants have no inputs)
    outputA_matrices.clear();
    std::cout << "[BoundedConstantNode::boundBackward] Cleared outputA_matrices (no inputs)" << std::endl;
    std::cout << "[BoundedConstantNode::boundBackward] Final lbias shape: " << lbias.sizes() << std::endl;
    std::cout << "[BoundedConstantNode::boundBackward] Final ubias shape: " << ubias.sizes() << std::endl;
}

BoundedTensor<torch::Tensor> NLR::BoundedConstantNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    (void)inputBounds; // Suppress unused parameter warning
    
    // Constants have exact bounds (no uncertainty)
    return BoundedTensor<torch::Tensor>(_constantValue, _constantValue);
}

void NLR::BoundedConstantNode::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& NLR::BoundedConstantNode::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

void NLR::BoundedConstantNode::setInputSize(unsigned size) {
    // Constants have no inputs, so this is a no-op
    (void)size; // Suppress unused parameter warning
}

void NLR::BoundedConstantNode::setOutputSize(unsigned size) {
    // For constants, output size is determined by the constant tensor
    // This method is mainly for interface compatibility
    (void)size; // Suppress unused parameter warning
}