#ifndef __BOUNDED_RELU_NODE_H__
#define __BOUNDED_RELU_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedReLUNode : public NLR::BoundedTorchNode {
public:
    BoundedReLUNode(const torch::nn::ReLU& reluModule, const String& name = "");
    
    // Node identification
    NLR::NodeType getNodeType() const override { return NLR::NodeType::RELU; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }
    
    // Forward pass
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // Backward bound propagation
    void boundBackward(
        const torch::Tensor& last_lA,
        const torch::Tensor& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;
    
    // IBP
    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;
    
    // Module information
    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override { return false; }
    
    // Size setters for initialization
    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;
    
    // Variable mapping
    void setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) override;
    const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const override;
    
    // Node state
    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    
    // Helper method for backward relaxation
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
        _backwardRelaxation(const torch::Tensor& input_lower, const torch::Tensor& input_upper);

private:
    std::shared_ptr<torch::nn::ReLU> _reluModule;
};

} // namespace NLR

#endif // __BOUNDED_RELU_NODE_H__