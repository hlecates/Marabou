// Marabou/src/nlr/BoundedTorchNode.h
#ifndef __BOUNDED_TORCH_NODE_H__
#define __BOUNDED_TORCH_NODE_H__

#include "Map.h"
#include "Vector.h"
#include "MString.h"
#include "InputQueryBuilder.h"
#include "BoundedTensor.h"

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

// Redefine Warning macro for CVC4 compatibility
#ifndef Warning
#define Warning (! ::CVC4::WarningChannel.isOn()) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

namespace NLR {

enum class NodeType { INPUT, CONSTANT, LINEAR, RELU, RESHAPE, IDENTITY };

class BoundedTorchNode : public torch::nn::Module {
public:
    virtual ~BoundedTorchNode() = default;
    
    // Node ID getters
    virtual NodeType getNodeType() const = 0;
    virtual String getNodeName() const = 0;
    virtual unsigned getNodeIndex() const = 0;

    // Standard PyTorch forward pass (evaluation mode without bound computation)
    virtual torch::Tensor forward(const torch::Tensor& input) = 0;
    virtual torch::Tensor forward(const std::vector<torch::Tensor>& inputs) {
        if (inputs.size() == 1) {
            return forward(inputs[0]);
        } else {
            throw std::runtime_error("Multi-input forward not implemented for this module");
        }
    }
    
    virtual BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) = 0;
    
    virtual void boundBackward(
        const torch::Tensor& last_lA,
        const torch::Tensor& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) = 0;

    // Node Info
    virtual unsigned getInputSize() const = 0;
    virtual unsigned getOutputSize() const = 0;
    virtual bool isPerturbed() const = 0;
    
    // Size setters for initialization
    virtual void setInputSize(unsigned size) = 0;
    virtual void setOutputSize(unsigned size) = 0;
    
    // Variable mapping
    virtual void setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) = 0;
    virtual const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const = 0;
    
    // Node state
    virtual void setNodeIndex(unsigned index) = 0;
    virtual void setNodeName(const String& name) = 0;

protected:
    unsigned _nodeIndex;
    String _nodeName;
    unsigned _input_size;
    unsigned _output_size;
    Map<unsigned, Vector<Variable>> _neuronToMarabouMap;
};

} // namespace NLR

#endif // __TORCH_MODULE_BOUNDED_H__