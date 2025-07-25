// Marabou/src/nlr/TorchBoundedModule.h
#ifndef __TORCH_MODULE_BOUNDED_H__
#define __TORCH_MODULE_BOUNDED_H__

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

// Structure to hold linear bounds for CROWN analysis
struct LinearBound {
    torch::Tensor lw;  // Lower weight matrix
    torch::Tensor uw;  // Upper weight matrix
    torch::Tensor lb;  // Lower bias
    torch::Tensor ub;  // Upper bias
    
    LinearBound() = default;
    
    LinearBound(const torch::Tensor& lw, const torch::Tensor& uw, 
                const torch::Tensor& lb, const torch::Tensor& ub)
        : lw(lw), uw(uw), lb(lb), ub(ub) {}
};

// Base interface for Torch bounded modules that handle CROWN and IBP analysis
// Each bounded module corresponds to a specific Torch layer type (Linear, Conv2d, etc.)
// Inherits from torch::nn::Module to take advantage of PyTorch's module system
class ITorchModuleBounded : public torch::nn::Module {
public:
    virtual ~ITorchModuleBounded() = default;
    
    // Standard PyTorch forward pass (evaluation mode without bound computation)
    // This is the standard torch::nn::Module::forward() method
    virtual torch::Tensor forward(const torch::Tensor& input) = 0;
    
    // Multi-input forward pass for modules that need multiple inputs
    virtual torch::Tensor forward(const std::vector<torch::Tensor>& inputs) {
        if (inputs.size() == 1) {
            return forward(inputs[0]);
        } else {
            throw std::runtime_error("Multi-input forward not implemented for this module");
        }
    }
    
    // IBP (Interval Bound Propagation): Fast interval-based bound computation
    // Used for initial bound estimation and fast approximate verification
    virtual std::pair<torch::Tensor, torch::Tensor> computeIntervalBoundPropagation(
        const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) = 0;
    
    // Auto-LiRPA style bound_backward method
    // Returns: A matrices for inputs, lower bias, upper bias
    // This follows auto-LiRPA's approach
    virtual std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, 
                       torch::Tensor, torch::Tensor> 
        boundBackward(const torch::Tensor& last_lA, 
                       const torch::Tensor& last_uA,
                       const Vector<BoundedTensor<torch::Tensor>>& input_bounds) = 0;
    
    // Module information
    virtual unsigned getInputSize() const = 0;
    virtual unsigned getOutputSize() const = 0;
    virtual String getModuleType() const = 0;
    
    // Variable mapping
    virtual void setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) = 0;
    virtual const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const = 0;
    
    // Bounds
    virtual torch::Tensor getLowerBound() const = 0;
    virtual torch::Tensor getUpperBound() const = 0;
    virtual bool hasComputedBounds() const = 0;
    
    virtual void setWorkingMemory(torch::Tensor* workingMemoryLower1,
                                 torch::Tensor* workingMemoryUpper1,
                                 torch::Tensor* workingMemoryLower2,
                                 torch::Tensor* workingMemoryUpper2) = 0;

};

} // namespace NLR

#endif // __TORCH_MODULE_BOUNDED_H__