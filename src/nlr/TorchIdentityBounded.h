#ifndef __TORCH_IDENTITY_BOUNDED_H__
#define __TORCH_IDENTITY_BOUNDED_H__

#include "TorchModuleBounded.h"

namespace NLR {

class TorchIdentityBounded : public ITorchModuleBounded {
public:
    TorchIdentityBounded(const torch::nn::Identity& identity_module);
    
    // Standard PyTorch forward pass
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // CROWN Backward Mode
    LinearBound computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha, 
                                               const torch::Tensor& lastUpperAlpha,
                                               const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputs);
    
    // IBP
    std::pair<torch::Tensor, torch::Tensor> computeIntervalBoundPropagation(
        const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputs) override;
    
    // Auto-LiRPA style boundBackward method (NEW)
    // Returns: A matrices for inputs, lower bias, upper bias
    std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, 
               torch::Tensor, torch::Tensor> 
        boundBackward(const torch::Tensor& last_lA, 
                       const torch::Tensor& last_uA,
                       const Vector<BoundedTensor<torch::Tensor>>& input_bounds) override;
    
    // Module information
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    String getModuleType() const override { return "Identity"; }
    
    // Variable mapping
    void setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& mapping) override;
    const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const override;
    
    // Bound state management
    torch::Tensor getLowerBound() const override { return _lower_bound; }
    torch::Tensor getUpperBound() const override { return _upper_bound; }
    bool hasComputedBounds() const override { return _bounds_computed; }
    
    // Working memory management
    void setWorkingMemory(torch::Tensor* work1_lower,
                         torch::Tensor* work1_upper,
                         torch::Tensor* work2_lower,
                         torch::Tensor* work2_upper) override;

private:
    torch::nn::Identity _identity_module;
    unsigned _input_size;
    unsigned _output_size;
    bool _bounds_computed;
    
    // Working memory pointers
    torch::Tensor* _work1_lower;
    torch::Tensor* _work1_upper;
    torch::Tensor* _work2_lower;
    torch::Tensor* _work2_upper;
    
    // Variable mapping
    Map<unsigned, Vector<Variable>> _neuronToMarabouMap;
    
    // Bound storage
    torch::Tensor _lower_bound;
    torch::Tensor _upper_bound;
};

} // namespace NLR

#endif // __TORCH_IDENTITY_BOUNDED_H__