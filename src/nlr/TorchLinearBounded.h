// Marabou/src/nlr/bounded_modules/TorchLinearModule.h
#ifndef __BOUNDED_LINEAR_MODULE_H__
#define __BOUNDED_LINEAR_MODULE_H__

#include "TorchModuleBounded.h"

namespace NLR {

class TorchLinearModule : public ITorchModuleBounded {
public:
    TorchLinearModule(const torch::nn::Linear& linearModule, float alpha = 1.0f);
    
    // Standard PyTorch forward pass (inherited from torch::nn::Module)
    torch::Tensor forward(const torch::Tensor& input) override;
    
    // IBP (Interval Bound Propagation): Fast interval-based bound computation for linear layer
    // Used for initial bound estimation and fast approximate verification
    std::pair<torch::Tensor, torch::Tensor> computeIntervalBoundPropagation(
        const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) override;
    
    // Auto-LiRPA style bound_backward method (NEW)
    // Returns: A matrices for inputs, lower bias, upper bias
    std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, 
               torch::Tensor, torch::Tensor> 
        boundBackward(const torch::Tensor& last_lA, 
                       const torch::Tensor& last_uA,
                       const Vector<BoundedTensor<torch::Tensor>>& input_bounds) override;
    
    // Module information
    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    String getModuleType() const override { return "Linear"; }
    
    // Variable mapping (using integer indices)
    void setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) override;
    const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const override;
    
    // Bound state management
    torch::Tensor getLowerBound() const override;
    torch::Tensor getUpperBound() const override;
    bool hasComputedBounds() const override { return _bounds_computed; }
    
    // Working memory management
    void setWorkingMemory(torch::Tensor* workingMemoryLower1,
                         torch::Tensor* workingMemoryUpper1,
                         torch::Tensor* workingMemoryLower2,
                         torch::Tensor* workingMemoryUpper2) override;

    // IBP helper methods for linear layer bound computation (public for testing)
    torch::Tensor computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, 
                                            const torch::Tensor& inputUpperBound);
    torch::Tensor computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, 
                                            const torch::Tensor& inputUpperBound);
    

private:
    // Linear module (weights and bias are already preprocessed during ONNX conversion)
    torch::nn::Linear _linear_module;
    
    // Alpha scaling factor for forward pass
    float _alpha;
    
    // Module information
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

#endif // __BOUNDED_LINEAR_MODULE_H__