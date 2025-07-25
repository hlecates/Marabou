#include "TorchIdentityBounded.h"

namespace NLR {

TorchIdentityBounded::TorchIdentityBounded(const torch::nn::Identity& identityModule) 
    : _identity_module(identityModule),
      _input_size(0),  // Will be set dynamically
      _output_size(0), // Will be set dynamically
      _bounds_computed(false),
      _work1_lower(nullptr),
      _work1_upper(nullptr),
      _work2_lower(nullptr),
      _work2_upper(nullptr) {
}

// Standard PyTorch forward pass
torch::Tensor TorchIdentityBounded::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }
    
    // Apply identity transformation
    return _identity_module->forward(input);
}

// Auto-LiRPA style boundBackward method (NEW)
// Identity layers simply pass through the A matrices without modification
std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, torch::Tensor, torch::Tensor>
TorchIdentityBounded::boundBackward(const torch::Tensor& last_lA, 
                                     const torch::Tensor& last_uA,
                                     const Vector<BoundedTensor<torch::Tensor>>& input_bounds) {
    
    if (input_bounds.size() < 1) {
        throw std::runtime_error("TorchIdentityBounded expects at least one input");
    }

    // Identity layers don't modify the linear relationships
    // Simply pass through the A matrices
    Vector<std::pair<torch::Tensor, torch::Tensor>> A_matrices;
    A_matrices.append(std::make_pair(last_lA, last_uA));
    
    // Identity layers don't add bias
    torch::Tensor lbias = torch::zeros({1});
    torch::Tensor ubias = torch::zeros({1});
    
    return std::make_tuple(A_matrices, lbias, ubias);
}

// CROWN Backward Mode: Propagate bounds backward through Identity
LinearBound TorchIdentityBounded::computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha, 
                                                                  const torch::Tensor& lastUpperAlpha,
                                                                  const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    (void)inputBounds; // Suppress unused parameter warning
    // Identity layer just passes through the alpha matrices
    return LinearBound(lastLowerAlpha, torch::zeros({lastLowerAlpha.size(0)}), 
                      lastUpperAlpha, torch::zeros({lastUpperAlpha.size(0)}));
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for Identity
std::pair<torch::Tensor, torch::Tensor> TorchIdentityBounded::computeIntervalBoundPropagation(
    const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Identity module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.first;
    torch::Tensor inputUpperBound = inputBoundsPair.second;
    
    // Identity layer just passes through the bounds
    return std::make_pair(inputLowerBound, inputUpperBound);
}

// Variable mapping management
void TorchIdentityBounded::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& TorchIdentityBounded::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

// Working memory management
void TorchIdentityBounded::setWorkingMemory(torch::Tensor* work1_lower,
                                            torch::Tensor* work1_upper,
                                            torch::Tensor* work2_lower,
                                            torch::Tensor* work2_upper) {
    _work1_lower = work1_lower;
    _work1_upper = work1_upper;
    _work2_lower = work2_lower;
    _work2_upper = work2_upper;
}

} // namespace NLR 