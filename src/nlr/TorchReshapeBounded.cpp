#include "TorchReshapeBounded.h"

namespace NLR {

TorchReshapeBounded::TorchReshapeBounded(const std::shared_ptr<Operations::ReshapeWrapper>& reshape_module) 
    : _reshape_module(reshape_module),
      _input_size(0),  // Will be set dynamically
      _output_size(0), // Will be set dynamically
      _bounds_computed(false),
      _work1_lower(nullptr),
      _work1_upper(nullptr),
      _work2_lower(nullptr),
      _work2_upper(nullptr) {
}

// Standard PyTorch forward pass
torch::Tensor TorchReshapeBounded::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel(); // Reshape preserves element count
    }
    
    // Apply reshape operation
    return _reshape_module->forward(input);
}

// Auto-LiRPA style boundBackward method (NEW)
// Reshape operations don't change the linear relationships, just pass through A matrices
std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, torch::Tensor, torch::Tensor>
TorchReshapeBounded::boundBackward(const torch::Tensor& last_lA, 
                                     const torch::Tensor& last_uA,
                                     const Vector<BoundedTensor<torch::Tensor>>& input_bounds) {
    
    if (input_bounds.size() < 1) {
        throw std::runtime_error("TorchReshapeBounded expects at least one input");
    }

    // Reshape operations don't change the linear relationships
    // Simply pass through the A matrices
    Vector<std::pair<torch::Tensor, torch::Tensor>> A_matrices;
    A_matrices.append(std::make_pair(last_lA, last_uA));
    
    // Reshape operations don't add bias
    torch::Tensor lbias = torch::zeros({1});
    torch::Tensor ubias = torch::zeros({1});
    
    return std::make_tuple(A_matrices, lbias, ubias);
}

// CROWN Backward Mode: Propagate bounds backward through Reshape
LinearBound TorchReshapeBounded::computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha, 
                                                                  const torch::Tensor& lastUpperAlpha,
                                                                  const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    (void)inputBounds; // Suppress unused parameter warning
    // Reshape layer just passes through the alpha matrices (reshape doesn't change the values)
    return LinearBound(lastLowerAlpha, torch::zeros({lastLowerAlpha.size(0)}), 
                      lastUpperAlpha, torch::zeros({lastUpperAlpha.size(0)}));
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for Reshape
std::pair<torch::Tensor, torch::Tensor> TorchReshapeBounded::computeIntervalBoundPropagation(
    const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Reshape module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.first;
    torch::Tensor inputUpperBound = inputBoundsPair.second;
    
    // For now, just pass through the bounds without reshaping
    // TODO: Implement proper reshape logic using the shape tensor
    return std::make_pair(inputLowerBound, inputUpperBound);
}

// Variable mapping
void TorchReshapeBounded::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& TorchReshapeBounded::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

// Bound state management
torch::Tensor TorchReshapeBounded::getLowerBound() const {
    if (!_bounds_computed) {
        throw std::runtime_error("Bounds not computed yet");
    }
    return _lower_bound;
}

torch::Tensor TorchReshapeBounded::getUpperBound() const {
    if (!_bounds_computed) {
        throw std::runtime_error("Bounds not computed yet");
    }
    return _upper_bound;
}

// Working memory management
void TorchReshapeBounded::setWorkingMemory(torch::Tensor* workingMemoryLower1,
                                          torch::Tensor* workingMemoryUpper1,
                                          torch::Tensor* workingMemoryLower2,
                                          torch::Tensor* workingMemoryUpper2) {
    _work1_lower = workingMemoryLower1;
    _work1_upper = workingMemoryUpper1;
    _work2_lower = workingMemoryLower2;
    _work2_upper = workingMemoryUpper2;
}

} // namespace NLR 