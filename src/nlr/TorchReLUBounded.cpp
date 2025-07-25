#include "TorchReLUBounded.h"

namespace NLR {

TorchReLUBounded::TorchReLUBounded(const torch::nn::ReLU& reluModule) 
    : _relu_module(reluModule),
      _input_size(0),  // Will be set dynamically
      _output_size(0), // Will be set dynamically
      _bounds_computed(false),
      _work1_lower(nullptr),
      _work1_upper(nullptr),
      _work2_lower(nullptr),
      _work2_upper(nullptr) {
}

// Standard PyTorch forward pass
torch::Tensor TorchReLUBounded::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }
    
    // Apply ReLU transformation
    return _relu_module->forward(input);
}


std::tuple<Vector<std::pair<torch::Tensor, torch::Tensor>>, torch::Tensor, torch::Tensor>
TorchReLUBounded::boundBackward(const torch::Tensor& last_lA, 
                                 const torch::Tensor& last_uA,
                                 const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("TorchReLUBounded expects at least one input");
    }

    // Get input bounds from BoundedTensor
    torch::Tensor input_lower = inputBounds[0].lower();
    torch::Tensor input_upper = inputBounds[0].upper();
    
    // Get ReLU relaxation parameters 
    auto [upper_d, upper_b, lower_d, lower_b] = _backwardRelaxation(input_lower, input_upper);
    
    // Initialize A matrices with correct shape based on output size
    torch::Tensor new_lA = torch::zeros_like(last_lA);
    torch::Tensor new_uA = torch::zeros_like(last_uA);
    
    // Create masks for different neuron states
    torch::Tensor always_active = (input_lower >= 0);
    torch::Tensor always_inactive = (input_upper <= 0);
    torch::Tensor unstable = torch::logical_and(input_lower < 0, input_upper > 0);

    // Handle always active neurons: pass through with slope 1
    torch::Tensor active_mask = always_active.to(torch::kFloat);
    new_lA += last_lA * active_mask.unsqueeze(0);
    new_uA += last_uA * active_mask.unsqueeze(0);

    // NOTE: Always inactive neurons dont have to be explicitly handled since they have slope of zero, 
    // which is already handled in the creation of the new_lA and new_uA zero tensors

    // Handle unstable neurons: apply ReLU relaxation
    if (unstable.any().item<bool>()) {
        torch::Tensor unstable_mask = unstable.to(torch::kFloat);
        
        // Apply relaxation slopes to unstable neurons only
        upper_d = upper_d * unstable_mask;
        upper_b = upper_b * unstable_mask;
        lower_d = lower_d * unstable_mask;
        lower_b = lower_b * unstable_mask;

        // Sign-based multiplication following auto-LiRPA's approach
        // For lower bounds: A_new = d_pos * A_pos + d_neg * A_neg

        // Only pos elements from prev A lower
        torch::Tensor A_pos_lower = last_lA.clamp(0, std::numeric_limits<float>::max());
        // Only the neg elements from prev A lower
        torch::Tensor A_neg_lower = last_lA.clamp(std::numeric_limits<float>::lowest(), 0);
        torch::Tensor A_pos_upper = last_uA.clamp(0, std::numeric_limits<float>::max());
        torch::Tensor A_neg_upper = last_uA.clamp(std::numeric_limits<float>::lowest(), 0);

        // Apply relaxation slopes to the correct sign masks
        // Similar idea to the linear sign based propagation
        // Positive A coefficients have the relxation slops applied since they contribution to the bound
        // Negative A coeffiecnts have zero slope applied since they have no contribution to the bound
        new_lA += (lower_d.unsqueeze(0) * A_pos_lower + torch::zeros_like(A_neg_lower) * A_neg_lower);
        
        new_uA += (upper_d.unsqueeze(0) * A_pos_upper + torch::zeros_like(A_neg_upper) * A_neg_upper);
    }

    // ReLU layers don't add bias --> bias is handled by linear layers
    // But bias should have the same size as the output neurons
    unsigned output_size = last_lA.size(1);
    torch::Tensor lbias = torch::zeros({output_size});
    torch::Tensor ubias = torch::zeros({output_size});

    // Construct the retun object
    // Return A matrices for inputs, lower bias, upper bias
    Vector<std::pair<torch::Tensor, torch::Tensor>> A_matrices;
    A_matrices.append(std::make_pair(new_lA, new_uA));
    
    return std::make_tuple(A_matrices, lbias, ubias);
}

// Helper method for ReLU relaxation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TorchReLUBounded::_backwardRelaxation(const torch::Tensor& input_lower, const torch::Tensor& input_upper) {
    
    // This is from auto lirpa logic
    torch::Tensor lb_r = input_lower.clamp(/* min */ std::numeric_limits<float>::lowest(), /* max */ 0);
    torch::Tensor ub_r = input_upper.clamp(/* min */ 0, /* max */ std::numeric_limits<float>::max());
    
    // Numerical stability: ensure ub_r > lb_r (prevents division by zero)
    ub_r = torch::max(ub_r, lb_r + 1e-8);
    
    // Upper bound computation 
    torch::Tensor upper_d = ub_r / (ub_r - lb_r);
    torch::Tensor upper_b = -lb_r * upper_d;
    
    // Lower bound computation
    // For unstable neurons: if upper slope > 0.5, use slope 1, else use 0
    torch::Tensor lower_d = (upper_d > 0.5).to(torch::kFloat);
    torch::Tensor lower_b = torch::zeros_like(input_lower);
    
    return std::make_tuple(upper_d, upper_b, lower_d, lower_b);
}


// IBP (Interval Bound Propagation): Fast interval-based bound computation for ReLU
std::pair<torch::Tensor, torch::Tensor> TorchReLUBounded::computeIntervalBoundPropagation(
    const Vector<std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) 
    {
        throw std::runtime_error("TorchReLUModule expects at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.first;
    torch::Tensor inputUpperBound = inputBoundsPair.second; 

    // Set size dynamically for the CROWN 
    if( inputLowerBound.numel() > 0 )
    {
        _input_size = inputLowerBound.size(-1); // Feature dim
        _output_size = inputUpperBound.size(-1); // ReLU preserves sizing
    }
    else 
    {
        // assume some default value
        _input_size = 1;
        _output_size = 1;
    }

    // Want to apply ReLU to every neuron
    // ReLU(x) = max(0, x)
    // - For interval [lower, upper]:
    //   * If lower >= 0: ReLU([lower, upper]) = [lower, upper]
    //   * If upper <= 0: ReLU([lower, upper]) = [0, 0]  
    //   * If lower < 0 < upper: ReLU([lower, upper]) = [0, upper]
    torch::Tensor lowerBound = torch::relu(inputLowerBound);
    torch::Tensor upperBound = torch::relu(inputUpperBound);

    _lower_bound = lowerBound;
    _upper_bound = upperBound;
    _bounds_computed = true;

    return std::make_pair(lowerBound, upperBound);
}

// Variable mapping management
void TorchReLUBounded::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& TorchReLUBounded::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

// Working memory management
void TorchReLUBounded::setWorkingMemory(torch::Tensor* work1_lower,
                                         torch::Tensor* work1_upper,
                                         torch::Tensor* work2_lower,
                                         torch::Tensor* work2_upper) {
    _work1_lower = work1_lower;
    _work1_upper = work1_upper;
    _work2_lower = work2_lower;
    _work2_upper = work2_upper;
}

} // namespace NLR 