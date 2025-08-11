#include "BoundedReLUNode.h"

namespace NLR {

NLR::BoundedReLUNode::BoundedReLUNode(const torch::nn::ReLU& reluModule, const String& name)
    : _reluModule(std::make_shared<torch::nn::ReLU>(reluModule)) {
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
}

// Forward pass through the ReLU layer
torch::Tensor BoundedReLUNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }
    
    // Apply ReLU transformation
    return (*_reluModule)(input);
}

// Auto-LiRPA style boundBackward method
void BoundedReLUNode::boundBackward(
    const torch::Tensor& last_lA, 
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    // Debug prints for input A matrices and bias
    std::cout << "\n=== BoundedReLUNode::boundBackward Debug ===" << std::endl;
    std::cout << "[ReLU INPUT] Node: " << _nodeName << " (index " << _nodeIndex << ")" << std::endl;
    
    if (last_lA.defined()) {
        std::cout << "[ReLU INPUT] last_lA shape: " << last_lA.sizes() << std::endl;
        std::cout << "[ReLU INPUT] last_lA:\n" << last_lA << std::endl;
    } else {
        std::cout << "[ReLU INPUT] last_lA: undefined" << std::endl;
    }
    
    if (last_uA.defined()) {
        std::cout << "[ReLU INPUT] last_uA shape: " << last_uA.sizes() << std::endl;
        std::cout << "[ReLU INPUT] last_uA:\n" << last_uA << std::endl;
    } else {
        std::cout << "[ReLU INPUT] last_uA: undefined" << std::endl;
    }
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReLUNode expects at least one input");
    }

    const auto& inputBound = inputBounds[0];
    torch::Tensor input_lower = inputBound.lower();
    torch::Tensor input_upper = inputBound.upper();
    
    // Compute the relaxation parameters (slopes and biases)
    auto [d_lower, d_upper, bias_lower, bias_upper] = 
        _backwardRelaxation(input_lower, input_upper);
    
    // Extract slopes from diagonal matrices
    torch::Tensor alpha_lower, alpha_upper;
    if (d_lower.dim() == 2) {
        // Extract diagonal elements
        alpha_lower = torch::diag(d_lower);
        alpha_upper = torch::diag(d_upper);
    } else {
        // Already 1D
        alpha_lower = d_lower;
        alpha_upper = d_upper;
    }
    
    // Compute vector slopes/biases once
    torch::Tensor aL = alpha_lower;                    // (n,)
    torch::Tensor aU = alpha_upper;                    // (n,)
    torch::Tensor bL = bias_lower;                     // (n,)
    torch::Tensor bU = bias_upper;                     // (n,)

    // Debug: print relaxed slope values
    std::cout << "[ReLU RELAX] aL (lower slopes) shape: " << aL.sizes() << std::endl;
    std::cout << "[ReLU RELAX] aL (lower slopes):\n" << aL << std::endl;
    std::cout << "[ReLU RELAX] aU (upper slopes) shape: " << aU.sizes() << std::endl;
    std::cout << "[ReLU RELAX] aU (upper slopes):\n" << aU << std::endl;

    auto expand_like = [](const torch::Tensor &v, const torch::Tensor &A) {
        return v.unsqueeze(0).expand_as(A);            // broadcast to A's shape (spec, n)
    };

    torch::Tensor new_lA, new_uA;
    // Removed local redeclaration of lbias and ubias to use reference params

    if (last_lA.defined()) {
        std::cout << "[ReLU RELAX] prev lA shape: " << last_lA.sizes() << std::endl;
        std::cout << "[ReLU RELAX] prev lA:\n" << last_lA << std::endl;
        auto Apos = torch::clamp_min(last_lA, 0);      // A⁺
        auto Aneg = torch::clamp_max(last_lA, 0);      // A⁻
        new_lA = Apos * expand_like(aL, last_lA) + Aneg * expand_like(aU, last_lA);
        std::cout << "[ReLU RELAX] new_lA (after applying slopes) shape: " << new_lA.sizes() << std::endl;
        std::cout << "[ReLU RELAX] new_lA (after applying slopes):\n" << new_lA << std::endl;
        lbias = (Apos * expand_like(bL, last_lA) + Aneg * expand_like(bU, last_lA)).sum(-1);
    }

    if (last_uA.defined()) {
        std::cout << "[ReLU RELAX] prev uA shape: " << last_uA.sizes() << std::endl;
        std::cout << "[ReLU RELAX] prev uA:\n" << last_uA << std::endl;
        auto Apos = torch::clamp_min(last_uA, 0);      // A⁺
        auto Aneg = torch::clamp_max(last_uA, 0);      // A⁻
        new_uA = Apos * expand_like(aU, last_uA) + Aneg * expand_like(aL, last_uA);
        std::cout << "[ReLU RELAX] new_uA (after applying slopes) shape: " << new_uA.sizes() << std::endl;
        std::cout << "[ReLU RELAX] new_uA (after applying slopes):\n" << new_uA << std::endl;
        ubias = (Apos * expand_like(bU, last_uA) + Aneg * expand_like(bL, last_uA)).sum(-1);
    }

    
    // Set output A matrices
    outputA_matrices.clear();
    outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(new_lA, new_uA));
    /*
    // Debug prints for computed output A matrices
    std::cout << "[ReLU OUTPUT] Computed A matrices:" << std::endl;
    if (new_lA.defined()) {
        std::cout << "[ReLU OUTPUT] new_lA shape: " << new_lA.sizes() << std::endl;
        std::cout << "[ReLU OUTPUT] new_lA:\n" << new_lA << std::endl;
    } else {
        std::cout << "[ReLU OUTPUT] new_lA: undefined" << std::endl;
    }
    
    if (new_uA.defined()) {
        std::cout << "[ReLU OUTPUT] new_uA shape: " << new_uA.sizes() << std::endl;
        std::cout << "[ReLU OUTPUT] new_uA:\n" << new_uA << std::endl;
    } else {
        std::cout << "[ReLU OUTPUT] new_uA: undefined" << std::endl;
    }
    
    // Bias computation - transform biases using A matrices
    if (last_lA.defined() && bias_lower.defined()) {
        // For lower bound bias computation:
        if (last_lA.defined() && bias_lower.defined()) {
            torch::Tensor pos_mask = last_lA >= 0;
            torch::Tensor neg_mask = last_lA < 0;
            
            // Expand biases to match A matrix dimensions
            torch::Tensor bias_lower_expanded = bias_lower.unsqueeze(0).expand_as(last_lA);
            torch::Tensor bias_upper_expanded = bias_upper.unsqueeze(0).expand_as(last_lA);
            
            // Use the actual relaxation bias terms, not slope * bound
            torch::Tensor selected_biases = torch::where(
                pos_mask, 
                bias_lower_expanded,  // Positive A → use lower bound relaxation bias
                bias_upper_expanded   // Negative A → use upper bound relaxation bias
            );
            
            // Sum the bias contributions
            //lbias = torch::sum(last_lA * selected_biases, -1);
            
            lbias = (last_lA * selected_biases).sum(-1);
        }
    } else {
        lbias = torch::zeros({last_lA.size(0)});
    }
    
    if (last_uA.defined() && bias_upper.defined()) {
        // For upper bound bias computation:
        if (last_uA.defined() && bias_upper.defined()) {
            torch::Tensor pos_mask = last_uA >= 0;
            torch::Tensor neg_mask = last_uA < 0;
            
            // Expand biases to match A matrix dimensions  
            torch::Tensor bias_lower_expanded = bias_lower.unsqueeze(0).expand_as(last_uA);
            torch::Tensor bias_upper_expanded = bias_upper.unsqueeze(0).expand_as(last_uA);
            
            // For upper bound: reverse the bias selection
            torch::Tensor selected_biases = torch::where(
                pos_mask,
                bias_upper_expanded,  // Positive A → use upper bound relaxation bias
                bias_lower_expanded   // Negative A → use lower bound relaxation bias
            );
            
            // Sum the bias contributions
            //ubias = torch::sum(last_uA * selected_biases, -1);
            
            
            ubias = (last_uA * selected_biases).sum(-1);
        }
    } else {
        ubias = torch::zeros({last_uA.size(0)});
    }
    */
    
    // Debug prints for computed bias terms
    std::cout << "[ReLU OUTPUT] Computed bias terms:" << std::endl;
    if (lbias.defined()) {
        std::cout << "[ReLU OUTPUT] lbias shape: " << lbias.sizes() << std::endl;
        std::cout << "[ReLU OUTPUT] lbias: " << lbias << std::endl;
    } else {
        std::cout << "[ReLU OUTPUT] lbias: undefined" << std::endl;
    }
    
    if (ubias.defined()) {
        std::cout << "[ReLU OUTPUT] ubias shape: " << ubias.sizes() << std::endl;
        std::cout << "[ReLU OUTPUT] ubias: " << ubias << std::endl;
    } else {
        std::cout << "[ReLU OUTPUT] ubias: undefined" << std::endl;
    }
    
    std::cout << "=== End BoundedReLUNode::boundBackward Debug ===\n" << std::endl;
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for ReLU
BoundedTensor<torch::Tensor> BoundedReLUNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("ReLU module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    // Set input size from the input tensor during IBP
    if (_input_size == 0 && inputLowerBound.defined()) {
        _input_size = inputLowerBound.numel();
    }
    
    // ReLU: y = max(0, x)
    torch::Tensor lowerBound = torch::clamp_min(inputLowerBound, 0);  // max(0, lower)
    torch::Tensor upperBound = torch::clamp_min(inputUpperBound, 0);  // max(0, upper)
    
    // Set output size from the computed bounds during IBP
    if (_output_size == 0 && lowerBound.defined()) {
        _output_size = lowerBound.numel();
    }
    
    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Variable mapping management
void BoundedReLUNode::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& BoundedReLUNode::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

// Node information
unsigned BoundedReLUNode::getInputSize() const {
    return _input_size;
}

unsigned BoundedReLUNode::getOutputSize() const {
    // If output size is not set, try to infer from input size
    if (_output_size == 0 && _input_size > 0) {
        return _input_size; // ReLU preserves input size
    }
    // If still 0, return a reasonable default for testing
    if (_output_size == 0) {
        return 2; // Default size for testing
    }
    return _output_size;
}

void BoundedReLUNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedReLUNode::setOutputSize(unsigned size) {
    _output_size = size;
}



// Helper method for backward relaxation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
NLR::BoundedReLUNode::_backwardRelaxation(const torch::Tensor& input_lower, const torch::Tensor& input_upper) {
    // std::cout << "[DEBUG] _backwardRelaxation called" << std::endl;
    // std::cout << "[DEBUG] input_lower shape: " << input_lower.sizes() << std::endl;
    // std::cout << "[DEBUG] input_upper shape: " << input_upper.sizes() << std::endl;
    // std::cout << "[DEBUG] input_lower values: " << input_lower << std::endl;
    // std::cout << "[DEBUG] input_upper values: " << input_upper << std::endl;
    
    // Initialize slopes and biases for the three ReLU cases
    torch::Tensor slopes_lower = torch::zeros_like(input_lower);  // Lower bound slopes
    torch::Tensor slopes_upper = torch::zeros_like(input_upper);  // Upper bound slopes  
    torch::Tensor bias_lower = torch::zeros_like(input_lower);    // Lower bound biases
    torch::Tensor bias_upper = torch::zeros_like(input_upper);    // Upper bound biases
    
    // Case 1: input_lower >= 0 (always active) - ReLU passes through
    auto always_active_mask = input_lower >= 0;
    slopes_lower = torch::where(always_active_mask, torch::ones_like(slopes_lower), slopes_lower);
    slopes_upper = torch::where(always_active_mask, torch::ones_like(slopes_upper), slopes_upper);
    // Bias is 0 for always active neurons
    bias_lower = torch::where(always_active_mask, torch::zeros_like(bias_lower), bias_lower);
    bias_upper = torch::where(always_active_mask, torch::zeros_like(bias_upper), bias_upper);
    
    // Case 2: input_upper <= 0 (always inactive) - ReLU outputs 0
    auto always_inactive_mask = input_upper <= 0;
    slopes_lower = torch::where(always_inactive_mask, torch::zeros_like(slopes_lower), slopes_lower);
    slopes_upper = torch::where(always_inactive_mask, torch::zeros_like(slopes_upper), slopes_upper);
    // Bias is 0 for always inactive neurons
    bias_lower = torch::where(always_inactive_mask, torch::zeros_like(bias_lower), bias_lower);
    bias_upper = torch::where(always_inactive_mask, torch::zeros_like(bias_upper), bias_upper);
    
    // Case 3: input_lower < 0 < input_upper (uncertain) - need relaxation
    auto uncertain_mask = (input_lower < 0) & (input_upper > 0);
    if (uncertain_mask.any().item<bool>()) {
        // Apply ReLU function to input bounds for triangle relaxation
        torch::Tensor lb_r = input_lower;
        torch::Tensor ub_r = input_upper;
        
        // Add small epsilon for numerical stability (following auto-LiRPA)
        ub_r = torch::max(ub_r, lb_r + 1e-8);
        
        // UPPER BOUND: Triangle relaxation (CROWN) - same as auto-LiRPA
        // Upper bound slope: u/(u-l) for uncertain neurons
        torch::Tensor upper_slope = ub_r / (ub_r - lb_r);
        slopes_upper = torch::where(uncertain_mask, upper_slope, slopes_upper);
        
        // Upper bound bias: -upper_slope * lb_r
        torch::Tensor lb = input_lower;
        torch::Tensor upper_bias = -upper_slope * lb;
        bias_upper = torch::where(uncertain_mask, upper_bias, bias_upper);
        
        // LOWER BOUND: Use adaptive approach (CROWN-IBP)
        // If upper_slope > 0.5, use slope = 1, otherwise use slope = 0
        auto adaptive_mask = upper_slope > 0.5;
        torch::Tensor lower_slope = torch::where(
            adaptive_mask,
            torch::ones_like(input_lower),  // slope = 1
            torch::zeros_like(input_lower)  // slope = 0
        );
        slopes_lower = torch::where(uncertain_mask, lower_slope, slopes_lower);
        
        // Lower bound bias: always 0 for CROWN-IBP
        bias_lower = torch::where(uncertain_mask, torch::zeros_like(bias_lower), bias_lower);
    }
    
    // Return slope vectors (flattened) and bias vectors directly to avoid constructing diagonals
    torch::Tensor d_lower = slopes_lower.flatten();
    torch::Tensor d_upper = slopes_upper.flatten();
    torch::Tensor b_lower = bias_lower.flatten();
    torch::Tensor b_upper = bias_upper.flatten();
    
    return std::make_tuple(d_lower, d_upper, b_lower, b_upper);
}

} // namespace NLR 