// Marabou/src/nlr/bounded_modules/BoundedLinearNode.cpp
#include "BoundedLinearNode.h"

namespace NLR {

NLR::BoundedLinearNode::BoundedLinearNode(const torch::nn::Linear& linearModule, 
    float alpha, const String& name)
    : _linearModule(linearModule),
      _alpha(alpha) {
    
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    
    // Try to set sizes from weight matrix during construction
    if (_linearModule && _linearModule->weight.defined()) {
        auto weight = _linearModule->weight;
        setInputSize(weight.size(1));  // Weight matrix columns
        setOutputSize(weight.size(0)); // Weight matrix rows
    }
}

// Forward pass through the linear layer
torch::Tensor BoundedLinearNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically if needed
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = _linearModule->weight.size(0);
    }
    
    // Convert input and weight to float32 for consistency
    torch::Tensor inputFloat = input.to(torch::kFloat32);
    torch::Tensor weight = _linearModule->weight.to(torch::kFloat32);
    
    // Apply linear transformation: y = alpha * (W * x + b)
    torch::Tensor matmul_result = torch::matmul(inputFloat, weight.t());
    torch::Tensor alpha_scaled = _alpha * matmul_result;
    
    // Add bias if defined
    torch::Tensor result = alpha_scaled;
    if (_linearModule->bias.defined()) {
        torch::Tensor bias = _linearModule->bias.to(torch::kFloat32);
        result = result + bias;
    }
    
    return result;
}

// Auto-LiRPA style boundBackward method
void BoundedLinearNode::boundBackward(
    const torch::Tensor& last_lA, 
    const torch::Tensor& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<torch::Tensor, torch::Tensor>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {
    
    // Debug prints for input A matrices and bias
    std::cout << "\n=== BoundedLinearNode::boundBackward Debug ===" << std::endl;
    std::cout << "[LINEAR INPUT] Node: " << _nodeName << " (index " << _nodeIndex << ")" << std::endl;
    
    if (last_lA.defined()) {
        std::cout << "[LINEAR INPUT] last_lA shape: " << last_lA.sizes() << std::endl;
        std::cout << "[LINEAR INPUT] last_lA:\n" << last_lA << std::endl;
    } else {
        std::cout << "[LINEAR INPUT] last_lA: undefined" << std::endl;
    }
    
    if (last_uA.defined()) {
        std::cout << "[LINEAR INPUT] last_uA shape: " << last_uA.sizes() << std::endl;
        std::cout << "[LINEAR INPUT] last_uA:\n" << last_uA << std::endl;
    } else {
        std::cout << "[LINEAR INPUT] last_uA: undefined" << std::endl;
    }
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedLinearNode expects at least one input");
    }

    // Extract weight and bias from the linear module
    auto weight = _linearModule->weight.to(torch::kFloat32);
    auto bias = _linearModule->bias.defined() ? _linearModule->bias.to(torch::kFloat32) : torch::Tensor();
    
    // Debug print original weight and bias
    std::cout << "[LINEAR WEIGHTS] Original weight shape: " << weight.sizes() << std::endl;
    std::cout << "[LINEAR WEIGHTS] Original weight:\n" << weight << std::endl;
    
    if (bias.defined()) {
        std::cout << "[LINEAR WEIGHTS] Original bias shape: " << bias.sizes() << std::endl;
        std::cout << "[LINEAR WEIGHTS] Original bias: " << bias << std::endl;
    } else {
        std::cout << "[LINEAR WEIGHTS] Original bias: undefined" << std::endl;
    }
    
    std::cout << "[LINEAR WEIGHTS] Alpha scaling factor: " << _alpha << std::endl;
    
    // Scale weight by alpha
    weight = _alpha * weight;
    
    // Debug print scaled weight
    std::cout << "[LINEAR WEIGHTS] Scaled weight shape: " << weight.sizes() << std::endl;
    std::cout << "[LINEAR WEIGHTS] Scaled weight:\n" << weight << std::endl;
    
    // For linear layers, A matrices are computed as: A = last_A @ weight
    // where last_A represents the transformation from final output to current layer input
    // and weight represents the transformation from current layer input to current layer output
    
    // Debug output for A matrix operations
    std::string last_lA_shape = "[";
    for (int i = 0; i < last_lA.dim(); i++) {
        if (i > 0) last_lA_shape += ", ";
        last_lA_shape += std::to_string(last_lA.size(i));
    }
    last_lA_shape += "]";
    
    std::string weight_shape = "[";
    for (int i = 0; i < weight.dim(); i++) {
        if (i > 0) weight_shape += ", ";
        weight_shape += std::to_string(weight.size(i));
    }
    weight_shape += "]";
    
    std::cout << "BoundedLinearNode::boundBackward: last_lA shape: " << last_lA_shape 
              << ", weight shape: " << weight_shape << std::endl;
    
    // Compute A matrices for linear layer
    torch::Tensor lA = torch::matmul(last_lA, weight);
    torch::Tensor uA = torch::matmul(last_uA, weight);
    
    // Debug output for computed A matrices
    std::string lA_shape = "[";
    for (int i = 0; i < lA.dim(); i++) {
        if (i > 0) lA_shape += ", ";
        lA_shape += std::to_string(lA.size(i));
    }
    lA_shape += "]";
    
    std::string uA_shape = "[";
    for (int i = 0; i < uA.dim(); i++) {
        if (i > 0) uA_shape += ", ";
        uA_shape += std::to_string(uA.size(i));
    }
    uA_shape += "]";
    
    std::cout << "BoundedLinearNode::boundBackward: computed lA shape: " << lA_shape 
              << ", uA shape: " << uA_shape << std::endl;
    
    outputA_matrices.append(Pair<torch::Tensor, torch::Tensor>(lA, uA));
    
    // Compute bias contribution following auto-LiRPA's approach
    // The key insight: bias terms must be transformed to output space dimensions
    // so they can be accumulated across different layers
    if (bias.defined()) {
        if (last_lA.defined() && last_lA.numel() > 0) {
            // Auto-LiRPA approach: bias gets transformed by the A matrix to output dimensions
            // last_lA shape: [batch, final_output_size, current_layer_output_size] 
            // bias shape: [current_layer_output_size]
            // Result should have shape: [batch, final_output_size] (final output dimensions)
            
            // Transform bias using A matrix multiplication
            // For our case: last_lA: [1, final_output_size, current_layer_output_size]
            // bias: [current_layer_output_size]
            // We need to reshape bias to match the A matrix dimensions
            // last_lA: [1, final_output_size, current_layer_output_size]
            // bias: [current_layer_output_size] -> [1, current_layer_output_size, 1]
            
            // Reshape bias for matrix multiplication
            torch::Tensor bias_reshaped = bias.unsqueeze(0).unsqueeze(-1); // [1, current_layer_output_size, 1]
            
            // Matrix multiplication: [1, final_output_size, current_layer_output_size] @ [1, current_layer_output_size, 1]
            // This gives us: [1, final_output_size, 1]
            torch::Tensor transformed_lbias = torch::matmul(last_lA, bias_reshaped).squeeze(-1).squeeze(0); // [final_output_size]
            torch::Tensor transformed_ubias = torch::matmul(last_uA, bias_reshaped).squeeze(-1).squeeze(0); // [final_output_size]
            
            // Debug output
            std::cout << "BoundedLinearNode::boundBackward: bias shape: [" << bias.size(0) << "]" << std::endl;
            std::cout << "BoundedLinearNode::boundBackward: bias_reshaped shape: [" << bias_reshaped.size(0) << ", " << bias_reshaped.size(1) << ", " << bias_reshaped.size(2) << "]" << std::endl;
            std::cout << "BoundedLinearNode::boundBackward: transformed_lbias shape: [" << transformed_lbias.size(0) << "]" << std::endl;
            
            lbias = transformed_lbias;
            ubias = transformed_ubias;
        } else {
            // If no A matrix, do not accumulate bias (not valid for CROWN backward)
            lbias = torch::Tensor();
            ubias = torch::Tensor();
        }
    }
    
    // Debug prints for computed output A matrices and bias
    std::cout << "[LINEAR OUTPUT] Computed A matrices:" << std::endl;
    if (outputA_matrices.size() > 0) {
        auto& outputPair = outputA_matrices[0];
        torch::Tensor out_lA = outputPair.first();
        torch::Tensor out_uA = outputPair.second();
        
        if (out_lA.defined()) {
            std::cout << "[LINEAR OUTPUT] lA shape: " << out_lA.sizes() << std::endl;
            std::cout << "[LINEAR OUTPUT] lA:\n" << out_lA << std::endl;
        } else {
            std::cout << "[LINEAR OUTPUT] lA: undefined" << std::endl;
        }
        
        if (out_uA.defined()) {
            std::cout << "[LINEAR OUTPUT] uA shape: " << out_uA.sizes() << std::endl;
            std::cout << "[LINEAR OUTPUT] uA:\n" << out_uA << std::endl;
        } else {
            std::cout << "[LINEAR OUTPUT] uA: undefined" << std::endl;
        }
    }
    
    std::cout << "[LINEAR OUTPUT] Computed bias terms:" << std::endl;
    if (lbias.defined()) {
        std::cout << "[LINEAR OUTPUT] lbias shape: " << lbias.sizes() << std::endl;
        std::cout << "[LINEAR OUTPUT] lbias: " << lbias << std::endl;
    } else {
        std::cout << "[LINEAR OUTPUT] lbias: undefined" << std::endl;
    }
    
    if (ubias.defined()) {
        std::cout << "[LINEAR OUTPUT] ubias shape: " << ubias.sizes() << std::endl;
        std::cout << "[LINEAR OUTPUT] ubias: " << ubias << std::endl;
    } else {
        std::cout << "[LINEAR OUTPUT] ubias: undefined" << std::endl;
    }
    
    std::cout << "=== End BoundedLinearNode::boundBackward Debug ===\n" << std::endl;
}

// Enhanced IBP with size setting fallback
BoundedTensor<torch::Tensor> BoundedLinearNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("Linear module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower().to(torch::kFloat32);
    torch::Tensor inputUpperBound = inputBoundsPair.upper().to(torch::kFloat32);
    
    // Set input size from input bounds if not already set
    if (_input_size == 0 && inputLowerBound.defined()) {
        setInputSize(inputLowerBound.numel());
        std::cout << "[BoundedLinearNode::computeIntervalBoundPropagation] Set input size to " 
                  << _input_size << " from input bounds" << std::endl;
    }
    
    // Extract weight and bias
    auto weight = _linearModule->weight.to(torch::kFloat32);
    auto bias = _linearModule->bias.defined() ? _linearModule->bias.to(torch::kFloat32) : torch::Tensor();
    
    // Scale weight by alpha
    weight = _alpha * weight;
    
    // Compute IBP bounds: y = alpha * (W * x + b)
    torch::Tensor lowerBound = computeLinearIBPLowerBound(inputLowerBound, inputUpperBound);
    torch::Tensor upperBound = computeLinearIBPUpperBound(inputLowerBound, inputUpperBound);
    
    // Add bias if defined
    if (bias.defined()) {
        torch::Tensor bias_scaled = _alpha * bias;
        lowerBound = lowerBound + bias_scaled;
        upperBound = upperBound + bias_scaled;
    }
    
    // Set output size from computed bounds if not already set
    if (_output_size == 0 && lowerBound.defined()) {
        setOutputSize(lowerBound.numel());
        std::cout << "[BoundedLinearNode::computeIntervalBoundPropagation] Set output size to " 
                  << _output_size << " from computed bounds" << std::endl;
    }
    
    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Variable mapping management
void BoundedLinearNode::setNeuronToMarabouMap(const Map<unsigned, Vector<Variable>>& neuronMapping) {
    _neuronToMarabouMap = neuronMapping;
}

const Map<unsigned, Vector<Variable>>& BoundedLinearNode::getNeuronToMarabouMap() const {
    return _neuronToMarabouMap;
}

// Node information
unsigned BoundedLinearNode::getInputSize() const {
    if (_input_size > 0) {
        return _input_size;
    }
    
    // Fallback: try to infer from weight matrix
    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(1); // Input size is weight matrix columns
    }
    
    return 0;
}

unsigned BoundedLinearNode::getOutputSize() const {
    if (_output_size > 0) {
        return _output_size;
    }
    
    // Fallback: try to infer from weight matrix
    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(0); // Output size is weight matrix rows
    }
    
    return 0;
}

void BoundedLinearNode::setInputSize(unsigned size) {
    if (size > 0) {
        _input_size = size;
        std::cout << "[BoundedLinearNode::setInputSize] Set input size to " 
                  << size << " for node " << _nodeName << std::endl;
    }
}

void BoundedLinearNode::setOutputSize(unsigned size) {
    if (size > 0) {
        _output_size = size;
        std::cout << "[BoundedLinearNode::setOutputSize] Set output size to " 
                  << size << " for node " << _nodeName << std::endl;
    }
}



// IBP computation methods
torch::Tensor BoundedLinearNode::computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    auto weight = _linearModule->weight.to(torch::kFloat32);
    weight = _alpha * weight;
    
    // Convert input tensors to float32 if needed
    torch::Tensor inputLower = inputLowerBound.to(torch::kFloat32);
    torch::Tensor inputUpper = inputUpperBound.to(torch::kFloat32);
    
    // For linear layers, IBP is straightforward
    // y_lower = W_positive * x_lower + W_negative * x_upper
    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);
    
    // Fix: Use weight directly, not transpose
    torch::Tensor term1 = torch::matmul(inputLower, weight.t());
    torch::Tensor term2 = torch::matmul(inputUpper, weight.t());
    
    // For lower bound: use positive weights with lower input, negative weights with upper input
    torch::Tensor positive_contribution = torch::matmul(inputLower, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputUpper, W_negative.t());
    
    return positive_contribution + negative_contribution;
}

torch::Tensor BoundedLinearNode::computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    auto weight = _linearModule->weight.to(torch::kFloat32);
    weight = _alpha * weight;
    
    // Convert input tensors to float32 if needed
    torch::Tensor inputLower = inputLowerBound.to(torch::kFloat32);
    torch::Tensor inputUpper = inputUpperBound.to(torch::kFloat32);
    
    // For linear layers, IBP is straightforward
    // y_upper = W_positive * x_upper + W_negative * x_lower
    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);
    
    // Fix: Use weight directly, not transpose
    torch::Tensor term1 = torch::matmul(inputUpper, weight.t());
    torch::Tensor term2 = torch::matmul(inputLower, weight.t());
    
    // For upper bound: use positive weights with upper input, negative weights with lower input
    torch::Tensor positive_contribution = torch::matmul(inputUpper, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputLower, W_negative.t());
    
    return positive_contribution + negative_contribution;
}

} // namespace NLR