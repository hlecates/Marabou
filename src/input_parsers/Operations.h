#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include <torch/torch.h>

namespace Operations {

class ReshapeImpl : public torch::nn::Module {
public:
    ReshapeImpl() {}
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& shape_tensor);
};
TORCH_MODULE(Reshape);

class ReshapeWrapper : public torch::nn::Module {
private:
    torch::Tensor shape_tensor;
public:
    ReshapeWrapper(torch::Tensor shape) : shape_tensor(shape) {
        register_buffer("shape", this->shape_tensor);
    }
    torch::Tensor forward(const torch::Tensor& input) {
        // Simple reshape implementation
        torch::Tensor flattened_shape = shape_tensor.flatten();
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < flattened_shape.numel(); ++i) {
            new_shape.push_back(flattened_shape[i].item<int64_t>());
        }
        
        // Handle batch dimension properly
        // If input has batch dimension (first dim = 1), preserve it
        if (input.dim() > 0 && input.size(0) == 1) {
            // Insert batch size as first dimension
            new_shape.insert(new_shape.begin(), 1);
        }
        
        return input.reshape(new_shape);
    }
};

class Constant : public torch::nn::Module {
    torch::Tensor value;
public:
    Constant(torch::Tensor value) : value(value) {
        register_buffer("value", this->value);
    }
    torch::Tensor forward();
};

} // namespace Operations

#endif // __OPERATIONS_H__ 