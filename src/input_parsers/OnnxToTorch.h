#ifndef __OnnxToTorchParser_h__
#define __OnnxToTorchParser_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "Query.h"
#include "InputQueryBuilder.h"
#include "onnx.proto3.pb.h"
#include "TorchModuleBounded.h"  // Add bounded module include

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

// Forward declarations
class TorchModel;

using TensorShape = Vector<unsigned int>;

// Error handling functions for OnnxToTorch parser
void onnxToTorchMissingAttributeError(const onnx::NodeProto &node, const String &attributeName);
void onnxToTorchUnimplementedOperationError(const onnx::NodeProto &node);
void onnxToTorchUnimplementedAttributeError(const onnx::NodeProto &node, const String &attributeName);
void onnxToTorchUnsupportedOperationError(const onnx::NodeProto &node);
void onnxToTorchMissingNodeError(const String &missingNodeName);
void onnxToTorchUnexpectedNumberOfInputs(const onnx::NodeProto &node, 
                                        unsigned int actualNumberOfInputs,
                                        unsigned int lowerBound, 
                                        unsigned int upperBound);
void onnxToTorchInvalidTensorShapeError(const String &nodeName, const String &reason);
void onnxToTorchUnsupportedDataTypeError(const onnx::TensorProto_DataType &dataType);
void onnxToTorchInvalidConstantNodeError(const onnx::NodeProto &node, const String &reason);
void onnxToTorchTopologicalSortError(const String &reason);
void onnxToTorchBoundedModuleCreationError(const String &operationType, const String &reason);
void onnxToTorchFileReadError(const String &filename, const String &reason);
void onnxToTorchModelParseError(const String &filename, const String &reason);
void onnxToTorchGraphProcessingError(const String &reason);
void onnxToTorchTensorConversionError(const String &tensorName, const String &reason);
void onnxToTorchAttributeProcessingError(const onnx::NodeProto &node, const String &attributeName, const String &reason);
void onnxToTorchShapeMismatchError(const String &operation, const TensorShape &expectedShape, const TensorShape &actualShape);
void onnxToTorchDimensionMismatchError(const String &operation, unsigned int expectedDim, unsigned int actualDim);
void onnxToTorchInvalidBroadcastError(const String &operation, const TensorShape &shape1, const TensorShape &shape2);
void onnxToTorchUnsupportedActivationError(const String &activationType);
void onnxToTorchInvalidWeightBiasError(const String &operation, const String &reason);
void onnxToTorchMemoryAllocationError(const String &operation, const String &reason);
void onnxToTorchPyTorchError(const String &operation, const String &pytorchError);

class OnnxToTorchParser
{
public:
    static std::shared_ptr<TorchModel> parse(const String &path, const Map<String, Vector<Variable>>& marabouVarMap);
private:
    OnnxToTorchParser(const String &path);
    std::shared_ptr<TorchModel> processGraph(const Map<String, Vector<Variable>>& marabouVarMap);
    onnx::ModelProto _onnx_model;
};


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


namespace AttributeUtils {
    Map<String, torch::IValue> extractAttributes(onnx::NodeProto &node);
    float getFloatAttribute(const onnx::NodeProto &node, const String &name, float defaultValue = 0.0f);
    int getIntAttribute(const onnx::NodeProto &node, const String &name, int defaultValue = 0);
    Vector<int> getIntsAttribute(onnx::NodeProto &node, const String &name, const Vector<int> &defaultValue = {});
    String getStringAttribute(onnx::NodeProto &node, const String &name, const String &defaultValue = "");
}

namespace GraphUtils {
    Vector<String> computeTopologicalOrder(
        const Map<String, onnx::NodeProto>& name_to_node,
        const Map<String, onnx::ValueInfoProto>& name_to_input,
        const Map<String, onnx::TensorProto>& name_to_initializer
    );

    Map<String, Set<String>> computeActivationDependencies(const onnx::GraphProto& graph);

    std::vector<int64_t> instantiateReshapeTemplate(
        const torch::Tensor& input, 
        const torch::Tensor& shape_tensor
    );
}

namespace ConstantProcessor {
    torch::Tensor processInitializer(const onnx::TensorProto& tensor);
    torch::Tensor processConstantNode(const onnx::NodeProto& node);
}

// New namespace for bounded module conversion
namespace BoundedOperationConverter {
    std::shared_ptr<NLR::ITorchModuleBounded> convertGemm(const onnx::NodeProto& node, 
                                                     const Map<String, torch::Tensor>& constants);
    std::shared_ptr<NLR::ITorchModuleBounded> convertRelu(const onnx::NodeProto& node);
    std::shared_ptr<NLR::ITorchModuleBounded> convertIdentity(const onnx::NodeProto& node);
    std::shared_ptr<NLR::ITorchModuleBounded> convertReshape(const onnx::NodeProto& node);
    std::shared_ptr<NLR::ITorchModuleBounded> convertConstant(const torch::Tensor& value);
}

#endif // __OnnxToTorchParser_h__