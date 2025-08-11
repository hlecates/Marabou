// Handles ONNX file parsing and entry to TorchModel

/*
 * Important Operations to Add (Before integrating to pipeline):
 * Add/Sub/Mul/Div
 * MatMul
 * Sigmoid
 *
 * Supported Operations:
 * Identity
 * Reshape
 * Gemm
 * Relu
 */


#include "OnnxToTorch.h"
#include "../nlr/TorchModel.h"
#include "../nlr/BoundedTorchNode.h"
#include "../nlr/BoundedConstantNode.h"
#include "../nlr/BoundedInputNode.h"
#include "../nlr/BoundedLinearNode.h"
#include "../nlr/BoundedReLUNode.h"
#include "../nlr/BoundedIdentityNode.h"
#include "../nlr/BoundedReshapeNode.h"
#include "MarabouError.h"
#include "File.h"
#include "MString.h"
#include "Vector.h"
#include "Map.h"
#include "Set.h"
#include "InputQueryBuilder.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <onnx.proto3.pb.h>

#include "FloatUtils.h"
#include "InputParserError.h"
#include "MString.h"
#include "Query.h"
#include "ReluConstraint.h"
#include "TensorUtils.h"
#include "onnx.proto3.pb.h"

#include <fstream>
#include <iostream>
#include <math.h>

#include "OnnxToTorch.h"
#include "Debug.h"

/******************
 * Error handling functions *
 ******************/

void onnxToTorchMissingAttributeError(const onnx::NodeProto &node, const String &attributeName)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx node of type %s is missing the expected attribute %s",
                                   node.op_type().c_str(),
                                   attributeName.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchUnimplementedOperationError(const onnx::NodeProto &node)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx '%s' operation not yet implemented for TorchModel conversion. Should be relatively easy to add.",
                                   node.op_type().c_str());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchUnimplementedAttributeError(const onnx::NodeProto &node, const String &attributeName)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx '%s' operation with non-default value for attribute '%s' not yet supported.",
                                   node.op_type().c_str(),
                                   attributeName.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchUnsupportedOperationError(const onnx::NodeProto &node)
{
    String errorMessage = Stringf("OnnxToTorch: Onnx operation %s not currently supported by TorchModel conversion",
                                   node.op_type().c_str());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchMissingNodeError(const String &missingNodeName)
{
    String errorMessage = Stringf("OnnxToTorch: Internal invariant violated: missing node '%s' not found",
                                   missingNodeName.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchUnexpectedNumberOfInputs(const onnx::NodeProto &node,
                                        unsigned int actualNumberOfInputs,
                                        unsigned int lowerBound,
                                        unsigned int upperBound)
{
    String errorMessage;
    if (lowerBound == upperBound)
    {
        errorMessage = Stringf("OnnxToTorch: %s node expected to have exactly %d inputs, but found %d",
                                node.op_type().c_str(),
                                lowerBound,
                                actualNumberOfInputs);
    }
    else
    {
        errorMessage = Stringf("OnnxToTorch: %s node expected to have between %d and %d inputs, but found %d",
                                node.op_type().c_str(),
                                lowerBound,
                                upperBound,
                                actualNumberOfInputs);
    }
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidTensorShapeError(const String &nodeName, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Invalid tensor shape for node '%s': %s",
                                   nodeName.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchUnsupportedDataTypeError(const onnx::TensorProto_DataType &dataType)
{
    String errorMessage = Stringf("OnnxToTorch: Support for Onnx constants of type '%s' not yet implemented.",
                                   TensorProto_DataType_Name(dataType).c_str());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidConstantNodeError(const onnx::NodeProto &node, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Invalid Constant node '%s': %s",
                                   node.name().c_str(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchTopologicalSortError(const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Topological sort failed: %s",
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchBoundedModuleCreationError(const String &operationType, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to create bounded module for operation '%s': %s",
                                   operationType.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchFileReadError(const String &filename, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to read file '%s': %s",
                                   filename.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchModelParseError(const String &filename, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to parse ONNX model from file '%s': %s",
                                   filename.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchGraphProcessingError(const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Graph processing failed: %s",
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchTensorConversionError(const String &tensorName, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to convert tensor '%s': %s",
                                   tensorName.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchAttributeProcessingError(const onnx::NodeProto &node, const String &attributeName, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Failed to process attribute '%s' for node '%s': %s",
                                   attributeName.ascii(),
                                   node.op_type().c_str(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchShapeMismatchError(const String &operation, const TensorShape &expectedShape, const TensorShape &actualShape)
{
    String expectedStr = "[";
    for (unsigned int i = 0; i < expectedShape.size(); ++i) {
        if (i > 0) expectedStr += ", ";
        expectedStr += Stringf("%u", expectedShape[i]);
    }
    expectedStr += "]";
    
    String actualStr = "[";
    for (unsigned int i = 0; i < actualShape.size(); ++i) {
        if (i > 0) actualStr += ", ";
        actualStr += Stringf("%u", actualShape[i]);
    }
    actualStr += "]";
    
    String errorMessage = Stringf("OnnxToTorch: Shape mismatch in operation '%s': expected %s, got %s",
                                   operation.ascii(),
                                   expectedStr.ascii(),
                                   actualStr.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchDimensionMismatchError(const String &operation, unsigned int expectedDim, unsigned int actualDim)
{
    String errorMessage = Stringf("OnnxToTorch: Dimension mismatch in operation '%s': expected %u dimensions, got %u",
                                   operation.ascii(),
                                   expectedDim,
                                   actualDim);
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidBroadcastError(const String &operation, const TensorShape &shape1, const TensorShape &shape2)
{
    String shape1Str = "[";
    for (unsigned int i = 0; i < shape1.size(); ++i) {
        if (i > 0) shape1Str += ", ";
        shape1Str += Stringf("%u", shape1[i]);
    }
    shape1Str += "]";
    
    String shape2Str = "[";
    for (unsigned int i = 0; i < shape2.size(); ++i) {
        if (i > 0) shape2Str += ", ";
        shape2Str += Stringf("%u", shape2[i]);
    }
    shape2Str += "]";
    
    String errorMessage = Stringf("OnnxToTorch: Invalid broadcast in operation '%s': cannot broadcast shapes %s and %s",
                                   operation.ascii(),
                                   shape1Str.ascii(),
                                   shape2Str.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchUnsupportedActivationError(const String &activationType)
{
    String errorMessage = Stringf("OnnxToTorch: Unsupported activation function '%s'",
                                   activationType.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchInvalidWeightBiasError(const String &operation, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Invalid weight/bias configuration in operation '%s': %s",
                                   operation.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchMemoryAllocationError(const String &operation, const String &reason)
{
    String errorMessage = Stringf("OnnxToTorch: Memory allocation failed in operation '%s': %s",
                                   operation.ascii(),
                                   reason.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

void onnxToTorchPyTorchError(const String &operation, const String &pytorchError)
{
    String errorMessage = Stringf("OnnxToTorch: PyTorch error in operation '%s': %s",
                                   operation.ascii(),
                                   pytorchError.ascii());
    throw MarabouError(MarabouError::ONNX_PARSER_ERROR, errorMessage.ascii());
}

/******************
 * Public methods *
 ******************/

using namespace Operations;
using namespace AttributeUtils;

using TensorShape = Vector<unsigned int>;

OnnxToTorchParser::OnnxToTorchParser(const String &path) {
    std::cerr << "[OnnxToTorchParser] Loading ONNX file: " << path << std::endl << std::flush;
    std::ifstream input(path.ascii(), std::ios::ate | std::ios::binary);
    if (!input.is_open()) {
        onnxToTorchFileReadError(path, "Could not open file");
    }
    std::streamsize size = input.tellg();
    std::cerr << "[OnnxToTorchParser] File size: " << size << " bytes" << std::endl << std::flush;

    input.seekg(0, std::ios::beg);
    Vector<char> buffer(size);
    input.read(buffer.data(), size);
    if (input.gcount() != size) {
        onnxToTorchFileReadError(path, Stringf("Failed to read entire file: expected %ld bytes, got %ld", 
                                               (long)size, (long)input.gcount()));
    }
    
    onnx::ModelProto model;
    if (!model.ParseFromArray(buffer.data(), size)) {
        onnxToTorchModelParseError(path, "Failed to parse ONNX protobuf");
    }
    std::cerr << "[OnnxToTorchParser] Successfully parsed ONNX model with " << model.graph().node_size() << " nodes" << std::endl << std::flush;
    
    // Debug: Print all nodes and their attributes
    for (int i = 0; i < model.graph().node_size(); ++i) {
        const auto& node = model.graph().node(i);
        std::cerr << "[OnnxToTorchParser] Node " << i << " op_type: " << node.op_type() << std::endl << std::flush;
        for (int j = 0; j < node.attribute_size(); ++j) {
            const auto& attr = node.attribute(j);
            std::cerr << "[OnnxToTorchParser]   Attribute " << j << " name: " << attr.name() << " type: " << attr.type() << std::endl << std::flush;
        }
    }
    
    _onnx_model = model;
}

std::shared_ptr<NLR::TorchModel> OnnxToTorchParser::parse(const String &path, const Map<String, Vector<Variable>>& marabouVarMap) {
    OnnxToTorchParser parser(path);
    return parser.processGraph(marabouVarMap);
}

std::shared_ptr<NLR::TorchModel> OnnxToTorchParser::processGraph(const Map<String, Vector<Variable>>& marabouVarMap) {
    // Process initializers
    Map<String, onnx::TensorProto> name_to_initializer;
    for (const auto& initializer : _onnx_model.graph().initializer()) {
        name_to_initializer[initializer.name()] = initializer;
    }

    // Process inputs
    Map<String, onnx::ValueInfoProto> name_to_input;
    for (const auto& input : _onnx_model.graph().input()) {
        name_to_input[input.name()] = input;
    }

    // Process nodes - store by output name for lookup
    Map<String, onnx::NodeProto> name_to_node;
    for (const auto& node : _onnx_model.graph().node()) {
        for (int i = 0; i < node.output_size(); ++i) {
            name_to_node[node.output(i)] = node;
        }
    }

    // Build a complete list of all tensors in processing order
    // This mimics the Python approach of processing in graph order
    Vector<String> processingOrder;
    
    // Add inputs first
    for (const auto& input : _onnx_model.graph().input()) {
        processingOrder.append(input.name());
    }
    
    // Add initializers
    for (const auto& initializer : _onnx_model.graph().initializer()) {
        processingOrder.append(initializer.name());
    }
    
    // Add node outputs in the order they appear in the graph
    for (const auto& node : _onnx_model.graph().node()) {
        for (int i = 0; i < node.output_size(); ++i) {
            String outputName = node.output(i);
            // Avoid duplicates
            bool already_exists = false;
            for (unsigned j = 0; j < processingOrder.size(); ++j) {
                if (processingOrder[j] == outputName) {
                    already_exists = true;
                    break;
                }
            }
            if (!already_exists) {
                processingOrder.append(outputName);
            }
        }
    }

    std::cerr << "[OnnxToTorchParser] Processing order:";
    for (unsigned i = 0; i < processingOrder.size(); ++i) {
        std::cerr << " " << processingOrder[i];
    }
    std::cerr << std::endl << std::flush;

    // Build model components using unified nodes
    Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    Vector<Vector<Variable>> marabouVars(processingOrder.size());
    Vector<unsigned> inputIndices;
    unsigned outputIndex = 0;

    // Debug: Show expected output tensor name
    String expectedOutputName;
    if (_onnx_model.graph().output_size() > 0) {
        expectedOutputName = _onnx_model.graph().output(0).name();
        std::cerr << "[OnnxToTorchParser] Expected output tensor name: " << expectedOutputName << std::endl << std::flush;
    } else {
        std::cerr << "[OnnxToTorchParser] No outputs found in graph" << std::endl << std::flush;
    }

    // Create constants map for operations
    Map<String, torch::Tensor> constantsMap;

    // Map input names to their corresponding constants for operations
    // This needs to happen BEFORE node processing so constants are available
    for (const auto& node : _onnx_model.graph().node()) {
        for (int i = 0; i < node.input_size(); ++i) {
            String inputName = node.input(i);
            // If this input name corresponds to a constant, add it to the constants map
            if (name_to_initializer.exists(inputName) && !constantsMap.exists(inputName)) {
                torch::Tensor constant = ConstantProcessor::processInitializer(name_to_initializer[inputName]);
                constantsMap[inputName] = constant;
            }
        }
    }

    // Map names to indices for lookup
    Map<String, unsigned> nameToIndex;
    for (unsigned i = 0; i < processingOrder.size(); ++i) {
        nameToIndex[processingOrder[i]] = i;
    }

    // Construct complete neuron-to-Marabou mapping
    Map<unsigned, Vector<Variable>> neuronToMarabouMap;

    // Build dependency map: node index -> input node indices
    Map<unsigned, Vector<unsigned>> dependencies;

    // Process each tensor in processing order
    for (unsigned i = 0; i < processingOrder.size(); ++i) {
        const String& tensorName = processingOrder[i];
        std::cerr << "[OnnxToTorchParser] Processing tensor " << i << ": " << tensorName << std::endl << std::flush;

        // Handle initializers (constants)
        if (name_to_initializer.exists(tensorName)) {
            torch::Tensor constant = ConstantProcessor::processInitializer(name_to_initializer[tensorName]);
            auto constantNode = std::make_shared<NLR::BoundedConstantNode>(constant, tensorName);
            constantNode->setNodeIndex(i);
            nodes.append(constantNode);
            
            if (marabouVarMap.exists(tensorName)) {
                neuronToMarabouMap[i] = marabouVarMap[tensorName];
                marabouVars[i] = marabouVarMap[tensorName];
            }
            continue;
        }

        // Handle Constant nodes
        if (name_to_node.exists(tensorName)) {
            const auto& node = name_to_node[tensorName];
            std::cerr << "  Found node with op_type: " << node.op_type() << std::endl << std::flush;
            if (node.op_type() == "Constant") {
                std::cerr << "  Processing as Constant node" << std::endl << std::flush;
                try {
                    torch::Tensor constant = ConstantProcessor::processConstantNode(node);
                    auto constantNode = std::make_shared<NLR::BoundedConstantNode>(constant, tensorName);
                    constantNode->setNodeIndex(i);
                    nodes.append(constantNode);
                    
                    if (marabouVarMap.exists(tensorName)) {
                        neuronToMarabouMap[i] = marabouVarMap[tensorName];
                        marabouVars[i] = marabouVarMap[tensorName];
                    }
                    std::cerr << "  Constant node processing completed successfully" << std::endl << std::flush;
                } catch (const MarabouError& e) {
                    // Re-throw MarabouError exceptions as they are already properly formatted
                    throw;
                } catch (const std::exception& e) {
                    // Convert other exceptions to OnnxToTorch specific errors
                    onnxToTorchInvalidConstantNodeError(node, e.what());
                }
                continue;
            }
        }

        // Handle inputs
        if (name_to_input.exists(tensorName) && !name_to_initializer.exists(tensorName)) {
            inputIndices.append(i);
            
            // Get input size from the input info
            unsigned inputSize = 1; // Default, should be extracted from input info
            TensorShape inputShape = BoundedOperationConverter::extractShapeFromNode(onnx::NodeProto(), name_to_input, name_to_initializer, tensorName);
            if (!inputShape.empty()) {
                inputSize = BoundedOperationConverter::computeTensorSize(inputShape);
                std::cout << "[DEBUG] Input tensor " << tensorName << " shape: ";
                for (unsigned dim : inputShape) {
                    std::cout << dim << " ";
                }
                std::cout << ", computed size: " << inputSize << std::endl;
            }
            auto inputNode = std::make_shared<NLR::BoundedInputNode>(i, inputSize, tensorName);
            inputNode->setNodeIndex(i);
            nodes.append(inputNode);
            
            if (marabouVarMap.exists(tensorName)) {
                neuronToMarabouMap[i] = marabouVarMap[tensorName];
                marabouVars[i] = marabouVarMap[tensorName];
            }
            continue;
        }

        // Handle node outputs - create bounded nodes
        if (name_to_node.exists(tensorName)) {
            const auto& node = name_to_node[tensorName];

            // Build input dependencies for this node
            Vector<unsigned> deps;
            for (int j = 0; j < node.input_size(); ++j) {
                String inputName = node.input(j);
                if (nameToIndex.exists(inputName)) {
                    unsigned inputIndex = nameToIndex[inputName];
                    if (!name_to_initializer.exists(inputName)) {
                        deps.append(inputIndex);
                    }
                }
            }
            if (!deps.empty()) {
                dependencies[i] = deps;
                std::cerr << "  Element dependencies: ";
                for (unsigned d = 0; d < deps.size(); ++d) std::cerr << deps[d] << " ";
                std::cerr << std::endl << std::flush;
            }

            // Convert node to bounded node with enhanced size setting
            std::shared_ptr<NLR::BoundedTorchNode> boundedNode;
            
            try {
                if (node.op_type() == "Identity") {
                    boundedNode = BoundedOperationConverter::convertIdentity(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Gemm") {
                    boundedNode = BoundedOperationConverter::convertGemm(node, constantsMap, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Relu") {
                    boundedNode = BoundedOperationConverter::convertRelu(node, name_to_input, name_to_initializer);
                } else if (node.op_type() == "Reshape") {
                    boundedNode = BoundedOperationConverter::convertReshape(node, name_to_input, name_to_initializer);
                } else {
                    onnxToTorchUnsupportedOperationError(node);
                }
                
                // Set node metadata
                boundedNode->setNodeIndex(i);
                boundedNode->setNodeName(tensorName);
                
            } catch (const MarabouError& e) {
                throw;
            } catch (const std::exception& e) {
                onnxToTorchBoundedModuleCreationError(node.op_type(), e.what());
            }
            
            nodes.append(boundedNode);
            
            if (marabouVarMap.exists(tensorName)) {
                neuronToMarabouMap[i] = marabouVarMap[tensorName];
                marabouVars[i] = marabouVarMap[tensorName];
            }
            continue;
        }

        // Track output index
        if (_onnx_model.graph().output_size() > 0 && tensorName == _onnx_model.graph().output(0).name()) {
            outputIndex = i;
            std::cerr << "  Output index set to " << i << " for tensor " << tensorName << std::endl << std::flush;
        }
    }

    return std::make_shared<NLR::TorchModel>(
        nodes,
        marabouVars,
        inputIndices,
        outputIndex,
        neuronToMarabouMap,
        dependencies
    );
}

namespace AttributeUtils {

float getFloatAttribute(const onnx::NodeProto &node, const String &name, float defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            return attr.f();
        }
    }
    return defaultValue;
}

int getIntAttribute(const onnx::NodeProto &node, const String &name, int defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            return attr.i();
        }
    }
    return defaultValue;
}

Vector<int> getIntsAttribute(onnx::NodeProto &node, const String &name, const Vector<int> &defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            Vector<int> result;
            for (int i = 0; i < attr.ints_size(); i++) {
                result.append(attr.ints(i));
            }
            return result;
        }
    }
    return defaultValue;
}

String getStringAttribute(onnx::NodeProto &node, const String &name, const String &defaultValue)
{
    for (const auto &attr : node.attribute()) {
        if (attr.name() == name.ascii()) {
            return String(attr.s());
        }
    }
    return defaultValue;
}

Map<String, torch::IValue> extractAttributes(onnx::NodeProto &node)
{
    Map<String, torch::IValue> kwargs;
    
    for (const auto &attr : node.attribute()) {
        String attr_name = attr.name();
        
        if (attr.type() == onnx::AttributeProto::INT) {
            kwargs[attr_name] = torch::IValue(static_cast<int64_t>(attr.i()));
        } else if (attr.type() == onnx::AttributeProto::FLOAT) {
            kwargs[attr_name] = torch::IValue(static_cast<double>(attr.f()));
        } else if (attr.type() == onnx::AttributeProto::STRING) {
            kwargs[attr_name] = torch::IValue(attr.s());
        } else if (attr.type() == onnx::AttributeProto::INTS) {
            std::vector<int64_t> ints;
            for (int i = 0; i < attr.ints_size(); i++) {
                ints.push_back(attr.ints(i));
            }
            kwargs[attr_name] = torch::IValue(ints);
        } else if (attr.type() == onnx::AttributeProto::FLOATS) {
            std::vector<double> floats;
            for (int i = 0; i < attr.floats_size(); i++) {
                floats.push_back(attr.floats(i));
            }
            kwargs[attr_name] = torch::IValue(floats);
        } else if (attr.type() == onnx::AttributeProto::TENSOR) {
            // Convert tensor attribute to torch::Tensor
            // This is simplified - real implementation needs full tensor conversion
            kwargs[attr_name] = torch::IValue(torch::zeros({1})); // Placeholder
        }
        // Add more attribute types as needed
    }
    
    return kwargs;
}
}

namespace Operations {

torch::Tensor Constant::forward()
{
    return value;
}

torch::Tensor ReshapeImpl::forward(const torch::Tensor& input, const torch::Tensor& shape_tensor) {
    std::vector<int64_t> shape = GraphUtils::instantiateReshapeTemplate(input, shape_tensor);
    return input.reshape(shape);
}

} // namespace Operations

namespace GraphUtils {

Vector<String> computeTopologicalOrder(const Map<String, onnx::NodeProto>& name_to_node,
                                      const Map<String, onnx::ValueInfoProto>& name_to_input,
                                      const Map<String, onnx::TensorProto>& name_to_initializer)
{
    std::cerr << "[GraphUtils] Starting simplified topological sort" << std::endl << std::flush;
    
    Vector<String> order;
    
    // First, add all inputs and initializers (sources)
    for (auto it = name_to_input.begin(); it != name_to_input.end(); ++it) {
        order.append(it->first);
        std::cerr << "[GraphUtils] Added input: " << it->first << std::endl << std::flush;
    }
    
    for (auto it = name_to_initializer.begin(); it != name_to_initializer.end(); ++it) {
        order.append(it->first);
        std::cerr << "[GraphUtils] Added initializer: " << it->first << std::endl << std::flush;
    }
    
    // Then add all node outputs in the order they appear in the graph
    // This mimics the Python approach where nodes are processed in order
    for (auto it = name_to_node.begin(); it != name_to_node.end(); ++it) {
        const onnx::NodeProto& node = it->second;
        for (int i = 0; i < node.output_size(); ++i) {
            String output = node.output(i);
            // Only add if not already in order (avoid duplicates)
            bool already_exists = false;
            for (unsigned j = 0; j < order.size(); ++j) {
                if (order[j] == output) {
                    already_exists = true;
                    break;
                }
            }
            if (!already_exists) {
                order.append(output);
                std::cerr << "[GraphUtils] Added node output: " << output << std::endl << std::flush;
            }
        }
    }
    
    std::cerr << "[GraphUtils] Final order size: " << order.size() << std::endl << std::flush;
    std::cerr << "[GraphUtils] Topological order: ";
    for (unsigned i = 0; i < order.size(); ++i) {
        std::cerr << order[i] << " ";
    }
    std::cerr << std::endl << std::flush;
    
    return order;
}

Map<String, Set<String>> computeActivationDependencies(const onnx::GraphProto& graph) {
    /*
     * Compute activation dependencies, mapping each tensor to its dependents.
     * This mimics the Python implementation's compute_activation_dependencies function.
     */
    Map<String, Set<String>> needed_by;
    
    for (const auto& node : graph.node()) {
        String out_op_id = node.output(0);
        for (int i = 0; i < node.input_size(); ++i) {
            String in_op_id = node.input(i);
            needed_by[in_op_id].insert(out_op_id);
        }
        // TODO: Handle Loop nodes if needed
    }
    
    return needed_by;
}

std::vector<int64_t> instantiateReshapeTemplate(const torch::Tensor& input, const torch::Tensor& shape_tensor) {
    std::vector<int64_t> oldShape(input.sizes().begin(), input.sizes().end());
    std::vector<int64_t> newShapeTemplate;
    
    // Convert shape tensor to vector - handle different tensor shapes
    torch::Tensor flattened_shape = shape_tensor.flatten();
    for (int64_t i = 0; i < flattened_shape.numel(); ++i) {
        newShapeTemplate.push_back(flattened_shape[i].item<int64_t>());
    }
    
    std::vector<int64_t> newShape;
    int inferredIndex = -1;
    int64_t knownProduct = 1;
    
    for (size_t i = 0; i < newShapeTemplate.size(); ++i) {
        int64_t dim = newShapeTemplate[i];
        if (dim == 0) {
            // Copy from input shape
            dim = (i < oldShape.size()) ? oldShape[i] : 1;
        }
        if (dim == -1) {
            inferredIndex = i;
            newShape.push_back(1); // Placeholder
        } else {
            newShape.push_back(dim);
            knownProduct *= dim;
        }
    }
    
    if (inferredIndex != -1) {
        int64_t total = input.numel();
        int64_t inferred = total / knownProduct;
        newShape[inferredIndex] = inferred;
    }
    
    return newShape;
}

} // namespace GraphUtils

namespace ConstantProcessor {

torch::Tensor processInitializer(const onnx::TensorProto& tensor) {
    std::cerr << "      [ConstantProcessor] Processing initializer with data type: " << tensor.data_type() << std::endl << std::flush;
    std::cerr << "      [ConstantProcessor] Tensor has " << tensor.dims_size() << " dimensions" << std::endl << std::flush;
    
    // Determine tensor shape
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor.dims_size(); ++i) {
        shape.push_back(tensor.dims(i));
        std::cerr << "      [ConstantProcessor] Dimension " << i << ": " << tensor.dims(i) << std::endl << std::flush;
    }
    
    // Convert based on data type
    switch (tensor.data_type()) {
        case onnx::TensorProto_DataType_FLOAT: {
            std::cerr << "      [ConstantProcessor] Processing FLOAT tensor with " << tensor.float_data_size() << " elements" << std::endl << std::flush;
            std::vector<float> data;
            for (int i = 0; i < tensor.float_data_size(); ++i) {
                data.push_back(tensor.float_data(i));
            }
            torch::Tensor result = torch::tensor(data, torch::kFloat32).reshape(shape);
            std::cerr << "      [ConstantProcessor] Created FLOAT tensor with shape: ";
            for (auto dim : result.sizes()) {
                std::cerr << dim << " ";
            }
            std::cerr << std::endl << std::flush;
            return result;
        }
        case onnx::TensorProto_DataType_INT64: {
            std::cerr << "      [ConstantProcessor] Processing INT64 tensor" << std::endl << std::flush;
            std::vector<int64_t> data;
            for (int i = 0; i < tensor.int64_data_size(); ++i) {
                data.push_back(tensor.int64_data(i));
            }
            return torch::tensor(data, torch::kInt64).reshape(shape);
        }
        case onnx::TensorProto_DataType_INT32: {
            std::cerr << "      [ConstantProcessor] Processing INT32 tensor" << std::endl << std::flush;
            std::vector<int32_t> data;
            for (int i = 0; i < tensor.int32_data_size(); ++i) {
                data.push_back(tensor.int32_data(i));
            }
            return torch::tensor(data, torch::kInt32).reshape(shape);
        }
        case onnx::TensorProto_DataType_DOUBLE: {
            std::cerr << "      [ConstantProcessor] Processing DOUBLE tensor" << std::endl << std::flush;
            std::vector<double> data;
            for (int i = 0; i < tensor.double_data_size(); ++i) {
                data.push_back(tensor.double_data(i));
            }
            return torch::tensor(data, torch::kFloat64).reshape(shape);
        }
        default:
            std::cerr << "      [ConstantProcessor] Unsupported data type: " << tensor.data_type() << std::endl << std::flush;
            onnxToTorchUnsupportedDataTypeError(static_cast<onnx::TensorProto_DataType>(tensor.data_type()));
            return torch::Tensor(); // This line will never be reached, but satisfies compiler
    }
}

torch::Tensor processConstantNode(const onnx::NodeProto& node) {
    std::cerr << "    [ConstantProcessor] Processing Constant node with " << node.attribute_size() << " attributes" << std::endl << std::flush;
    for (const auto& attr : node.attribute()) {
        std::cerr << "    [ConstantProcessor] Checking attribute: " << attr.name() << " (type: " << attr.type() << ")" << std::endl << std::flush;
        if (attr.name() == "value") {
            std::cerr << "    [ConstantProcessor] Found value attribute, checking if tensor data is present..." << std::endl << std::flush;
            
            // Check if the tensor data is actually present, regardless of type
            if (attr.has_t()) {
                std::cerr << "    [ConstantProcessor] Tensor data is present, processing..." << std::endl << std::flush;
                try {
                    torch::Tensor result = processInitializer(attr.t());
                    std::cerr << "    [ConstantProcessor] Successfully processed constant tensor" << std::endl << std::flush;
                    return result;
                } catch (const std::exception& e) {
                    std::cerr << "    [ConstantProcessor] Exception processing tensor: " << e.what() << std::endl << std::flush;
                    throw;
                }
            } else {
                std::cerr << "    [ConstantProcessor] No tensor data found in attribute" << std::endl << std::flush;
            }
        }
    }
    std::cerr << "    [ConstantProcessor] No valid value attribute found" << std::endl << std::flush;
    onnxToTorchInvalidConstantNodeError(node, "No valid value attribute found");
    return torch::Tensor(); // This line will never be reached, but satisfies compiler
}

} // namespace ConstantProcessor

namespace BoundedOperationConverter {

    // Helper function to extract shape information from ONNX
    TensorShape extractShapeFromNode(const onnx::NodeProto& node, 
                                   const Map<String, onnx::ValueInfoProto>& name_to_input,
                                   const Map<String, onnx::TensorProto>& name_to_initializer,
                                   const String& tensorName) {
        (void)node; // Suppress unused parameter warning
        // Try to get shape from input info
        if (name_to_input.exists(tensorName)) {
            const auto& inputInfo = name_to_input[tensorName];
            if (inputInfo.type().tensor_type().has_shape()) {
                const auto& shape = inputInfo.type().tensor_type().shape();
                TensorShape result;
                for (int i = 0; i < shape.dim_size(); ++i) {
                    if (shape.dim(i).has_dim_value()) {
                        result.append(shape.dim(i).dim_value());
                    }
                }
                return result;
            }
        }
        
        // Try to get shape from initializer
        if (name_to_initializer.exists(tensorName)) {
            const auto& initializer = name_to_initializer[tensorName];
            TensorShape result;
            for (int i = 0; i < initializer.dims_size(); ++i) {
                result.append(initializer.dims(i));
            }
            return result;
        }
        
        return TensorShape();
    }
    
    // Helper function to compute tensor size from shape
    unsigned computeTensorSize(const TensorShape& shape) {
        if (shape.empty()) return 0;
        
        unsigned size = 1;
        for (unsigned dim : shape) {
            size *= dim;
        }
        return size;
    }

    std::shared_ptr<NLR::BoundedTorchNode> convertGemm(const onnx::NodeProto& node, 
                                                     const Map<String, torch::Tensor>& constants,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        (void)name_to_input; // Suppress unused parameter warning
        (void)name_to_initializer; // Suppress unused parameter warning
        // Extract weights and bias from constants
        if (node.input_size() < 2) {
            onnxToTorchUnexpectedNumberOfInputs(node, node.input_size(), 2, 3);
            return nullptr;
        }
        
        String weightName = node.input(1);
        String biasName = (node.input_size() > 2) ? node.input(2) : "";
        
        // Try to find weight tensor
        torch::Tensor weights;
        bool foundWeights = false;
        
        if (constants.exists(weightName)) {
            weights = constants[weightName];
            foundWeights = true;
        } else {
            Vector<String> possibleNames = {weightName, "weight", "W", "weights"};
            for (const auto& name : possibleNames) {
                if (constants.exists(name)) {
                    weights = constants[name];
                    foundWeights = true;
                    break;
                }
            }
        }
        
        if (!foundWeights) {
            onnxToTorchInvalidWeightBiasError("Gemm", "Weight tensor not found in constants");
            return nullptr;
        }
        
        // Handle bias tensor
        torch::Tensor bias;
        if (biasName.length() > 0 && constants.exists(biasName)) {
            bias = constants[biasName];
        } else {
            bias = torch::zeros({weights.size(0)});
        }
        
        // Extract attributes
        float alpha = AttributeUtils::getFloatAttribute(node, "alpha", 1.0f);
        float beta = AttributeUtils::getFloatAttribute(node, "beta", 1.0f);
        int transB = AttributeUtils::getIntAttribute(node, "transB", 0);
        
        // ONNX Gemm preprocessing
        if (transB == 0) {
            weights = weights.transpose(-2, -1);
        }
        
        if (beta != 1.0f) {
            bias = beta * bias;
        }
        
        // Create linear module
        auto linear_module = torch::nn::Linear(weights.size(1), weights.size(0));
        linear_module->weight = weights;
        linear_module->bias = bias;
        
        // Create bounded linear node - sizes will be set automatically in constructor
        auto boundedNode = std::make_shared<NLR::BoundedLinearNode>(linear_module, alpha);
        
        std::cout << "[OnnxToTorch::convertGemm] Created Gemm node with sizes: input=" 
                  << boundedNode->getInputSize() << ", output=" << boundedNode->getOutputSize() << std::endl;
        
        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertRelu(const onnx::NodeProto& node,
                                                     const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                     const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer input size from the input tensor
        unsigned inputSize = 0;
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            std::cout << "[DEBUG] ReLU input tensor name: " << inputName << std::endl;
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            std::cout << "[DEBUG] ReLU input shape: ";
            for (unsigned dim : inputShape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
            inputSize = computeTensorSize(inputShape);
            std::cout << "[DEBUG] ReLU computed input size: " << inputSize << std::endl;
        }
        
        auto relu_module = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(false));
        auto boundedNode = std::make_shared<NLR::BoundedReLUNode>(relu_module);
        
        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
            boundedNode->setOutputSize(inputSize); // ReLU preserves input size
            std::cout << "[OnnxToTorch::convertRelu] Set sizes for ReLU node: input=output=" 
                      << inputSize << std::endl;
        }
        
        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertIdentity(const onnx::NodeProto& node,
                                                         const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                         const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer input size from the input tensor
        unsigned inputSize = 0;
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            inputSize = computeTensorSize(inputShape);
        }
        
        auto identity_module = torch::nn::Identity();
        auto boundedNode = std::make_shared<NLR::BoundedIdentityNode>(identity_module);
        
        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
            boundedNode->setOutputSize(inputSize); // Identity preserves input size
            std::cout << "[OnnxToTorch::convertIdentity] Set sizes for Identity node: input=output=" 
                      << inputSize << std::endl;
        }
        
        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertReshape(const onnx::NodeProto& node,
                                                        const Map<String, onnx::ValueInfoProto>& name_to_input,
                                                        const Map<String, onnx::TensorProto>& name_to_initializer) {
        // Try to infer sizes from input and output shapes
        unsigned inputSize = 0;
        unsigned outputSize = 0;
        
        if (node.input_size() > 0) {
            String inputName = node.input(0);
            TensorShape inputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, inputName);
            inputSize = computeTensorSize(inputShape);
        }
        
        if (node.output_size() > 0) {
            String outputName = node.output(0);
            TensorShape outputShape = extractShapeFromNode(node, name_to_input, name_to_initializer, outputName);
            outputSize = computeTensorSize(outputShape);
        }
        
        torch::Tensor default_shape = torch::tensor({-1});
        auto reshape_module = Operations::ReshapeWrapper(default_shape);
        auto boundedNode = std::make_shared<NLR::BoundedReshapeNode>(reshape_module);
        
        // Set sizes if we can infer them
        if (inputSize > 0) {
            boundedNode->setInputSize(inputSize);
        }
        if (outputSize > 0) {
            boundedNode->setOutputSize(outputSize);
        }
        
        if (inputSize > 0 || outputSize > 0) {
            std::cout << "[OnnxToTorch::convertReshape] Set sizes for Reshape node: input=" 
                      << inputSize << ", output=" << outputSize << std::endl;
        }
        
        return boundedNode;
    }
    
    std::shared_ptr<NLR::BoundedTorchNode> convertConstant(const torch::Tensor& value) {
        auto boundedNode = std::make_shared<NLR::BoundedConstantNode>(value, "");
        
        // Set sizes from constant value
        if (value.defined()) {
            unsigned size = value.numel();
            boundedNode->setInputSize(0); // Constants have no input
            boundedNode->setOutputSize(size);
            std::cout << "[OnnxToTorch::convertConstant] Set sizes for Constant node: output=" 
                      << size << std::endl;
        }
        
        return boundedNode;
    }
    
} // namespace BoundedOperationConverter