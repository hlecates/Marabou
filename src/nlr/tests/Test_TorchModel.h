#ifndef __TEST_TORCH_MODEL_H__
#define __TEST_TORCH_MODEL_H__

#include "../TorchModel.h"
#include "../TorchLinearBounded.h"
#include "MarabouError.h"
#include "Vector.h"
#include "Map.h"
#include "Set.h"
#include <cxxtest/TestSuite.h>
#include <torch/torch.h>

class TorchModelTestSuite : public CxxTest::TestSuite
{
public:
    const double DELTA = 0.0001;

    void test_constructor()
    {
        // Test basic constructor functionality with bounded modules
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        unsigned outputIndex = 0;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;

        // Add a simple linear bounded module
        auto linear_module = torch::nn::Linear(2, 3);
        linear_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        linear_module->bias = torch::tensor({0.1, 0.2, 0.3});
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear_module));
        
        // Add input index
        inputIndices.append(0);
        
        // Add Marabou variables
        Vector<Variable> testVars;
        testVars.append(Variable(0));
        testVars.append(Variable(1));
        marabouVars.append(testVars);
        
        // Add to mapping
        neuronToMarabouMap[0] = testVars;

        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE;
        elementToBoundedModuleIndex[0] = 0;

        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        TS_ASSERT_EQUALS(model.getSize(), 1u);
        TS_ASSERT_EQUALS(model.getOutputIndex(), 0u);
        TS_ASSERT_EQUALS(model.getInputIndices().size(), 1u);
        TS_ASSERT_EQUALS(model.getInputIndices()[0], 0u);
        TS_ASSERT_EQUALS(model.getVariables().size(), 1u);
        TS_ASSERT_EQUALS(model.getNeuronToMarabouMap().size(), 1u);
    }

    void test_forward_single_linear()
    {
        // Test forward pass with a single linear bounded module
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;
        
        auto linear = torch::nn::Linear(2, 3);
        linear->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        linear->bias = torch::tensor({0.1, 0.2, 0.3});
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear));
        
        inputIndices.append(1); // Input is at element index 1
        
        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE; // Linear
        elementTypes[1] = ElementType::INPUT;  // Input
        
        elementToBoundedModuleIndex[0] = 0; // Linear -> bounded module 0
        elementToInputIndex[1] = 0;  // Input -> input 0
        
        dependencies[0] = Vector<unsigned>{1}; // Linear depends on input
        
        TorchModel model(boundedModules, constants, marabouVars, inputIndices, 0, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        torch::Tensor input = torch::tensor({1.0, 2.0});
        torch::Tensor expected = torch::matmul(input, linear->weight.t()) + linear->bias;
        
        // Use the correct input mapping: element index 1 maps to input index 0
        Map<unsigned, torch::Tensor> inputs;
        inputs[0] = input; // Use input index 0, not element index 1
        torch::Tensor output = model.forward(inputs);
        TS_ASSERT_EQUALS(output.sizes().size(), expected.sizes().size());
        TS_ASSERT_EQUALS(output.numel(), expected.numel());
        for (int i = 0; i < output.numel(); ++i) {
            TS_ASSERT_DELTA(output[i].item<double>(), expected[i].item<double>(), DELTA);
        }
    }

    void test_forward_linear_chain()
    {
        // Test forward pass with two linear bounded modules in sequence
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;
        
        auto linear1 = torch::nn::Linear(2, 3);
        linear1->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        linear1->bias = torch::tensor({0.1, 0.2, 0.3});
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear1));
        
        auto linear2 = torch::nn::Linear(3, 1);
        linear2->weight = torch::tensor({{1.0, 2.0, 3.0}});
        linear2->bias = torch::tensor({0.5});
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear2));
        
        inputIndices.append(2); // Input is at element index 2
        
        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE; // Linear1
        elementTypes[1] = ElementType::MODULE; // Linear2
        elementTypes[2] = ElementType::INPUT;  // Input
        
        elementToBoundedModuleIndex[0] = 0; // Linear1 -> bounded module 0
        elementToBoundedModuleIndex[1] = 1; // Linear2 -> bounded module 1
        elementToInputIndex[2] = 0;  // Input -> input 0
        
        dependencies[0] = Vector<unsigned>{2}; // Linear1 depends on input
        dependencies[1] = Vector<unsigned>{0}; // Linear2 depends on Linear1
        
        TorchModel model(boundedModules, constants, marabouVars, inputIndices, 1, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        torch::Tensor input = torch::tensor({1.0, 2.0});
        torch::Tensor layer1_output = torch::matmul(input, linear1->weight.t()) + linear1->bias;
        torch::Tensor expected = torch::matmul(layer1_output, linear2->weight.t()) + linear2->bias;
        
        // Use the correct input mapping: element index 2 maps to input index 0
        Map<unsigned, torch::Tensor> inputs;
        inputs[0] = input; // Use input index 0, not element index 2
        torch::Tensor output = model.forward(inputs);
        TS_ASSERT_EQUALS(output.sizes().size(), expected.sizes().size());
        TS_ASSERT_EQUALS(output.numel(), expected.numel());
        for (int i = 0; i < output.numel(); ++i) {
            TS_ASSERT_DELTA(output[i].item<double>(), expected[i].item<double>(), DELTA);
        }
    }

    void test_eliminate_variable()
    {
        // Test variable elimination functionality
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        unsigned outputIndex = 0;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;

        auto linear = torch::nn::Linear(2, 3);
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear));
        
        Vector<Variable> testVars;
        testVars.append(Variable(0));
        testVars.append(Variable(1));
        marabouVars.append(testVars);
        
        neuronToMarabouMap[0] = testVars;
        
        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE;
        elementToBoundedModuleIndex[0] = 0;

        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        // Test elimination
        model.eliminateVariable(0, 5.0);
        TS_ASSERT(model.neuronEliminated(0));
        TS_ASSERT_DELTA(model.getEliminatedNeuronValue(0), 5.0, DELTA);
    }

    void test_update_variable_indices()
    {
        // Test variable index updating functionality
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        unsigned outputIndex = 0;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;

        auto linear = torch::nn::Linear(2, 3);
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear));
        
        Vector<Variable> testVars;
        testVars.append(Variable(0));
        testVars.append(Variable(1));
        marabouVars.append(testVars);
        
        neuronToMarabouMap[0] = testVars;
        
        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE;
        elementToBoundedModuleIndex[0] = 0;

        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        // Create index mapping
        Map<unsigned, unsigned> oldIndexToNewIndex;
        oldIndexToNewIndex[0] = 10;
        oldIndexToNewIndex[1] = 11;
        
        Map<unsigned, unsigned> mergedVariables;
        mergedVariables[2] = 10;
        mergedVariables[3] = 11;
        
        model.updateVariableIndices(oldIndexToNewIndex, mergedVariables);
        
        // Test that the mapping was updated correctly
        TS_ASSERT_EQUALS(model.getNeuronToMarabouMap().size(), 1u);
    }

    void test_getters()
    {
        // Test all getter methods
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        unsigned outputIndex = 1;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;

        // Add bounded modules
        auto linear1 = torch::nn::Linear(2, 3);
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear1));
        
        auto linear2 = torch::nn::Linear(3, 1);
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear2));
        
        // Add constants
        constants.append(torch::tensor({1.0, 2.0}));
        constants.append(torch::tensor({})); // Empty tensor
        
        // Add input indices
        inputIndices.append(0);
        
        // Add Marabou variables
        Vector<Variable> inputVars;
        inputVars.append(Variable(0));
        inputVars.append(Variable(1));
        marabouVars.append(inputVars);
        
        Vector<Variable> outputVars;
        outputVars.append(Variable(2));
        marabouVars.append(outputVars);
        
        neuronToMarabouMap[0] = inputVars;
        neuronToMarabouMap[1] = outputVars;

        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE;
        elementTypes[1] = ElementType::MODULE;
        elementToBoundedModuleIndex[0] = 0;
        elementToBoundedModuleIndex[1] = 1;

        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        // Test all getters
        TS_ASSERT_EQUALS(model.getBoundedModules().size(), 2u);
        TS_ASSERT_EQUALS(model.getVariables().size(), 2u);
        TS_ASSERT_EQUALS(model.getInputIndices().size(), 1u);
        TS_ASSERT_EQUALS(model.getOutputIndex(), 1u);
        TS_ASSERT_EQUALS(model.getSize(), 2u);
        TS_ASSERT_EQUALS(model.getNeuronToMarabouMap().size(), 2u);
        
        // Test specific values
        TS_ASSERT_EQUALS(model.getInputIndices()[0], 0u);
        TS_ASSERT_EQUALS(model.getVariables()[0].size(), 2u);
        TS_ASSERT_EQUALS(model.getVariables()[1].size(), 1u);
    }

    void test_error_handling()
    {
        // Test error handling for invalid inputs
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices; // Empty input indices
        unsigned outputIndex = 0;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;

        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        // Test forward pass with no input indices
        torch::Tensor input = torch::tensor({1.0, 2.0});
        TS_ASSERT_THROWS_EQUALS(
            model.forward(input),
            const MarabouError& e,
            e.getCode(),
            MarabouError::TORCH_MODEL_ERROR
        );
    }

    void test_eliminated_neuron_value()
    {
        // Test getting eliminated neuron values
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        unsigned outputIndex = 0;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;

        auto linear = torch::nn::Linear(2, 3);
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear));
        inputIndices.append(0);
        
        Vector<Variable> testVars;
        testVars.append(Variable(0));
        marabouVars.append(testVars);
        
        neuronToMarabouMap[0] = testVars;

        // Set up element types and mappings
        elementTypes[0] = ElementType::MODULE;
        elementToBoundedModuleIndex[0] = 0;

        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        // Eliminate a variable
        model.eliminateVariable(Variable(0), 7.5);
        
        // Test getting the eliminated value
        TS_ASSERT_DELTA(model.getEliminatedNeuronValue(0), 7.5, DELTA);
        
        // Test that non-eliminated neurons throw error
        TS_ASSERT_THROWS(model.getEliminatedNeuronValue(1), const MarabouError&);
    }
};

#endif // __TEST_TORCH_MODEL_H__
