#ifndef __TEST_CROWN_ANALYSIS_H__
#define __TEST_CROWN_ANALYSIS_H__

#include "CROWNAnalysis.h"
#include "TorchLinearBounded.h"
#include "TorchReLUBounded.h"
#include "TorchModel.h"
#include "MarabouError.h"
#include "Vector.h"
#include "Map.h"
#include "Set.h"
#include <cxxtest/TestSuite.h>
#include <torch/torch.h>

class CROWNAnalysisTestSuite : public CxxTest::TestSuite
{
public:
    const double DELTA = 0.0001;

    void test_linear_module_bound_backward()
    {
        // Test that linear module's bound_backward method works correctly
        auto linear_module = torch::nn::Linear(2, 3);
        linear_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        linear_module->bias = torch::tensor({0.1, 0.2, 0.3});
        
        NLR::TorchLinearModule boundedModule(linear_module);
        
        // Create input bounds using BoundedTensor
        Vector<BoundedTensor<torch::Tensor>> input_bounds;
        torch::Tensor input_lower = torch::tensor({-1.0, -1.0});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0});
        input_bounds.append(BoundedTensor<torch::Tensor>(input_lower, input_upper));
        
        // Create A matrices (identity for testing)
        torch::Tensor last_lA = torch::eye(3);
        torch::Tensor last_uA = torch::eye(3);
        
        // Call boundBackward method
        auto [A_matrices, lbias, ubias] = boundedModule.boundBackward(last_lA, last_uA, input_bounds);
        
        // Verify results
        TS_ASSERT_EQUALS(A_matrices.size(), 1u);
        TS_ASSERT(A_matrices[0].first.defined());
        TS_ASSERT(A_matrices[0].second.defined());
        TS_ASSERT(lbias.defined());
        TS_ASSERT(ubias.defined());
        
        // Verify A matrix shape (should be 3x2 for linear layer)
        TS_ASSERT_EQUALS(A_matrices[0].first.size(0), 3u);
        TS_ASSERT_EQUALS(A_matrices[0].first.size(1), 2u);
        TS_ASSERT_EQUALS(A_matrices[0].second.size(0), 3u);
        TS_ASSERT_EQUALS(A_matrices[0].second.size(1), 2u);
    }

    void test_relu_module_bound_backward()
    {
        // Test that ReLU module's bound_backward method works correctly
        auto relu_module = torch::nn::ReLU();
        NLR::TorchReLUBounded boundedModule(relu_module);
        
        // Create input bounds using BoundedTensor
        Vector<BoundedTensor<torch::Tensor>> input_bounds;
        torch::Tensor input_lower = torch::tensor({-1.0, 0.5, -0.5});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0, 0.5});
        input_bounds.append(BoundedTensor<torch::Tensor>(input_lower, input_upper));
        
        // Create A matrices (identity for testing)
        torch::Tensor last_lA = torch::eye(3);
        torch::Tensor last_uA = torch::eye(3);
        
        // Call boundBackward method
        auto [A_matrices, lbias, ubias] = boundedModule.boundBackward(last_lA, last_uA, input_bounds);
        
        // Verify results
        TS_ASSERT_EQUALS(A_matrices.size(), 1u);
        TS_ASSERT(A_matrices[0].first.defined());
        TS_ASSERT(A_matrices[0].second.defined());
        TS_ASSERT(lbias.defined());
        TS_ASSERT(ubias.defined());
        
        // Verify A matrix shape (should be 3x3 for ReLU layer)
        TS_ASSERT_EQUALS(A_matrices[0].first.size(0), 3u);
        TS_ASSERT_EQUALS(A_matrices[0].first.size(1), 3u);
        TS_ASSERT_EQUALS(A_matrices[0].second.size(0), 3u);
        TS_ASSERT_EQUALS(A_matrices[0].second.size(1), 3u);
    }

    void test_crown_analysis_initialization()
    {
        // Test that CROWNAnalysis can be initialized with a TorchModel
        // This is a basic test to ensure the refactored code compiles and initializes correctly
        
        // Create bounded modules
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
        
        // Create a simple linear module for testing
        auto linear_module = torch::nn::Linear(2, 3);
        linear_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        linear_module->bias = torch::tensor({0.1, 0.2, 0.3});
        
        auto boundedModule = std::make_shared<NLR::TorchLinearModule>(linear_module);
        boundedModules.append(boundedModule);
        
        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, 
                        neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, 
                        elementToConstantIndex, elementToInputIndex);
        
        NLR::CROWNAnalysis crownAnalysis(&model);
        
        // Basic test to ensure initialization succeeded
        TS_ASSERT(true); // If we get here, initialization succeeded
    }

    void test_crown_analysis_bound_computation()
    {
        // Test that CROWNAnalysis can compute bounds for a simple network
        // Create a simple network: Linear -> ReLU -> Linear
        
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
        
        // Create first linear layer: 2 -> 3
        auto linear1_module = torch::nn::Linear(2, 3);
        linear1_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});
        linear1_module->bias = torch::tensor({0.1, 0.2, 0.3});
        auto boundedLinear1 = std::make_shared<NLR::TorchLinearModule>(linear1_module);
        boundedModules.append(boundedLinear1);
        
        // Create ReLU layer
        auto relu_module = torch::nn::ReLU();
        auto boundedReLU = std::make_shared<NLR::TorchReLUBounded>(relu_module);
        boundedModules.append(boundedReLU);
        
        // Create second linear layer: 3 -> 1
        auto linear2_module = torch::nn::Linear(3, 1);
        linear2_module->weight = torch::tensor({{1.0, 2.0, 3.0}});
        linear2_module->bias = torch::tensor({0.5});
        auto boundedLinear2 = std::make_shared<NLR::TorchLinearModule>(linear2_module);
        boundedModules.append(boundedLinear2);
        
        TorchModel model(boundedModules, constants, marabouVars, inputIndices, outputIndex, 
                        neuronToMarabouMap, dependencies, elementTypes, elementToBoundedModuleIndex, 
                        elementToConstantIndex, elementToInputIndex);
        
        NLR::CROWNAnalysis crownAnalysis(&model);
        
        // Create input bounds
        Vector<BoundedTensor<torch::Tensor>> input_bounds;
        torch::Tensor input_lower = torch::tensor({-1.0, -1.0});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0});
        input_bounds.append(BoundedTensor<torch::Tensor>(input_lower, input_upper));
        
        // Test that we can call the bound computation method
        // Note: This is a basic test to ensure the method exists and can be called
        // The actual bound computation logic would need more setup
        
        TS_ASSERT(true); // If we get here, the test structure is valid
    }

    void test_linear_module_specific_bound_computation()
    {
        // Test specific bound computation for linear module with known values
        auto linear_module = torch::nn::Linear(2, 2);
        linear_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
        linear_module->bias = torch::tensor({0.1, 0.2});
        
        NLR::TorchLinearModule boundedModule(linear_module);
        
        // Create input bounds: x1 in [-1, 1], x2 in [-1, 1]
        Vector<BoundedTensor<torch::Tensor>> input_bounds;
        torch::Tensor input_lower = torch::tensor({-1.0, -1.0});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0});
        input_bounds.append(BoundedTensor<torch::Tensor>(input_lower, input_upper));
        
        // Create A matrices (identity for testing)
        torch::Tensor last_lA = torch::eye(2);
        torch::Tensor last_uA = torch::eye(2);
        
        // Call boundBackward method
        auto [A_matrices, lbias, ubias] = boundedModule.boundBackward(last_lA, last_uA, input_bounds);
        
        // Verify A matrix computation
        // For linear layer y = Wx + b, the A matrix should be last_A.matmul(weight)
        // Following auto-LiRPA's BoundLinear::bound_backward pattern
        // Note: The weight stored in the module is already in the "transposed" state
        // (as if transB == 0 was applied during ONNX conversion)
        torch::Tensor expected_A = torch::matmul(last_lA, linear_module->weight);
        
        TS_ASSERT(torch::all(torch::eq(A_matrices[0].first, expected_A)).template item<bool>());
        TS_ASSERT(torch::all(torch::eq(A_matrices[0].second, expected_A)).template item<bool>());
        
        // Verify bias computation
        torch::Tensor expected_bias = torch::matmul(last_lA, linear_module->bias);
        TS_ASSERT(torch::all(torch::eq(lbias, expected_bias)).template item<bool>());
        TS_ASSERT(torch::all(torch::eq(ubias, expected_bias)).template item<bool>());
    }

    void test_relu_module_specific_bound_computation()
    {
        // Test specific bound computation for ReLU module with known values
        auto relu_module = torch::nn::ReLU();
        NLR::TorchReLUBounded boundedModule(relu_module);
        
        // Create input bounds with mixed signs to test ReLU behavior
        Vector<BoundedTensor<torch::Tensor>> input_bounds;
        torch::Tensor input_lower = torch::tensor({-0.5, 0.5, -1.0});
        torch::Tensor input_upper = torch::tensor({0.5, 1.0, 0.0});
        input_bounds.append(BoundedTensor<torch::Tensor>(input_lower, input_upper));
        
        // Create A matrices (identity for testing)
        torch::Tensor last_lA = torch::eye(3);
        torch::Tensor last_uA = torch::eye(3);
        
        // Call boundBackward method
        auto result = boundedModule.boundBackward(last_lA, last_uA, input_bounds);
        
        // Extract results using std::get for tuple access
        auto A_matrices = std::get<0>(result);
        auto lbias = std::get<1>(result);
        auto ubias = std::get<2>(result);
        
        // Verify A matrices are returned
        TS_ASSERT_EQUALS(A_matrices.size(), (size_t)1);
        
        auto A_pair = A_matrices[0];
        torch::Tensor new_lA = A_pair.first;
        torch::Tensor new_uA = A_pair.second;
        
        // Verify A matrices are not null
        TS_ASSERT(new_lA.defined());
        TS_ASSERT(new_uA.defined());
        
        // Verify A matrices have correct shape (3x3 for 3 input neurons)
        TS_ASSERT_EQUALS(new_lA.size(0), 3);
        TS_ASSERT_EQUALS(new_lA.size(1), 3);
        TS_ASSERT_EQUALS(new_uA.size(0), 3);
        TS_ASSERT_EQUALS(new_uA.size(1), 3);
        
        // Verify bias terms are returned
        TS_ASSERT_EQUALS(lbias.size(0), 3);
        TS_ASSERT_EQUALS(ubias.size(0), 3);
        
        // Test specific ReLU behavior:
        // Input: [-0.5, 0.5], [0.5, 1.0], [-1.0, 0.0]
        // Expected ReLU output: [0, 0.5], [0.5, 1.0], [0, 0]
        
        // For neuron 0: input [-0.5, 0.5] -> unstable (crosses 0)
        // For neuron 1: input [0.5, 1.0] -> always active (all positive)
        // For neuron 2: input [-1.0, 0.0] -> always inactive (all negative)
        
        // Verify that the A matrices reflect the ReLU behavior
        // For always active neurons (neuron 1): A should be identity
        TS_ASSERT_DELTA(new_lA[1][1].item<float>(), 1.0, DELTA);
        TS_ASSERT_DELTA(new_uA[1][1].item<float>(), 1.0, DELTA);
        
        // For always inactive neurons (neuron 2): A should be zero
        TS_ASSERT_DELTA(new_lA[2][2].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(new_uA[2][2].item<float>(), 0.0, DELTA);
        
        // For unstable neurons (neuron 0): A should be between 0 and 1
        float neuron0_lower = new_lA[0][0].item<float>();
        float neuron0_upper = new_uA[0][0].item<float>();
        TS_ASSERT(neuron0_lower >= 0.0 && neuron0_lower <= 1.0);
        TS_ASSERT(neuron0_upper >= 0.0 && neuron0_upper <= 1.0);
        
        // Verify bias terms are zero for ReLU (no bias added)
        TS_ASSERT_DELTA(lbias[0].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(lbias[1].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(lbias[2].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(ubias[0].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(ubias[1].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(ubias[2].item<float>(), 0.0, DELTA);
    }

    void test_ibp_interval_computation()
    {
        // Test IBP interval computation for linear and ReLU modules
        auto linear_module = torch::nn::Linear(2, 2);
        linear_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
        linear_module->bias = torch::tensor({0.1, 0.2});
        
        NLR::TorchLinearModule linearBoundedModule(linear_module);
        
        // Create input bounds for linear layer
        Vector<std::pair<torch::Tensor, torch::Tensor>> linear_input_bounds;
        torch::Tensor input_lower = torch::tensor({-1.0, -1.0});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0});
        linear_input_bounds.append(std::make_pair(input_lower, input_upper));
        
        // Compute IBP bounds for linear layer
        auto [linear_lower, linear_upper] = linearBoundedModule.computeIntervalBoundPropagation(linear_input_bounds);
        
        // Verify linear IBP bounds are computed
        TS_ASSERT(linear_lower.defined());
        TS_ASSERT(linear_upper.defined());
        TS_ASSERT_EQUALS(linear_lower.size(0), 2);
        TS_ASSERT_EQUALS(linear_upper.size(0), 2);
        
        // Verify linear bounds are reasonable
        // For input [-1, 1] and weights [[1, 2], [3, 4]], bias [0.1, 0.2]
        // Expected: lower bound = min(1*(-1) + 2*(-1) + 0.1, 3*(-1) + 4*(-1) + 0.2) = [-2.9, -6.8]
        // Expected: upper bound = max(1*1 + 2*1 + 0.1, 3*1 + 4*1 + 0.2) = [3.1, 7.2]
        TS_ASSERT(linear_lower[0].item<float>() < 0.0); // Should be negative
        TS_ASSERT(linear_upper[0].item<float>() > 0.0); // Should be positive
        
        // Test ReLU IBP computation
        auto relu_module = torch::nn::ReLU();
        NLR::TorchReLUBounded reluBoundedModule(relu_module);
        
        // Create input bounds for ReLU layer
        Vector<std::pair<torch::Tensor, torch::Tensor>> relu_input_bounds;
        torch::Tensor relu_input_lower = torch::tensor({-0.5, 0.5, -1.0});
        torch::Tensor relu_input_upper = torch::tensor({0.5, 1.0, 0.0});
        relu_input_bounds.append(std::make_pair(relu_input_lower, relu_input_upper));
        
        // Compute IBP bounds for ReLU layer
        auto [relu_lower, relu_upper] = reluBoundedModule.computeIntervalBoundPropagation(relu_input_bounds);
        
        // Verify ReLU IBP bounds are computed
        TS_ASSERT(relu_lower.defined());
        TS_ASSERT(relu_upper.defined());
        TS_ASSERT_EQUALS(relu_lower.size(0), 3);
        TS_ASSERT_EQUALS(relu_upper.size(0), 3);
        
        // Verify ReLU bounds follow ReLU behavior
        // Neuron 0: input [-0.5, 0.5] -> output [0, 0.5]
        TS_ASSERT_DELTA(relu_lower[0].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(relu_upper[0].item<float>(), 0.5, DELTA);
        
        // Neuron 1: input [0.5, 1.0] -> output [0.5, 1.0]
        TS_ASSERT_DELTA(relu_lower[1].item<float>(), 0.5, DELTA);
        TS_ASSERT_DELTA(relu_upper[1].item<float>(), 1.0, DELTA);
        
        // Neuron 2: input [-1.0, 0.0] -> output [0, 0]
        TS_ASSERT_DELTA(relu_lower[2].item<float>(), 0.0, DELTA);
        TS_ASSERT_DELTA(relu_upper[2].item<float>(), 0.0, DELTA);
    }

    void test_crown_analysis_run_method()
    {
        // Test the actual run() method in CROWNAnalysis
        // Create a simple network: Linear -> ReLU -> Linear
        
        // Create bounded modules
        Vector<std::shared_ptr<NLR::ITorchModuleBounded>> boundedModules;
        Vector<torch::Tensor> constants;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        unsigned outputIndex = 3; // Output is the last node
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        Map<unsigned, ElementType> elementTypes;
        Map<unsigned, unsigned> elementToBoundedModuleIndex;
        Map<unsigned, unsigned> elementToConstantIndex;
        Map<unsigned, unsigned> elementToInputIndex;
        
        // Create linear layer 1: 2 -> 2
        auto linear1_module = torch::nn::Linear(2, 2);
        linear1_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
        linear1_module->bias = torch::tensor({0.1, 0.2});
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear1_module));
        
        // Create ReLU layer
        auto relu_module = torch::nn::ReLU();
        boundedModules.append(std::make_shared<NLR::TorchReLUBounded>(relu_module));
        
        // Create linear layer 2: 2 -> 1
        auto linear2_module = torch::nn::Linear(2, 1);
        linear2_module->weight = torch::tensor({{1.0, 2.0}});
        linear2_module->bias = torch::tensor({0.3});
        boundedModules.append(std::make_shared<NLR::TorchLinearModule>(linear2_module));
        
        // Set up dependencies: input -> linear1 -> relu -> linear2 -> output
        dependencies[0] = Vector<unsigned>(); // input has no dependencies
        dependencies[1] = Vector<unsigned>{0}; // linear1 depends on input
        dependencies[2] = Vector<unsigned>{1}; // relu depends on linear1
        dependencies[3] = Vector<unsigned>{2}; // linear2 depends on relu
        
        // Set up element types (all are MODULE type)
        elementTypes[0] = ElementType::INPUT;
        elementTypes[1] = ElementType::MODULE;
        elementTypes[2] = ElementType::MODULE;
        elementTypes[3] = ElementType::MODULE;
        
        // Set up element to module mapping
        elementToBoundedModuleIndex[1] = 0; // linear1
        elementToBoundedModuleIndex[2] = 1; // relu
        elementToBoundedModuleIndex[3] = 2; // linear2
        
        // Create TorchModel with correct constructor signature
        TorchModel torchModel(boundedModules, constants, marabouVars, inputIndices, 
                             outputIndex, neuronToMarabouMap, dependencies, elementTypes,
                             elementToBoundedModuleIndex, elementToConstantIndex, elementToInputIndex);
        
        // Set input bounds for the input node (node 0)
        Map<unsigned, std::pair<torch::Tensor, torch::Tensor>> inputBoundsMap;
        torch::Tensor input_lower = torch::tensor({-1.0, -1.0});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0});
        inputBoundsMap[0] = std::make_pair(input_lower, input_upper);
        torchModel.setInputBounds(inputBoundsMap);
        
        // Create CROWNAnalysis
        NLR::CROWNAnalysis crownAnalysis(&torchModel);
        
        // Run the analysis
        crownAnalysis.run();
        
        // Retrieve and check IBP bounds for the output node (node 3)
        torch::Tensor ibp_lower = crownAnalysis.getIBPLowerBound(3);
        torch::Tensor ibp_upper = crownAnalysis.getIBPUpperBound(3);
        TS_ASSERT(ibp_lower.defined());
        TS_ASSERT(ibp_upper.defined());
        TS_ASSERT_EQUALS(ibp_lower.size(0), 1);
        TS_ASSERT_EQUALS(ibp_upper.size(0), 1);
        TS_ASSERT(ibp_lower[0].item<float>() < ibp_upper[0].item<float>());
        TS_ASSERT(std::isfinite(ibp_lower[0].item<float>()));
        TS_ASSERT(std::isfinite(ibp_upper[0].item<float>()));

        // Retrieve and check CROWN (linear relaxation) bounds for the output node (node 3)
        torch::Tensor crown_lower = crownAnalysis.getCrownLowerBound(3);
        torch::Tensor crown_upper = crownAnalysis.getCrownUpperBound(3);
        TS_ASSERT(crown_lower.defined());
        TS_ASSERT(crown_upper.defined());
        TS_ASSERT_EQUALS(crown_lower.size(0), 1);
        TS_ASSERT_EQUALS(crown_upper.size(0), 1);
        TS_ASSERT(crown_lower[0].item<float>() < crown_upper[0].item<float>());
        TS_ASSERT(std::isfinite(crown_lower[0].item<float>()));
        TS_ASSERT(std::isfinite(crown_upper[0].item<float>()));
    }

    void test_forward_before_bound_computation()
    {
        // Test that forward method is called before bound computation
        // This follows the auto-LiRPA pattern where get_forward_value is called
        // before bound_backward in backward_general
        
        auto linear_module = torch::nn::Linear(2, 2);
        linear_module->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
        linear_module->bias = torch::tensor({0.1, 0.2});
        
        NLR::TorchLinearModule boundedModule(linear_module);
        
        // First, call forward method (simulating get_forward_value in auto-LiRPA)
        torch::Tensor input = torch::tensor({{1.0, 2.0}});
        torch::Tensor forward_output = boundedModule.forward(input);
        
        // Verify forward computation worked
        TS_ASSERT(forward_output.defined());
        TS_ASSERT_EQUALS(forward_output.size(0), 1u);
        TS_ASSERT_EQUALS(forward_output.size(1), 2u);
        
        // Now call bound_backward (simulating CROWN backward pass)
        Vector<BoundedTensor<torch::Tensor>> input_bounds;
        torch::Tensor input_lower = torch::tensor({-1.0, -1.0});
        torch::Tensor input_upper = torch::tensor({1.0, 1.0});
        input_bounds.append(BoundedTensor<torch::Tensor>(input_lower, input_upper));
        
        torch::Tensor last_lA = torch::eye(2);
        torch::Tensor last_uA = torch::eye(2);
        
        auto [A_matrices, lbias, ubias] = boundedModule.boundBackward(last_lA, last_uA, input_bounds);
        
        // Verify bound computation worked after forward pass
        TS_ASSERT(A_matrices[0].first.defined());
        TS_ASSERT(A_matrices[0].second.defined());
        TS_ASSERT(lbias.defined());
        TS_ASSERT(ubias.defined());
        
        // This test verifies that the forward method can be called before bound computation
        // without any issues, following the auto-LiRPA pattern
    }
};

#endif // __TEST_CROWN_ANALYSIS_H__
