
#include "CROWNAnalysis.h"
#include "TorchModel.h"
#include "BoundedInputNode.h"
#include "BoundedLinearNode.h"
#include "BoundedReLUNode.h"
#include "BoundedConstantNode.h"
#include "FloatUtils.h"
#include "InputQueryBuilder.h"
#include "MStringf.h"

#include <cxxtest/TestSuite.h>
#include <torch/torch.h>

class CROWNAnalysisTestSuite : public CxxTest::TestSuite
{
public:
    void setUp()
    {
    }

    void tearDown()
    {
    }

    void testLinearLayerBoundBackward()
    {
        // Test parameters:
        // Weight matrix: [[1,1],[0,1]]
        // Bias: [1,0]
        // Input bounds: [0,4] and [0,2]
        // Expected: A matrices should equal weight matrix, bias should equal layer bias
        
        // Create weight matrix [[1,1],[0,1]]
        torch::Tensor weight = torch::tensor({{1.0f, 1.0f}, {0.0f, 1.0f}}, torch::kFloat32);
        
        // Create bias [1,0]
        torch::Tensor bias = torch::tensor({1.0f, 0.0f}, torch::kFloat32);
        
        // Create linear module
        torch::nn::Linear linearModule(2, 2);
        linearModule->weight = weight;
        linearModule->bias = bias;
        
        // Create bounded linear node
        std::shared_ptr<NLR::BoundedLinearNode> linearNode = 
            std::make_shared<NLR::BoundedLinearNode>(linearModule, 1.0f, "test_linear");
        
        // Set node properties
        linearNode->setNodeIndex(1);
        linearNode->setInputSize(2);
        linearNode->setOutputSize(2);
        
        // Create input bounds: [0,4] and [0,2]
        torch::Tensor inputLower = torch::tensor({0.0f, 0.0f}, torch::kFloat32);
        torch::Tensor inputUpper = torch::tensor({4.0f, 2.0f}, torch::kFloat32);
        BoundedTensor<torch::Tensor> inputBounds(inputLower, inputUpper);
        
        // Create input bounds vector
        Vector<BoundedTensor<torch::Tensor>> inputBoundsVector;
        inputBoundsVector.append(inputBounds);
        
        // Create identity A matrices (2x2 identity for 2 outputs)
        torch::Tensor identityA = torch::eye(2, torch::kFloat32).unsqueeze(0); // Shape [1, 2, 2]
        
        // Output containers
        Vector<Pair<torch::Tensor, torch::Tensor>> outputA_matrices;
        torch::Tensor lbias, ubias;
        
        // Call boundBackward method
        linearNode->boundBackward(identityA, identityA, inputBoundsVector, outputA_matrices, lbias, ubias);
        
        // Verify results
        TS_ASSERT_EQUALS(outputA_matrices.size(), (unsigned)1);
        
        // Get the computed A matrices
        torch::Tensor computed_lA = outputA_matrices[0].first();
        torch::Tensor computed_uA = outputA_matrices[0].second();
        
        // Expected A matrices should equal the weight matrix
        torch::Tensor expectedA = weight; // [[1,1],[0,1]]
        
        // Debug output
        std::cout << "Test Linear Layer:" << std::endl;
        std::cout << "Weight matrix: " << weight << std::endl;
        std::cout << "Bias: " << bias << std::endl;
        std::cout << "Input bounds - Lower: " << inputLower << ", Upper: " << inputUpper << std::endl;
        std::cout << "Identity A matrix: " << identityA << std::endl;
        std::cout << "Computed lA: " << computed_lA << std::endl;
        std::cout << "Computed uA: " << computed_uA << std::endl;
        std::cout << "Expected A: " << expectedA << std::endl;
        std::cout << "Computed lbias: " << lbias << std::endl;
        std::cout << "Computed ubias: " << ubias << std::endl;
        std::cout << "Expected bias: " << bias << std::endl;
        
        // Check A matrix shapes
        TS_ASSERT_EQUALS(computed_lA.dim(), 3); // Should be [batch, spec, input_dim]
        TS_ASSERT_EQUALS(computed_uA.dim(), 3);
        TS_ASSERT_EQUALS(computed_lA.size(0), 1); // batch size
        TS_ASSERT_EQUALS(computed_lA.size(1), 2); // spec size (output dimension)
        TS_ASSERT_EQUALS(computed_lA.size(2), 2); // input dimension
        
        // Remove batch dimension for comparison
        torch::Tensor lA_2d = computed_lA.squeeze(0); // Shape [2, 2]
        torch::Tensor uA_2d = computed_uA.squeeze(0); // Shape [2, 2]
        
        // Check that A matrices equal the weight matrix
        TS_ASSERT(torch::allclose(lA_2d, expectedA, 1e-6));
        TS_ASSERT(torch::allclose(uA_2d, expectedA, 1e-6));
        
        // Check bias terms
        TS_ASSERT(lbias.defined());
        TS_ASSERT(ubias.defined());
        TS_ASSERT_EQUALS(lbias.numel(), 2);
        TS_ASSERT_EQUALS(ubias.numel(), 2);
        
        // Expected bias should be the layer bias transformed by the identity A matrix
        // Since A is identity, bias should remain the same
        torch::Tensor expectedBias = bias; // [1, 0]
        
        TS_ASSERT(torch::allclose(lbias, expectedBias, 1e-6));
        TS_ASSERT(torch::allclose(ubias, expectedBias, 1e-6));
        
        std::cout << "Linear layer test PASSED!" << std::endl;
    }

    void testLinearLayerIBP()
    {
        // Test IBP computation for the same linear layer
        
        // Create weight matrix [[1,1],[0,1]]
        torch::Tensor weight = torch::tensor({{1.0f, 1.0f}, {0.0f, 1.0f}}, torch::kFloat32);
        
        // Create bias [1,0]
        torch::Tensor bias = torch::tensor({1.0f, 0.0f}, torch::kFloat32);
        
        // Create linear module
        torch::nn::Linear linearModule(2, 2);
        linearModule->weight = weight;
        linearModule->bias = bias;
        
        // Create bounded linear node
        std::shared_ptr<NLR::BoundedLinearNode> linearNode = 
            std::make_shared<NLR::BoundedLinearNode>(linearModule, 1.0f, "test_linear_ibp");
        
        // Set node properties
        linearNode->setNodeIndex(1);
        linearNode->setInputSize(2);
        linearNode->setOutputSize(2);
        
        // Create input bounds: [0,4] and [0,2]
        torch::Tensor inputLower = torch::tensor({0.0f, 0.0f}, torch::kFloat32);
        torch::Tensor inputUpper = torch::tensor({4.0f, 2.0f}, torch::kFloat32);
        BoundedTensor<torch::Tensor> inputBounds(inputLower, inputUpper);
        
        // Create input bounds vector
        Vector<BoundedTensor<torch::Tensor>> inputBoundsVector;
        inputBoundsVector.append(inputBounds);
        
        // Call IBP method
        BoundedTensor<torch::Tensor> ibpResult = linearNode->computeIntervalBoundPropagation(inputBoundsVector);
        
        // Manual calculation for verification:
        // y = W*x + b where W = [[1,1],[0,1]], b = [1,0]
        // For input bounds [0,4] and [0,2]:
        // y1 = 1*x1 + 1*x2 + 1 = x1 + x2 + 1
        // y2 = 0*x1 + 1*x2 + 0 = x2
        
        // For IBP with positive weights:
        // y1_lower = 1*0 + 1*0 + 1 = 1
        // y1_upper = 1*4 + 1*2 + 1 = 7
        // y2_lower = 0*0 + 1*0 + 0 = 0  
        // y2_upper = 0*4 + 1*2 + 0 = 2
        
        torch::Tensor expectedLower = torch::tensor({1.0f, 0.0f}, torch::kFloat32);
        torch::Tensor expectedUpper = torch::tensor({7.0f, 2.0f}, torch::kFloat32);
        
        // Debug output
        std::cout << "IBP Test:" << std::endl;
        std::cout << "Input bounds - Lower: " << inputLower << ", Upper: " << inputUpper << std::endl;
        std::cout << "Computed IBP - Lower: " << ibpResult.lower() << ", Upper: " << ibpResult.upper() << std::endl;
        std::cout << "Expected IBP - Lower: " << expectedLower << ", Upper: " << expectedUpper << std::endl;
        
        // Check IBP results
        TS_ASSERT(torch::allclose(ibpResult.lower(), expectedLower, 1e-6));
        TS_ASSERT(torch::allclose(ibpResult.upper(), expectedUpper, 1e-6));
        
        std::cout << "Linear layer IBP test PASSED!" << std::endl;
    }

    void testReLULayerBoundBackward()
    {
        // Test parameters based on auto-LiRPA computation:
        // Input bounds: [0,4] and [-2,2]  
        // Previous A matrix: [[1,1],[0,1]] (weight matrix from linear layer)
        // Previous bias: [1,0] (will be handled by bias accumulation in full CROWN pass)
        // Expected D_lower: [[1,0],[0,0]] (diagonal matrix)
        // Expected D_upper: [[1,0],[0,0.5]] (diagonal matrix)
        // Expected A_lower: [[1,0],[0,0]]
        // Expected A_upper: [[1,0.5],[0,0.5]]
        // Expected ReLU lower bias: [0,0] (ReLU lower bounds have no bias)
        // Expected ReLU upper bias: [1,1] (from ReLU relaxation)
        
        // Create ReLU module
        torch::nn::ReLU reluModule;
        
        // Create bounded ReLU node
        std::shared_ptr<NLR::BoundedReLUNode> reluNode = 
            std::make_shared<NLR::BoundedReLUNode>(reluModule, "test_relu");
        
        // Set node properties
        reluNode->setNodeIndex(2);
        reluNode->setInputSize(2);
        reluNode->setOutputSize(2);
        
        // Create input bounds: [0,4] and [-2,2]
        torch::Tensor inputLower = torch::tensor({0.0f, -2.0f}, torch::kFloat32);
        torch::Tensor inputUpper = torch::tensor({4.0f, 2.0f}, torch::kFloat32);
        BoundedTensor<torch::Tensor> inputBounds(inputLower, inputUpper);
        
        // Create input bounds vector
        Vector<BoundedTensor<torch::Tensor>> inputBoundsVector;
        inputBoundsVector.append(inputBounds);
        
        // Create previous A matrix: [[1,1],[0,1]] (weight matrix from linear layer)
        torch::Tensor prevA = torch::tensor({{1.0f, 1.0f}, {0.0f, 1.0f}}, torch::kFloat32).unsqueeze(0); // Shape [1, 2, 2]
        
        // Create previous bias: [1,0]
        torch::Tensor prevBias = torch::tensor({1.0f, 0.0f}, torch::kFloat32);
        
        // Output containers
        Vector<Pair<torch::Tensor, torch::Tensor>> outputA_matrices;
        torch::Tensor lbias, ubias;
        
        // Call boundBackward method
        reluNode->boundBackward(prevA, prevA, inputBoundsVector, outputA_matrices, lbias, ubias);
        
        // Verify results
        TS_ASSERT_EQUALS(outputA_matrices.size(), (unsigned)1);
        
        // Get the computed A matrices
        torch::Tensor computed_lA = outputA_matrices[0].first();
        torch::Tensor computed_uA = outputA_matrices[0].second();
        
        // Expected A matrices based on the specifications
        torch::Tensor expected_lA = torch::tensor({{1.0f, 0.0f}, {0.0f, 0.0f}}, torch::kFloat32);
        torch::Tensor expected_uA = torch::tensor({{1.0f, 0.5f}, {0.0f, 0.5f}}, torch::kFloat32);
        
        // Expected bias terms (based on auto-LiRPA computation):
        // ReLU only adds its relaxation bias, previous layer bias handled separately
        torch::Tensor expected_lbias = torch::tensor({0.0f, 0.0f}, torch::kFloat32); // ReLU lower has no bias
        torch::Tensor expected_ubias = torch::tensor({1.0f, 1.0f}, torch::kFloat32); // ReLU upper relaxation bias
        
        // Debug output
        std::cout << "Test ReLU Layer:" << std::endl;
        std::cout << "Input bounds - Lower: " << inputLower << ", Upper: " << inputUpper << std::endl;
        std::cout << "Previous A matrix: " << prevA.squeeze(0) << std::endl;
        std::cout << "Previous bias: " << prevBias << std::endl;
        std::cout << "Computed lA: " << computed_lA.squeeze(0) << std::endl;
        std::cout << "Computed uA: " << computed_uA.squeeze(0) << std::endl;
        std::cout << "Expected lA: " << expected_lA << std::endl;
        std::cout << "Expected uA: " << expected_uA << std::endl;
        std::cout << "Computed lbias: " << lbias << std::endl;
        std::cout << "Computed ubias: " << ubias << std::endl;
        std::cout << "Expected lbias: " << expected_lbias << std::endl;
        std::cout << "Expected ubias: " << expected_ubias << std::endl;
        
        // Check A matrix shapes
        TS_ASSERT_EQUALS(computed_lA.dim(), 3); // Should be [batch, spec, input_dim]
        TS_ASSERT_EQUALS(computed_uA.dim(), 3);
        TS_ASSERT_EQUALS(computed_lA.size(0), 1); // batch size
        TS_ASSERT_EQUALS(computed_lA.size(1), 2); // spec size (output dimension)
        TS_ASSERT_EQUALS(computed_lA.size(2), 2); // input dimension
        
        // Remove batch dimension for comparison
        torch::Tensor lA_2d = computed_lA.squeeze(0); // Shape [2, 2]
        torch::Tensor uA_2d = computed_uA.squeeze(0); // Shape [2, 2]
        
        // Check that A matrices match expected values
        TS_ASSERT(torch::allclose(lA_2d, expected_lA, 1e-6));
        TS_ASSERT(torch::allclose(uA_2d, expected_uA, 1e-6));
        
        // Check bias terms
        TS_ASSERT(lbias.defined());
        TS_ASSERT(ubias.defined());
        TS_ASSERT_EQUALS(lbias.numel(), 2);
        TS_ASSERT_EQUALS(ubias.numel(), 2);
        
        // Check bias values
        TS_ASSERT(torch::allclose(lbias, expected_lbias, 1e-6));
        TS_ASSERT(torch::allclose(ubias, expected_ubias, 1e-6));
        
        // Test the _backwardRelaxation helper method directly
        auto [d_lower, d_upper, bias_lower, bias_upper] = reluNode->_backwardRelaxation(inputLower, inputUpper);
        
        // Expected slope vectors from _backwardRelaxation (no diagonal construction)
        torch::Tensor expected_d_lower = torch::tensor({1.0f, 0.0f}, torch::kFloat32);
        torch::Tensor expected_d_upper = torch::tensor({1.0f, 0.5f}, torch::kFloat32);
        
        std::cout << "Direct _backwardRelaxation test:" << std::endl;
        std::cout << "Computed d_lower: " << d_lower << std::endl;
        std::cout << "Computed d_upper: " << d_upper << std::endl;
        std::cout << "Expected d_lower: " << expected_d_lower << std::endl;
        std::cout << "Expected d_upper: " << expected_d_upper << std::endl;
        std::cout << "Computed bias_lower: " << bias_lower << std::endl;
        std::cout << "Computed bias_upper: " << bias_upper << std::endl;
        
        // Check slope vectors
        TS_ASSERT_EQUALS(d_lower.dim(), 1);
        TS_ASSERT_EQUALS(d_upper.dim(), 1);
        TS_ASSERT_EQUALS(d_lower.numel(), 2);
        TS_ASSERT_EQUALS(d_upper.numel(), 2);
        TS_ASSERT(torch::allclose(d_lower, expected_d_lower, 1e-6));
        TS_ASSERT(torch::allclose(d_upper, expected_d_upper, 1e-6));
        
        std::cout << "ReLU layer test PASSED!" << std::endl;
    }

    void testReLULayerIBP()
    {
        // Test IBP computation for the ReLU layer
        
        // Create ReLU module
        torch::nn::ReLU reluModule;
        
        // Create bounded ReLU node
        std::shared_ptr<NLR::BoundedReLUNode> reluNode = 
            std::make_shared<NLR::BoundedReLUNode>(reluModule, "test_relu_ibp");
        
        // Set node properties
        reluNode->setNodeIndex(2);
        reluNode->setInputSize(2);
        reluNode->setOutputSize(2);
        
        // Create input bounds: [0,4] and [-2,2]
        torch::Tensor inputLower = torch::tensor({0.0f, -2.0f}, torch::kFloat32);
        torch::Tensor inputUpper = torch::tensor({4.0f, 2.0f}, torch::kFloat32);
        BoundedTensor<torch::Tensor> inputBounds(inputLower, inputUpper);
        
        // Create input bounds vector
        Vector<BoundedTensor<torch::Tensor>> inputBoundsVector;
        inputBoundsVector.append(inputBounds);
        
        // Call IBP method
        BoundedTensor<torch::Tensor> ibpResult = reluNode->computeIntervalBoundPropagation(inputBoundsVector);
        
        // Manual calculation for verification:
        // ReLU: y = max(0, x)
        // For input bounds [0,4] and [-2,2]:
        // y1_lower = max(0, 0) = 0
        // y1_upper = max(0, 4) = 4
        // y2_lower = max(0, -2) = 0
        // y2_upper = max(0, 2) = 2
        
        torch::Tensor expectedLower = torch::tensor({0.0f, 0.0f}, torch::kFloat32);
        torch::Tensor expectedUpper = torch::tensor({4.0f, 2.0f}, torch::kFloat32);
        
        // Debug output
        std::cout << "ReLU IBP Test:" << std::endl;
        std::cout << "Input bounds - Lower: " << inputLower << ", Upper: " << inputUpper << std::endl;
        std::cout << "Computed IBP - Lower: " << ibpResult.lower() << ", Upper: " << ibpResult.upper() << std::endl;
        std::cout << "Expected IBP - Lower: " << expectedLower << ", Upper: " << expectedUpper << std::endl;
        
        // Check IBP results
        TS_ASSERT(torch::allclose(ibpResult.lower(), expectedLower, 1e-6));
        TS_ASSERT(torch::allclose(ibpResult.upper(), expectedUpper, 1e-6));
        
        std::cout << "ReLU layer IBP test PASSED!" << std::endl;
    }

};
