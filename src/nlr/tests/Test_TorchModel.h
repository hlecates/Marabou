#include "TorchModel.h"
#include "BoundedInputNode.h"
#include "BoundedLinearNode.h"
#include "BoundedReLUNode.h"
#include "BoundedConstantNode.h"
#include "FloatUtils.h"
#include "InputQuery.h"
#include "MStringf.h"

#include <cxxtest/TestSuite.h>
#include <torch/torch.h>

class TorchModelTestSuite : public CxxTest::TestSuite
{
public:
    void setUp()
    {
    }

    void tearDown()
    {
    }

    // Helper function to create a simple test model
    std::shared_ptr<NLR::TorchModel> createSimpleTestModel()
    {
        // Create a simple model: Input -> Linear -> ReLU -> Output
        // Model structure:
        // Node 0: Input (size 2)
        // Node 1: Linear (input 2, output 3)
        // Node 2: ReLU (input 3, output 3)
        
        Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
        Vector<Vector<Variable>> marabouVars;
        Vector<unsigned> inputIndices;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        Map<unsigned, Vector<unsigned>> dependencies;
        
        // Create input node (index 0)
        auto inputNode = std::make_shared<NLR::BoundedInputNode>(0, 2, "input");
        inputNode->setNodeIndex(0);
        inputNode->setInputSize(2);
        inputNode->setOutputSize(2);
        nodes.append(inputNode);
        
        // Create linear node (index 1) - need to create a torch::nn::Linear module
        torch::nn::Linear linearModule = torch::nn::Linear(2, 3);
        // Set weights and bias manually
        linearModule->weight = torch::tensor({{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, torch::kFloat32);
        linearModule->bias = torch::tensor({0.1, 0.2, 0.3}, torch::kFloat32);
        
        auto linearNode = std::make_shared<NLR::BoundedLinearNode>(linearModule, 1.0f, "linear");
        linearNode->setNodeIndex(1);
        linearNode->setInputSize(2);
        linearNode->setOutputSize(3);
        nodes.append(linearNode);
        
        // Create ReLU node (index 2) - need to create a torch::nn::ReLU module
        torch::nn::ReLU reluModule = torch::nn::ReLU();
        auto reluNode = std::make_shared<NLR::BoundedReLUNode>(reluModule, "relu");
        reluNode->setNodeIndex(2);
        reluNode->setInputSize(3);
        reluNode->setOutputSize(3);
        nodes.append(reluNode);
        
        // Set input indices
        inputIndices.append(0);
        
        // Set dependencies
        dependencies[1] = Vector<unsigned>{0};  // Linear depends on Input
        dependencies[2] = Vector<unsigned>{1};  // ReLU depends on Linear
        
        // Create Marabou variables (simplified for testing)
        for (unsigned i = 0; i < nodes.size(); ++i) {
            Vector<Variable> vars;
            unsigned outputSize = nodes[i]->getOutputSize();
            for (unsigned j = 0; j < outputSize; ++j) {
                vars.append(Variable(i * 10 + j));  // Simple variable assignment
            }
            marabouVars.append(vars);
            neuronToMarabouMap[i] = vars;
        }
        
        // Output index is the last node (ReLU)
        unsigned outputIndex = 2;
        
        return std::make_shared<NLR::TorchModel>(nodes, marabouVars, inputIndices, 
                                                outputIndex, neuronToMarabouMap, dependencies);
    }

    void test_constructor_and_basic_getters()
    {
        std::shared_ptr<NLR::TorchModel> model = createSimpleTestModel();
        
        // Test basic getters
        TS_ASSERT_EQUALS(model->getNumNodes(), 3u);
        TS_ASSERT_EQUALS(model->getInputSize(), 2u);
        TS_ASSERT_EQUALS(model->getOutputSize(), 3u);
        TS_ASSERT_EQUALS(model->getOutputIndex(), 2u);
        
        // Test input indices
        const Vector<unsigned>& inputIndices = model->getInputIndices();
        TS_ASSERT_EQUALS(inputIndices.size(), 1u);
        TS_ASSERT_EQUALS(inputIndices[0], 0u);
        
        // Test node access
        auto inputNode = model->getNode(0);
        TS_ASSERT(inputNode != nullptr);
        TS_ASSERT_EQUALS(inputNode->getNodeType(), NLR::NodeType::INPUT);
        TS_ASSERT_EQUALS(inputNode->getOutputSize(), 2u);
        
        auto linearNode = model->getNode(1);
        TS_ASSERT(linearNode != nullptr);
        TS_ASSERT_EQUALS(linearNode->getNodeType(), NLR::NodeType::LINEAR);
        TS_ASSERT_EQUALS(linearNode->getInputSize(), 2u);
        TS_ASSERT_EQUALS(linearNode->getOutputSize(), 3u);
        
        auto reluNode = model->getNode(2);
        TS_ASSERT(reluNode != nullptr);
        TS_ASSERT_EQUALS(reluNode->getNodeType(), NLR::NodeType::RELU);
        TS_ASSERT_EQUALS(reluNode->getInputSize(), 3u);
        TS_ASSERT_EQUALS(reluNode->getOutputSize(), 3u);
        
        // Test node indices
        Vector<unsigned> allIndices = model->getAllNodeIndices();
        TS_ASSERT_EQUALS(allIndices.size(), 3u);
        TS_ASSERT_EQUALS(allIndices[0], 0u);
        TS_ASSERT_EQUALS(allIndices[1], 1u);
        TS_ASSERT_EQUALS(allIndices[2], 2u);
        
        // Test nodes by type
        Vector<unsigned> inputNodes = model->getNodesByType(NLR::NodeType::INPUT);
        TS_ASSERT_EQUALS(inputNodes.size(), 1u);
        TS_ASSERT_EQUALS(inputNodes[0], 0u);
        
        Vector<unsigned> linearNodes = model->getNodesByType(NLR::NodeType::LINEAR);
        TS_ASSERT_EQUALS(linearNodes.size(), 1u);
        TS_ASSERT_EQUALS(linearNodes[0], 1u);
        
        Vector<unsigned> reluNodes = model->getNodesByType(NLR::NodeType::RELU);
        TS_ASSERT_EQUALS(reluNodes.size(), 1u);
        TS_ASSERT_EQUALS(reluNodes[0], 2u);
        
        // Test dependencies
        Vector<unsigned> deps1 = model->getDependencies(1);
        TS_ASSERT_EQUALS(deps1.size(), 1u);
        TS_ASSERT_EQUALS(deps1[0], 0u);
        
        Vector<unsigned> deps2 = model->getDependencies(2);
        TS_ASSERT_EQUALS(deps2.size(), 1u);
        TS_ASSERT_EQUALS(deps2[0], 1u);
        
        // Test dependents
        Vector<unsigned> dependents0 = model->getDependents(0);
        TS_ASSERT_EQUALS(dependents0.size(), 1u);
        TS_ASSERT_EQUALS(dependents0[0], 1u);
        
        Vector<unsigned> dependents1 = model->getDependents(1);
        TS_ASSERT_EQUALS(dependents1.size(), 1u);
        TS_ASSERT_EQUALS(dependents1[0], 2u);
        
        // Test degrees
        TS_ASSERT_EQUALS(model->getDegreeIn(0), 0u);   // Input has no dependencies
        TS_ASSERT_EQUALS(model->getDegreeIn(1), 1u);   // Linear has 1 dependency
        TS_ASSERT_EQUALS(model->getDegreeIn(2), 1u);   // ReLU has 1 dependency
        
        TS_ASSERT_EQUALS(model->getDegreeOut(0), 1u);  // Input has 1 dependent
        TS_ASSERT_EQUALS(model->getDegreeOut(1), 1u);  // Linear has 1 dependent
        TS_ASSERT_EQUALS(model->getDegreeOut(2), 0u);  // ReLU has no dependents
        
        // Test roots and leaves
        Vector<unsigned> roots = model->getRoots();
        TS_ASSERT_EQUALS(roots.size(), 1u);
        TS_ASSERT_EQUALS(roots[0], 0u);
        
        Vector<unsigned> leaves = model->getLeaves();
        TS_ASSERT_EQUALS(leaves.size(), 1u);
        TS_ASSERT_EQUALS(leaves[0], 2u);
        
        // Test topological sort
        Vector<unsigned> topoSort = model->topologicalSort();
        TS_ASSERT_EQUALS(topoSort.size(), 3u);
        TS_ASSERT_EQUALS(topoSort[0], 0u);  // Input first
        TS_ASSERT_EQUALS(topoSort[1], 1u);  // Linear second
        TS_ASSERT_EQUALS(topoSort[2], 2u);  // ReLU last
    }

    void test_forward_pass_and_activations()
    {
        std::shared_ptr<NLR::TorchModel> model = createSimpleTestModel();
        
        // Create test input tensor
        torch::Tensor input = torch::tensor({1.0, 2.0}, torch::kFloat32);
        
        // Test forward pass with activations
        Map<unsigned, torch::Tensor> activations = model->forwardAndStoreActivations(input);
        
        // Verify that activations were computed for all nodes
        TS_ASSERT_EQUALS(activations.size(), 3u);
        TS_ASSERT(activations.exists(0));
        TS_ASSERT(activations.exists(1));
        TS_ASSERT(activations.exists(2));
        
        // Test input node activation (should be the input itself)
        torch::Tensor inputActivation = activations[0];
        TS_ASSERT(inputActivation.defined());
        TS_ASSERT_EQUALS(inputActivation.numel(), 2);
        TS_ASSERT_DELTA(inputActivation[0].item<float>(), 1.0f, 1e-6);
        TS_ASSERT_DELTA(inputActivation[1].item<float>(), 2.0f, 1e-6);
        
        // Test linear node activation
        // Expected: linear(x) = W * x + b
        // W = [[1, 2], [3, 4], [5, 6]], b = [0.1, 0.2, 0.3]
        // x = [1, 2]
        // Expected: [1*1 + 2*2 + 0.1, 1*3 + 2*4 + 0.2, 1*5 + 2*6 + 0.3]
        // = [1 + 4 + 0.1, 3 + 8 + 0.2, 5 + 12 + 0.3]
        // = [5.1, 11.2, 17.3]
        torch::Tensor linearActivation = activations[1];
        TS_ASSERT(linearActivation.defined());
        TS_ASSERT_EQUALS(linearActivation.numel(), 3);
        TS_ASSERT_DELTA(linearActivation[0].item<float>(), 5.1f, 1e-6);
        TS_ASSERT_DELTA(linearActivation[1].item<float>(), 11.2f, 1e-6);
        TS_ASSERT_DELTA(linearActivation[2].item<float>(), 17.3f, 1e-6);
        
        // Test ReLU node activation
        // Expected: ReLU([5.1, 11.2, 17.3]) = [5.1, 11.2, 17.3] (all positive)
        torch::Tensor reluActivation = activations[2];
        TS_ASSERT(reluActivation.defined());
        TS_ASSERT_EQUALS(reluActivation.numel(), 3);
        TS_ASSERT_DELTA(reluActivation[0].item<float>(), 5.1f, 1e-6);
        TS_ASSERT_DELTA(reluActivation[1].item<float>(), 11.2f, 1e-6);
        TS_ASSERT_DELTA(reluActivation[2].item<float>(), 17.3f, 1e-6);
        
        // Test with negative input to verify ReLU works
        torch::Tensor negativeInput = torch::tensor({-1.0, -2.0}, torch::kFloat32);
        Map<unsigned, torch::Tensor> negativeActivations = model->forwardAndStoreActivations(negativeInput);
        
        // Linear output with negative input: W * [-1, -2] + b
        // = [(-1)*1 + (-2)*2 + 0.1, (-1)*3 + (-2)*4 + 0.2, (-1)*5 + (-2)*6 + 0.3]
        // = [-1 - 4 + 0.1, -3 - 8 + 0.2, -5 - 12 + 0.3]
        // = [-4.9, -10.8, -16.7]
        torch::Tensor negativeLinearActivation = negativeActivations[1];
        TS_ASSERT_DELTA(negativeLinearActivation[0].item<float>(), -4.9f, 1e-6);
        TS_ASSERT_DELTA(negativeLinearActivation[1].item<float>(), -10.8f, 1e-6);
        TS_ASSERT_DELTA(negativeLinearActivation[2].item<float>(), -16.7f, 1e-6);
        
        // ReLU output with negative input: ReLU([-4.9, -10.8, -16.7]) = [0, 0, 0]
        torch::Tensor negativeReluActivation = negativeActivations[2];
        TS_ASSERT_DELTA(negativeReluActivation[0].item<float>(), 0.0f, 1e-6);
        TS_ASSERT_DELTA(negativeReluActivation[1].item<float>(), 0.0f, 1e-6);
        TS_ASSERT_DELTA(negativeReluActivation[2].item<float>(), 0.0f, 1e-6);
    }

    void test_input_bounds_functionality()
    {
        std::shared_ptr<NLR::TorchModel> model = createSimpleTestModel();
        
        // Initially, no input bounds should be set
        TS_ASSERT(!model->hasInputBounds());
        
        // Create input bounds
        torch::Tensor inputLower = torch::tensor({-1.0, -2.0}, torch::kFloat32);
        torch::Tensor inputUpper = torch::tensor({1.0, 2.0}, torch::kFloat32);
        BoundedTensor<torch::Tensor> inputBounds(inputLower, inputUpper);
        
        // Set input bounds
        model->setInputBounds(inputBounds);
        
        // Verify input bounds are set
        TS_ASSERT(model->hasInputBounds());
        
        // Test getting input bounds
        BoundedTensor<torch::Tensor> retrievedBounds = model->getInputBounds();
        TS_ASSERT(retrievedBounds.lower().defined());
        TS_ASSERT(retrievedBounds.upper().defined());
        TS_ASSERT_EQUALS(retrievedBounds.lower().numel(), 2);
        TS_ASSERT_EQUALS(retrievedBounds.upper().numel(), 2);
        
        // Test individual bound access
        torch::Tensor retrievedLower = model->getInputLowerBounds();
        torch::Tensor retrievedUpper = model->getInputUpperBounds();
        
        TS_ASSERT_DELTA(retrievedLower[0].item<float>(), -1.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedLower[1].item<float>(), -2.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpper[0].item<float>(), 1.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpper[1].item<float>(), 2.0f, 1e-6);
        
        // Test with different bounds
        torch::Tensor newInputLower = torch::tensor({0.0, 0.5}, torch::kFloat32);
        torch::Tensor newInputUpper = torch::tensor({3.0, 3.5}, torch::kFloat32);
        BoundedTensor<torch::Tensor> newInputBounds(newInputLower, newInputUpper);
        
        model->setInputBounds(newInputBounds);
        
        // Verify new bounds are set
        TS_ASSERT(model->hasInputBounds());
        
        torch::Tensor newRetrievedLower = model->getInputLowerBounds();
        torch::Tensor newRetrievedUpper = model->getInputUpperBounds();
        
        TS_ASSERT_DELTA(newRetrievedLower[0].item<float>(), 0.0f, 1e-6);
        TS_ASSERT_DELTA(newRetrievedLower[1].item<float>(), 0.5f, 1e-6);
        TS_ASSERT_DELTA(newRetrievedUpper[0].item<float>(), 3.0f, 1e-6);
        TS_ASSERT_DELTA(newRetrievedUpper[1].item<float>(), 3.5f, 1e-6);
    }

    void test_concrete_bounds_functionality()
    {
        std::shared_ptr<NLR::TorchModel> model = createSimpleTestModel();
        
        // Initially, no concrete bounds should be set for any node
        TS_ASSERT(!model->hasConcreteBounds(0));
        TS_ASSERT(!model->hasConcreteBounds(1));
        TS_ASSERT(!model->hasConcreteBounds(2));
        
        // Set concrete bounds for linear node (index 1)
        torch::Tensor linearLower = torch::tensor({-5.0, -10.0, -15.0}, torch::kFloat32);
        torch::Tensor linearUpper = torch::tensor({5.0, 10.0, 15.0}, torch::kFloat32);
        BoundedTensor<torch::Tensor> linearBounds(linearLower, linearUpper);
        
        model->setConcreteBounds(1, linearBounds);
        
        // Verify concrete bounds are set for linear node
        TS_ASSERT(model->hasConcreteBounds(1));
        TS_ASSERT(!model->hasConcreteBounds(0));
        TS_ASSERT(!model->hasConcreteBounds(2));
        
        // Test getting concrete bounds
        BoundedTensor<torch::Tensor> retrievedLinearBounds = model->getConcreteBounds(1);
        TS_ASSERT(retrievedLinearBounds.lower().defined());
        TS_ASSERT(retrievedLinearBounds.upper().defined());
        TS_ASSERT_EQUALS(retrievedLinearBounds.lower().numel(), 3);
        TS_ASSERT_EQUALS(retrievedLinearBounds.upper().numel(), 3);
        
        TS_ASSERT_DELTA(retrievedLinearBounds.lower()[0].item<float>(), -5.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedLinearBounds.lower()[1].item<float>(), -10.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedLinearBounds.lower()[2].item<float>(), -15.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedLinearBounds.upper()[0].item<float>(), 5.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedLinearBounds.upper()[1].item<float>(), 10.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedLinearBounds.upper()[2].item<float>(), 15.0f, 1e-6);
        
        // Set concrete bounds for ReLU node (index 2)
        torch::Tensor reluLower = torch::tensor({0.0, 0.0, 0.0}, torch::kFloat32);
        torch::Tensor reluUpper = torch::tensor({10.0, 20.0, 30.0}, torch::kFloat32);
        BoundedTensor<torch::Tensor> reluBounds(reluLower, reluUpper);
        
        model->setConcreteBounds(2, reluBounds);
        
        // Verify concrete bounds are set for ReLU node
        TS_ASSERT(model->hasConcreteBounds(2));
        
        // Test getting ReLU concrete bounds
        BoundedTensor<torch::Tensor> retrievedReluBounds = model->getConcreteBounds(2);
        TS_ASSERT_DELTA(retrievedReluBounds.lower()[0].item<float>(), 0.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedReluBounds.lower()[1].item<float>(), 0.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedReluBounds.lower()[2].item<float>(), 0.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedReluBounds.upper()[0].item<float>(), 10.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedReluBounds.upper()[1].item<float>(), 20.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedReluBounds.upper()[2].item<float>(), 30.0f, 1e-6);
        
        // Test updating existing bounds
        torch::Tensor updatedLinearLower = torch::tensor({-3.0, -8.0, -12.0}, torch::kFloat32);
        torch::Tensor updatedLinearUpper = torch::tensor({3.0, 8.0, 12.0}, torch::kFloat32);
        BoundedTensor<torch::Tensor> updatedLinearBounds(updatedLinearLower, updatedLinearUpper);
        
        model->setConcreteBounds(1, updatedLinearBounds);
        
        // Verify updated bounds
        BoundedTensor<torch::Tensor> retrievedUpdatedBounds = model->getConcreteBounds(1);
        TS_ASSERT_DELTA(retrievedUpdatedBounds.lower()[0].item<float>(), -3.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpdatedBounds.lower()[1].item<float>(), -8.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpdatedBounds.lower()[2].item<float>(), -12.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpdatedBounds.upper()[0].item<float>(), 3.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpdatedBounds.upper()[1].item<float>(), 8.0f, 1e-6);
        TS_ASSERT_DELTA(retrievedUpdatedBounds.upper()[2].item<float>(), 12.0f, 1e-6);
    }
};
