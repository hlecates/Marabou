#include <cxxtest/TestSuite.h>

#include "../CROWNAnalysis.h"
#include "../TorchModel.h"
#include "../BoundedInputNode.h"
#include "../BoundedLinearNode.h"
#include "../BoundedReLUNode.h"
#include "../../common/Vector.h"
#include "../../common/Map.h"
#include "../../input_parsers/InputQueryBuilder.h"

#include <torch/torch.h>
#include <memory>
#include <cmath>

class CROWNAnalysisTestSuite : public CxxTest::TestSuite {
public:
    void setUp() {}
    void tearDown() {}

private:
    // Helper: allclose for tensors
    void assertAllClose(const torch::Tensor &a, const torch::Tensor &b, double atol = 1e-5, double rtol = 1e-5) {
        TS_ASSERT(a.defined());
        TS_ASSERT(b.defined());
        TS_ASSERT_EQUALS(a.sizes(), b.sizes());
        TS_ASSERT(torch::allclose(a, b, rtol, atol));
    }

    // Helper: build a minimal TorchModel with explicit nodes/deps
    std::unique_ptr<NLR::TorchModel> buildModel(const Vector<std::shared_ptr<NLR::BoundedTorchNode>> &nodes,
                                                const Map<unsigned, Vector<unsigned>> &deps,
                                                const Vector<unsigned> &inputIndices,
                                                unsigned outputIndex) {
        Vector<Vector<Variable>> marabouVars;
        Map<unsigned, Vector<Variable>> neuronToMarabouMap;
        for (unsigned i = 0; i < nodes.size(); ++i) {
            unsigned sz = nodes[i]->getOutputSize() > 0 ? nodes[i]->getOutputSize() : nodes[i]->getInputSize();
            Vector<Variable> vars;
            for (unsigned j = 0; j < (sz ? sz : 1); ++j) vars.append(Variable(i * 100 + j));
            marabouVars.append(vars);
            neuronToMarabouMap[i] = vars;
        }
        auto model = std::make_unique<NLR::TorchModel>(nodes, marabouVars, inputIndices, outputIndex, neuronToMarabouMap, deps);
        return model;
    }

public:
    // ReLU backward: always active (lb >= 0)
    void test_relu_backward_always_active() {
        torch::nn::ReLU reluOptions{};
        auto reluNode = std::make_shared<NLR::BoundedReLUNode>(reluOptions, "relu");
        reluNode->setInputSize(3);
        reluNode->setOutputSize(3);

        torch::Tensor lb = torch::tensor({0.1f, 2.0f, 5.0f});
        torch::Tensor ub = torch::tensor({1.0f, 3.0f, 6.0f});
        Vector<BoundedTensor<torch::Tensor>> inBounds;
        inBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

        torch::Tensor last_lA = torch::tensor({{1.0f, -2.0f, 0.5f}}); // (1,3)
        torch::Tensor last_uA = torch::tensor({{-1.0f, 4.0f, -0.5f}});
        Vector<Pair<torch::Tensor, torch::Tensor>> outA;
        torch::Tensor lbias, ubias;
        reluNode->boundBackward(last_lA, last_uA, inBounds, outA, lbias, ubias);

        TS_ASSERT_EQUALS(outA.size(), 1U);
        auto new_lA = outA[0].first();
        auto new_uA = outA[0].second();

        assertAllClose(new_lA, last_lA);
        assertAllClose(new_uA, last_uA);
        TS_ASSERT(lbias.defined());
        TS_ASSERT(ubias.defined());
        assertAllClose(lbias, torch::zeros({last_lA.size(0)}));
        assertAllClose(ubias, torch::zeros({last_uA.size(0)}));
    }

    // ReLU backward: always inactive (ub <= 0)
    void test_relu_backward_always_inactive() {
        torch::nn::ReLU reluOptions{};
        auto reluNode = std::make_shared<NLR::BoundedReLUNode>(reluOptions, "relu");
        reluNode->setInputSize(2);
        reluNode->setOutputSize(2);

        torch::Tensor lb = torch::tensor({-3.0f, -1.0f});
        torch::Tensor ub = torch::tensor({-0.1f, -0.5f});
        Vector<BoundedTensor<torch::Tensor>> inBounds;
        inBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

        torch::Tensor last_lA = torch::tensor({{2.0f, -1.0f}});
        torch::Tensor last_uA = torch::tensor({{-3.0f, 0.5f}});
        Vector<Pair<torch::Tensor, torch::Tensor>> outA;
        torch::Tensor lbias, ubias;
        reluNode->boundBackward(last_lA, last_uA, inBounds, outA, lbias, ubias);

        TS_ASSERT_EQUALS(outA.size(), 1U);
        auto new_lA = outA[0].first();
        auto new_uA = outA[0].second();

        assertAllClose(new_lA, torch::zeros_like(last_lA));
        assertAllClose(new_uA, torch::zeros_like(last_uA));
        assertAllClose(lbias, torch::zeros({last_lA.size(0)}));
        assertAllClose(ubias, torch::zeros({last_uA.size(0)}));
    }

    // ReLU backward: uncertain cases - one with upper_slope > 0.5 (lower slope=1), one with <= 0.5 (lower slope=0)
    void test_relu_backward_uncertain_cases() {
        torch::nn::ReLU reluOptions{};
        auto reluNode = std::make_shared<NLR::BoundedReLUNode>(reluOptions, "relu");
        reluNode->setInputSize(2);
        reluNode->setOutputSize(2);

        // lb=[-1,-0.6], ub=[2,0.4] => upper slopes [2/3, 0.4/1.0] = [0.6666, 0.4]
        torch::Tensor lb = torch::tensor({-1.0f, -0.6f});
        torch::Tensor ub = torch::tensor({ 2.0f,  0.4f});
        Vector<BoundedTensor<torch::Tensor>> inBounds;
        inBounds.append(BoundedTensor<torch::Tensor>(lb, ub));

        // Mix of + and - entries to test Apos/Aneg selection
        torch::Tensor last_lA = torch::tensor({{ 1.0f, -2.0f}}); // (1,2)
        torch::Tensor last_uA = torch::tensor({{-1.5f,  3.0f}});

        // Expected slopes and biases per node logic
        float aU0 = 2.0f / (2.0f - (-1.0f)); // 2/3
        float aU1 = 0.4f / (0.4f - (-0.6f)); // 0.4/1.0=0.4
        float aL0 = 1.0f;                    // > 0.5
        float aL1 = 0.0f;                    // <= 0.5
        float bU0 = -aU0 * (-1.0f);          // -aU*lb
        float bU1 = -aU1 * (-0.6f);
        float bL0 = 0.0f, bL1 = 0.0f;

        torch::Tensor aL = torch::tensor({aL0, aL1}).unsqueeze(0); // (1,2)
        torch::Tensor aU = torch::tensor({aU0, aU1}).unsqueeze(0);
        torch::Tensor bL = torch::tensor({bL0, bL1}).unsqueeze(0);
        torch::Tensor bU = torch::tensor({bU0, bU1}).unsqueeze(0);

        auto Apos_l = torch::clamp_min(last_lA, 0);
        auto Aneg_l = torch::clamp_max(last_lA, 0);
        auto Apos_u = torch::clamp_min(last_uA, 0);
        auto Aneg_u = torch::clamp_max(last_uA, 0);

        torch::Tensor expected_lA = Apos_l * aL + Aneg_l * aU;
        torch::Tensor expected_uA = Apos_u * aU + Aneg_u * aL;
        torch::Tensor expected_lbias = (Apos_l * bL + Aneg_l * bU).sum(-1);
        torch::Tensor expected_ubias = (Apos_u * bU + Aneg_u * bL).sum(-1);

        Vector<Pair<torch::Tensor, torch::Tensor>> outA;
        torch::Tensor lbias, ubias;
        reluNode->boundBackward(last_lA, last_uA, inBounds, outA, lbias, ubias);

        TS_ASSERT_EQUALS(outA.size(), 1U);
        assertAllClose(outA[0].first(), expected_lA);
        assertAllClose(outA[0].second(), expected_uA);
        assertAllClose(lbias, expected_lbias);
        assertAllClose(ubias, expected_ubias);
    }

    // Linear backward: A propagation and bias transformation (3D A for consistency with implementation)
    void test_linear_backward_A_and_bias() {
        // Linear layer: out=2, in=3
        torch::nn::Linear lin(torch::nn::LinearOptions(3, 2));
        lin->weight = torch::tensor({{2.0f, -1.0f, 0.0f}, {0.0f, 3.0f, 1.0f}});
        lin->bias = torch::tensor({1.0f, -2.0f});
        auto linNode = std::make_shared<NLR::BoundedLinearNode>(lin, 1.0f, "lin");
        linNode->setInputSize(3);
        linNode->setOutputSize(2);

        // last A shape (spec=1, out=2) to match BoundedLinearNode expectation (2D ok)
        torch::Tensor last_lA = torch::tensor({{ 1.0f, -2.0f}});
        torch::Tensor last_uA = torch::tensor({{-1.0f,  4.0f}});

        Vector<BoundedTensor<torch::Tensor>> inBounds; // provide dummy to satisfy interface
        inBounds.append(BoundedTensor<torch::Tensor>(torch::zeros({3}), torch::ones({3})));
        Vector<Pair<torch::Tensor, torch::Tensor>> outA;
        torch::Tensor lbias, ubias;
        linNode->boundBackward(last_lA, last_uA, inBounds, outA, lbias, ubias);

        TS_ASSERT_EQUALS(outA.size(), 1U);
        auto lA = outA[0].first();
        auto uA = outA[0].second();

        torch::Tensor expected_lA = torch::matmul(last_lA, lin->weight); // (1,3)
        torch::Tensor expected_uA = torch::matmul(last_uA, lin->weight);
        assertAllClose(lA, expected_lA);
        assertAllClose(uA, expected_uA);

        // Bias transformed to final output dims: (spec=1)
        // For 2D last_A, bias contribution reduces to row-wise dot with bias
        torch::Tensor expected_lbias = torch::matmul(last_lA, lin->bias.unsqueeze(-1)).squeeze(-1);
        torch::Tensor expected_ubias = torch::matmul(last_uA, lin->bias.unsqueeze(-1)).squeeze(-1);
        assertAllClose(lbias, expected_lbias);
        assertAllClose(ubias, expected_ubias);
    }

    // Input node backward: passthrough A, no bias
    void test_input_backward_passthrough() {
        auto inNode = std::make_shared<NLR::BoundedInputNode>(0, 3, "input");
        inNode->setNodeIndex(0);

        torch::Tensor last_lA = torch::tensor({{1.0f, -2.0f, 0.5f}});
        torch::Tensor last_uA = torch::tensor({{-1.0f, 4.0f, -0.5f}});
        Vector<BoundedTensor<torch::Tensor>> inBounds; // unused
        Vector<Pair<torch::Tensor, torch::Tensor>> outA;
        torch::Tensor lbias, ubias;
        inNode->boundBackward(last_lA, last_uA, inBounds, outA, lbias, ubias);

        TS_ASSERT_EQUALS(outA.size(), 1U);
        assertAllClose(outA[0].first(), last_lA);
        assertAllClose(outA[0].second(), last_uA);
        TS_ASSERT(!lbias.defined());
        TS_ASSERT(!ubias.defined());
    }

    // IBP: input node returns preset bounds
    void test_ibp_input_node() {
        auto inNode = std::make_shared<NLR::BoundedInputNode>(0, 2, "input");
        torch::Tensor lb = torch::tensor({-1.0f, 0.5f});
        torch::Tensor ub = torch::tensor({ 2.0f, 1.5f});
        inNode->setInputBounds(BoundedTensor<torch::Tensor>(lb, ub));

        Vector<BoundedTensor<torch::Tensor>> inBounds; // unused
        auto res = inNode->computeIntervalBoundPropagation(inBounds);
        assertAllClose(res.lower(), lb);
        assertAllClose(res.upper(), ub);
    }

    // IBP: linear node matches analytical bound rules
    void test_ibp_linear_node() {
        torch::nn::Linear lin(torch::nn::LinearOptions(3, 2));
        lin->weight = torch::tensor({{ 2.0f, -1.0f, 0.5f}, {-3.0f, 4.0f, -2.0f}});
        lin->bias = torch::tensor({1.0f, -2.0f});
        auto linNode = std::make_shared<NLR::BoundedLinearNode>(lin, 1.0f, "lin");

        torch::Tensor xL = torch::tensor({-1.0f, 0.0f, 2.0f});
        torch::Tensor xU = torch::tensor({ 3.0f, 1.0f, 4.0f});
        Vector<BoundedTensor<torch::Tensor>> inBounds;
        inBounds.append(BoundedTensor<torch::Tensor>(xL, xU));

        auto res = linNode->computeIntervalBoundPropagation(inBounds);

        // Compute expected via pos/neg decomposition
        auto W = lin->weight;
        auto Wpos = torch::clamp_min(W, 0);
        auto Wneg = torch::clamp_max(W, 0);
        torch::Tensor expectedL = torch::matmul(xL, Wpos.t()) + torch::matmul(xU, Wneg.t()) + lin->bias;
        torch::Tensor expectedU = torch::matmul(xU, Wpos.t()) + torch::matmul(xL, Wneg.t()) + lin->bias;
        assertAllClose(res.lower(), expectedL);
        assertAllClose(res.upper(), expectedU);
    }

    // IBP: ReLU clamp behavior
    void test_ibp_relu_node() {
        torch::nn::ReLU reluOptions{};
        auto reluNode = std::make_shared<NLR::BoundedReLUNode>(reluOptions, "relu");

        torch::Tensor xL = torch::tensor({-1.0f, 0.5f, -0.2f});
        torch::Tensor xU = torch::tensor({ 3.0f, 1.0f,  0.1f});
        Vector<BoundedTensor<torch::Tensor>> inBounds;
        inBounds.append(BoundedTensor<torch::Tensor>(xL, xU));

        auto res = reluNode->computeIntervalBoundPropagation(inBounds);
        assertAllClose(res.lower(), torch::clamp_min(xL, 0));
        assertAllClose(res.upper(), torch::clamp_min(xU, 0));
    }

    // CROWNAnalysis: computeConcreteLowerBound/UpperBound arithmetic
    void test_compute_concrete_bounds_from_A_bias_and_input() {
        // A (spec=1, n=3)
        torch::Tensor lA = torch::tensor({{ 1.0f, -2.0f, 0.5f}});
        torch::Tensor uA = torch::tensor({{-1.0f,  4.0f, -0.5f}});
        torch::Tensor lBias = torch::tensor({0.3f});
        torch::Tensor uBias = torch::tensor({-0.2f});
        torch::Tensor xL = torch::tensor({-1.0f, 0.0f, 2.0f});
        torch::Tensor xU = torch::tensor({ 3.0f, 1.0f, 4.0f});

        // Build minimal model to satisfy constructor; not used in this test beyond instantiation
        Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
        auto inNode = std::make_shared<NLR::BoundedInputNode>(0, 3, "input");
        inNode->setNodeIndex(0);
        nodes.append(inNode);
        Map<unsigned, Vector<unsigned>> deps;
        Vector<unsigned> inputs; inputs.append(0);
        auto model = buildModel(nodes, deps, inputs, 0);
        NLR::CROWNAnalysis crown(model.get());

        // Expected formulas
        auto AposL = torch::clamp_min(lA.unsqueeze(0), 0); // (1,1,3)
        auto AnegL = torch::clamp_max(lA.unsqueeze(0), 0);
        auto AposU = torch::clamp_min(uA.unsqueeze(0), 0);
        auto AnegU = torch::clamp_max(uA.unsqueeze(0), 0);
        auto xL3 = xL.unsqueeze(0).unsqueeze(-1);
        auto xU3 = xU.unsqueeze(0).unsqueeze(-1);
        auto lB3 = lBias.unsqueeze(0).unsqueeze(-1);
        auto uB3 = uBias.unsqueeze(0).unsqueeze(-1);

        torch::Tensor expectedLower = (AposL.bmm(xL3) + AnegL.bmm(xU3) + lB3).squeeze(-1).squeeze(0);
        torch::Tensor expectedUpper = (AposU.bmm(xU3) + AnegU.bmm(xL3) + uB3).squeeze(-1).squeeze(0);

        auto gotLower = crown.computeConcreteLowerBound(lA, lBias, xL, xU);
        auto gotUpper = crown.computeConcreteUpperBound(uA, uBias, xL, xU);
        assertAllClose(gotLower, expectedLower);
        assertAllClose(gotUpper, expectedUpper);
    }

    // CROWNAnalysis helpers: preprocessC and addA/addBound/addBias
    void test_helpers_additions_and_concretize_with_bias() {
        // Build minimal model and analysis (single input node as output)
        Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
        auto inNode = std::make_shared<NLR::BoundedInputNode>(0, 2, "input");
        inNode->setNodeIndex(0);
        nodes.append(inNode);
        Map<unsigned, Vector<unsigned>> deps;
        Vector<unsigned> inputs; inputs.append(0);
        auto model = buildModel(nodes, deps, inputs, 0);
        NLR::CROWNAnalysis crown(model.get());

        // addA utility
        torch::Tensor A1 = torch::tensor({{1.0f, 2.0f}}); // (1,2)
        torch::Tensor A2 = torch::tensor({{3.0f, 4.0f}});
        auto sum = crown.addA(A1, A2);
        assertAllClose(sum, torch::tensor({{4.0f, 6.0f}}));

        // Prepare final A and bias at the input index and set input bounds
        torch::Tensor final_lA = torch::tensor({{ 1.0f, -2.0f}});
        torch::Tensor final_uA = torch::tensor({{-1.0f,  4.0f}});
        torch::Tensor lBias = torch::tensor({0.3f, -0.1f}); // spec=2
        torch::Tensor uBias = torch::tensor({-0.2f, 0.5f});
        crown.addBound(0, final_lA, final_uA);
        crown.addBias(0, lBias, uBias);

        torch::Tensor xL = torch::tensor({-1.0f, 0.0f});
        torch::Tensor xU = torch::tensor({ 3.0f, 1.0f});
        model->setInputBounds(BoundedTensor<torch::Tensor>(xL, xU));

        // Concretize and check output bounds
        crown.concretizeBounds();
        auto out = crown.getOutputBounds();
        auto lA3 = final_lA.unsqueeze(0); // (1,2,2) after ensure3A inside
        auto uA3 = final_uA.unsqueeze(0);
        auto xL3 = xL.unsqueeze(0).unsqueeze(-1);
        auto xU3 = xU.unsqueeze(0).unsqueeze(-1);
        auto lB3 = lBias.unsqueeze(0).unsqueeze(-1);
        auto uB3 = uBias.unsqueeze(0).unsqueeze(-1);
        torch::Tensor expectedLower = (torch::clamp_min(lA3, 0).bmm(xL3) + torch::clamp_max(lA3, 0).bmm(xU3) + lB3).squeeze(-1).squeeze(0);
        torch::Tensor expectedUpper = (torch::clamp_min(uA3, 0).bmm(xU3) + torch::clamp_max(uA3, 0).bmm(xL3) + uB3).squeeze(-1).squeeze(0);
        assertAllClose(out.lower(), expectedLower);
        assertAllClose(out.upper(), expectedUpper);
    }

    // IBP propagation end-to-end on a tiny graph: Input(2) -> Linear(2)
    void test_ibp_end_to_end_small_graph() {
        // Nodes
        auto input = std::make_shared<NLR::BoundedInputNode>(0, 2, "input");
        input->setNodeIndex(0);
        torch::nn::Linear lin(torch::nn::LinearOptions(2, 2));
        lin->weight = torch::tensor({{1.0f, -1.0f}, {0.5f, 2.0f}});
        lin->bias = torch::tensor({0.5f, -1.0f});
        auto linNode = std::make_shared<NLR::BoundedLinearNode>(lin, 1.0f, "lin");
        linNode->setNodeIndex(1);
        linNode->setInputSize(2);
        linNode->setOutputSize(2);

        Vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
        nodes.append(input);
        nodes.append(linNode);

        // Dependencies: lin depends on input
        Map<unsigned, Vector<unsigned>> deps;
        deps[0] = Vector<unsigned>();
        Vector<unsigned> d; d.append(0); deps[1] = d;

        Vector<unsigned> inputs; inputs.append(0);
        auto model = buildModel(nodes, deps, inputs, 1);

        // Set input bounds on the model
        torch::Tensor xL = torch::tensor({-1.0f, 2.0f});
        torch::Tensor xU = torch::tensor({ 3.0f, 5.0f});
        model->setInputBounds(BoundedTensor<torch::Tensor>(xL, xU));

        NLR::CROWNAnalysis crown(model.get());
        crown.computeIBPBounds();

        // IBP for linear
        auto W = lin->weight;
        auto Wpos = torch::clamp_min(W, 0);
        auto Wneg = torch::clamp_max(W, 0);
        torch::Tensor expectedL = torch::matmul(xL, Wpos.t()) + torch::matmul(xU, Wneg.t()) + lin->bias;
        torch::Tensor expectedU = torch::matmul(xU, Wpos.t()) + torch::matmul(xL, Wneg.t()) + lin->bias;

        auto linBounds = crown.getNodeIBPBounds(1);
        assertAllClose(linBounds.lower(), expectedL);
        assertAllClose(linBounds.upper(), expectedU);
    }
};
