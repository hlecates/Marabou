#include "src/nlr/CROWNAnalysis.h"
#include "src/nlr/TorchModel.h"
#include "src/nlr/BoundedInputNode.h"
#include "src/nlr/BoundedLinearNode.h"
#include "src/nlr/BoundedReLUNode.h"
#include "src/common/Vector.h"
#include "src/common/Map.h"

#include <torch/torch.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <utility>

NLR::TorchModel* createDeepPolyModel() {
    std::cout << "Creating DeepPoly network model..." << std::endl;
    
    std::vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    std::vector<std::vector<Variable>> marabouVars;
    std::vector<unsigned> inputIndices;
    std::map<unsigned, std::vector<Variable>> neuronToMarabouMap;
    std::map<unsigned, std::vector<unsigned>> dependencies;
    
    // Model architecture based on DeepPoly paper Fig. 3:
    // Input(2) -> Linear(2,2) -> ReLU -> Linear(2,2) -> ReLU -> Linear(2,2) -> Output
    
    // Create input node (2 input neurons)
    auto inputNode = std::make_shared<NLR::BoundedInputNode>(0, 2, "input");
    inputNode->setNodeIndex(0);
    nodes.push_back(inputNode);
    
    // Create first linear layer (2 -> 2 neurons) 
    torch::nn::Linear linear1(torch::nn::LinearOptions(2, 2));
    // Set weights according to the diagram: [1,1; 1,-1]
    linear1->weight = torch::tensor({{1.0, 1.0}, {1.0, -1.0}}, torch::kFloat64);
    linear1->bias = torch::zeros({2}, torch::kFloat64);  // All biases are 0
    auto linearNode1 = std::make_shared<NLR::BoundedLinearNode>(linear1, 1.0f, "linear1");
    linearNode1->setNodeIndex(1);
    nodes.push_back(linearNode1);

    std::cout << "Linear1 weight:\n" << linear1->weight << std::endl;
    std::cout << "Linear1 bias:\n" << linear1->bias << std::endl;
    
    // Create first ReLU activation
    torch::nn::ReLU relu1(torch::nn::ReLUOptions{});
    auto reluNode1 = std::make_shared<NLR::BoundedReLUNode>(relu1, "relu1");
    reluNode1->setNodeIndex(2);
    nodes.push_back(reluNode1);
    
    // Create second linear layer (2 -> 2 neurons)
    torch::nn::Linear linear2(torch::nn::LinearOptions(2, 2));
    // Set weights according to the diagram: [1,1; 1,-1]
    linear2->weight = torch::tensor({{1.0, 1.0}, {1.0, -1.0}}, torch::kFloat64);
    linear2->bias = torch::zeros({2}, torch::kFloat64);  // All biases are 0
    auto linearNode2 = std::make_shared<NLR::BoundedLinearNode>(linear2, 1.0f, "linear2");
    linearNode2->setNodeIndex(3);
    nodes.push_back(linearNode2);

    std::cout << "Linear2 weight:\n" << linear2->weight << std::endl;
    std::cout << "Linear2 bias:\n" << linear2->bias << std::endl;
    
    // Create second ReLU activation
    torch::nn::ReLU relu2(torch::nn::ReLUOptions{});
    auto reluNode2 = std::make_shared<NLR::BoundedReLUNode>(relu2, "relu2");
    reluNode2->setNodeIndex(4);
    nodes.push_back(reluNode2);
    
    // Create third linear layer (2 -> 2 neurons)
    torch::nn::Linear linear3(torch::nn::LinearOptions(2, 2));
    // Set weights according to the diagram: [1,0; 1,1] with bias [1,0]
    linear3->weight = torch::tensor({{1.0, 1.0}, {0.0, 1.0}}, torch::kFloat64);
    linear3->bias = torch::tensor({1.0, 0.0}, torch::kFloat64);  // Bias [1,0] as specified
    auto linearNode3 = std::make_shared<NLR::BoundedLinearNode>(linear3, 1.0f, "linear3");
    linearNode3->setNodeIndex(5);
    nodes.push_back(linearNode3);

    std::cout << "Linear3 weight:\n" << linear3->weight << std::endl;
    std::cout << "Linear3 bias:\n" << linear3->bias << std::endl;
    
    // Set up dependencies: input -> linear1 -> relu1 -> linear2 -> relu2 -> linear3
    std::vector<unsigned> inputDeps;  // Input has no dependencies
    dependencies[0] = inputDeps;
    
    std::vector<unsigned> linear1Deps;
    linear1Deps.push_back(0);  // linear1 depends on input
    dependencies[1] = linear1Deps;
    
    std::vector<unsigned> relu1Deps;
    relu1Deps.push_back(1);  // relu1 depends on linear1
    dependencies[2] = relu1Deps;
    
    std::vector<unsigned> linear2Deps;
    linear2Deps.push_back(2);  // linear2 depends on relu1
    dependencies[3] = linear2Deps;
    
    std::vector<unsigned> relu2Deps;
    relu2Deps.push_back(3);  // relu2 depends on linear2
    dependencies[4] = relu2Deps;
    
    std::vector<unsigned> linear3Deps;
    linear3Deps.push_back(4);  // linear3 depends on relu2
    dependencies[5] = linear3Deps;
    
    // Set input indices
    inputIndices.push_back(0);
    
    // Set output index (last node)
    unsigned outputIndex = 5;
    
    // Set up variable mapping for Marabou (12 variables total as in the original)
    // Node 0 (input): variables 0, 1
    std::vector<Variable> inputVars;
    inputVars.push_back(Variable(0));
    inputVars.push_back(Variable(1));
    marabouVars.push_back(inputVars);
    neuronToMarabouMap[0] = inputVars;
    
    // Node 1 (linear1): variables 2, 3
    std::vector<Variable> linear1Vars;
    linear1Vars.push_back(Variable(2));
    linear1Vars.push_back(Variable(3));
    marabouVars.push_back(linear1Vars);
    neuronToMarabouMap[1] = linear1Vars;
    
    // Node 2 (relu1): variables 4, 5
    std::vector<Variable> relu1Vars;
    relu1Vars.push_back(Variable(4));
    relu1Vars.push_back(Variable(5));
    marabouVars.push_back(relu1Vars);
    neuronToMarabouMap[2] = relu1Vars;
    
    // Node 3 (linear2): variables 6, 7
    std::vector<Variable> linear2Vars;
    linear2Vars.push_back(Variable(6));
    linear2Vars.push_back(Variable(7));
    marabouVars.push_back(linear2Vars);
    neuronToMarabouMap[3] = linear2Vars;
    
    // Node 4 (relu2): variables 8, 9
    std::vector<Variable> relu2Vars;
    relu2Vars.push_back(Variable(8));
    relu2Vars.push_back(Variable(9));
    marabouVars.push_back(relu2Vars);
    neuronToMarabouMap[4] = relu2Vars;
    
    // Node 5 (linear3): variables 10, 11
    std::vector<Variable> linear3Vars;
    linear3Vars.push_back(Variable(10));
    linear3Vars.push_back(Variable(11));
    marabouVars.push_back(linear3Vars);
    neuronToMarabouMap[5] = linear3Vars;
    
    // Set sizes for nodes
    inputNode->setInputSize(2);
    inputNode->setOutputSize(2);
    
    linearNode1->setInputSize(2);
    linearNode1->setOutputSize(2);
    
    reluNode1->setInputSize(2);
    reluNode1->setOutputSize(2);
    
    linearNode2->setInputSize(2);
    linearNode2->setOutputSize(2);
    
    reluNode2->setInputSize(2);
    reluNode2->setOutputSize(2);
    
    linearNode3->setInputSize(2);
    linearNode3->setOutputSize(2);
    
    std::cout << "DeepPoly model created with " << nodes.size() << " nodes:" << std::endl;
    std::cout << "  - Input node (2 neurons)" << std::endl;
    std::cout << "  - Linear1 node (2->2 neurons)" << std::endl;
    std::cout << "  - ReLU1 node (2 neurons)" << std::endl;
    std::cout << "  - Linear2 node (2->2 neurons)" << std::endl;
    std::cout << "  - ReLU2 node (2 neurons)" << std::endl;
    std::cout << "  - Linear3 node (2->2 neurons)" << std::endl;
    
    // Convert std containers to Marabou containers
    Vector<std::shared_ptr<NLR::BoundedTorchNode>> marabouNodes;
    for (const auto& node : nodes) {
        marabouNodes.append(node);
    }
    
    Vector<Vector<Variable>> marabouVarsVec;
    for (const auto& vars : marabouVars) {
        Vector<Variable> marabouVarsInner;
        for (const auto& var : vars) {
            marabouVarsInner.append(var);
        }
        marabouVarsVec.append(marabouVarsInner);
    }
    
    Vector<unsigned> marabouInputIndices;
    for (const auto& idx : inputIndices) {
        marabouInputIndices.append(idx);
    }
    
    Map<unsigned, Vector<Variable>> marabouNeuronMap;
    for (const auto& pair : neuronToMarabouMap) {
        Vector<Variable> marabouVarsInner;
        for (const auto& var : pair.second) {
            marabouVarsInner.append(var);
        }
        marabouNeuronMap[pair.first] = marabouVarsInner;
    }
    
    Map<unsigned, Vector<unsigned>> marabouDependencies;
    for (const auto& pair : dependencies) {
        Vector<unsigned> marabouDeps;
        for (const auto& dep : pair.second) {
            marabouDeps.append(dep);
        }
        marabouDependencies[pair.first] = marabouDeps;
    }
    
    // Create and return the TorchModel
    NLR::TorchModel* model = new NLR::TorchModel(
        marabouNodes,
        marabouVarsVec,
        marabouInputIndices,
        outputIndex,
        marabouNeuronMap,
        marabouDependencies
    );
    
    return model;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "DeepPoly Network CROWN Analysis" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    try {
        // Step 1: Create DeepPoly model
        NLR::TorchModel* model = createDeepPolyModel();
        
        // Step 2: Set input bounds using BoundedTensor
        std::cout << "\nSetting input bounds..." << std::endl;
        unsigned inputSize = model->getInputSize();
        torch::Tensor lowerBounds = torch::full({(long)inputSize}, -1.0, torch::kFloat64);
        torch::Tensor upperBounds = torch::ones({(long)inputSize}, torch::kFloat64);
        
        BoundedTensor<torch::Tensor> inputBounds(lowerBounds, upperBounds);
        model->setInputBounds(inputBounds);
        
        std::cout << "Input bounds set: [-1.0, 1.0] for all " << inputSize << " inputs" << std::endl;
        
        // Step 3: Create CROWN analysis and run it
        std::cout << "\nRunning CROWN analysis..." << std::endl;
        NLR::CROWNAnalysis crownAnalysis(model);
        crownAnalysis.run();
        
        // Step 4: Print CROWN bounds for all neurons
        std::cout << "\nNeuron Bounds:" << std::endl;
        
        // Get the number of nodes in the model
        unsigned numNodes = model->getNumNodes();
        
        // Print bounds for each node
        for (unsigned nodeIndex = 0; nodeIndex < numNodes; ++nodeIndex) {
            if (crownAnalysis.hasConcreteBounds(nodeIndex)) {
                BoundedTensor<torch::Tensor> concreteBounds = crownAnalysis.getNodeConcreteBounds(nodeIndex);
                
                if (concreteBounds.lower().defined() && concreteBounds.upper().defined()) {
                    std::cout << "Node " << nodeIndex << ": ";
                    
                    // Handle the case where bounds might be tensors with multiple elements
                    torch::Tensor lowerBound = concreteBounds.lower();
                    torch::Tensor upperBound = concreteBounds.upper();
                    
                    // Ensure we have 1D tensors
                    if (lowerBound.dim() == 0) {
                        lowerBound = lowerBound.unsqueeze(0);
                    }
                    if (upperBound.dim() == 0) {
                        upperBound = upperBound.unsqueeze(0);
                    }
                    
                    // Print each element
                    for (int i = 0; i < lowerBound.size(0); ++i) {
                        if (i > 0) std::cout << " ";
                        
                        // Extract the i-th element and ensure it's a scalar
                        torch::Tensor lowerElem = lowerBound[i];
                        torch::Tensor upperElem = upperBound[i];
                        
                        // If the element is still a tensor, take the first element
                        if (lowerElem.dim() > 0) {
                            lowerElem = lowerElem.flatten()[0];
                        }
                        if (upperElem.dim() > 0) {
                            upperElem = upperElem.flatten()[0];
                        }
                        
                        std::cout << "[" << lowerElem.item<double>() 
                                 << ", " << upperElem.item<double>() << "]";
                    }
                    std::cout << std::endl;
                }
            }
        }
        
        // Step 5: Print output bounds specifically
        std::cout << "\nOutput Bounds:" << std::endl;
        
        BoundedTensor<torch::Tensor> outputBounds = crownAnalysis.getOutputBounds();
        
        if (outputBounds.lower().defined() && outputBounds.upper().defined()) {
            torch::Tensor lowerBound = outputBounds.lower();
            torch::Tensor upperBound = outputBounds.upper();
            
            // Ensure we have 1D tensors
            if (lowerBound.dim() == 0) {
                lowerBound = lowerBound.unsqueeze(0);
            }
            if (upperBound.dim() == 0) {
                upperBound = upperBound.unsqueeze(0);
            }
            
            for (int i = 0; i < lowerBound.size(0); ++i) {
                if (i > 0) std::cout << " ";
                
                // Extract the i-th element and ensure it's a scalar
                torch::Tensor lowerElem = lowerBound[i];
                torch::Tensor upperElem = upperBound[i];
                
                // If the element is still a tensor, take the first element
                if (lowerElem.dim() > 0) {
                    lowerElem = lowerElem.flatten()[0];
                }
                if (upperElem.dim() > 0) {
                    upperElem = upperElem.flatten()[0];
                }
                
                std::cout << "[" << lowerElem.item<double>() 
                         << ", " << upperElem.item<double>() << "]";
            }
            std::cout << std::endl;
        }
        
        // Also print IBP bounds for comparison
        BoundedTensor<torch::Tensor> ibpBounds = crownAnalysis.getOutputIBPBounds();
        if (ibpBounds.lower().defined() && ibpBounds.upper().defined()) {
            std::cout << "IBP Bounds: ";
            
            torch::Tensor lowerBound = ibpBounds.lower();
            torch::Tensor upperBound = ibpBounds.upper();
            
            // Ensure we have 1D tensors
            if (lowerBound.dim() == 0) {
                lowerBound = lowerBound.unsqueeze(0);
            }
            if (upperBound.dim() == 0) {
                upperBound = upperBound.unsqueeze(0);
            }
            
            for (int i = 0; i < lowerBound.size(0); ++i) {
                if (i > 0) std::cout << " ";
                
                // Extract the i-th element and ensure it's a scalar
                torch::Tensor lowerElem = lowerBound[i];
                torch::Tensor upperElem = upperBound[i];
                
                // If the element is still a tensor, take the first element
                if (lowerElem.dim() > 0) {
                    lowerElem = lowerElem.flatten()[0];
                }
                if (upperElem.dim() > 0) {
                    upperElem = upperElem.flatten()[0];
                }
                
                std::cout << "[" << lowerElem.item<double>() 
                         << ", " << upperElem.item<double>() << "]";
            }
            std::cout << std::endl;
        }
        
        // Cleanup
        delete model;
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "DeepPoly CROWN Analysis Completed" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n==========================================" << std::endl;
        std::cerr << "ERROR: DeepPoly CROWN Analysis Failed" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        std::cerr << "==========================================" << std::endl;
        return 1;
    }
} 