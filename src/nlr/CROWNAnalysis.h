#ifndef __CROWNAnalysis_h__
#define __CROWNAnalysis_h__

#include "TorchModel.h"
#include "BoundedTorchNode.h"
#include "BoundedTensor.h"
#include "Map.h"
#include "Vector.h"
#include "Set.h"
#include "Queue.h"

#include <torch/torch.h>
#include <memory>

namespace NLR {

class CROWNAnalysis
{
public:
    CROWNAnalysis( TorchModel *torchModel );
    ~CROWNAnalysis();

    // Analysis execution
    void run();
    
    // Node access
    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    unsigned getInputSize() const;
    unsigned getOutputSize() const;
    unsigned getOutputIndex() const;

    // Public access methods for testing
    torch::Tensor getIBPLowerBound(unsigned nodeIndex);
    torch::Tensor getIBPUpperBound(unsigned nodeIndex);
    torch::Tensor getCrownLowerBound(unsigned nodeIndex);
    torch::Tensor getCrownUpperBound(unsigned nodeIndex);
    bool hasIBPBounds(unsigned nodeIndex);
    bool hasCrownBounds(unsigned nodeIndex);
    unsigned getNumNodes() const;

    // Concrete bound access methods
    torch::Tensor getConcreteLowerBound(unsigned nodeIndex);
    torch::Tensor getConcreteUpperBound(unsigned nodeIndex);
    bool hasConcreteBounds(unsigned nodeIndex);

    // Output bound access methods
    BoundedTensor<torch::Tensor> getOutputBounds() const;
    BoundedTensor<torch::Tensor> getOutputIBPBounds() const;

    // Model access for testing 
    TorchModel* getModel() const { return _torchModel; }

    // Additional public methods for testing
    Vector<BoundedTensor<torch::Tensor>> getInputBoundsForNode(unsigned nodeIndex);

    // Processing state
    void resetProcessingState();
    void markProcessed(unsigned nodeIndex);
    bool isProcessed(unsigned nodeIndex) const;


    void computeIBPBounds();
    void computeCrownBackwardPropagation();
    void concretizeBounds();

    // Compute the forward pass vlaues via the torch model for concretizing the bounds
    void computeForwardPassValues();

    // Updated concrete bound method signatures
    torch::Tensor computeConcreteLowerBound(const torch::Tensor& lA, const torch::Tensor& lBias,
                                           const torch::Tensor& xLower, const torch::Tensor& xUpper);
    torch::Tensor computeConcreteUpperBound(const torch::Tensor& uA, const torch::Tensor& uBias,
                                           const torch::Tensor& xLower, const torch::Tensor& xUpper);


    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);
    BoundedTensor<torch::Tensor> getNodeIBPBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getNodeCrownBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getNodeConcreteBounds(unsigned nodeIndex) const;

    // Helper functions for A matrix accumulation (following auto-LiRPA's approach)
    torch::Tensor addA(const torch::Tensor& A1, const torch::Tensor& A2);
    void addBound(unsigned nodeIndex, const torch::Tensor& lA, const torch::Tensor& uA);
    void addBias(unsigned nodeIndex, const torch::Tensor& lBias, const torch::Tensor& uBias);

private:
    TorchModel *_torchModel;

    // Node-centric graph structure (delegated to TorchModel)
    // ie all graph management is done by torch model
    Map<unsigned, std::shared_ptr<BoundedTorchNode>> _nodes;
    
    // A matrix storage following auto-LiRPA's approach
    Map<unsigned, torch::Tensor> _lA;  // lower bound A matrices
    Map<unsigned, torch::Tensor> _uA;  // upper bound A matrices

    // Bias accumulation following auto-LiRPA's approach
    // The following were global bias accumulation, since we need bounds on every nueron, need the bias for each nodes individual scope
    // torch::Tensor _lowerBias;
    // torch::Tensor _upperBias;
    Map<unsigned, torch::Tensor> _lowerBias;
    Map<unsigned, torch::Tensor> _upperBias;

    Map<unsigned, BoundedTensor<torch::Tensor>> _ibpBounds;

    // Concrete Bounds 
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;

    // Forward value for concretizing bounds
    Map<unsigned, torch::Tensor> _forwardPassValues;

    // Concretize Bounds
    void computeConcreteBounds(const torch::Tensor& lA, const torch::Tensor& uA,
                              const torch::Tensor& lBias, const torch::Tensor& uBias,
                              const torch::Tensor& nodeLower, const torch::Tensor& nodeUpper,
                              torch::Tensor& concreteLower, torch::Tensor& concreteUpper);

    // Utility methods
    void log( const String &message );
    std::string nodeTypeToString(NodeType type) {
        switch (type) {
            case NodeType::INPUT: return "INPUT";
            case NodeType::CONSTANT: return "CONSTANT";
            case NodeType::LINEAR: return "LINEAR";
            case NodeType::RELU: return "RELU";
            case NodeType::RESHAPE: return "RESHAPE";
            case NodeType::IDENTITY: return "IDENTITY";
            default: return "UNKNOWN";
        }
    }

    // Helper function for establishing consistent tensor format
    torch::Tensor preprocessC(const torch::Tensor& C, unsigned outputSize);

};

} // namespace NLR

#endif // __CROWNAnalysis_h__