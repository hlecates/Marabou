// goal is to have
// CROWN Analysis ←→ TorchModel ←→ NLR/Engine
// Such that the torch model acts as a torch "delegator"/"convertor"

#ifndef __TorchModel_h__
#define __TorchModel_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "Query.h"
#include "InputQueryBuilder.h"
#include "BoundedTorchNode.h"
#include "BoundedInputNode.h"
#include "BoundedLinearNode.h"
#include "BoundedReLUNode.h"
#include "BoundedIdentityNode.h"
#include "BoundedConstantNode.h"
#include "BoundedReshapeNode.h"
#include "Tightening.h"
#include "List.h"
#include "ITableau.h"
#include "LayerOwner.h"

// Forward declaration to avoid circular dependency
class CROWNAnalysis;

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

namespace NLR {

class TorchModel {
public:
    TorchModel(const Vector<std::shared_ptr<BoundedTorchNode>>& nodes, 
               const Vector<Vector<Variable>>& marabouVars,
               const Vector<unsigned>& inputIndices,
               unsigned outputIndex,
               const Map<unsigned, Vector<Variable>>& neuronToMarabouMap,
               const Map<unsigned, Vector<unsigned>>& dependencies);
    
    // Forward pass through the entire model
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor forward(unsigned nodeIndex, Map<unsigned, torch::Tensor>& activations, 
                         const Map<unsigned, torch::Tensor>& inputs);

    
    
    // forward pass that returns activations for all nodes
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const torch::Tensor& input);
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const Map<unsigned, torch::Tensor>& inputs);            

    // Get model information
    unsigned getInputSize() const { return _input_size; }
    unsigned getOutputSize() const { return _output_size; }
    unsigned getNumNodes() const { return _nodes.size(); }
    
    // Access to nodes
    const Vector<std::shared_ptr<BoundedTorchNode>>& getNodes() const { return _nodes; }
    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    Vector<unsigned> getAllNodeIndices() const;
    Vector<unsigned> getNodesByType(NodeType type) const;
    const Vector<unsigned>& getInputIndices() const { return _inputIndices; }
    unsigned getOutputIndex() const { return _outputIndex; }
    
    // PRIMARY BOUND MANAGEMENT INTERFACE
    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);
    
    // CONCRETE BOUND STORAGE (for CROWN analysis to call)
    void setConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds);
    // FOR TESTING AND OUTPUTTING
    BoundedTensor<torch::Tensor> getConcreteBounds(unsigned nodeIndex) const;
    bool hasConcreteBounds(unsigned nodeIndex) const;
    
    // Input bound access
    BoundedTensor<torch::Tensor> getInputBounds() const;
    bool hasInputBounds() const;
    torch::Tensor getInputLowerBounds() const;
    torch::Tensor getInputUpperBounds() const;
    
    // NLR/ENGINE COMMUNICATION INTERFACE (for future Marabou integration)
    // void setModelOwner(LayerOwner* layerOwner) { _modelOwner = layerOwner; }
    // LayerOwner* getModelOwner() const { return _modelOwner; }
    
    // NLR BOUND MANAGEMENT
    // void obtainCurrentBoundsFromNLR();
    // void updateNLRWithTighterBounds();
    
    // BOUND COMPARISON AND COMMUNICATION
    // void communicateTighterBound(unsigned nodeIndex, unsigned neuronIndex, double bound, Tightening::BoundType type);
    // bool hasTighterBounds() const;
    
    // Variable mapping - simplified to just store the passed maps
    const Vector<Vector<Variable>>& getVariables() const { return _marabouVars; }
    const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const { return _neuronToMarabouMap; }
    const Map<unsigned, Vector<unsigned>>& getDependenciesMap() const { return _dependencies; }

    // Full graph 
    void buildDependencyGraph();
    void buildDependents();
    void computeDegrees();

    // Traversal 
    Vector<unsigned> topologicalSort() const;
    Vector<unsigned> getRoots() const;
    Vector<unsigned> getLeaves() const;
    Vector<unsigned> getDependents(unsigned nodeIndex) const;
    Vector<unsigned> getDependencies(unsigned nodeIndex) const;

    // Degree and Processing states
    unsigned getDegreeOut(unsigned nodeIndex) const;
    unsigned getDegreeIn(unsigned nodeIndex) const;
    void resetProcessingState();
    bool isProcessed(unsigned nodeIndex) const;
    void markProcessed(unsigned nodeIndex);

    // Logging
    void log(const String& message) const;

private:
    Vector<std::shared_ptr<BoundedTorchNode>> _nodes;
    Vector<Vector<Variable>> _marabouVars;
    Vector<unsigned> _inputIndices;
    unsigned _outputIndex;
    Map<unsigned, Vector<Variable>> _neuronToMarabouMap;
    Map<unsigned, Vector<unsigned>> _dependencies;
    
    // Graph traversal state
    Map<unsigned, Vector<unsigned>> _dependents;
    Map<unsigned, unsigned> _degreeOut;
    Map<unsigned, unsigned> _degreeIn;
    Map<unsigned, bool> _processed;
    
    // Model dimensions
    unsigned _input_size;
    unsigned _output_size;
    
    // NLR COMMUNICATION
    // LayerOwner* _modelOwner;
    
    BoundedTensor<torch::Tensor> _inputBounds;      // Input bounds for the model
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;  // CROWN concrete bounds
    // Map<unsigned, BoundedTensor<torch::Tensor>> _nlrBounds;       // NLR bounds for comparison
    
    // error checking
    void validateNodeIndex(unsigned nodeIndex) const;
};

} // namespace NLR

#endif // __TorchModel_h__ 