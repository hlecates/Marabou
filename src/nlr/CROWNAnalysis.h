#ifndef __CROWNAnalysis_h__
#define __CROWNAnalysis_h__

#include "TorchModel.h"
#include "TorchModuleBounded.h"
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

    void run();

    // Public access methods for testing
    torch::Tensor getIBPLowerBound(unsigned elementIndex);
    torch::Tensor getIBPUpperBound(unsigned elementIndex);
    torch::Tensor getCrownLowerBound(unsigned elementIndex);
    torch::Tensor getCrownUpperBound(unsigned elementIndex);
    bool hasIBPBounds(unsigned elementIndex);
    bool hasCrownBounds(unsigned elementIndex);
    unsigned getNumElements() const;

    // Concrete bound access methods
    torch::Tensor getConcreteLowerBound(unsigned elementIndex);
    torch::Tensor getConcreteUpperBound(unsigned elementIndex);
    bool hasConcreteBounds(unsigned elementIndex);

private:
    TorchModel *_torchModel;

    // Graph structure following auto-LiRPA's approach
    Map<unsigned, std::shared_ptr<ITorchModuleBounded>> _boundedElements;
    Map<unsigned, Vector<unsigned>> _dependencies;  // layer -> input layers
    Map<unsigned, Vector<unsigned>> _dependents;    // layer -> output layers
    Map<unsigned, unsigned> _degreeIn;              // in-degree
    Map<unsigned, unsigned> _degreeOut;             // out-degree
    Map<unsigned, bool> _processed;                 // track processed nodes
    
    // A matrix storage following auto-LiRPA's approach
    Map<unsigned, torch::Tensor> _lA;  // lower bound A matrices
    Map<unsigned, torch::Tensor> _uA;  // upper bound A matrices
    
    // Working memory
    torch::Tensor *_workLowerBounds;
    torch::Tensor *_workUpperBounds;
    torch::Tensor *_workLinearWeights;
    torch::Tensor *_workLinearBias;

    // Bound computation state
    Map<unsigned, torch::Tensor> _lowerBounds;
    Map<unsigned, torch::Tensor> _upperBounds;
    Map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _ibpBounds;
    Map<unsigned, NLR::LinearBound> _linearBounds;

    // Bias accumulation following auto-LiRPA's approach
    torch::Tensor _lowerBias;
    torch::Tensor _upperBias;

    // Concrete Bounds 
    Map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _concreteBounds;

    // Helper methods
    Vector<BoundedTensor<torch::Tensor>> getInputBoundsForElement(unsigned elementIndex);
    Map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _inputBounds;

    // Graph construction and traversal methods
    void buildDependencyGraph();
    Vector<unsigned> topologicalSort();

    // Processing State management
    void resetProcessingState();
    bool isProcessed(unsigned elementIndex) const;
    void markProcessed(unsigned elementIndex);
    
    // Memory management
    void allocateMemory();
    void freeMemoryIfNeeded();

    // Bound computation methods
    void computeIBPBounds();
    void computeCrownBackwardPropagation();

    // Concretize Bounds
    void concretizeBounds();
    torch::Tensor computeConcreteLowerBound(const torch::Tensor& lA, const torch::Tensor& lBias,
                                           const torch::Tensor& center, const torch::Tensor& eps);
    torch::Tensor computeConcreteUpperBound(const torch::Tensor& uA, const torch::Tensor& uBias,
                                           const torch::Tensor& center, const torch::Tensor& eps);
    void computeConcreteBounds(const torch::Tensor& lA, const torch::Tensor& uA,
                              const torch::Tensor& lBias, const torch::Tensor& uBias,
                              const torch::Tensor& center, const torch::Tensor& eps,
                              torch::Tensor& concreteLower, torch::Tensor& concreteUpper);


    // Utility methods
    void log( const String &message );

    // Helper functions for A matrix accumulation (following auto-LiRPA's approach)
    torch::Tensor addA(const torch::Tensor& A1, const torch::Tensor& A2);
    void addBound(unsigned elementIndex, const torch::Tensor& lA, const torch::Tensor& uA);
    
    // Helper function for establishing consistent tensor format
    torch::Tensor preprocessC(const torch::Tensor& C, unsigned outputSize);
    
    // Helper method to get output index
    unsigned getOutputIndex() const;
};

} // namespace NLR

#endif // __CROWNAnalysis_h__