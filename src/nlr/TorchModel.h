#ifndef __TorchModel_h__
#define __TorchModel_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "Query.h"
#include "InputQueryBuilder.h"
#include "TorchModuleBounded.h"

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

// Enum for element types
enum class ElementType { MODULE, CONSTANT, INPUT };

class TorchModel : public torch::nn::Module {
public:
    TorchModel(
        const Vector<std::shared_ptr<NLR::ITorchModuleBounded>>& boundedModules,
        const Vector<torch::Tensor>& constants,
        const Vector<Vector<Variable>>& marabouVars,
        const Vector<unsigned>& inputIndices,
        unsigned outputIndex,
        const Map<unsigned, Vector<Variable>>& neuronToMarabouMap,
        const Map<unsigned, Vector<unsigned>>& dependencies,
        const Map<unsigned, ElementType>& elementTypes,
        const Map<unsigned, unsigned>& elementToBoundedModuleIndex,
        const Map<unsigned, unsigned>& elementToConstantIndex,
        const Map<unsigned, unsigned>& elementToInputIndex
    );

    // Forward pass methods
    torch::Tensor forward(const Map<unsigned, torch::Tensor>& inputs);
    torch::Tensor forward(torch::Tensor input); // Legacy 

    // Getters for model information
    const Vector<std::shared_ptr<NLR::ITorchModuleBounded>>& getBoundedModules() const { return _boundedModules; }
    const Vector<Vector<Variable>>& getVariables() const { return _marabouVars; }
    const Vector<unsigned>& getInputIndices() const { return _inputIndices; }
    unsigned getOutputIndex() const { return _outputIndex; }
    unsigned getSize() const { return _boundedModules.size(); }
    const Map<unsigned, Vector<Variable>>& getNeuronToMarabouMap() const { return _neuronToMarabouMap; }
    const Map<unsigned, Vector<unsigned>>& getDependencies() const { return _dependencies; }

    // Bound computation methods
    void computeBounds(const Map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& inputBounds);
    std::pair<torch::Tensor, torch::Tensor> getOutputBounds() const;
    NLR::LinearBound getLinearBounds(unsigned elementIndex) const;
    
    // Variable elimination and reindexing
    void eliminateVariable(unsigned variable, double value);
    void updateVariableIndices(const Map<unsigned, unsigned>& oldIndexToNewIndex,
                              const Map<unsigned, unsigned>& mergedVariables);
    bool neuronEliminated(unsigned neuron) const;
    double getEliminatedNeuronValue(unsigned neuron) const;
    void reduceIndexFromAllMaps(unsigned startIndex);
    void adjustMapIndexing(Map<unsigned, Vector<Variable>>& map, unsigned startIndex);

    // Add setter for input bounds
    void setInputBounds(const Map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& inputBounds) {
        _inputBounds = inputBounds;
    }
    
    // Getter for input bounds
    const Map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& getInputBounds() const {
        return _inputBounds;
    }
    
    // Getter for element to bounded module index mapping
    const Map<unsigned, unsigned>& getElementToBoundedModuleIndex() const {
        return _elementToBoundedModuleIndex;
    }

private:
    // Core model components - only bounded modules
    Vector<std::shared_ptr<NLR::ITorchModuleBounded>> _boundedModules;
    Vector<torch::Tensor> _constants;
    Vector<Vector<Variable>> _marabouVars;
    Vector<unsigned> _inputIndices;
    unsigned _outputIndex;
    Map<unsigned, Vector<Variable>> _neuronToMarabouMap;
    Map<unsigned, Vector<unsigned>> _dependencies;
    
    // Mapping system for type safety
    Map<unsigned, ElementType> _elementTypes;
    Map<unsigned, unsigned> _elementToBoundedModuleIndex;
    Map<unsigned, unsigned> _elementToConstantIndex;
    Map<unsigned, unsigned> _elementToInputIndex;
    
    // Track eliminated neurons and their values
    Map<unsigned, double> _eliminatedNeurons;

    // Bound computation state
    Map<unsigned, torch::Tensor> _lowerBounds;
    Map<unsigned, torch::Tensor> _upperBounds;
    Map<unsigned, NLR::LinearBound> _linearBounds;
    Map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _inputBounds;

    // Helper method for recursive forward pass
    torch::Tensor forward(unsigned elementIndex, Map<unsigned, torch::Tensor>& activations, const Map<unsigned, torch::Tensor>& inputs);
};

#endif // __TorchModel_h__ 