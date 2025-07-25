#include "TorchModel.h"
#include "MarabouError.h"
#include <iostream>

TorchModel::TorchModel(
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
) : _boundedModules(boundedModules),
    _constants(constants),
    _marabouVars(marabouVars),
    _inputIndices(inputIndices),
    _outputIndex(outputIndex),
    _neuronToMarabouMap(neuronToMarabouMap),
    _dependencies(dependencies),
    _elementTypes(elementTypes),
    _elementToBoundedModuleIndex(elementToBoundedModuleIndex),
    _elementToConstantIndex(elementToConstantIndex),
    _elementToInputIndex(elementToInputIndex)
{
    // Set up neuron-to-Marabou mapping for all bounded modules
    for (auto& boundedModule : _boundedModules) {
        // Pass the mapping to each bounded module
        boundedModule->setNeuronToMarabouMap(_neuronToMarabouMap);
    }
}

torch::Tensor TorchModel::forward(unsigned elementIndex, Map<unsigned, torch::Tensor>& activations, 
                                 const Map<unsigned, torch::Tensor>& inputs) {
    // Return cached result if available
    if (activations.exists(elementIndex)) {
        return activations[elementIndex];
    }

    // Get element type
    if (!_elementTypes.exists(elementIndex)) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
            (String("Element index not found: ") + std::to_string(elementIndex)).ascii());
    }
    
    ElementType type = _elementTypes[elementIndex];

    // Handle based on element type
    switch (type) {
        case ElementType::INPUT: {
            if (!_elementToInputIndex.exists(elementIndex)) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("Input element not found in mapping: ") + std::to_string(elementIndex)).ascii());
            }
            unsigned inputIndex = _elementToInputIndex[elementIndex];
            if (!inputs.exists(inputIndex)) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("Input index not found: ") + std::to_string(inputIndex)).ascii());
            }
            activations[elementIndex] = inputs[inputIndex];
            return inputs[inputIndex];
        }
        
        case ElementType::CONSTANT: {
            if (!_elementToConstantIndex.exists(elementIndex)) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("Constant element not found in mapping: ") + std::to_string(elementIndex)).ascii());
            }
            unsigned constantIndex = _elementToConstantIndex[elementIndex];
            if (constantIndex >= _constants.size()) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("Constant index out of bounds: ") + std::to_string(constantIndex)).ascii());
            }
            activations[elementIndex] = _constants[constantIndex];
            return _constants[constantIndex];
        }
        
        case ElementType::MODULE: {
            // Get dependencies for this module
            if (!_dependencies.exists(elementIndex)) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("No dependencies found for module at element index: ") + std::to_string(elementIndex)).ascii());
            }
            
            Vector<unsigned> deps = _dependencies[elementIndex];
            
            // Recursively compute all input activations
            std::vector<torch::Tensor> inputTensors;
            for (unsigned dep : deps) {
                inputTensors.push_back(forward(dep, activations, inputs));
            }

            // All computation nodes must have at least one input
            if (inputTensors.empty()) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("No input tensors for module at element index: ") + std::to_string(elementIndex)).ascii());
            }

            // Get bounded module index and call the module
            if (!_elementToBoundedModuleIndex.exists(elementIndex)) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("Bounded module element not found in mapping: ") + std::to_string(elementIndex)).ascii());
            }
            unsigned boundedModuleIndex = _elementToBoundedModuleIndex[elementIndex];
            
            if (boundedModuleIndex >= _boundedModules.size()) {
                throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
                    (String("Bounded module index out of bounds: ") + std::to_string(boundedModuleIndex)).ascii());
            }
            
            auto& boundedModule = _boundedModules[boundedModuleIndex];
            torch::Tensor result;
            
            if (inputTensors.size() == 1) {
                result = boundedModule->forward(inputTensors[0]);
            } else {
                // For multiple inputs, we need to handle them appropriately
                // For now, we'll use the first input and log a warning
                // TODO: Implement proper multi-input handling for bounded modules
                std::cerr << "Warning: Multiple inputs detected for bounded module. Using first input only." << std::endl;
                result = boundedModule->forward(inputTensors[0]);
            }

            // Cache and return result
            activations[elementIndex] = result;
            return result;
        }
    }
    
    // This should never be reached
    throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
        (String("Unknown element type for element index: ") + std::to_string(elementIndex)).ascii());
}

torch::Tensor TorchModel::forward(torch::Tensor input) {
    Map<unsigned, torch::Tensor> inputs;
    if (_inputIndices.size() > 0) {
        inputs[_inputIndices[0]] = input;
    } else {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
            "No input indices defined for model");
    }
    return forward(inputs);
}

torch::Tensor TorchModel::forward(const Map<unsigned, torch::Tensor>& inputs) {
    Map<unsigned, torch::Tensor> activations;
    return forward(_outputIndex, activations, inputs);
}

void TorchModel::eliminateVariable(unsigned variable, double value){
    // Find which neuron this variable corresponds to
    for (auto& pair : _neuronToMarabouMap)
    {
        unsigned neuron = pair.first;
        const Vector<Variable>& variables = pair.second;
        
        for (unsigned i = 0; i < variables.size(); ++i)
        {
            if (variables[i] == variable)
            {
                // Mark this neuron as eliminated with the given value
                _eliminatedNeurons[neuron] = value;
                return;
            }
        }
    }
}

bool TorchModel::neuronEliminated(unsigned neuron) const{
    return _eliminatedNeurons.exists(neuron);
}

double TorchModel::getEliminatedNeuronValue(unsigned neuron) const{
    if (!_eliminatedNeurons.exists(neuron)) {
        throw MarabouError(MarabouError::TORCH_MODEL_ERROR, 
            (String("Neuron ") + std::to_string(neuron) + " is not eliminated").ascii());
    }
    return _eliminatedNeurons[neuron];
}

void TorchModel::reduceIndexFromAllMaps(unsigned startIndex){
    // Adjust input indices
    for (unsigned i = 0; i < _inputIndices.size(); ++i)
    {
        if (_inputIndices[i] >= startIndex)
            --_inputIndices[i];
    }
    
    // Adjust output index
    if (_outputIndex >= startIndex)
        --_outputIndex;
    
    // Adjust neuronToMarabouMap indices
    adjustMapIndexing(_neuronToMarabouMap, startIndex);
}

void TorchModel::adjustMapIndexing(Map<unsigned, Vector<Variable>>& map, unsigned startIndex){
    Map<unsigned, Vector<Variable>> copyOfMap = map;
    map.clear();
    
    for (const auto& pair : copyOfMap)
    {
        unsigned key = pair.first;
        const Vector<Variable>& value = pair.second;
        
        if (key >= startIndex)
            map[key - 1] = value;
        else
            map[key] = value;
    }
}

void TorchModel::updateVariableIndices(const Map<unsigned, unsigned>& oldIndexToNewIndex,
                                      const Map<unsigned, unsigned>& mergedVariables)
{
    // Update _neuronToMarabouMap (the single source of truth)
    for (auto& pair : _neuronToMarabouMap) {
        Vector<Variable>& vars = pair.second;
        for (unsigned i = 0; i < vars.size(); ++i) {
            unsigned idx = vars[i];
            if (oldIndexToNewIndex.exists(idx))
                vars[i] = Variable(oldIndexToNewIndex[idx]);
            else if (mergedVariables.exists(idx))
                vars[i] = Variable(mergedVariables[idx]);
        }
    }
    
    // Update _marabouVars
    for (unsigned i = 0; i < _marabouVars.size(); ++i) {
        Vector<Variable>& vars = _marabouVars[i];
        for (unsigned j = 0; j < vars.size(); ++j) {
            unsigned idx = vars[j];
            if (oldIndexToNewIndex.exists(idx))
                vars[j] = Variable(oldIndexToNewIndex[idx]);
            else if (mergedVariables.exists(idx))
                vars[j] = Variable(mergedVariables[idx]);
        }
    }
    
}

