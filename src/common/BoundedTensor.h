// This is rough template, instead of using marabou's Pair wrapper this would be a more directed wrapper
// Would follow auto lirpa's approach nearly directly
//  Allow for easier porting, and also readbility of code

// This Vector<std::pair<torch::Tensor, torch::Tensor>> inputBounds;
// Becomes vector<BoundedTenrsor> inputBounds;


// Do not want to change all the std::pair to either Marabou' Pair or this since I dont want to do the switch multiple times

// not sure if making these template/wrappers is encouraged if will only be used in the torch CROWN


#ifndef __BoundedTensor_h__
#define __BoundedTensor_h__

#include <iostream>

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>

// Redefine Warning macro for CVC4 compatibility
#ifndef Warning
#define Warning (! ::CVC4::WarningChannel.isOn()) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

template <class T = torch::Tensor> class BoundedTensor
{
    typedef std::pair<T, T> Super;

public:
    BoundedTensor()
    {
    }

    BoundedTensor( const T &lower, const T &upper )
        : _container( lower, upper )
    {
    }

    // Copy constructor to fix deprecation warning
    BoundedTensor( const BoundedTensor<T> &other )
        : _container( other._container )
    {
    }

    // Access methods
    T &lower()
    {
        return _container.first;
    }

    const T &lower() const
    {
        return _container.first;
    }

    T &upper()
    {
        return _container.second;
    }

    const T &upper() const
    {
        return _container.second;
    }

    // Assignment
    BoundedTensor<T> &operator=( const BoundedTensor<T> &other )
    {
        _container = other._container;
        return *this;
    }

    // Width of the bound
    T width() const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            return upper() - lower();
        } else {
            return upper() - lower();
        }
    }

    // Center of the bound
    T center() const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            return (lower() + upper()) / 2.0;
        } else {
            return (lower() + upper()) / 2.0;
        }
    }

    // Comparisons
    bool operator==( const BoundedTensor<T> &other ) const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            return torch::all(torch::eq(lower(), other.lower())).template item<bool>() &&
                   torch::all(torch::eq(upper(), other.upper())).template item<bool>();
        } else {
            return _container == other._container;
        }
    }

    bool operator!=( const BoundedTensor<T> &other ) const
    {
        return !(*this == other);
    }

    bool operator<( const BoundedTensor<T> &other ) const
    {
        if constexpr (std::is_same_v<T, torch::Tensor>) {
            // Compare by center, then by width
            T thisCenter = center();
            T otherCenter = other.center();
            T thisWidth = width();
            T otherWidth = other.width();
            
            if (torch::all(torch::eq(thisCenter, otherCenter)).template item<bool>()) {
                return torch::all(torch::lt(thisWidth, otherWidth)).template item<bool>();
            }
            return torch::all(torch::lt(thisCenter, otherCenter)).template item<bool>();
        } else {
            return _container < other._container;
        }
    }

protected:
    Super _container;
};

template <class T> std::ostream &operator<<( std::ostream &stream, const BoundedTensor<T> &boundedTensor )
{
    return stream << "[" << boundedTensor.lower() << "," << boundedTensor.upper() << "]";
}

#endif // __BoundedTensor_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
