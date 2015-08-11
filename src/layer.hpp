#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <boost/range/iterator_range.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/generate.hpp>

// base class of all layers
template<class T>
class ILayer
{
public:
	typedef ILayer<T> this_t;
	typedef std::weak_ptr<const ILayer>  WP_ILayer;
	typedef boost::iterator_range<T*> range_t;
	typedef boost::iterator_range<const T*> const_range_t;
	
	virtual ~ILayer() {};
	virtual void forward() = 0;
	virtual void backward() = 0;
	
	template<class Cont>
	void setOutput(const Cont& container)
	{
		assert( container.size() == getNumNeurons() );
		boost::copy( container, getOutputMutable().begin() );
	}
	
	template<class Cont>
	void setGradient(const Cont& container)
	{
		assert( container.size() == getNumInputs() );
		boost::copy( container, getGradientMutable().begin() );
	}
	
	template<class Cont>
	void getOutput( Cont& cont )
	{
		boost::copy( getOutput(), cont.begin() );
	}
	
	template<class Func>
	void randomizeWeights( const Func& distribution )
	{
		boost::generate( getWeightsMutable(), distribution );		
	}
	
	template<class IT>
	void setWeights( IT iterator )
	{
		std::copy(iterator, iterator + getWeightsMutable().size(), getWeightsMutable().begin());
	}

	// build up connections
	virtual void setPreviousLayer( const WP_ILayer& prev ) = 0;
	virtual void setNextLayer( const WP_ILayer& next ) = 0;
	
	// infos
	virtual unsigned getNumNeurons()   const = 0;
	virtual unsigned getNumInputs()    const = 0;
	virtual const char* getLayerType() const = 0;
	
	virtual const WP_ILayer& getPreviousLayer() const = 0;
	virtual const WP_ILayer& getNextLayer()     const = 0;
	
	// get access to layer data
	const_range_t getOutput()   const { return const_cast<this_t*>(this)->getOutputMutable(); };
	const_range_t getWeights()  const { return const_cast<this_t*>(this)->getWeightsMutable(); };
	const_range_t getBias()     const { return const_cast<this_t*>(this)->getBiasMutable(); };
	const_range_t getGradient()    const { return const_cast<this_t*>(this)->getGradientMutable(); };
	
protected:
	// mutable access to layer data
	virtual range_t getOutputMutable() = 0;
	virtual range_t getWeightsMutable() = 0;
	virtual range_t getBiasMutable() = 0;
	virtual range_t getGradientMutable() = 0;
};

#endif // LAYER_HPP_INCLUDED
