#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

#include "config.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <boost/range/iterator_range.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/algorithm/generate.hpp>
#include <boost/range/algorithm/fill.hpp>

// base class of all layers

/// \todo change gradient to be saved alongside output instead of input.
class ILayer
{
public:
	typedef std::weak_ptr<const ILayer>  WP_ILayer;
	typedef boost::iterator_range<float_t*> range_t;
	typedef boost::iterator_range<const float_t*> const_range_t;

	virtual ~ILayer() {};
	virtual void forward() = 0;
	virtual void backward() = 0;

	// clones this layer.
	virtual std::shared_ptr<ILayer> clone() const = 0;

	/// calculates the learning slopes. they will be added to
	/// \p target, and the pointer will be incremented.
	virtual void calcLearningSlopes( T*& target ) = 0;

	void updateWeights( T*& source )
	{
		auto weights = getWeightsMutable();
		for( auto& w : weights )
			w += *(source++);
	}

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

	void resetGradient()
	{
		boost::fill(getGradientMutable(), T(0));
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
	const_range_t getGradient() const { return const_cast<this_t*>(this)->getGradientMutable(); };

protected:
	// mutable access to layer data
	virtual range_t getOutputMutable() = 0;
	virtual range_t getWeightsMutable() = 0;
	virtual range_t getGradientMutable() = 0;
};

#endif // LAYER_HPP_INCLUDED
