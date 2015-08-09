#ifndef LAYER_HPP_INCLUDED
#define LAYER_HPP_INCLUDED

#include <vector>
#include <algorithm>
#include <memory>
#include <functional>
#include <boost/range/iterator_range.hpp>
#include <boost/range/algorithm/copy.hpp>

// base class of all layers
template<class T>
class ILayer
{
public:
	typedef std::weak_ptr<const ILayer>  WP_ILayer;
	typedef boost::iterator_range<T*> range_t;
	typedef boost::iterator_range<const T*> const_range_t;
	
	virtual ~ILayer() {};
	virtual void forward() = 0;
	
	void setOutput(const std::vector<T>& container)
	{
		assert( container.size() == getNumNeurons() );
		setOutput( container.data() );
	}
	
	template<class Cont>
	void getOutput( Cont& cont )
	{
		boost::copy( getOutput(), cont.begin() );
	}
	
	virtual void setOutput(const T* out) = 0; // forces the layers output to be \p out. A safer version (overload) of this function is provided 
											  // that checks length compatibility
	
	virtual void randomizeWeights( const std::function<T()>& distribution ) = 0;
	
	// build up connections
	virtual void setPreviousLayer( const WP_ILayer& prev ) = 0;
	virtual void setNextLayer( const WP_ILayer& next ) = 0;
	
	// infos
	virtual unsigned getNumNeurons() const = 0;
	virtual unsigned getNumInputs()  const = 0;
	
	virtual const WP_ILayer& getPreviousLayer() const = 0;
	virtual const WP_ILayer& getNextLayer()     const = 0;
	
	// get access to layer data
	virtual const_range_t getOutput()   const = 0;
	virtual const_range_t getNeuronIn() const = 0; 
	virtual const_range_t getWeights()  const = 0;
	virtual const_range_t getBias()     const = 0;
	virtual const_range_t getError()    const = 0;
};

#endif // LAYER_HPP_INCLUDED
