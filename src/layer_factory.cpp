#include "layer_factory.hpp"

#include "input_layer.hpp"
#include "output_layer.hpp"
#include "fc_layer.hpp"
#include "nl_layer.hpp"

// would be cool to use boost.hana, but gcc requirements too high right now
#include <boost/algorithm/string/trim_all.hpp>
#include <boost/algorithm/string/split.hpp>

namespace detail
{

	// creator functions
	template<class T>
	using LP = std::shared_ptr<ILayer<T>>;


	template<class T>
	LP<T> createInputLayer(int before, int neurons, std::string parameter)
	{
		return std::make_shared<InputLayer<T>>( neurons );
	}

	template<class T>
	LP<T> createOutputLayer(int before, int neurons, std::string parameter)
	{
		return std::make_shared<OutputLayer<T>>( before );
	}

	template<class T>
	LP<T> createFCLayer(int before, int neurons, std::string parameter)
	{
		return std::make_shared<FCLayer<T>>( before, neurons );
	}

	template<class T>
	LP<T> createNLLayer(int before, int neurons, std::string parameter)
	{
		if( parameter == "tanh" )
			return std::make_shared<NLLayer<T, activation::tanh>>( neurons );
	}

	template<class T>
	LP<T> make_layer( int inputs, int neurons, std::string config )
	{
		boost::algorithm::trim_all(config);
		if( config == "input" )
		{
			return createInputLayer<T>(inputs, neurons, "");
		} else if( config == "output" )
		{
			return createOutputLayer<T>(inputs, neurons, "");
		} else if( config == "fc" )
		{
			return createFCLayer<T>(inputs, neurons, "");
		}
		
		std::vector< std::string > parts;
		
		/// \todo there should be an is_space
		boost::algorithm::split(parts, config, boost::is_any_of(" "));
		
		if( parts[0] == "nl" )
		{
			assert( parts.size() == 2 );
			return createNLLayer<T>(inputs, neurons, parts[1]);
		}
		
		assert(0);
	}

	boost::any createLayer_Internal(std::type_index type, int inputs, int neurons, std::string configuration)
	{
		/// \todo use boost mpl/hana for dispatch
		if( type == typeid(float) )
		{
			return make_layer<float>( inputs, neurons, configuration );
		} 
		else if( type == typeid(double) )
		{
			return make_layer<double>( inputs, neurons, configuration );
		} 
		else if( type == typeid(long double) )
		{
			return make_layer<long double>( inputs, neurons, configuration );
		}
		
		assert(0);
	}

}
