#pragma once

#include <memory>
#include <string>
#include <boost/any.hpp>
#include <typeindex>

template<class T>
class ILayer;

namespace detail
{
	boost::any createLayer_Internal(std::type_index type, int inputs, int neurons, std::string configuration);
}

template<class T>
std::shared_ptr<ILayer<T>> createLayer(int inputs, int neurons, const std::string& configuration)
{
	boost::any layer = detail::createLayer_Internal( typeid(T), inputs, neurons, configuration );
	return boost::any_cast<std::shared_ptr<ILayer<T>>>(layer);
}

/// overload for 1 to 1 layers / layers that have only one significant size
template<class T>
std::shared_ptr<ILayer<T>> createLayer(int neurons, const std::string& configuration)
{
	boost::any layer = detail::createLayer_Internal( typeid(T), neurons, neurons, configuration );
	return boost::any_cast<std::shared_ptr<ILayer<T>>>(layer);
}

