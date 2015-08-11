#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <iostream>

#include "../src/network.hpp"

// layer tests
BOOST_AUTO_TEST_SUITE(network)

typedef boost::mpl::list<float, double, long double> float_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(network_ctor, T, float_types)
{
	const int NUM_IN = 5;
	const int NUM_OUT = 4;
	
	LayerInfo first = {"input", NUM_IN};
	LayerInfo second = {"fc", NUM_OUT};
	LayerInfo third =  {"nl tanh", NUM_OUT};
	
	Network<T> net = {std::vector<LayerInfo>{first, second, third}};
	
	BOOST_CHECK_EQUAL(net.getNumInputs(), NUM_IN);
	BOOST_CHECK_EQUAL(net.getNumOutputs(), NUM_OUT);
	BOOST_CHECK_EQUAL(net.getNumLayers(), 4);
	
	BOOST_CHECK_EQUAL( net.getLayer(0)->getLayerType(), std::string("input") );
	BOOST_CHECK_EQUAL( net.getLayer(1)->getLayerType(), std::string("fc") );
	BOOST_CHECK_EQUAL( net.getLayer(2)->getLayerType(), std::string("nl") );
	BOOST_CHECK_EQUAL( net.getLayer(3)->getLayerType(), std::string("output") );
	
	/// ! \todo add tests for wrong configs
}

BOOST_AUTO_TEST_CASE_TEMPLATE(forward, T, float_types)
{
	const int NUM_IN = 2;
	const int NUM_OUT = 2;
	
	LayerInfo first = {"input", NUM_IN};
	LayerInfo second = {"fc", NUM_OUT};
	LayerInfo third =  {"nl tanh", NUM_OUT};
	Network<T> net = {std::vector<LayerInfo>{first, second, third}};
	
	std::vector<T> weights = {1.0, 0.0, 1.0, 1.0};
	net.getLayer(1)->setWeights( weights.begin() );	
	
	std::vector<T> input = {1.0, -1.0};
	net.forward(input);
	
	std::vector<T> output = {std::tanh(T(1.0)), std::tanh(T(0))};
	BOOST_CHECK( net.getOutput() == output );
}

BOOST_AUTO_TEST_SUITE_END()

