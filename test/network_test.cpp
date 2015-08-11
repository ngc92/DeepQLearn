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
	
	/// ! \todo add tests for wrong configs
}

BOOST_AUTO_TEST_SUITE_END()

