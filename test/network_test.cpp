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
	std::vector<T> bias= {0.0, 0.5};
	net.getLayer(2)->setWeights( bias.begin() );
	
	std::vector<T> input = {1.0, -1.0};
	net.forward(input);
	
	std::vector<T> output = {std::tanh(T(1.0)), std::tanh(T(0.5))};
	BOOST_CHECK( net.getOutput() == output );
}

BOOST_AUTO_TEST_CASE_TEMPLATE(backward, T, float_types)
{
	// THIS IS CURRENTLY USED TO TEST LEARNING
	const int NUM_IN = 2;
	const int NUM_OUT = 2;
	
	LayerInfo first = {"input", NUM_IN};
	LayerInfo second = {"fc", NUM_OUT};
	LayerInfo third =  {"nl tanh", NUM_OUT};
	Network<T> net = {std::vector<LayerInfo>{first, second, third}};
	
	std::vector<T> weights = {1.0, 0.0, 1.0, 1.0};
	net.getLayer(1)->setWeights( weights.begin() );	
	std::vector<T> bias= {0.0, 0.5};
	net.getLayer(2)->setWeights( bias.begin() );
	/*
	for(int i = 0; i < 10; ++i)
	{
		std::vector<T> input = {1.0, -1.0};
		net.forward(input);
		
		// desired output
		std::vector<T> out = {std::tanh(T(1.0)), std::tanh(T(0.0))};
		auto rout = net.getOutput();
		std::cout << net.getLayer(2)->getWeights()[1] << "\n";
		T sqerrsum = 0;
		for( unsigned i = 0; i < out.size(); ++i)
		{
			out[i] -= rout[i];
			sqerrsum += out[i] * out[i];
		}
		net.backward(out);
		for(int j = 0; j < 4; ++j)
		{
//			net.getLayer(j)->updateWeights(0.5);
		}
		
		std::cout << rout[0] << " " << rout[1] << " " << sqerrsum << "\n";
	
	}
	*/
	
	//BOOST_CHECK( net.getOutput() == output );
}

BOOST_AUTO_TEST_SUITE_END()

