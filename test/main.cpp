#define BOOST_TEST_MODULE deep_q_net

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "../src/fc_layer.hpp"
#include "../src/input_layer.hpp"
#include "../src/nl_layer.hpp"

// layer tests
BOOST_AUTO_TEST_SUITE(layer)

typedef boost::mpl::list<float, double, long double> float_types;

// constructors
BOOST_AUTO_TEST_CASE_TEMPLATE(fc_layer_ctor, T, float_types)
{
	const int NUM_IN = 3;
	const int NUM_OUT = 5;
	
	// test
	typedef FCLayer<T> fc_layer_t;
	
	fc_layer_t layer(NUM_IN, NUM_OUT);
	
	BOOST_CHECK_EQUAL( layer.getNumNeurons(), NUM_OUT );
	BOOST_CHECK_EQUAL( layer.getNumInputs(), NUM_IN );
	
	BOOST_CHECK( layer.getPreviousLayer().expired() );
	BOOST_CHECK( layer.getNextLayer().expired() );

	BOOST_CHECK( !layer.getOutput().empty() );
	BOOST_CHECK( !layer.getWeights().empty() );
	BOOST_CHECK( layer.getBias().empty() );
	BOOST_CHECK( !layer.getGradient().empty() ); /// \todo change, when we make error allocation optional
};

BOOST_AUTO_TEST_CASE_TEMPLATE(input_layer_ctor, T, float_types)
{
	const int NUM_OUT = 5;
	
	// test
	typedef InputLayer<T> layer_t;
	
	layer_t layer(NUM_OUT);
	
	BOOST_CHECK_EQUAL( layer.getNumNeurons(), NUM_OUT );
	BOOST_CHECK_EQUAL( layer.getNumInputs(), 0 );
	
	BOOST_CHECK( layer.getPreviousLayer().expired() );
	BOOST_CHECK( layer.getNextLayer().expired() );
	
	BOOST_CHECK( !layer.getOutput().empty() );
	BOOST_CHECK( layer.getWeights().empty() );
	BOOST_CHECK( layer.getBias().empty() );
	BOOST_CHECK( layer.getGradient().empty() );
};

BOOST_AUTO_TEST_CASE_TEMPLATE(nl_layer_ctor, T, float_types)
{
	const int NUM_OUT = 5;
	
	// test
	typedef NLLayer<T, activation::tanh> layer_t;
	
	layer_t layer(NUM_OUT);
	
	BOOST_CHECK_EQUAL( layer.getNumNeurons(), NUM_OUT );
	BOOST_CHECK_EQUAL( layer.getNumInputs(), NUM_OUT );
	
	BOOST_CHECK( layer.getPreviousLayer().expired() );
	BOOST_CHECK( layer.getNextLayer().expired() );
	
	BOOST_CHECK( !layer.getOutput().empty() );
	BOOST_CHECK( layer.getWeights().empty() ); /// \todo allow weights to configure steepness
	BOOST_CHECK( !layer.getBias().empty() );
	BOOST_CHECK( !layer.getGradient().empty() );
};

// set output tests
BOOST_AUTO_TEST_CASE_TEMPLATE(input_layer, T, float_types)
{
	const int NUM_OUT = 5;
	
	// test
	typedef InputLayer<T> layer_t;
	
	layer_t layer(NUM_OUT);
	
	std::vector<T> output = {1.2, 1.5, 3.6, -1.7, 0.0};
	((ILayer<T>*)(&layer))->setOutput( output );
	
 	auto result = layer.getOutput();
	
	for(int i = 0; i < NUM_OUT; ++i)
	{
		BOOST_CHECK_EQUAL( output[i], result[i] );
	}
	
/// \todo set previous layer currently asserts. change to exception
//	auto prev = std::make_shared<layer_t>(3);
//	layer.setPreviousLayer(prev);
	auto next = std::make_shared<FCLayer<T>>(NUM_OUT, 3);
	layer.setNextLayer(next);
	
//	BOOST_CHECK_EQUAL( layer.getPreviousLayer().lock(), prev );
	BOOST_CHECK_EQUAL( layer.getNextLayer().lock(), next );
};


BOOST_AUTO_TEST_CASE_TEMPLATE(forward, T, float_types)
{
	const int NUM_IN = 2;
	const int NUM_OUT = 2;
	
	typedef InputLayer<T> layer_t;
	auto ip = std::make_shared<layer_t>(NUM_IN);
	
	std::vector<T> inp = {1.0, -1.0};
	ip->setOutput(inp);
	
	// test fully connected
	auto fc = std::make_shared<FCLayer<T>>(NUM_IN, NUM_OUT);
	fc->setPreviousLayer( ip );
	fc->forward(); // all weights are zero prior to initialization
	bool equal = fc->getOutput() == std::vector<T>{0.0, 0.0} ;
	BOOST_CHECK( equal );
	
	std::vector<T> weights = {1.0, 1.0, 2.0, 0.0};
	int i = 0;
	auto f = [weights, i]() mutable
	{
		return weights[i++];
	};
	
	fc->randomizeWeights( f );
	equal = fc->getWeights() == weights;
	BOOST_CHECK(equal);
	
	fc->forward();
	equal = fc->getOutput() == std::vector<T>{0.0, 2.0} ;
	BOOST_CHECK( equal );
	
	// test nonlinearity
	auto nl = std::make_shared<NLLayer<T, activation::tanh>>(NUM_IN);
	nl->setPreviousLayer( ip );
	nl->forward();
	equal = nl->getOutput() == std::vector<T>{std::tanh(T(1.0)), std::tanh(T(-1.0))};
	BOOST_CHECK( equal );
}

BOOST_AUTO_TEST_SUITE_END()

