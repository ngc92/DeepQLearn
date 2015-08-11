#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <iostream>

#include "../src/layer.hpp"
#include "../src/layer_factory.hpp"

// layer tests
BOOST_AUTO_TEST_SUITE(factory)

typedef boost::mpl::list<float, double, long double> float_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(factory, T, float_types)
{
	auto input = createLayer<T>(10, "input");
	auto output = createLayer<T>(10, "output");
	auto fc = createLayer<T>(7, 10, "fc");
	auto nl = createLayer<T>(9, "nl tanh");
	
	// check results
	BOOST_CHECK_EQUAL( input->getNumNeurons(), 10 );
	BOOST_CHECK_EQUAL( input->getNumInputs(), 0 );
	BOOST_CHECK_EQUAL( input->getLayerType(), "input" );
	
	BOOST_CHECK_EQUAL( output->getNumNeurons(), 10 );
	BOOST_CHECK_EQUAL( output->getNumInputs(), 10 );
	BOOST_CHECK_EQUAL( output->getLayerType(), "output" );
	
	BOOST_CHECK_EQUAL( fc->getNumNeurons(), 10 );
	BOOST_CHECK_EQUAL( fc->getNumInputs(), 7 );
	BOOST_CHECK_EQUAL( fc->getLayerType(), "fc" );
	
	BOOST_CHECK_EQUAL( nl->getNumNeurons(), 9 );
	BOOST_CHECK_EQUAL( nl->getNumInputs(), 9 );
	BOOST_CHECK_EQUAL( nl->getLayerType(), "nl" );
}

BOOST_AUTO_TEST_SUITE_END()

