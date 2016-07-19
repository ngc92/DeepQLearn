#include "qconfig.hpp"
#include <algorithm>

namespace qlearn 
{
Config::Config( std::size_t input_size, std::size_t action_count, std::size_t memory_length ) :
	mInputSize( input_size ), mActionCount( action_count ), mMemoryLength( memory_length )
{
	
}

Config& Config::batch_size( std::size_t size )
{
	mMiniBatchSize = size;
	return *this;
}

Config& Config::steps_per_batch( std::size_t steps )
{
	mStepsPerBatch = steps;
	return *this;
}

Config& Config::discount_factor( double factor )
{
	mDiscountFactor = factor;
	return *this;
}

Config& Config::update_interval( std::size_t interval )
{
	mNetUpdateFrq = interval;
	return *this;
}

Config& Config::init_memory_size( std::size_t init_mem )
{
	mInitMemorySize = init_mem;
	return *this;
}

Config& Config::init_epsilon_time( std::size_t initeps )
{
	mEpsilonStart = initeps;
	return *this;
}
 
Config& Config::epsilon_steps( std::size_t steps )
{
	mEpsilonSteps = steps;
	return *this;
}

float Config::getStepEpsilon( std::size_t num_step ) const
{
	if(num_step > mEpsilonSteps + mEpsilonStart)
		return mFinalEpsilon;
	if( num_step < mEpsilonStart )
		return 1;

	double f = (double)(num_step - mEpsilonStart) / mEpsilonSteps;
	
	return mFinalEpsilon * f + (1-f);
}
}
