#include "qconfig.hpp"

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
 
Config& Config::epsilon_steps( std::size_t steps )
{
	mEpsilonSteps = steps;
	return *this;
}

float Config::getStepEpsilon( std::size_t num_step ) const
{
	if(num_step > mEpsilonSteps)
		return mFinalEpsilon;
	
	return mFinalEpsilon + (1 - mFinalEpsilon) * (double)num_step / mEpsilonSteps;
}
}
