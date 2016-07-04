#pragma once

#include <cstdint>

namespace qlearn
{
	class Config
	{
	public:
		Config( std::size_t input_size, std::size_t action_count, std::size_t memory_length );
		
		// setter functions
		Config& batch_size( std::size_t size );
		Config& steps_per_batch( std::size_t steps );
		Config& discount_factor( double factor );
		Config& update_interval( std::size_t interval );
		Config& epsilon_steps( std::size_t steps );
		
		// get info
		float getStepEpsilon( std::size_t num_step ) const;
		
		std::size_t init_memory_size() const { return mInitMemorySize; };
		std::size_t batch_size() const { return mMiniBatchSize; };
		std::size_t action_count() const { return mActionCount; }
		double      gamma() const { return mDiscountFactor; } 
		std::size_t update_interval(  ) const { return mNetUpdateFrq; }
		std::size_t memory(  ) const { return mMemoryLength; }
	private:
		// problem config
		std::size_t mInputSize;
		std::size_t mActionCount;
		std::size_t mHistoryLength  = 1;

		// learning parameters
		std::size_t mMiniBatchSize  = 32;
		std::size_t mStepsPerBatch  = 4;

		// q algorithm params
		std::size_t mMemoryLength;
		double      mDiscountFactor = 0.99;
		std::size_t mNetUpdateFrq   = 10000;
		std::size_t mInitMemorySize = 1000;
		
		// strategy annealing
		float       mFinalEpsilon   = 0.1;
		std::size_t mEpsilonSteps   = 1e6;
	};
}
