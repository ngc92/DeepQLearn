#pragma once

#include <cstdint>

class QLearnerConfig
{
public:
	QLearnerConfig( std::size_t input_size, std::size_t action_count, std::size_t memory_length );
	
	// setter functions
	QLearnerConfig& batch_size( std::size_t size );
	QLearnerConfig& steps_per_batch( std::size_t steps );
	QLearnerConfig& discount_factor( double factor );
	QLearnerConfig& update_interval( std::size_t interval );
	QLearnerConfig& epsilon_steps( std::size_t steps );
protected:
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
	
	// strategy annealing
	double      mFinalEpsilon   = 0.1;
	std::size_t mEpsilonSteps   = 1e6;
};
