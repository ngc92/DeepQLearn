#pragma once

#include "config.h"
#include <vector>
#include <memory>
#include <boost/circular_buffer.hpp>

#include "qconfig.hpp"
#include "action.h"

namespace net
{
	class Solver;
	class ComputationGraph;
}

namespace qlearn
{
	class MemoryCache;
	
	class QCore
	{
	public:
		
		QCore( Config cfg );
		~QCore( );
		
		std::size_t getRandomAction(); 
		
		// propagate a game state forward through th graph, and get the action according to the current policy.
		Action forward( net::ComputationGraph& policy, const Vector& input, bool learn = true );
		
		// save the result of the action that was propagated by forward.
		void backward( float reward, bool terminal );
		
		// accumulates gradients of policy in the solver.
		// returns mse of minibatch. 
		float learn(net::ComputationGraph& policy, net::ComputationGraph& target, net::Solver& solver);

		std::size_t getSteps() const { return mStepCounter; }
		float getEpsilon() const;
		
	private:
		Config mConfig;
		
		std::unique_ptr<MemoryCache> mMemory;
		std::size_t mStepCounter = 0;
		std::size_t mLearningSteps = 0;
		
		// cache the last situations
		boost::circular_buffer<Vector> mLastStates;
		boost::circular_buffer<float> mLastRewards;
		boost::circular_buffer<bool> mLastTerminal;
		boost::circular_buffer<std::size_t> mLastActions;
		
		Vector mErrorCache; // cache vector to prevent reallocation
		
		// random engine
		std::default_random_engine mRandom;
	};
}

