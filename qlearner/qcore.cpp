#include "qcore.hpp"
#include "net/computation_graph.hpp"
#include "memory.hpp"
#include <iostream>

namespace qlearn
{
	using namespace net;
	
	QCore::QCore( Config cfg ) : mConfig( std::move(cfg) ),
	mMemory( std::make_unique<MemoryCache>( mConfig.memory() ) )
	{
		mLastStates.set_capacity(3);
		mLastRewards.set_capacity(3);
		mLastTerminal.set_capacity(3);
		mLastActions.set_capacity(3);
	}
	
	QCore::~QCore() {}
	
	Action getAction(ComputationGraph& graph, const Vector& situation)
	{
		const auto& result = graph.forward( situation );
		// greedy algorithm that generates the next action.
		int row, col;
		float quality = result.maxCoeff(&row,&col);
		return {row, quality};
	}
	
	std::size_t QCore::getRandomAction()
	{
		auto ind_dst = std::uniform_int_distribution<int>(0, mConfig.action_count()-1); // this is inclusive, so we need -1
		return ind_dst(mRandom);
	}
	
	Action QCore::forward( net::ComputationGraph& policy, const Vector& input, bool learning )
	{
		++mStepCounter;
		float eps = mConfig.getStepEpsilon( mLearningSteps );
		
		if(!learning) 		eps = 0.f;
		
		// this performs an assignment when the buffer is full, so we soon stop allocating new memory.
		if(learning)	 mLastStates.push_front( input );
		
		// with certain probability choose a random action
		auto random_action = std::discrete_distribution<int>({1 - eps, eps});
		if( random_action(mRandom))
		{
			Action action;
			action.id = getRandomAction();
			action.score = 0;
			if( learning ) mLastActions.push_front( action.id );
			return action;
		}
		else 
		{
			auto ac = getAction( policy, input );
			if( learning ) mLastActions.push_front( ac.id );
			return std::move(ac);
		}
	}
	
	void QCore::backward( float reward, bool terminal )
	{
		mLastRewards.push_front( reward );
		mLastTerminal.push_front( terminal );
		if(mLastStates.size() < 2) return;
		
		auto& old_state = mLastStates[1];
		float old_rewd  = mLastRewards[1];
		float old_term  = mLastTerminal[1];
		std::size_t old_act = mLastActions[1];
		auto& new_state = mLastStates[0];
		
		mMemory->emplace( old_state, old_act, new_state, old_rewd, old_term );
	}
	
	float getTargetQValue(const Experience& experience, ComputationGraph& target_q, float gamma)
	{
		// best value that can be reached from here
		float y = 0;
		if( !experience.terminal )
		{
			auto best = getAction(target_q, experience.future);
			y = best.score * gamma;
		}
		y += experience.reward;
		
		return y;
	}
	
	float QCore::learn(net::ComputationGraph& policy, net::ComputationGraph& target, Solver& solver)
	{
		// check if we are allowed to learn
		if(mMemory->size() < mConfig.init_memory_size())
			return 0;
	
		++mLearningSteps;
		
		float mse = 0;

		// train an epoch
		for(unsigned i = 0; i < mConfig.batch_size(); ++i)
		{
			/// \attention this line allocates!
			const auto& trans = mMemory->get_random(mRandom);
			float target_value = getTargetQValue( trans, target, mConfig.gamma() );
			
			const auto& result = policy.forward( trans.situation );
			mErrorCache = Vector::Zero( result.size() );
			float delta = result[trans.action] - target_value;
			mErrorCache[trans.action] = delta;
			policy.backpropagate(mErrorCache, solver );
			mse += delta * delta;
		}
		
		mse /= mConfig.batch_size();
		
		return mse;
	}
}

