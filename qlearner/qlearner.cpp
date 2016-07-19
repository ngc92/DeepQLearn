#include "qlearner.hpp"
#include "qcore.hpp"
#include "stats.h"

// helpers
/*Vector concat(const boost::circular_buffer<Vector>& b)
{
	
	std::size_t len = 0;
	for(const auto& v : b)
	{
		len += v.rows();
	}
	Vector result(len);
	int p = 0;
	for(const auto& v : b)
	{
		for(int i = 0; i < v.rows(); ++i)
		{
			result[p+i] = v[i];
		}
		p += v.rows();
	}
	return result;
}
*/

namespace qlearn 
{
	using namespace net;
	
	QLearner::QLearner( Config cfg, Network net ) : 
		mConfig( cfg ), 
		mCore( std::make_unique<QCore>( std::move(cfg)) ),
		mStats( std::make_unique<Stats>( 10000 ) ),
		mNetwork( net.clone() ),
		mNetworkGraph( mNetwork ),
		mTargetNet( std::move(net) ),
		mTargetGraph( mTargetNet )
	{
	}
	
	QLearner::~QLearner()
	{
		
	}
	
	int QLearner::learn_step( const Vector& situation, float reward, bool terminal, Solver& solver )
	{
		if(mCore->getSteps() % mConfig.update_interval() == 0)
		{
			if( mCallback )
				mCallback(*this, *mStats);
			
			// replace network
			mTargetNet = mNetwork.clone();
			mTargetGraph = ComputationGraph( mTargetNet );
		}
		
		mCore->backward( reward, terminal );
		auto action = mCore->forward( mNetworkGraph, situation );
		mStats->record(reward, action.score);
		/// \todo technically, this is wrong! reward is shifted by one vs the score!
		
		float mse = mCore->learn(mNetworkGraph, mTargetGraph, solver);
		mNetwork.update( solver );
		mStats->record_error(mse);
		return action.id;
	}
	
	float QLearner::getCurrentEpsilon() const
	{
		return mCore->getEpsilon();
	}
}
