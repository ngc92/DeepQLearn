#include "qlearner.hpp"
#include "qcore.hpp"

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
				mCallback(*this);

		/*	// reset reward stats
			mCurNetTotalReward = 0;
			mCurNetTotalEpisodes = 0;

			// replace network
			mQNetwork = std::make_shared<Network>(mLearningNetwork->clone());
			mQGraph = ComputationGraph( *mQNetwork );
		*/
		}
/*
		/// \attention This line allocates
		mCurrentHistory.push_back( situation );
		
		/// \attention This line allocates
		auto hist = concat(mCurrentHistory);*/
		
		mCore->backward( reward, terminal );
		auto action = mCore->forward( mNetworkGraph, situation );

		mCore->learn(mNetworkGraph, mTargetGraph, solver);
		return action.id;
	}
}
