#include "q_learner.hpp"
#include "net/network.hpp"
#include "net/computation_node.hpp"
#include "net/rmsprop.hpp"

#include <boost/range/algorithm/max_element.hpp>
#include <iostream>

// helpers
Vector concat(const boost::circular_buffer<Vector>& b)
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

QLearner::QLearner( QLearnerConfig config ):
	QLearnerConfig( std::move(config) ), mMemory( mMemoryLength )
{
	mCurrentHistory.set_capacity( mHistoryLength );
}

void QLearner::setQNetwork(Network net)
{
	mQNetwork = std::make_shared<Network>(std::move(net.clone()));
	mLearningNetwork = std::make_shared<Network>(std::move(net.clone()));
}

QLearner::~QLearner()
{
}

double QLearner::getAverageEpisodeReward() const
{
	if(mCurNetTotalEpisodes == 0)
		return 0;

	return mCurNetTotalReward / mCurNetTotalEpisodes;
}

int QLearner::learn_step( const Vector& situation, float reward, bool terminal, Solver& solver )
{
	mStepCounter++;
	if(mLearningNetwork && mStepCounter % mNetUpdateFrq == 0)
	{
		if( mCallback )
			mCallback(*this);

		// reset reward stats
		mCurNetTotalReward = 0;
		mCurNetTotalEpisodes = 0;

		// replace network
		mQNetwork = std::make_shared<Network>(mLearningNetwork->clone());
	}

	/// \attention This line allocates
	mCurrentHistory.push_back( situation );
	
	/// \attention This line allocates
	auto hist = concat(mCurrentHistory);
	/// \attention This line allocates
	mMemory.push( hist, terminal, reward );
	mCurrenEpisodeReward  += reward;

	if( terminal )
	{
		mCurNetTotalReward += mCurrenEpisodeReward;
		mCurNetTotalEpisodes++;
		mCurrenEpisodeReward = 0;
	}
	/// \attention This line allocates
	int action = getAction( *mQNetwork, hist, mCurrentQuality );

//	std::cout << mCurrentQuality << "\n";
//Eigen::internal::set_is_malloc_allowed(false);
	mAverageQuality = mFloatingMean * mAverageQuality + (1-mFloatingMean) * mCurrentQuality;

	// update the strategy: adapt epsilon
	if( mCurrentEpsilon > mFinalEpsilon)
		mCurrentEpsilon -= (1.0 - mFinalEpsilon) / mEpsilonSteps;

	// with certain probability choose a random action
	auto random_action = std::discrete_distribution<int>({1 - mCurrentEpsilon, mCurrentEpsilon});
	if( random_action(mRandom))
	{
		auto ind_dst = std::uniform_int_distribution<int>(0, mActionCount-1); // this is inclusive, so we need -1
		action = ind_dst(mRandom);
	}
	
	mMemory.prepare_next( std::move(hist), action );
	learn(solver);
	
//	Eigen::internal::set_is_malloc_allowed(true);
	return action;
}

Vector QLearner::assess( const Vector& situation ) const
{
	return (*mQNetwork)(situation).output();
}

int QLearner::getAction( const Network& network, Vector situation, float& quality )
{
	/// \attention this causes dynamic memory allocation.
	auto result = network( std::move(situation) );
	// greedy algorithm that generates the next action.
	int row, col;
	quality = result.output().maxCoeff(&row,&col);
	return row;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// 					training preparation thread
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// helper function for gathering the data for a mini batch
std::vector<LearningEntry> QLearner::build_mini_batch(  )
{
	std::vector<LearningEntry> dataset;
	dataset.reserve( mMiniBatchSize );
	for(unsigned i = 0; i < mMiniBatchSize; ++i)
	{
		/// \attention this line allocates!
		auto trans = mMemory.get_random(mRandom);

		LearningEntry entry;
		entry.situation = std::move(trans.situation);
		entry.action = trans.action;
		entry.q_values.resize( mActionCount );

		// best value that can be reached from here
		float y = 0;
		if( !trans.terminal )
		{
			int best = getAction(*mQNetwork, std::move(trans.future), y);
			// plus current reward
			y *= mDiscountFactor;
		}
		y += trans.reward;

		// get current output
		auto result = (*mLearningNetwork)(entry.situation);
		entry.q_values = result.output();
		entry.q_values[trans.action] = y;

		dataset.push_back(entry);
	}
	return dataset;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -					learning thread								 -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void QLearner::learn(Solver& solver)
{
	if(mMemory.size() < 1)
		return;
	
	mLearnStepCounter++;
	// set learning parameters
	auto batch = build_mini_batch();

	float mse = 0;
	
	// train an epoch
	for(auto& entry : batch )
	{
		auto result = (*mLearningNetwork)( entry.situation );
		auto error = result.output()  - entry.q_values;
		result.backpropagate(error, solver );
		mse += error.squaredNorm();
	}
	
	mse /= batch.size();

	mLearningNetwork->update(solver);
	/// \todo ensure that error does not diverge
	mAverageError = mFloatingMean * mAverageError + (1-mFloatingMean)*mse;

}
