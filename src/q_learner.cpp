#include "q_learner.hpp"
#include "network.hpp"
#include "rmsprop.hpp"

#include <boost/range/algorithm/max_element.hpp>
#include <iostream>

// helpers
std::vector<float> concat(const boost::circular_buffer<std::vector<float>>& b)
{
	std::vector<float> result;
	std::size_t len = 0;
	for(const auto& v : b)
	{
		len += v.size();
	}
	result.reserve(len);
	for(const auto& v : b)
	{
		std::copy(v.begin(), v.end(), std::back_inserter(result));
	}
	return result;
}

QLearner::QLearner(int input_size, int action_count, int memory_length, int history):
	mInputSize(input_size * history),
	mNumActions(action_count),
	mHistoryLength( history )
{
	mMemory.set_capacity( memory_length );
	mCurrentHistory.set_capacity( mHistoryLength );
	
	mInitMemoryPop = std::min( memory_length, 100*mMiniBatchSize );
}

void QLearner::setQNetwork(std::shared_ptr<Network<float_type>> net)
{
	mQNetwork = net;
	mLearningNetwork = std::make_shared<Network<float_type>>(std::move(net->clone()));
	
	mTeacher = std::make_shared<BatchTeacher<float_type>>( mLearningNetwork, std::make_shared<RMSPROPWeightUpdater<float_type>>() );
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

int QLearner::learn_step( const std::vector<float>& situation, float reward, bool terminal )
{
	mStepCounter++;
	if(mLearningNetwork && mStepCounter % mNetUpdateFrq == 0)
	{
		// reset reward stats
		mCurNetTotalReward = 0;
		mCurNetTotalEpisodes = 0;

		// replace network
		std::cout << "update q net\n";
		mQNetwork = std::make_shared<Network<float_type>>(mLearningNetwork->clone());
	}

	mCurrentHistory.push_back( situation );
	auto hist = concat(mCurrentHistory);
	
	mMemoryCache.future   =  hist;
	mMemoryCache.terminal =  terminal;
	mMemoryCache.reward   =  reward;
	mCurrenEpisodeReward  += reward;
	if( mMemoryCache.situation.size() != 0)
		push_memory();

	if( terminal )
	{
		mCurNetTotalReward += mCurrenEpisodeReward;
		mCurNetTotalEpisodes++;
		mCurrenEpisodeReward = 0;
	}

	int action = getAction( mQNetwork, hist, mCurrentQuality );
	if(mCurrentHistory.size() == mHistoryLength)
		mMemoryCache.situation = hist;

//	std::cout << mCurrentQuality << "\n";

	mAverageQuality = mFloatingMean * mAverageQuality + (1-mFloatingMean) * mCurrentQuality;

	// update the strategy: adapt epsilon
	if( mCurrentEpsilon > mFinalEpsilon && mMemory.size() >  mInitMemoryPop)
		mCurrentEpsilon -= (1.0 - mFinalEpsilon) / mEpsilonSteps;

	// with certain probability choose a random action
	auto random_action = std::discrete_distribution<int>({1 - mCurrentEpsilon, mCurrentEpsilon});
	if( random_action(mRandom))
	{
		auto ind_dst = std::uniform_int_distribution<int>(0, mNumActions-1); // this is inclusive, so we need -1
		action = ind_dst(mRandom);
	}

	mMemoryCache.action = action;
	
	
	// 
	if( mMemory.size() > mInitMemoryPop)
		learn();
	
	return action;
}

float* QLearner::assess( const std::vector<float>& situation )
{
	mQNetwork->forward(situation);
	return nullptr;
}

int QLearner::getAction( const std::shared_ptr<Network<float_type>>& network, const std::vector<float>& situation, float& quality )
{
	network->forward(situation);
	// greedy algorithm that generates the next action.
	auto scores = network->getOutput();
	auto max = boost::max_element(scores);
	quality = *max;
	return  max - scores.begin();
}

void QLearner::push_memory()
{
	mMemory.push_back( mMemoryCache );
}

auto QLearner::get_memory( int index ) -> MemoryEntry
{
	return mMemory[index]; /// \todo this requires coying and memory allocation.
	/// not nice for performance
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// 					training preparation thread
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// helper function for gathering the data for a mini batch
std::vector<LearningEntry> QLearner::build_mini_batch(  )
{
	std::vector<LearningEntry> dataset;
	for(unsigned i = 0; i < mMiniBatchSize; ++i)
	{
		int sample = std::uniform_int_distribution<int>(0, mMemory.size() - 1)(mRandom);
		MemoryEntry trans = get_memory( sample );
		
		LearningEntry entry;
		entry.situation = trans.situation;
		entry.action = trans.action;
		entry.q_values.resize( mNumActions );

		// best value that can be reached from here
		float y = 0;
		if( !trans.terminal )
		{
			int best = getAction(mLearningNetwork, trans.future, y);
			// plus current reward
			y *= mDiscountFactor;
		}
		y += trans.reward;

		// get current output
		mQNetwork->forward(trans.situation);
		boost::copy(mQNetwork->getOutput(), entry.q_values.begin());
		entry.q_values[trans.action] = y;
		
		dataset.push_back(entry);
	}
	return dataset;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -					learning thread								 -
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void QLearner::learn()
{
	mLearnStepCounter++;
	// set learning parameters
	auto batch = build_mini_batch();

	// train an epoch
	for(auto& entry : batch )
	{
		mTeacher->sample_supervised( entry.situation, entry.q_values );
	}
	
	float error = mTeacher->finishMiniBatch();
	/// \todo ensure that error does not diverge
	mAverageError = mFloatingMean * mAverageError + (1-mFloatingMean)*error;
}
