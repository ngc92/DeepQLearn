#pragma once

typedef float float_type;

#include <boost/circular_buffer.hpp>
#include <vector>
#include <functional>
#include <string>
#include "fwd.hpp"

struct LearningEntry
{
	std::vector<float_type> situation;
	int action;
	std::vector<float_type> q_values;
};

class QLearner
{
public:
	QLearner( int input_size, int action_count, int memory_length, int history = 1 );
	~QLearner();
	
	void setQNetwork(std::shared_ptr<Network<float_type>> net);

	// this puts a new learning step into the Q learner. It returns the
	// action ID that the Q-learner wants to test next.
	// gets current situation and reward that the last step generated.
	int learn_step( const std::vector<float_type>& situation, float_type reward, bool terminal );
	
	// get the Q values of a configuration
	float* assess( const std::vector<float_type>& situation );
	
	// parameter setup
	void setMiniBatchSize  ( int    mbs ) { mMiniBatchSize  = mbs; };
	void setStepsPerBatch  ( int    spb ) { mStepsPerBatch  = spb; };
	void setDiscountFactor ( double dis ) { mDiscountFactor = dis; };
	void setLearningRate   ( double lrt ) { mLearningRate   = lrt; };
	void setEpsilonSteps   ( int    eps ) { mEpsilonSteps   = eps; };
	void setFinalEpsilon   ( double  eps ) { mFinalEpsilon   = eps; };
	void setNetUpdateRate  ( int    nur ) { mNetUpdateFrq   = nur; };
	void setInitMemoryPop  ( int    imp ) { mInitMemoryPop  = imp; };


	// success tracking
	double getAverageError()         const  { return mAverageError;     };
	double getAverageQuality()       const  { return mAverageQuality;   };
	double getAverageEpisodeReward() const;
	float  getCurrentQuality()       const  { return mCurrentQuality;   };

	double getCurrentEpsilon()       const  { return mCurrentEpsilon;   };
	int    getNumberLearningSteps()  const  { return mLearnStepCounter; };
	
	
	int getInputSize() const { return mInputSize; };
	int getNumActions() const { return mNumActions; };

private:
	// configuration
	int mInputSize;
	int mNumActions;

	int    mMiniBatchSize  = 32;
	int    mHistoryLength  = 1;
	float  mStepsPerBatch  = 4;
	double mDiscountFactor = 0.99;
	double mLearningRate   = 0.01;

	// neural net
	std::shared_ptr<Network<float_type>> mQNetwork;

	// strategy annealing
	double mCurrentEpsilon = 1.0;
	double mFinalEpsilon   = 0.1;
	int    mEpsilonSteps   = 1e6;
	int    mNetUpdateFrq   = 10000;
	int    mInitMemoryPop;
	int    mStepCounter    = 0;

	// success tracking
	double mFloatingMean        = 0.999;
	double mAverageError        = 0;
	double mAverageQuality      = 0;
	float  mCurrentQuality      = 0;
	float  mCurrenEpisodeReward = 0;     // cummulative reward of current episode
	float  mCurNetTotalReward   = 0;	 // total reward of current Q-Net
	int    mCurNetTotalEpisodes = 0;     // number of episodes played by the current net

	// random engine
	std::default_random_engine mRandom;

	struct MemoryEntry
	{
		std::vector<float> situation;
		int action;
		std::vector<float> future;
		float reward;
		bool terminal;
	};
	
	std::vector<LearningEntry> build_mini_batch(  );
	void learn();

	boost::circular_buffer<std::vector<float>> mCurrentHistory;
	boost::circular_buffer<MemoryEntry> mMemory;
	MemoryEntry mMemoryCache;

	// memory interface
	void push_memory(); // adds mMemoryCache to memory;
	MemoryEntry get_memory( int index ); // gets an element from memory
	
	int mLearnStepCounter = 0;
	std::shared_ptr<Network<float_type>> mLearningNetwork;
	std::shared_ptr<BatchTeacher<float_type>> mTeacher;
	
	int getAction( const std::shared_ptr<Network<float_type>>& network, const std::vector<float>& situation, float& quality );
};
