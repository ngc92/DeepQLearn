#pragma once

#include "config.h"
#include <boost/circular_buffer.hpp>
#include <vector>
#include <functional>
#include <string>

class Network;
class Solver;
class QLearner;
using qlearn_callback = std::function<void(const QLearner& learner)>;

struct LearningEntry
{
	Vector situation;
	int action;
	Vector q_values;
};

class QLearner
{
public:
	QLearner( int input_size, int action_count, int memory_length, int history = 1 );
	~QLearner();

	void setCallback( qlearn_callback cb ) { mCallback = cb; };

	void setQNetwork( Network net);

	// this puts a new learning step into the Q learner. It returns the
	// action ID that the Q-learner wants to test next.
	// gets current situation and reward that the last step generated.
	int learn_step( const Vector& situation, float reward, bool terminal, Solver& solver );

	// get the Q values of a configuration
	Vector assess( const Vector& situation ) const;

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
	
	const Network& getQNetwork() const { return *mQNetwork; };

private:
	// configuration
	int mInputSize;
	int mNumActions;

	int    mMiniBatchSize  = 32;
	int    mHistoryLength  = 1;
	float  mStepsPerBatch  = 4;
	double mDiscountFactor = 0.99;
	double mLearningRate   = 0.01;

	qlearn_callback mCallback;

	// neural net
	std::shared_ptr<Network> mQNetwork;

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
		Vector situation;
		int action;
		Vector future;
		float reward;
		bool terminal;
	};

	std::vector<LearningEntry> build_mini_batch(  );
	void learn(Solver& solver);

	boost::circular_buffer<Vector> mCurrentHistory;
	boost::circular_buffer<MemoryEntry> mMemory;
	MemoryEntry mMemoryCache;

	// memory interface
	void push_memory(); // adds mMemoryCache to memory;
	MemoryEntry get_memory( int index ); // gets an element from memory

	int mLearnStepCounter = 0;
	std::shared_ptr<Network> mLearningNetwork;
//	std::shared_ptr<BatchTeacher<float_type>> mTeacher;
public:
	static int getAction( const Network& network, const Vector& situation, float& quality );
};
