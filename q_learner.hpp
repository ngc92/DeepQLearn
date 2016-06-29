#pragma once

#include "config.h"
#include <vector>
#include <functional>
#include <string>

#include "q_learner_config.h"
#include "memory_cache.hpp"

class QLearner;
using qlearn_callback = std::function<void(const QLearner& learner)>;

using namespace net;

struct LearningEntry
{
	Vector situation;
	int action;
	Vector q_values;
};

class QLearner : private QLearnerConfig
{
public:
	QLearner( QLearnerConfig config );
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
	void setFinalEpsilon   ( double  eps ) { mFinalEpsilon   = eps; };


	// success tracking
	double getAverageError()         const  { return mAverageError;     };
	double getAverageQuality()       const  { return mAverageQuality;   };
	double getAverageEpisodeReward() const;
	float  getCurrentQuality()       const  { return mCurrentQuality;   };

	double getCurrentEpsilon()       const  { return mCurrentEpsilon;   };
	int    getNumberLearningSteps()  const  { return mLearnStepCounter; };


	int getInputSize() const { return mInputSize; };
	int getNumActions() const { return mActionCount; };
	
	const Network& getQNetwork() const { return *mQNetwork; };

private:
	qlearn_callback mCallback;

	// neural net
	std::shared_ptr<Network> mQNetwork;

	// strategy annealing
	double mCurrentEpsilon = 1.0;
	double mFinalEpsilon   = 0.1;
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

	std::vector<LearningEntry> build_mini_batch(  );
	void learn(Solver& solver);

	boost::circular_buffer<Vector> mCurrentHistory;
	MemoryCache mMemory;

	int mLearnStepCounter = 0;
	std::shared_ptr<Network> mLearningNetwork;
public:
	static int getAction( const Network& network, const Vector& situation, float& quality );
};
