#ifndef QLEARNER_H_INCLUDED
#define QLEARNER_H_INCLUDED

#include <boost/circular_buffer.hpp>
#include <vector>
#include <functional>
#include <string>
#include <atomic>
#include <mutex>
#include <thread>

class QLearner;
using qlearn_callback = std::function<void(const QLearner& learner)>;

extern "C"
{
	struct fann;
	struct fann_train_data;
}

class QLearner
{
public:
	QLearner( int input_size, int action_count, int memory_length );
	~QLearner();
	
	void setCallback( qlearn_callback cb ) { mCallback = cb; };

	void start_learning();
	void load(std::string name);
	void save(std::string file) const;

	// this puts a new learning step into the Q learner. It returns the
	// action ID that the Q-learner wants to test next.
	// gets current situation and reward that the last step generated.
	int learn_step( const std::vector<float>& situation, float reward, bool terminal );
	
	// get the Q values of a configuration
	float* assess( const std::vector<float>& situation );
	
	// parameter setup
	void setMiniBatchSize  ( int    mbs ) { mMiniBatchSize  = mbs; };
	void setStepsPerBatch  ( int    spb ) { mStepsPerBatch  = spb; };
	void setDiscountFactor ( double dis ) { mDiscountFactor = dis; };
	void setLearningRate   ( double lrt ) { mLearningRate   = lrt; };
	void setEpsilonSteps   ( int    eps ) { mEpsilonSteps   = eps; };
	void setNetUpdateRate  ( int    nur ) { mNetUpdateFrq   = nur; };


	// success tracking
	double getAverageError()         const  { return mAverageError;     };
	double getAverageQuality()       const  { return mAverageQuality;   };
	double getAverageEpisodeReward() const;
	float  getCurrentQuality()       const  { return mCurrentQuality;   };

	double getCurrentEpsilon()       const  { return mCurrentEpsilon;   };
	int    getNumberLearningSteps()  const  { return mLearnStepCounter; };

private:
	// configuration
	int mInputSize;
	int mNumActions;

	int    mMiniBatchSize  = 32;
	float  mStepsPerBatch  = 4;
	double mDiscountFactor = 0.99;
	double mLearningRate   = 0.01;
	
	qlearn_callback mCallback;

	// neural net
	fann* mQNetwork;
	std::mutex mLockNetwork;

	// strategy annealing
	double mCurrentEpsilon = 1.0;
	double mFinalEpsilon   = 0.1;
	int    mEpsilonSteps   = 1e6;
	int    mNetUpdateFrq   = 10000;
	int    mInitMemoryPop  = 50000;
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

	static int getAction( fann* net, const std::vector<float>& situation, float& quality );


	struct MemoryEntry
	{
		std::vector<float> situation;
		int action;
		std::vector<float> future;
		float reward;
		bool terminal;
	};

	boost::circular_buffer<MemoryEntry> mMemory;
	std::mutex mLockMemory;
	MemoryEntry mMemoryCache;

	// memory interface
	void push_memory(); // adds mMemoryCache to memory;
	MemoryEntry get_memory( int index ); // gets an element from memory

	// thread management
	std::atomic<bool> mRunLearning; // variable that determinates whether threads are allowed to run

	//  training data generation
	//  this thread is responsible for generating training data
	std::thread mPreparationThread;
	void prepare_training();
	void build_mini_batch( fann* network, fann_train_data* data );
	//  deque containing training sets that are to be learned, and corresponding mutex
	std::deque<fann_train_data*> mPreprocessedTrainingData;
	std::mutex mTrainingDataMutex;

	std::mutex mSpentTrainingMutex;
	std::deque<fann_train_data*> mSpentTrainingData;		// training data memory waiting to be reused

	//  learning thread
	std::thread mLearningThread;
	fann* mLearningNetwork;
	std::mutex mLearningMutex;
	int mLearnStepCounter;
	void learn_thread();
};

#endif // QLEARNER_H_INCLUDED
