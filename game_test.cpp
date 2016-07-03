#include "q_learner.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>
#include <irrlicht/irrlicht.h>
#include <boost/lexical_cast.hpp>

#include "config.h"
#include "net/fc_layer.hpp"
#include "net/relu_layer.hpp"
#include "net/tanh_layer.hpp"
#include "net/solver.hpp"
#include "net/rmsprop.hpp"
#include "net/network.hpp"

#include "games/collect.h"


using namespace net;
using namespace irr;

void build_image(const QLearner& l);

IrrlichtDevice* device;
std::mutex mTargetNet;

void learn_thread( Network& target_net, Game& game )
{
	game.restart();
	Vector state;
	game.getCurrentState( state );
	
	QLearner learner( QLearnerConfig( state.size(), game.getNumInputs(), 30000).epsilon_steps(200000).update_interval(2000).batch_size(64) );
	
	Network network;
	network << FcLayer((Matrix::Random(50, learner.getInputSize()).array() - 0.5) / 5);
	network << TanhLayer(Matrix::Zero(50, 1));
	network << FcLayer((Matrix::Random(50, 50).array() - 0.5) / 7);
	network << TanhLayer(Matrix::Zero(50, 1));
	network << FcLayer((Matrix::Random(game.getNumInputs(), 50).array() - 0.5) / 7);
	network << TanhLayer(Matrix::Zero(game.getNumInputs(), 1));
	
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	
	std::fstream rewf("reward.txt", std::fstream::out);
	learner.setQNetwork( std::move(network) );

	int ac = 2;
	auto last_time = std::chrono::high_resolution_clock::now();
	bool run = true;
	int episodes = 0;
	
	learner.setCallback( [&](const QLearner& l ) 
	{
		std::cout << learner.getCurrentEpsilon() << "\n";
		std::cout << learner.getAverageEpisodeReward() << " (" << learner.getAverageQuality() << ", " << learner.getAverageError() << ")\n";
		rewf << learner.getAverageEpisodeReward() << " " << learner.getAverageQuality() << " " << learner.getAverageError() << "\n";
		rewf.flush();
		std::cout << learner.getNumberLearningSteps() << "\n";
		std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(
					std::chrono::high_resolution_clock::now() - last_time).count() << " ms\n";
		last_time = std::chrono::high_resolution_clock::now();
		std::cout << " - - - - - - - - - - \n";
		std::lock_guard<std::mutex> lck(mTargetNet);
		target_net = learner.getQNetwork().clone();
		
		if(episodes > 160)
		{
			prop->setRate(0.00025);
		} else if(episodes > 80)
		{
			prop->setRate(0.0005);
		}
	} );
	
	
	while(run)
	{
		float r = game.step(ac);
		try
		{
			game.getCurrentState(state);
			ac = learner.learn_step( state, r, r != 0, solver );
		} catch( std::exception& e)
		{
			std::cout << "EXCEPTION " << e.what() << "\n";
		} catch( ... )
		{
			std::cout << "???";
		}
		
		if(game.isFinished())
			game.restart();
	}
}

extern int status;
int main()
{
	
	Collect game;
	game.restart();
	
	Collect learn_game;
	
	Network network;
	std::thread learner( learn_thread, std::ref(network), std::ref(learn_game) );
	learner.detach();
	
	std::fstream rewf("reward.txt", std::fstream::out);

	device = createDevice(video::EDT_SOFTWARE, core::dimension2du(800, 600));

	Vector state;
	while(device->run())
	{
		float v;
		{
			std::lock_guard<std::mutex> lck(mTargetNet);
			game.getCurrentState(state);
			int ac = QLearner::getAction(network, state, v);
			if(rand() % 100 < 5)
				ac = rand() % game.getNumInputs();
			game.step(ac);
		}
		device->getVideoDriver()->beginScene();
		game.visualize(*device->getVideoDriver(), core::recti(10, 10, 300, 300));
		device->getVideoDriver()->endScene();
		device->sleep(20);
	}
//	learner.save("final");
}


