#include "q_learner.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <irrlicht/irrlicht.h>
#include <boost/lexical_cast.hpp>

#include "config.h"
#include "fc_layer.hpp"
#include "relu_layer.hpp"
#include "tanh_layer.hpp"
#include "solver.hpp"
#include "rmsprop.hpp"
#include "network.hpp"

using namespace irr;

const float BAT_SIZE = 0.05;

struct PongGame
{
	float ballx;
	float bally;
	float bvx;
	float bvy;
	float posy;

	void reset()
	{
		ballx = (rand() % 101) / 100.f;
		bally = (rand() % 101) / 100.f;
		bvx = 1;
		bvy = 0;//(rand() % 101 - 50) / 20.f;
		posy = 0.5;
	}

	float step( int ac )
	{
		ballx += 0.01*bvx;
		bally += 0.01*bvy;

		if( ac == 1 )
			posy += 0.025;
		else if(ac == 2)
			posy -= 0.025;

		if(bally > 1)
		{
			bally = 2 - bally;
			bvy *= -1;
		}

		if(bally < 0)
		{
			bally = -bally;
			bvy *= -1;
		}

		if( ballx > 1.0 )
		{
			if( std::abs(bally - posy) < BAT_SIZE )
				return 1;
			else
				return -1;
		}

		return 0;
	}

	Vector data() const
	{
		auto vec = Vector(3);
		vec << ballx, bally, posy;
		return vec;
	}
};

void build_image(const QLearner& l);

IrrlichtDevice* device;
video::ITexture* texture = nullptr;

void learn_thread( Network& target_net )
{
	QLearner learner( QLearnerConfig(3, 3, 1e6) );
	
	Network network;
	network << FcLayer(Matrix::Random(30, learner.getInputSize()));;
	network << TanhLayer(Matrix::Zero(30, 1));
	network << FcLayer(Matrix::Random(3, 30));
	network << TanhLayer(Matrix::Zero(3, 1));
	
	auto prop = std::unique_ptr<RMSProp>(new RMSProp(0.9, 0.0005, 0.0001));
	RMSProp* rmsprop = prop.get();
	Solver solver( std::move(prop) );
	
	std::fstream rewf("reward.txt", std::fstream::out);
	learner.setQNetwork( std::move(network) );
	//learner.load("pnet");

	PongGame game;
	game.reset();
	int ac = 2;
	int games = 0;

	learner.setCallback( [&](const QLearner& l ) 
	{
			std::cout << games << ": " << learner.getCurrentEpsilon() << "\n";
			std::cout << learner.getAverageEpisodeReward() << " (" << learner.getAverageQuality() << ", " << learner.getAverageError() << ")\n";
			rewf << learner.getAverageEpisodeReward() << " " << learner.getAverageQuality() << " " << learner.getAverageError() << "\n";
			rewf.flush();
			std::cout << learner.getNumberLearningSteps() << "\n";
			build_image(learner);
			std::cout << " - - - - - - - - - - \n";
			
			target_net = learner.getQNetwork().clone();
	} );

	while(true)
	{
		if(games < 2000)
		{
			rmsprop->setRate(0.001);
		} else if ( games < 10000 )
		{
			rmsprop->setRate(0.0001);
		} else
		{
			rmsprop->setRate(0.000001);
		}

		float r = game.step(ac);
		ac = learner.learn_step( game.data(), r, game.ballx > 1, solver );
		if( game.ballx > 1.0 )
		{
			game.reset();
			games++;
		}
	}
}

int main()
{
	Network network;
	
	std::thread learner( learn_thread, std::ref(network) );
	
	std::fstream rewf("reward.txt", std::fstream::out);

	PongGame game;
	game.reset();
	int games = 0;

	device = createDevice(video::EDT_SOFTWARE, core::dimension2du(800, 600));

	int step = 0;
	while(device->run())
	{
		step++;
		float v;
		int ac = QLearner::getAction(network, game.data(), v);
		game.step(ac);
		
		
		device->getVideoDriver()->beginScene();
		device->getVideoDriver()->draw2DPolygon( core::position2di(game.ballx * 400, game.bally * 400+100), 10);
		device->getVideoDriver()->draw2DLine(core::position2di(400, (game.posy-BAT_SIZE)*400+100), core::position2di(400, (game.posy+BAT_SIZE)*400+100));

		//device->getVideoDriver()->draw2DLine( core::position2di(500, 600), core::position2di(500, 200-200*q));
		device->getVideoDriver()->draw2DImage(texture, core::position2di(600, 0));
		//device->getVideoDriver()->draw2DLine( core::position2di(520, 600), core::position2di(520, 200-200*r));
		device->getVideoDriver()->endScene();
		device->sleep(10);

		if( game.ballx > 1.0 )
		{
			game.reset();
			games++;
			device->sleep(100);
		}
	}
//	learner.save("final");
}

void build_image(const QLearner& l)
{
	std::vector<unsigned char> grayscale(100*100*3);

	PongGame game;
	game.reset();
	game.bvx = 1;
	auto v = [](float r) -> unsigned { 
		return std::min(255u, unsigned((r+1)*127)); 
	};
	for(int y = 0; y < 100; ++y)
	{
		for(int x = 0; x < 100; ++x)
		{
			game.ballx = x / 100.0;
			game.bally = y / 100.0;
			Vector r = l.assess( game.data() );
			grayscale[(100*y+x)*3] = (r[0] > r[1] && r[0] > r[2]) ? v(r[0]) : 0;
			grayscale[(100*y+x)*3+1] = (r[1] > r[0] && r[1] > r[2]) ? v(r[1]) : 0;
			grayscale[(100*y+x)*3+2] = (r[2] > r[0] && r[2] > r[1]) ? v(r[2]) : 0;
		}
	}

	auto img = device->getVideoDriver()->createImageFromData(irr::video::ECF_R8G8B8, irr::core::dimension2du(100, 100), &grayscale[0], false, false);
	texture = device->getVideoDriver()->addTexture("image", img);
}


