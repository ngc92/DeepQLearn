#include "q_learner.hpp"
#include <iostream>
#include <fstream>
#include <irrlicht/irrlicht.h>
#include <boost/lexical_cast.hpp>

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
		ballx = 0.6;
		bally = (rand() % 101) / 100.f;
		bvx = 1;
		bvy = (rand() % 101 - 50) / 20.f;
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

	std::vector<float> data()
	{
		return std::vector<float>{ballx, bally, posy};
	}
};

void build_image(QLearner& l);

IrrlichtDevice* device;
video::ITexture* texture = nullptr;

int main()
{
	std::fstream rewf("reward.txt", std::fstream::out);
	QLearner learner(3, 3, 1e6, 1);
	learner.setEpsilonSteps(1e6);
	learner.setQNetwork( std::make_shared<Network<float>>(std::vector<LayerInfo>{
														LayerInfo("input", learner.getInputSize()), 
														LayerInfo("fc", 30), 
														LayerInfo("nl tanh", 30),
														LayerInfo("fc", 3)
														}) );
	//learner.load("pnet");
	learner.setLearningRate(2.5e-4f);
	learner.setMiniBatchSize(32);
	learner.setNetUpdateRate(10000);

	PongGame game;
	game.reset();
	int ac = 2;
	int i = 0;
	int games = 0;
	float best = -1;
	bool drawing = true;
	int last = 0;
	int learn_anneal = 0;

/*	learner.setCallback( [&](const QLearner& l ) {
			std::cout << games << "\n";
			std::cout << learner.getCurrentEpsilon() << " - " << learner.getAverageEpisodeReward() << "\n";
			rewf << learner.getAverageEpisodeReward() << " " << learner.getAverageQuality() << " " << learner.getAverageError() << "\n";
			rewf.flush();
			std::cout << learner.getNumberLearningSteps() << "\n";
			build_image(learner);
			std::cout << " - - - - - - - - - - \n";
			if( learner.getAverageEpisodeReward() > best  )
			{
				build_image(learner);
				best = learner.getAverageEpisodeReward();
				learner.save("pnet");
				last = games / 1000;
			}
			if(learn_anneal < 100)
			{
				learn_anneal++;
			} 
			else if(learn_anneal == 100)
			{
				learner.setLearningRate(1e-5f);
			}
			else if( learn_anneal == 101 && learner.getAverageEpisodeReward() > 0.0 )
			{
				learner.setLearningRate(1e-6f);
				learn_anneal++;
			} else if( learn_anneal == 102 && learner.getAverageEpisodeReward() > 0.5 )
			{
				learner.setLearningRate(1e-7f);
				learn_anneal++;
			}
	} );*/
	
	device = createDevice(video::EDT_SOFTWARE, core::dimension2du(800, 600));
	
	int step = 0;
	while(device->run())
	{
		step++;
		float r = game.step(ac);
		ac = learner.learn_step( game.data(), r, r != 0 );
		step++;
		if(drawing && games % 10000 < 5 )
		{
			device->getVideoDriver()->beginScene();
			device->getVideoDriver()->draw2DPolygon( core::position2di(game.ballx * 400, game.bally * 400+100), 10);
			device->getVideoDriver()->draw2DLine(core::position2di(400, (game.posy-BAT_SIZE)*400+100), core::position2di(400, (game.posy+BAT_SIZE)*400+100));
			
			float q = learner.getCurrentQuality();
			float r = learner.getAverageEpisodeReward();
			device->getVideoDriver()->draw2DLine( core::position2di(500, 600), core::position2di(500, 200-200*q));
			device->getVideoDriver()->draw2DImage(texture, core::position2di(600, 0));
			device->getVideoDriver()->draw2DLine( core::position2di(520, 600), core::position2di(520, 200-200*r));
			device->getVideoDriver()->endScene();
			device->sleep(10);
		} else 
		{
			if(step % 1000 == 0)
			{
				device->getVideoDriver()->beginScene();
				float q = learner.getCurrentQuality();
				float r = learner.getAverageEpisodeReward();
				device->getVideoDriver()->draw2DLine( core::position2di(500, 600), core::position2di(500, 300-300*q));
				device->getVideoDriver()->draw2DLine( core::position2di(520, 600), core::position2di(520, 300-300*r));
				device->getVideoDriver()->draw2DImage(texture, core::position2di(600, 0));
				device->getVideoDriver()->endScene();
			}
		}
		
		
		
		if( r != 0)
		{
			game.reset();
			games++;
		}

		i++;
	}
//	learner.save("final");
}

void build_image(QLearner& l)
{
	std::vector<unsigned char> grayscale(100*100*3);
	
	PongGame game;
	game.reset();
	game.bvx = 1;
	for(int y = 0; y < 100; ++y)
	{
		for(int x = 0; x < 100; ++x)
		{
			game.ballx = x / 100.0;
			game.bally = y / 100.0;
			float* r = l.assess( game.data() );
			grayscale[(100*y+x)*3] = (r[0] > r[1] && r[0] > r[2]) ? unsigned((r[0]+1)*127) : 0;
			grayscale[(100*y+x)*3+1] = (r[1] > r[0] && r[1] > r[2]) ? unsigned((r[1]+1)*127) : 0;
			grayscale[(100*y+x)*3+2] = (r[2] > r[0] && r[2] > r[1]) ? unsigned((r[2]+1)*127) : 0;
		}
	}
	
	auto img = device->getVideoDriver()->createImageFromData(irr::video::ECF_R8G8B8, irr::core::dimension2du(100, 100), &grayscale[0], false, false);
	texture = device->getVideoDriver()->addTexture("image", img);
}


