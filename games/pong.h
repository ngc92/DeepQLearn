#pragma once

#include "game.h"

class Pong : public Game
{
public:
	int getNumInputs() const override;
	void getCurrentState( Vector& target ) const override;
	bool isFinished() const override;
	void restart() override;
	void step(int input) override;
	void visualize( IVideoDriver& driver ) const override;
private:
	// game state
	float mBallx;
	float mBally;
	float mBvx;
	float mBvy;
	float mPosy;
	
	// game config
	bool mHasVy;
};
