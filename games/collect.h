#ifndef COLLECT_H_INCLUDED
#define COLLECT_H_INCLUDED

#include <vector>
#include "game.h"

struct Object 
{
	float x;
	float y;
	int type;
};

class Collect : public Game
{
public:
	int getNumInputs() const override;
	void getCurrentState( Vector& target ) const override;
	bool isFinished() const override;
	void restart() override;
	float step(int input) override;
	void visualize( irr::video::IVideoDriver& driver, irr::core::recti area  ) const override;
private:
	// game state
	float mPosX;
	float mPosY;
	float mAngle;
	
	std::vector<Object> mObjects;
};


#endif // COLLECT_H_INCLUDED
