#pragma once

#include "config.h"
#include <irrlicht/rect.h>

namespace irr
{
	namespace video
	{
		class IVideoDriver;
	}
}

class Game
{
public:
	virtual int getNumInputs() const = 0;
	virtual void getCurrentState( Vector& target ) const = 0;
	virtual bool isFinished() const = 0;
	virtual void restart() = 0;
	virtual float step(int input) = 0;
	virtual void visualize( irr::video::IVideoDriver& driver, irr::core::recti area ) const = 0;
};
