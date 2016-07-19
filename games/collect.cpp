#include "collect.h"
#include <irrlicht/irrlicht.h>
#include <iostream>

const float RADIUS = 0.04;
const float SENSOR_ANGLE_DIFF = 0.3;
const float ROTATION_SPEED = 0.07;
const int NUM_EYES_DIR = 4;

// helper functions
float hitTest( float sx, float sy, float a, const Object& o );
float hitTest( float sx, float sy, float a, const std::vector<Object>& obs, int filter );




/** @brief getNumInputs  */
int Collect::getNumInputs() const
{
	return 5;
}

/** @brief getCurrentState  */
void Collect::getCurrentState(Vector& target) const
{
	const int size = (2*NUM_EYES_DIR+1) * 2;
	if(target.size() < size)
		target.resize(size);
	
	int i = 0;
	for(int d = -NUM_EYES_DIR; d <= NUM_EYES_DIR; ++d)
	{
		float da = d * SENSOR_ANGLE_DIFF;
		float l = hitTest(mPosX, mPosY, mAngle + da, mObjects, 0);
		target[i++] = l;
	}
	
	for(int d = -NUM_EYES_DIR; d <= NUM_EYES_DIR; ++d)
	{
		float da = d * SENSOR_ANGLE_DIFF;
		float l = hitTest(mPosX, mPosY, mAngle + da, mObjects, 1);
		target[i++] = l;
	}
}

/** @brief isFinished  */
bool Collect::isFinished() const
{
	return false;
}

/** @brief restart  */
void Collect::restart()
{
	mAngle = 0;
	mPosX = rand() % 100 / 100.f;
	mPosY = rand() % 100 / 100.f;
	
	for(int i = 0; i < 10; ++i)
	{
		Object ob;
		ob.x = rand() % 100 / 100.f;
		ob.y = rand() % 100 / 100.f;
		ob.type = i % 2;
		mObjects.push_back( ob );
	}
}

/** @brief step  */
float Collect::step(int input)
{
	switch(input)
	{
	case 1:
		mAngle += ROTATION_SPEED;
	case 2:
		mAngle += 2*ROTATION_SPEED;
		break;
	case 3:
		mAngle -= ROTATION_SPEED;
	case 4:
		mAngle -= 2*ROTATION_SPEED;
		break;
	}
	
	float vx = std::cos(mAngle);
	float vy = std::sin(mAngle);
	mPosX += 0.01 * vx;
	mPosY += 0.01 * vy;
	
	if(mPosX > 1) mPosX -= 1;
	if(mPosY > 1) mPosY -= 1;
	if(mPosX < 0) mPosX += 1;
	if(mPosY < 0) mPosY += 1;
	
	float score = 0;
	
	for(auto& ob : mObjects)
	{
		float dx = ob.x - mPosX;
		float dy = ob.y - mPosY;
		if( dx*dx + dy * dy < 4 * RADIUS * RADIUS )
		{
			score += ob.type == 0 ? 1 : -1;
			ob.x = rand() % 100 / 100.f;
			ob.y = rand() % 100 / 100.f;
		}
	}
	
	return score;
}

/** @brief visualize  */
void Collect::visualize(irr::video::IVideoDriver& driver , irr::core::recti area  ) const
{
	using namespace irr;
	float w = area.getWidth();
	float h = area.getHeight();
	float r = RADIUS * w;
	driver.draw2DRectangleOutline( area );
	driver.draw2DPolygon( core::vector2di(mPosX * w, mPosY * h) + area.UpperLeftCorner, r );
	for(const auto& ob : mObjects)
	{
		video::SColor col = ob.type == 1 ? video::SColor(255, 180, 0, 0) : video::SColor(255, 0, 180, 0);
		driver.draw2DPolygon( core::vector2di(ob.x * w, ob.y * h) + area.UpperLeftCorner, r, col );
	}
	
	for(int d = -NUM_EYES_DIR; d <= NUM_EYES_DIR; ++d)
	{
		float da = d * SENSOR_ANGLE_DIFF;
		auto start = core::vector2di(mPosX * w, mPosY * h) + area.UpperLeftCorner;
		float l = hitTest(mPosX, mPosY, mAngle + da, mObjects, -1);
		driver.draw2DLine( start, start + core::vector2di(w * std::cos(mAngle + da) * l, w * std::sin(mAngle + da) * l)  );
	}
}


// ----------------------------------------------------------------------------------
float hitTest( float sx, float sy, float a, const Object& o )
{
	float dx = sx - o.x;
	float dy = sy - o.y;
	float lx = std::cos(a);
	float ly = std::sin(a);
	
	float ld = lx*dx + ly * dy;
	float d = dx*dx+dy*dy - RADIUS*RADIUS;
	float D = ld*ld - d;
	if( D < 0)		return 1e12;
	
	float i1 = -ld - std::sqrt(D);
	float i2 = -ld + std::sqrt(D);
	if( i1 > 0 )	return i1;
	if( i2 > 0 )	return i2;
	
	return 1e12;
}

float hitTest( float sx, float sy, float a, const std::vector<Object>& obs, int filter )
{
	float hit = 0.3;
	for(auto& ob : obs )
	{
		if(filter != -1 && ob.type != filter) continue;
		float h = hitTest(sx, sy, a, ob);
		if(h < hit)
			hit = h;
	}
	return hit;
}

