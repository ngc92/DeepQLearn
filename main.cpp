//#include "config.h"
//#include "fc_layer.hpp"
//#include "relu_layer.hpp"
//#include "solver.hpp"
//#include "rmsprop.hpp"
//#include "network.hpp"
//#include <iostream>
//
//int main()
//{
//	Network network;
//    network << FcLayer(Matrix::Random(5, 5));;
//	network << ReLULayer(Matrix::Zero(5, 1));
//	network << FcLayer(Matrix::Random(5, 5));
//
//	Solver solver(std::unique_ptr<RMSProp>(new RMSProp(0.9, 1.0, 1.0)));
//	Vector in = Vector::Random(5);
//    std::cout << in << "\n\n";
//    std::cout << network(in).output() << "\n\n";
//    for(int i = 0; i < 1000; ++i)
//	{
//		auto proc = network(in);
//		proc.backpropagate(proc.output(), solver);
//		network.update( solver );
//	}
//    std::cout << network(in).output() << "\n\n";
//}
