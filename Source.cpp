#include "NeuralNetwork.h"
#include <vector>
#include <iostream>

int main()
{

	/*
	

	5000
	5
	1 100 100 100 1


	*/

	std::cin.tie(nullptr)->sync_with_stdio(false);
	srand(time_t(NULL));

	int N, M; std::cin >> M >> N;
	std::vector<int> topology(N);
	for (auto& val : topology)
	{
		std::cin >> val;
	}

	NeuralNetwork model(topology);
	std::vector<double> input(topology.front()), target(topology.back()), predict;

	for (int i = 0; i < M; ++i)
	{
		std::cout << "Training: " << i << " / " << M << '\n';

		for (int j = 0; j < input.size(); ++j)
		{
			input[j] = double(rand() % 360);
			std::cout << "input: " << input[j]<<"   ";
		}
		for (int j = 0; j < input.size(); ++j)
		{
			target.front() = sin(input[j] / 180.0 * 3.1415926535);
		}
		std::cout << "target: " << target.front() << " ";

		model.FeedForward(input);
		predict = model.GetResult();
		std::cout << "predict: ";
		for (auto val : predict)
		{
			std::cout << val << ' ';
		}
		model.BackPropagate(target);
	}
	std::cout << "Training completed!\n";

	while (true)
	{
		std::cout << "Input data:\n";
		for (auto& i : input)
		{
			std::cin >> i;
		}


		model.FeedForward(input);
		predict = model.GetResult();
		std::cout << "predict:\n";
		for (auto val : predict)
		{
			std::cout << val << ' ';
		}

		std::cout << '\n';
	}
}