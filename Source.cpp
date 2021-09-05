#include "NeuralNetwork.h"
#include <vector>
#include <iostream>

int main()
{

	/*
	

	1000
	3
	2 4 1


	*/

	std::cin.tie(nullptr)->sync_with_stdio(false);
	srand(time(nullptr));

	int N, M; std::cin >> M >> N;
	std::vector<int> topology(N);
	for (auto& val : topology)
	{
		std::cin >> val;
	}

	NeuralNetwork model(topology);
	std::vector<double> input(topology.front()), target, predict;

	for (int i = 0; i < M; ++i)
	{
		std::cout << "Training: " << i << " / " << M << '\n';

		for (int j = 0; j < input.size(); ++j)
		{
			input[j] = j;
			std::cout << "input: " << input[j]<<"   ";
		}

		//clear the target very imporatn lol xd lmao
		target.assign(topology.back(), 0);
		for (int j = 0; j < input.size(); ++j)
		{
			target.front() = int(target.front()) ^ int(input[j]);
		}
		 std::cout << "target: " << target.front() << " ";



		model.ForwardPropagate(input);
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


		model.ForwardPropagate(input);
		predict = model.GetResult();
		std::cout << "predict:\n";
		for (auto val : predict)
		{
			std::cout << val << ' ';
		}

		std::cout << '\n';
	}
}