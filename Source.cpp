#include "NeuralNetwork.h"
#include <vector>
#include <iostream>
#include <fstream>

int main()
{
	std::cin.tie(nullptr)->sync_with_stdio(false);

	std::ifstream input_file("training.txt");
	if (!input_file.is_open())
	{
		std::cerr << "Couldn't find the training data\n";
		return EXIT_FAILURE;
	}

	int layer_num; input_file >> layer_num;
	std::vector<int> topology(layer_num);
	for (auto& val : topology)
	{
		input_file >> val;
	}

	NeuralNetwork model(topology);
	std::vector<double> features(topology.front()), labeled_examples(topology.back()), predict;


	int test_case = 0;
	while (!input_file.eof())
	{
		std::cout <<"Test case: " << ++test_case << '\n';
		for (auto& feature : features)
		{
			input_file >> feature;
			std::cout << "f: " << feature << "   ";
		}

		for (auto& label : labeled_examples)
		{
			input_file >> label;
		}
		std::cout << "l: " << labeled_examples.front() << " ";


		model.ForwardPropagate(features);
		model.BackPropagate(labeled_examples);
	}
	std::cout << "Training completed!\n";

	while (true)
	{
		std::cout << "Input data:\n";
		for (auto& i : features)
		{
			std::cin >> i;
		}


		model.ForwardPropagate(features);
		predict = model.GetResult();
		std::cout << "predict:\n";
		for (auto val : predict)
		{
			std::cout << val << ' ';
		}

		std::cout << '\n';
	}
}