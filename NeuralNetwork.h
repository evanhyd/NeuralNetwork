#pragma once
#include <vector>
#include "Neuron.h"

class NeuralNetwork
{
	using Layer = std::vector<Neuron>;

private:
	std::vector<Layer> layers;
	double RMS; //root mean square error

public:
	NeuralNetwork(const std::vector<int>& topology);
	void ForwardPropagate(const std::vector<double>& features);
	void BackPropagate(const std::vector<double>& labeled_examples);
	std::vector<double> GetResult() const;

};

