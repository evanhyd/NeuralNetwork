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
	void FeedForward(const std::vector<double>& input);
	void BackPropagate(const std::vector<double>& target);
	std::vector<double> GetResult() const;

};

