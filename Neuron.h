#pragma once
#include <vector>
#include "Connection.h"

class Neuron
{
	using Layer = std::vector<Neuron>;

private:
	int neuron_index;
	double value;
	std::vector<Connection> output_weights;

public:
	Neuron(int neuron_index, int output_num);
	void SetValue(double new_value);
	double GetValue() const;
	void FeedForward(const Layer& previous_layer);

private:
	static double TransferFunction(double value);
	static double TransferFunctionDerv(double value);
};

