#include "Neuron.h"
#include <algorithm>

Neuron::Neuron(int new_neuron_index, int output_num) : neuron_index(new_neuron_index), value(0)
{
	this->output_weights.assign(output_num, Connection());
}

void Neuron::SetValue(double new_value)
{
	this->value = new_value;
}
double Neuron::GetValue() const
{
	return this->value;
}

void Neuron::FeedForward(const Layer& previous_layer)
{
	for (int neuron_index = 0; neuron_index < previous_layer.size(); ++neuron_index)
	{
		this->value += previous_layer[neuron_index].value * previous_layer[neuron_index].output_weights[this->neuron_index].weight;
	}

	this->value = TransferFunction(this->value);
}


double Neuron::TransferFunction(double value)
{
	//sigmoid 1.0 / (1.0 - std::exp(-value))

	return tanh(value);
}

double Neuron::TransferFunctionDerv(double value)
{
	double tanh_value = tanh(value);
	return 1 - tanh_value * tanh_value;
}