#include "NeuralNetwork.h"
#include <iostream>
#include <cassert>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology)
{
	//topology's size is the number of layer
	for (int layer_index = 0; layer_index < topology.size(); ++layer_index)
	{
		//add a new layer
		this->layers.push_back(Layer());

		//populate the new layer with neurons
		//plus equal for an extra bias neuron

		int output_num = (layer_index + 1 == topology.size() ? 1 : topology[layer_index + 1]);
		for (int neuron_index = 0; neuron_index <= topology[layer_index]; ++neuron_index)
		{
			this->layers.back().push_back(Neuron(neuron_index, output_num));
		}
	}
}


void NeuralNetwork::FeedForward(const std::vector<double>& input)
{
	//-1 to account for the bias neuron
	assert(input.size() == this->layers.front().size() - 1);

	for (int neuron_index = 0; neuron_index < input.size(); ++neuron_index)
	{
		this->layers.front()[neuron_index].SetValue(input[neuron_index]);
	}

	//forward propagate
	for (int layer_index = 1; layer_index < this->layers.size(); ++layer_index)
	{
		const Layer& previous_layer = this->layers[layer_index - 1];

		//-1 to skip the bias neuron
		for (int neuron_index = 0; neuron_index < this->layers[layer_index].size() - 1; ++neuron_index)
		{
			this->layers[layer_index][neuron_index].FeedForward(previous_layer);
		}
	}
}