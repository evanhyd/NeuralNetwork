#include "NeuralNetwork.h"
#include <iostream>
#include <cassert>

NeuralNetwork::NeuralNetwork(const std::vector<int>& topology) : RMS(0.0)
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

	//set bias neuron's value to 1
	this->layers.back().back().SetValue(1.0);
}


void NeuralNetwork::FeedForward(const std::vector<double>& input)
{
	//-1 to account for the bias neuron
	assert(input.size() == this->layers.front().size() - 1);

	Layer& input_layer = this->layers.front();
	for (int neuron_index = 0; neuron_index < input.size(); ++neuron_index)
	{
		input_layer[neuron_index].SetValue(input[neuron_index]);
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


void NeuralNetwork::BackPropagate(const std::vector<double>& target)
{
	//calculate the mean of square loss
	Layer& output_layer = this->layers.back();
	assert(target.size() == output_layer.size() - 1);
	
	//square root of mean error
	this->RMS = 0.0;

	//- 1 to skip the bias neuron
	for (int neuron_index = 0; neuron_index < output_layer.size() - 1; ++neuron_index)
	{
		double loss = target[neuron_index] - output_layer[neuron_index].GetValue();
		this->RMS += loss * loss;
	}

	this->RMS /= double(output_layer.size() - 1);
	this->RMS = sqrt(RMS);

	std::cout << "RMS: " << this->RMS << '\n';

	//calculate the output layer gradient
	for (int neuron_index = 0; neuron_index < output_layer.size() - 1; ++neuron_index)
	{
		output_layer[neuron_index].UpdateOutputLayerGradient(target[neuron_index]);
	}

	//calculate the hidden layer gradient
	for (int layer_index = output_layer.size() - 2; layer_index > 0; --layer_index)
	{
		Layer& curr_layer = this->layers[layer_index];
		const Layer& next_layer = this->layers[layer_index + 1];

		for (Neuron& curr_neuron : curr_layer)
		{
			curr_neuron.UpdateHiddenLayerGradient(next_layer);
		}
	}


	//update the weight
	for (int layer_index = this->layers.size() - 1; layer_index > 0; --layer_index)
	{
		Layer& curr_layer = this->layers[layer_index];
		Layer& prev_layer = this->layers[layer_index - 1];

		for (int neuron_index = 0; neuron_index < curr_layer.size() - 1; ++neuron_index)
		{
			curr_layer[neuron_index].UpdateLayerWeight(prev_layer);
		}
	}
}


std::vector<double> NeuralNetwork::GetResult() const
{
	const Layer& output_layer = this->layers.back();
	std::vector<double> results(output_layer.size() - 1);

	//-1 to skip the bias neuron
	for (int neuron_index = 0; neuron_index < output_layer.size() - 1; ++neuron_index)
	{
		results[neuron_index] = output_layer[neuron_index].GetValue();
	}

	return results;
}