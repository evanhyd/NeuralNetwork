#include "Connection.h"
#include <random>
#include <ctime>


Connection::Connection()
{
	srand(time(nullptr));

	this->weight = rand() / double(RAND_MAX);
	this->diff_weight = rand() / double(RAND_MAX);
}
