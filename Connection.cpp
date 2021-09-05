#include "Connection.h"
#include <random>
#include <ctime>


Connection::Connection()
{
	srand(time_t(NULL));

	this->weight = rand() / double(RAND_MAX);
	this->diff_weight = rand() / double(RAND_MAX);
}
