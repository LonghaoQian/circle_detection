#include "simpleSearch.h"

namespace simplesearch {

	simpleSearch::simpleSearch(bool isMultiThread, std::vector<Vec3f>& pointlist)
	{

	}
	distancepair simpleSearch::GetThreadEndPoint(size_t threadpointnumber, const distancepair& starting_pair) {
		distancepair end_pair = starting_pair;
		size_t numberpointscurrentstack = starting_pair.starting_j - starting_pair.current_i;
		size_t remainingpoints = threadpointnumber - numberpointscurrentstack;
		if (remainingpoints > 0) {// if greater than zero, means that the end point is in an other stack
			// calculate in which stack
			double t = static_cast<double>(end_pair.starting_j) + 1.0;
			double a = static_cast<double>(remainingpoints);
			size_t c1 = static_cast<size_t>(floor(0.5 - t + sqrt(0.25*(2.0*t - 1)*(2.0*t - 1) + 2.0*a)));
			size_t c2 = c1 + 1;
			// verify which stack number is correct
			size_t b2 = (2 * end_pair.starting_j + c2 + 1)*c2 / 2 - remainingpoints;

			if (((2 * end_pair.starting_j + c1 + 1)*c1 / 2) == remainingpoints) {
				end_pair(c1 + starting_pair.starting_j - 1, c1 + starting_pair.starting_j);
			}
			else {
				end_pair(c2 + starting_pair.starting_j - 1 - b2, c2 + starting_pair.starting_j);
			}

		}
		else {
			// if not in the same stack

			for (int i = 0; i < threadpointnumber; i++) {
				end_pair.Advance();
			}
		}
		return end_pair;
	}
	simpleSearch::~simpleSearch()
	{
	}
	void UpdatePointFeature(const std::vector<Vec3f>& pointlist, std::vector< pointfeature::PointEulerDistanceFeature>& featurelist) {
		// find the distance to the 4 closet points
		float EulerDistance = 0.0;
		distancepair pair;
		pair(0, 1);
		size_t totaldistancenumber = (featurelist.size() - 1)*featurelist.size() / 2;

		for (size_t i = 0; i < totaldistancenumber; i++) {
			EulerDistance = sqrt((pointlist[pair.current_i][VECTOR_X] - pointlist[pair.current_j][VECTOR_X]) * (pointlist[pair.current_i][VECTOR_X] - pointlist[pair.current_j][VECTOR_X])
				+ (pointlist[pair.current_i][VECTOR_Y] - pointlist[pair.current_j][VECTOR_Y]) * (pointlist[pair.current_i][VECTOR_Y] - pointlist[pair.current_j][VECTOR_Y]));
			if (featurelist[pair.current_i].Maxdistance >= EulerDistance) {
				featurelist[pair.current_i](EulerDistance, pair.current_j);
			}
			if (featurelist[pair.current_j].Maxdistance >= EulerDistance) {
				featurelist[pair.current_j](EulerDistance, pair.current_i);
			}
			pair.Advance();
		}
	}

	void Parallel_UpdateFeature2(const std::vector<Vec3f>& pointlist, std::vector<pointfeature::PointEulerDistanceFeature>& featurelist,
		distancepair pair, // staring index pair
		size_t numberofdistance) {
		float EulerDistance = 0.0;
		for (size_t i = 0; i < numberofdistance; i++) {
			EulerDistance = sqrt((pointlist[pair.current_i][VECTOR_X] - pointlist[pair.current_j][VECTOR_X]) * (pointlist[pair.current_i][VECTOR_X] - pointlist[pair.current_j][VECTOR_X])
				+ (pointlist[pair.current_i][VECTOR_Y] - pointlist[pair.current_j][VECTOR_Y]) * (pointlist[pair.current_i][VECTOR_Y] - pointlist[pair.current_j][VECTOR_Y]));
			if (featurelist[pair.current_i].Maxdistance >= EulerDistance) {
				mu.lock();
				featurelist[pair.current_i](EulerDistance, pair.current_j);
				mu.unlock();
			}
			if (featurelist[pair.current_j].Maxdistance >= EulerDistance) {
				mu.lock();
				featurelist[pair.current_j](EulerDistance, pair.current_i);
				mu.unlock();
			}
			pair.Advance();
		}
	}


}
