#pragma once
#include "pointFeature.h"
using namespace cv;
static std::mutex mu;
namespace simplesearch {
	struct distancepair {
		// determine the starting point of each thread
		size_t starting_i{ 0 };
		size_t starting_j{ 1 };
		size_t numberofpoints{ 0 };// total number of points
		size_t current_i{ 0 };
		size_t current_j{ 1 };
		void operator()(size_t i, size_t j) {
			starting_i = i;
			starting_j = j;

			if ((starting_j - 1) < i) {
				std::cout << "error in assigning staring points! start i:" << starting_i << ", start j: " << starting_j << "\n";
			}
			else {
				Reset();
			}

		};

		void operator = (const distancepair& T) {
			if (this != &T) {
				starting_i = T.starting_i;
				starting_j = T.starting_j;
				current_i = T.current_i;
				current_j = T.current_j;
				numberofpoints = T.numberofpoints;
			}
		}

		void Advance() {
			if ((current_j - 1) == current_i) {// means the current index has reach the top of the stack
				if (current_j == (numberofpoints - 1)) {// means the current the index has reach the last possible one
					std::cout << "error !, the index has reach the last one, can not advance further more!" << "\n";
				}
				else {
					current_j++;
					current_i = 0;
				}
			}
			else {
				current_i++;
			}
		}
		void Reset() {
			current_i = starting_i;
			current_j = starting_j;
		}
		void Display() {
			std::cout << "current i:" << current_i << ", current j: " << current_j << "\n";
		}
	};
  void UpdatePointFeature(const std::vector<Vec3f>& pointlist, std::vector< pointfeature::PointEulerDistanceFeature>& featurelist);
  class simpleSearch
  {
  public:
	simpleSearch(bool isMultiThread, std::vector<Vec3f>& pointlist);
	distancepair GetThreadEndPoint(size_t threadpointnumber, const distancepair& starting_pair);

	void Parallel_UpdateFeature2(const std::vector<Vec3f>& pointlist, std::vector<pointfeature::PointEulerDistanceFeature>& featurelist,
		distancepair pair, // staring index pair
		size_t numberofdistance);
	~simpleSearch();
  private:
  };
}



