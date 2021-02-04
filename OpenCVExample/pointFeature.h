#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
/*
A feature associated with a point is defined.
The key is the distance table 'Feature' containing 4 distance values.
The UpdatePointFeature function inputs 'Feature' with the values corresponds to the 4 closest point of a selected point.
For a special point, the mean value of 'Feature' is lower than 32. The grid stride is determined as 37.6 pixels.
*/

enum vectorindex
{
	VECTOR_X = 0,
	VECTOR_Y,
};

namespace pointfeature {
	struct PointEulerDistanceFeature {
		float Maxdistance{ 0.0 };  // the maximum distance
		float Mindistance{ 0.0 }; // the mimum distance
		size_t MaxDistanceIndex{ 0 }; // the index of the feature
		size_t MinDistanceIndex{ 0 }; // the index of the min distance
		cv::Vec4f Feature;             // the distance to the neighbour points
		cv::Vec4i Indexlist;           // the index in the point list of the neighbour points

		void operator()(float distance, size_t index) {
			Feature[MaxDistanceIndex] = distance;// record the euler distance
			Indexlist[MaxDistanceIndex] = index;// record the index of the point in the point list
			UpdateMaxDistance();
			UpdateMinDistance();
		}


		void UpdateMaxDistance() { // update the max distance if 'Feature' is modified
			Maxdistance = Feature[0];
			MaxDistanceIndex = 0;
			for (size_t i = 1; i < 4; i++) {
				if (Feature[i] >= Maxdistance) {
					MaxDistanceIndex = i;
					Maxdistance = Feature[i];
				}
			}
		}

		void UpdateMinDistance() {
			Mindistance = Feature[0];
			for (size_t i = 1; i < 4; i++) {
				if (Feature[i] <= Mindistance) {
					MinDistanceIndex = i;
					Mindistance = Feature[i];
				}
			}
		}

		void Reset(float initialvalue) {
			Maxdistance = initialvalue;
			for (size_t i = 0; i < 4; i++) {
				Feature[i] = initialvalue;
				Indexlist[i] = 0;
			}
			UpdateMaxDistance();
			UpdateMinDistance();
		}

	};
}