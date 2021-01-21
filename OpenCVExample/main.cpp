/*
The image.png is made up of a regular points grid with three special points.
Please write a C/C++ function that finds all points in the calibration image.

Your solution will be evaluated on efficiency, coding style, readability, and correctness.
In addition to correct results and time taken, optimal code is a significant factor in determining your grade.

Assume that this is for the PC platform.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#define THREAD 1
using namespace cv;

static std::mutex mu;

/* the Opencv Point type is used instead of this simple struct 
struct Point
{
	int x;
	int y;
};
*/
// name of index with directions to avoid errors
enum vectorindex
{
	VECTOR_X = 0,
	VECTOR_Y,
};

/*
A feature associated with a point is defined.
The key is the distance table 'Feature' containing 4 distance values.
The UpdatePointFeature function inputs 'Feature' with the values corresponds to the 4 closest point of a selected point.
For a special point, the mean value of 'Feature' is lower than 32. The grid stride is determined as 37.6 pixels.
*/
struct PointEulerDistanceFeature {// TO DO: use operator() to replace all the functions
	float Maxdistance{ 0.0 };  // the maximum distance
	float Mindistance{ 0.0 }; // the mimum distance
	size_t MaxDistanceIndex{ 0 }; // the index of the feature
	size_t MinDistanceIndex{ 0 }; // the index of the min distance
	Vec4f Feature;             // the distance to the neighbour points
	Vec4i Indexlist;           // the index in the point list of the neighbour points
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

};
// draw a rectangular box to a selected point
void HighLightAPoint(const Vec3f& pointinfo,  Mat& src) {
	Point topleft = Point(pointinfo[VECTOR_X] + 2.0*pointinfo[2] / sqrt(2), pointinfo[VECTOR_Y] + 2.0*pointinfo[2] / sqrt(2));
	Point bottomright = Point(pointinfo[VECTOR_X] - 2.0*pointinfo[2] / sqrt(2), pointinfo[VECTOR_Y] - 2.0*pointinfo[2] / sqrt(2));
	rectangle(src, topleft, bottomright, (0, 255, 0), 2);
}
// Update the feature for a selected point
#ifdef THREAD
void UpdatePointFeature(const std::vector<Vec3f>& pointlist, std::vector<PointEulerDistanceFeature>& featurelist, size_t index) {
	// find the distance to the 4 closet points
	float EulerDistance = 0.0;
	for (size_t i = 0; i < featurelist.size(); i++) {
		// distance of 
		if (i != index) {
			// only calculate this for points other than the index point
			// first, calculate the euler distance from the index point to all other points
			EulerDistance = sqrt((pointlist[i][VECTOR_X] - pointlist[index][VECTOR_X]) * (pointlist[i][VECTOR_X] - pointlist[index][VECTOR_X])
				+ (pointlist[i][VECTOR_Y] - pointlist[index][VECTOR_Y]) * (pointlist[i][VECTOR_Y] - pointlist[index][VECTOR_Y]));
			// second, check whether the distance is lower than the maximum value of the recorded distance table
			// if so, replace the maximum feature with this euler distance and update the feature info
			mu.lock();
			if (featurelist[index].Maxdistance >= EulerDistance) {
				featurelist[index].Feature[featurelist[index].MaxDistanceIndex] = EulerDistance;// record the euler distance
				featurelist[index].Indexlist[featurelist[index].MaxDistanceIndex] = i;// record the index of the point in the point list
				featurelist[index].UpdateMaxDistance();
				featurelist[index].UpdateMinDistance();
			}
			mu.unlock();
		}
	}
}

void Parallel_UpdateFeature(const std::vector<Vec3f>& pointlist, std::vector<PointEulerDistanceFeature>& featurelist, size_t startpoint, size_t endpoint) {
	for (size_t j = startpoint; j <= endpoint; j ++) {
		float EulerDistance = 0.0;
		for (size_t i = 0; i < featurelist.size(); i++) {
			// distance of 
			if (i != j) {
				// only calculate this for points other than the index point
				// first, calculate the euler distance from the index point to all other points
				EulerDistance = sqrt((pointlist[i][VECTOR_X] - pointlist[j][VECTOR_X]) * (pointlist[i][VECTOR_X] - pointlist[j][VECTOR_X])
					+ (pointlist[i][VECTOR_Y] - pointlist[j][VECTOR_Y]) * (pointlist[i][VECTOR_Y] - pointlist[j][VECTOR_Y]));
				// second, check whether the distance is lower than the maximum value of the recorded distance table
				// if so, replace the maximum feature with this euler distance and update the feature info
				if (featurelist[j].Maxdistance >= EulerDistance) {
					featurelist[j].Feature[featurelist[j].MaxDistanceIndex] = EulerDistance;// record the euler distance
					featurelist[j].Indexlist[featurelist[j].MaxDistanceIndex] = i;// record the index of the point in the point list
					featurelist[j].UpdateMaxDistance();
					featurelist[j].UpdateMinDistance();
				}
			}
		}
	}
}
#elif
void UpdatePointFeature(const std::vector<Vec3f>& pointlist, std::vector<PointEulerDistanceFeature>& featurelist, size_t index) {
	// find the distance to the 4 closet points
	float EulerDistance = 0.0;
	for (size_t i = 0; i < featurelist.size(); i++) {
		// distance of 
		if (i != index) {
			// only calculate this for points other than the index point
			// first, calculate the euler distance from the index point to all other points
			EulerDistance = sqrt((pointlist[i][VECTOR_X] - pointlist[index][VECTOR_X]) * (pointlist[i][VECTOR_X] - pointlist[index][VECTOR_X])
				+ (pointlist[i][VECTOR_Y] - pointlist[index][VECTOR_Y]) * (pointlist[i][VECTOR_Y] - pointlist[index][VECTOR_Y]));
			// second, check whether the distance is lower than the maximum value of the recorded distance table
			// if so, replace the maximum feature with this euler distance and update the feature info
			if (featurelist[index].Maxdistance >= EulerDistance) {
				featurelist[index].Feature[featurelist[index].MaxDistanceIndex] = EulerDistance;// record the euler distance
				featurelist[index].Indexlist[featurelist[index].MaxDistanceIndex] = i;// record the index of the point in the point list
				featurelist[index].UpdateMaxDistance();
				featurelist[index].UpdateMinDistance();
			}
		}
	}
}
#endif // ASYNC
// display the feature of a selected point
void DisplayPointFeatureInfo(const std::vector<PointEulerDistanceFeature>& featurelist, size_t index) {
	std::cout << "-------The feature of point #" << index << " ---------- \n";
	std::cout << "Maximum distance: " << featurelist[index].Maxdistance << "\n";
	std::cout << "Maximum index: " << featurelist[index].MaxDistanceIndex<< "\n";
	for (size_t i = 0; i < 4; i++) {
		std::cout << "Feature #: "<< i << " is: " << featurelist[index].Feature[i] << "\n";
		std::cout << "Point Index in the list is: " << featurelist[index].Indexlist[i] << "\n";
	}
	std::cout << "------------End of feature-------------- \n";
}

void find_points(std::vector<Point> &regular_points, 
				 std::vector<Point> &special_points, 
				 Mat& src)
{
	// preprocess image
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	medianBlur(gray, gray, 9);
	std::vector<Vec3f> circles;
	// use Hough transformation to detect circles
	// I set the hough transformation parameter by hand, I have no idea of an easy way to make this adaptive to any image input.
	// The adaptive hough requires a significant amount of work. 
	HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
		gray.rows / 48,  // change this value to detect circles with different distances to each other
		150, 15, 1, 30 // change the last two parameters
	// (min_radius & max_radius) to detect larger circles
	);
	// define and initialize the feature list
	std::vector<PointEulerDistanceFeature> Featurelist;
	Featurelist.reserve(circles.size());
	PointEulerDistanceFeature initialfeature;
	initialfeature.Maxdistance = 1000.0;
	for (size_t i = 0; i < 4; i++) {
		initialfeature.Feature[i] = 1000.0;
		initialfeature.Indexlist[i] = 0;
	}
	initialfeature.UpdateMaxDistance();
	initialfeature.UpdateMinDistance();
	// show all the points on the image
	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		Point center = Point(c[0], c[1]);
		// circle center
		circle(src, center, 1, Scalar(0, 100, 100), 2, LINE_AA);
		// circle outline
		int radius = c[2];
		circle(src, center, radius, Scalar(255, 0, 255), 2, LINE_AA);
		Featurelist.push_back(initialfeature);
	}
	std::cout << "There are " << circles.size() << " points \n";
	// determine the special points
	// step 1 calculate the mean of the minimum distance of all the points

	float MeanMin = 0.0;
#ifdef THREAD

	unsigned long const MinPointsPerThread = 30; // allocate number of points to be calculated by 1 thread
	unsigned long const MaxRequiredThread = ((unsigned long)circles.size() + MinPointsPerThread - 1) / MinPointsPerThread;
	unsigned long const HardwareThreads = std::thread::hardware_concurrency();
	unsigned long const NumThreads = std::min(HardwareThreads != 0 ? HardwareThreads : 2, MaxRequiredThread);
	
	std::cout << "minmum points per thread is: " << MinPointsPerThread << "\n";
	std::wcout << "numeber of required threads is: " << MaxRequiredThread << "\n";
	std::cout << "Hardware Threads are: " << HardwareThreads << "\n";
	std::cout << "Num Threads Threads is: " << NumThreads << "\n";
	size_t NumberPointsPerThread = (NumThreads == MaxRequiredThread ? (size_t)MinPointsPerThread : (circles.size()+ (size_t)NumThreads- 1)/(size_t)NumThreads );
	std::cout << "Num of Points of Threads Threads is: " << NumberPointsPerThread << "\n";
	std::vector<std::thread> disthread;
	disthread.reserve(NumThreads - 1);
	for (size_t i = 0; i < NumThreads - 1; i++) {
		disthread.emplace_back(Parallel_UpdateFeature, std::ref(circles), std::ref(Featurelist), i*NumberPointsPerThread, (i + 1)*NumberPointsPerThread - 1);
	}	
	
	// calculate the last block
	Parallel_UpdateFeature(circles, Featurelist, (NumThreads-1)*NumberPointsPerThread, circles.size()-1);
	// join all threads
	std::for_each(disthread.begin(), disthread.end(),std::mem_fn(&std::thread::join));
	for (const auto& feature : Featurelist) {// access by const reference 
		MeanMin += feature.Mindistance;
	}
#elif
	for (size_t i = 0; i < circles.size(); i++) {
		UpdatePointFeature(circles, Featurelist, i);
		MeanMin += Featurelist[i].Mindistance;
	}
#endif // ASYNC

	
	MeanMin /= (float)(circles.size());
	// step 2, assume that the number of special points are much less than regular points, all 
	// points with minmum distance less than the total mean is the special point
	special_points.clear();
	regular_points.reserve(circles.size());
	std::vector<size_t> specialpointindex;
	for (size_t i = 0; i < circles.size(); i++) {
		Vec3i location_v = circles[i];
		Point location_p = Point(location_v[0], location_v[1]);// convert it to points type
		// step 3: the special points should have both the maximum distance and the minimum distance at around 
		// 0.707 (sqrt(2)/2) of grid distance. Therefore, the we set 0.15 as the theshold to determine the specail points, i.e the ration : 0.7+0.1 = 0.85
		if ((Featurelist[i].Mindistance < MeanMin*0.85)&&(Featurelist[i].Maxdistance < MeanMin*0.85)) {
			special_points.push_back(location_p);
			specialpointindex.push_back(i);
		}
		else {
			// all the remaining points are regular points
			regular_points.push_back(location_p);
		}
	}

	// display detection result:
	std::cout << "total mean dist is: " << MeanMin << "\n";
	std::cout << "There are " << special_points.size() << " special points \n";
	std::cout << "There are " << regular_points.size() << " regular points \n";
	// hight light special points:
	for (size_t i = 0; i < specialpointindex.size(); i++) {
		DisplayPointFeatureInfo(Featurelist, specialpointindex[i]);
		HighLightAPoint(circles[specialpointindex[i]], src);
	}

	imshow("detected circles", src);
}


int main(int argc, char** argv) {

	const char* filename = argc >= 2 ? argv[1] : "smarties.png";
	// Loads an image
	Mat src = imread(samples::findFile(filename), IMREAD_COLOR);
	// Check if image is loaded fine
	if (src.empty()) {
		printf(" Error opening image\n");
		printf(" Program Arguments: [image_name -- default %s] \n", filename);
		return EXIT_FAILURE;
	}
	std::vector<Point> RegularPoints;
	std::vector<Point> SpecialPoints;

	find_points(RegularPoints, SpecialPoints, src);

	waitKey();
	return EXIT_SUCCESS;

}