/*
The image.png is made up of a regular points grid with three special points.
Please write a C/C++ function that finds all points in the calibration image.

Your solution will be evaluated on efficiency, coding style, readability, and correctness.
In addition to correct results and time taken, optimal code is a significant factor in determining your grade.

Assume that this is for the PC platform.
*/

#include <iostream>
#include "kdTreeSearch.h"
#include "kdTreeSearch.cpp"
#include "simpleSearch.h"
#include <random>

//#define THREAD 1
#define SPEEDTEST 1

#ifdef  SPEEDTEST
#include <chrono>
#endif

using namespace cv;
// display the feature of a selected point
void DisplayPointFeatureInfo(const std::vector<pointfeature::PointEulerDistanceFeature>& featurelist, size_t index) {
	std::cout << "-------The feature of point #" << index << " ---------- \n";
	std::cout << "Maximum distance: " << featurelist[index].Maxdistance << "\n";
	std::cout << "Maximum index: " << featurelist[index].MaxDistanceIndex << "\n";
	std::cout << "Minimum distance: " << featurelist[index].Mindistance << "\n";
	for (int i = 0; i < 4; i++) {
		std::cout << "Feature #: " << i << " is: " << featurelist[index].Feature[i] << "\n";
		std::cout << "Point Index in the list is: " << featurelist[index].Indexlist[i] << "\n";
	}
	std::cout << "------------End of feature-------------- \n";
}
// draw a rectangular box to a selected point
void HighLightAPoint(const Vec3f& pointinfo,  Mat& src) {
	Point topleft = Point(pointinfo[VECTOR_X] + 2.0*pointinfo[2] / sqrt(2), pointinfo[VECTOR_Y] + 2.0*pointinfo[2] / sqrt(2));
	Point bottomright = Point(pointinfo[VECTOR_X] - 2.0*pointinfo[2] / sqrt(2), pointinfo[VECTOR_Y] - 2.0*pointinfo[2] / sqrt(2));
	rectangle(src, topleft, bottomright, (0, 255, 0), 2);
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
	// use Hough transformation to detect circle: TO DO: (future work) adaptive dot detection (use dot)

	HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
		gray.rows / 48,  // change this value to detect circles with different distances to each other
		150, 15, 1, 30 // change the last two parameters
	// (min_radius & max_radius) to detect larger circles
	);
	// use a copy of the points for the kd tree build
	std::vector<Vec3f>temp(circles.begin(), circles.end());

	std::cout << "There are " << temp.size() << " points detected \n";
	// define and initialize the feature list
	std::vector<pointfeature::PointEulerDistanceFeature> Featurelist;
	Featurelist.reserve(temp.size());
#ifdef SPEEDTEST
	std::vector<pointfeature::PointEulerDistanceFeature> Featurelist2;
	Featurelist2.reserve(temp.size());
#endif
	pointfeature::PointEulerDistanceFeature initialfeature;
	initialfeature.Reset(1000.0*1000.0);
	// initialize all the 
	for (size_t i = 0; i < temp.size(); i++) {
		Featurelist.push_back(initialfeature);
#ifdef SPEEDTEST
		Featurelist2.push_back(initialfeature);
#endif
	}
	// show all the points on the image
	for (size_t i = 0; i < temp.size(); i++)
	{
		Vec3i c = temp[i];
		Point center = Point(c[0], c[1]);
		// circle center
		circle(src, center, 1, Scalar(0, 100, 100), 2, LINE_AA);
		// circle outline
		int radius = c[2];
		circle(src, center, radius, Scalar(255, 0, 255), 2, LINE_AA);
	}
	// step 1 calculate the mean of the minimum distance of all the points
	float MeanMin = 0.0;
#ifdef THREAD
	// calculate the optimum thread allocation using multi-threading
	unsigned long const MinPointsPerThread = 30; // allocate number of points to be calculated by 1 thread
	unsigned long const MaxRequiredThread = ((unsigned long)temp.size() + MinPointsPerThread - 1) / MinPointsPerThread;
	//unsigned long const HardwareThreads = std::thread::hardware_concurrency();
	unsigned long const HardwareThreads = 1;
	unsigned long const NumThreads = std::min(HardwareThreads != 0 ? HardwareThreads : 2, MaxRequiredThread);
	std::cout << "multi threading enabled... \n";
	std::cout << "minmum points per thread is: " << MinPointsPerThread << "\n";
	std::cout << "numeber of required threads is: " << MaxRequiredThread << "\n";
	std::cout << "hardware supported threads are: " << HardwareThreads << "\n";
	std::cout << "number of threads possible is: " << NumThreads << "\n";
	size_t numberOfcalculations = temp.size();
	size_t NumberPointsPerThread = (NumThreads == MaxRequiredThread ? (size_t)MinPointsPerThread : (numberOfcalculations + (size_t)NumThreads - 1) / (size_t)NumThreads);
	std::cout << "number of points allocated per thread is: " << NumberPointsPerThread << "\n";

	std::vector<size_t> NumberOfDistanceArray;
	NumberOfDistanceArray.reserve(NumThreads);
	for (size_t i = 0; i < NumThreads - 1; i++) {// 0 - N-2 same
		NumberOfDistanceArray.emplace_back(NumberPointsPerThread);
	}
	// final one is special
	NumberOfDistanceArray.emplace_back(numberOfcalculations - (NumThreads - 1)*NumberPointsPerThread);

	for (auto& t : NumberOfDistanceArray) {
		std::cout << "number of distances are: " << t << "\n";
	}
	// construct the thread object
	std::vector<std::thread> disthread;
	disthread.reserve(NumThreads - 1);// remaining one is the main thread

	// construct the kd tree
#ifdef SPEEDTEST
	auto start_kd = std::chrono::steady_clock::now();
#endif
	kdTree::kdTreeSearch<float, Vec3f, 2> kdTree1(temp);
	// construct the index list based on the 
	for (size_t i = 0; i < NumThreads - 1; i++) {
		disthread.emplace_back(&kdTree::kdTreeSearch<float, Vec3f, 2>::BatchSearchForNearestPoint, &kdTree1,
			std::ref(temp), std::ref(Featurelist), i*NumberPointsPerThread, NumberPointsPerThread);
	}
	// calculate the last block
	disthread.emplace_back(&kdTree::kdTreeSearch<float, Vec3f, 2>::BatchSearchForNearestPoint, &kdTree1,
		std::ref(temp), std::ref(Featurelist), (NumThreads-1)*NumberPointsPerThread, numberOfcalculations - (NumThreads - 1)*NumberPointsPerThread);
	std::for_each(disthread.begin(), disthread.end(), std::mem_fn(&std::thread::join));



#ifdef SPEEDTEST
	auto end_kd = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds_kd = end_kd - start_kd;
#endif

#else

#ifdef SPEEDTEST
	auto start_kd = std::chrono::steady_clock::now();
#endif
	// build the k-d tree using the temp data
	kdTree::kdTreeSearch<float, Vec3f, 2> kdTree1(temp);

	kdTree1.BatchSearchForNearestPoint(temp, Featurelist, 0,temp.size());
	/*
	for (size_t i = 0; i < temp.size(); i++) {
		kdTree1.SearchForNearestPoint(temp[i], Featurelist[i], i);
		// take the sqrt as the distance feature
		Featurelist[i].Mindistance = sqrt(Featurelist[i].Mindistance);
		Featurelist[i].Maxdistance = sqrt(Featurelist[i].Maxdistance);
		for (int j = 0; j < 4; j++) {
			Featurelist[i].Feature[j] = sqrt(Featurelist[i].Feature[j]);
		}
	}*/
#ifdef SPEEDTEST
	auto end_kd = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds_kd = end_kd - start_kd;
#endif


#endif // ASYNC
#ifdef SPEEDTEST
	auto start_simple = std::chrono::steady_clock::now();
	simplesearch::UpdatePointFeature(temp, Featurelist2);
	auto end_simple = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds_simple = end_simple - start_simple;
#endif
	// calculate the minimum distance
	for (size_t i = 0; i < temp.size(); i++) {
		MeanMin += Featurelist[i].Mindistance;
	}

	MeanMin /= static_cast<float>(temp.size());
	// step 2, assume that the number of special points are much less than regular points, all 
	// points with minmum distance less than the total mean is the special point
	special_points.clear();
	regular_points.reserve(temp.size());
	std::vector<size_t> specialpointindex;
	for (size_t i = 0; i < temp.size(); i++) {
		Vec3i location_v = temp[i];
		Point location_p = Point(location_v[0], location_v[1]);// convert it to points type
		// step 3: the special points should have both the maximum distance and the minimum distance at around 
		// 0.707 (sqrt(2)/2) of grid distance. Therefore, the we set 0.15 as the theshold to determine the specail points, i.e the ration : 0.7+0.1 = 0.85
		if ((Featurelist[i].Mindistance < MeanMin*0.85) && (Featurelist[i].Maxdistance < MeanMin*0.85)) {
			special_points.push_back(location_p);
			specialpointindex.push_back(i);
		}
		else {
			// all the remaining points are regular points
			regular_points.push_back(location_p);
		}
	}

	// display detection result:
#ifdef SPEEDTEST
	std::cout << "elapsed time (kd tree): " << elapsed_seconds_kd.count() << "s\n";
	std::cout << "elapsed time (simple method): " << elapsed_seconds_simple.count() << "s\n";
	std::cout << "speed ratio is: " << elapsed_seconds_simple.count()/ elapsed_seconds_kd.count() << "\n";
#endif
	std::cout << "total mean dist is: " << MeanMin << "\n";
	std::cout << "There are " << special_points.size() << " special points \n";
	std::cout << "There are " << regular_points.size() << " regular points \n";

	for (size_t i = 0; i < specialpointindex.size(); i++) {
		DisplayPointFeatureInfo(Featurelist, specialpointindex[i]);
		HighLightAPoint(temp[specialpointindex[i]], src);
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