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
struct PointEulerDistanceFeature {
	float Maxdistance{ 0.0 };  // the maximum distance
	float Mindistance{ 0.0 }; // the mimum distance
	size_t MaxDistanceIndex{ 0 }; // the index of the feature
	size_t MinDistanceIndex{ 0 }; // the index of the min distance
	Vec4f Feature;             // the distance to the neighbour points
	Vec4i Indexlist;           // the index in the point list of the neighbour points

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

};
struct distancepair{
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
	void Display(){
		std::cout << "current i:" << current_i << ", current j: " << current_j << "\n";
	}
};
distancepair GetThreadEndPoint (size_t threadpointnumber,const distancepair& starting_pair) {
	distancepair end_pair = starting_pair;
	size_t numberpointscurrentstack = starting_pair.starting_j - starting_pair.current_i;
	size_t remainingpoints = threadpointnumber - numberpointscurrentstack;
	if (remainingpoints > 0) {// if greater than zero, means that the end point is in an other stack
		// calculate in which stack
		double t = static_cast<double>(end_pair.starting_j)+1.0;
		double a = static_cast<double>(remainingpoints);
		size_t c1 = static_cast<size_t>(floor(0.5 - t + sqrt(0.25*(2.0*t-1)*(2.0*t-1) + 2.0*a)));
		size_t c2 = c1 + 1;
		// verify which stack number is correct
		size_t b2 = (2 * end_pair.starting_j + c2 + 1)*c2/2 - remainingpoints;

		if (((2 * end_pair.starting_j + c1 + 1)*c1/2) == remainingpoints) {
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
// draw a rectangular box to a selected point
void HighLightAPoint(const Vec3f& pointinfo,  Mat& src) {
	Point topleft = Point(pointinfo[VECTOR_X] + 2.0*pointinfo[2] / sqrt(2), pointinfo[VECTOR_Y] + 2.0*pointinfo[2] / sqrt(2));
	Point bottomright = Point(pointinfo[VECTOR_X] - 2.0*pointinfo[2] / sqrt(2), pointinfo[VECTOR_Y] - 2.0*pointinfo[2] / sqrt(2));
	rectangle(src, topleft, bottomright, (0, 255, 0), 2);
}
// Update the feature for a selected point
#ifdef THREAD
void Parallel_UpdateFeature2(const std::vector<Vec3f>& pointlist, std::vector<PointEulerDistanceFeature>& featurelist, 
							 distancepair pair, // staring index pair
							 size_t numberofdistance) {// number of required calculations
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

#else

void UpdatePointFeature(const std::vector<Vec3f>& pointlist, std::vector<PointEulerDistanceFeature>& featurelist) {
	// find the distance to the 4 closet points
	float EulerDistance = 0.0;
	distancepair pair;
	pair(0, 1);
	size_t totaldistancenumber = (featurelist.size()-1)*featurelist.size()/2;

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
	// calculate the optimum thread allocation
	unsigned long const MinPointsPerThread = 30; // allocate number of points to be calculated by 1 thread
	unsigned long const MaxRequiredThread = ((unsigned long)circles.size() + MinPointsPerThread - 1) / MinPointsPerThread;
	unsigned long const HardwareThreads = std::thread::hardware_concurrency();
	unsigned long const NumThreads = std::min(HardwareThreads != 0 ? HardwareThreads : 2, MaxRequiredThread);
	
	std::cout << "minmum points per thread is: " << MinPointsPerThread << "\n";
	std::wcout << "numeber of required threads is: " << MaxRequiredThread << "\n";
	std::cout << "Hardware Threads are: " << HardwareThreads << "\n";
	std::cout << "Num Threads Threads is: " << NumThreads << "\n";
	//size_t NumberPointsPerThread = (NumThreads == MaxRequiredThread ? (size_t)MinPointsPerThread : (circles.size()+ (size_t)NumThreads- 1)/(size_t)NumThreads );
	size_t totaldistancenumber = (circles.size() - 1)*circles.size() / 2;
	size_t NumberPointsPerThread = (NumThreads == MaxRequiredThread ? (size_t)MinPointsPerThread : (totaldistancenumber + (size_t)NumThreads- 1)/(size_t)NumThreads );
	std::vector<size_t> NumberOfDistanceArray;
	NumberOfDistanceArray.reserve(NumThreads);
	for (size_t i = 0; i < NumThreads - 1; i++) {// 0 - N-2 same
		NumberOfDistanceArray.emplace_back(NumberPointsPerThread);
	}
	// final one is special
	NumberOfDistanceArray.emplace_back(totaldistancenumber - (NumThreads - 1)*NumberPointsPerThread);

	for (auto& t : NumberOfDistanceArray) {
		std::cout << "number of distances are: " << t << "\n";
	}

	//std::cout << "Num of Points of threads is: " << NumberPointsPerThread << "\n";
	std::cout << "Num of distance calculations per thread is: " << NumberPointsPerThread << "\n";
	std::vector<std::thread> disthread;
	disthread.reserve(NumThreads - 1);
	// calculate the starting index point for each thread
	std::vector<distancepair> distancepairarray;
	distancepairarray.reserve(NumThreads);
	distancepair startpoint;
	startpoint(0, 1);
	distancepairarray.emplace_back(startpoint);// the first element is (0, 1)
	for (size_t i = 1; i < NumThreads; i++) {// determine the starting index for each thread
		startpoint = GetThreadEndPoint( NumberPointsPerThread, distancepairarray[i - 1]);
		startpoint.Advance();// advance one step to get the next start point
		distancepairarray.emplace_back(startpoint);
	}

	std::cout << "--- result 1-----\n";

	for (auto& t : distancepairarray) {
		t.Display();
	}
	std::cout << "--- result 2-----\n";
	// verify 
	distancepair testpoint;
	testpoint(0, 1);
	testpoint.Display();
	for (size_t i = 0; i < NumThreads-1; i++) {// determine the starting index for each thread
		for (size_t i = 0; i <= NumberPointsPerThread-1; i++) {// determine the starting index for each thread
			testpoint.Advance();
		}
		testpoint.Display();
	}

	for (size_t i = 0; i < NumThreads - 1; i++) {
		disthread.emplace_back(Parallel_UpdateFeature2, std::ref(circles), std::ref(Featurelist), distancepairarray[i], NumberOfDistanceArray[i]);
	}
	// calculate the last block
	Parallel_UpdateFeature2(circles, Featurelist, distancepairarray.back(), NumberOfDistanceArray.back());
	std::for_each(disthread.begin(), disthread.end(),std::mem_fn(&std::thread::join));
	for (const auto& feature : Featurelist) {// access by const reference 
		MeanMin += feature.Mindistance;
	}
#else

	UpdatePointFeature(circles, Featurelist);
	for (size_t i = 0; i < circles.size(); i++) {
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