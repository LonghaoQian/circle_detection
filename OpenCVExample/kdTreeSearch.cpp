#include "kdTreeSearch.h"


namespace kdTree {
	template< typename nodeType, typename inputtype, int k >
	kdTreeSearch<nodeType, inputtype, k>::kdTreeSearch(std::vector<inputtype>& pointlist)
	{
		kdroot_ = NULL;
		BatchInsertRecstd(kdroot_, pointlist, 0, pointlist.size() - 1, 0);// add the data point to the kd tree
		//treeaccesslist_(pointlist.size(), NULL);
	}

	template< typename nodeType, typename inputtype, int k >
	kdTreeSearch<nodeType, inputtype, k>::~kdTreeSearch()
	{
	}

	template< typename nodeType, typename inputtype, int k >
	void kdTreeSearch<nodeType, inputtype, k>::InsertRec(std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		const inputtype& point,
		size_t point_index,
		int depth) {
		// determine whether the tree is empty
		if (root == NULL) {// if the tree is empty, then append the node with the given value
			root = newNode<nodeType, inputtype, k>(point, point_index);
			return;
		}
		// Compare the new point with root on current dimension 'cd' 
		// and decide the left or right subtree 
		int cd = depth % k;
		if (point[cd] < root->point[cd]) {
			// if the k d value is less than the 
			InsertRec(root->left, point, point_index, depth + 1);
		}
		else {
			InsertRec(root->right, point, point_index, depth + 1);
		}
	}

	template< typename nodeType, typename inputtype, int k >
	void  kdTreeSearch<nodeType, inputtype, k>::BatchInsertRecstd(std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		std::vector<inputtype>& pointlist,
		size_t index_begin,
		size_t index_end,
		int depth) 
	{
		if (pointlist.empty()) {
			std::cout << "Error, the point list is empty! \n";
			return;
		}
		size_t vector_size = index_end - index_begin + 1;
		// Compare the new point with root on current dimension 'cd' 
		// and decide the left or right subtree 
		int cd = depth % k;
		// if not empty determine the value of the median
		std::sort(pointlist.begin() + index_begin,
			pointlist.begin() + index_end + 1,
			[&cd](const cv::Vec<float, 3>& t1, const cv::Vec<float, 3>& t2) {return t1[cd] < t2[cd]; });
		size_t median_location = index_begin + vector_size / 2;
		size_t median_location2 = median_location;
		//record the median value
		inputtype medianvalue = pointlist[median_location];
		// make sure the point after the median are strictly larger than the median
		for (size_t i = median_location; i < index_end; i++) {
			if (pointlist[i][cd] <= medianvalue[cd]) {
				median_location2 = i;
			}
		}

		// add the median to the tree
		InsertRec(root, pointlist[median_location2], median_location2, depth);

		// split the vector into right and left branch
		size_t number_before_median = median_location2 - index_begin;
		size_t number_after_median = index_end - median_location2;

		//std::cout << "number befor median is: " << number_before_median << "\n";
		//std::cout << "number after median is: " << number_after_median << "\n";
		//std::cout << "Index " << median_location << "reached \n";
		//std::cout << "Position X: " << pointlist[median_location][VECTOR_X] << ", Y: " << pointlist[median_location][VECTOR_Y] << "\n";
		//std::cout << "----------------- \n";
		if (number_before_median > 0) {// contains at least one element before the median
			BatchInsertRecstd(root->left, pointlist, index_begin, median_location2 - 1, depth + 1);
		}

		if (number_after_median > 0) {
			BatchInsertRecstd(root->right, pointlist, median_location2 + 1, index_end, depth + 1);
		}
		// if the branches are all finished 
		return;
	}
	template< typename nodeType, typename inputtype, int k >
	bool kdTreeSearch<nodeType, inputtype, k>::SimpleBoundingBoxCheck(const std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		const inputtype& center,
		const nodeType& radius_squared,
		const int feature_dimension) 
	{
		nodeType dist = center[feature_dimension] - root->point[feature_dimension];
		if (dist* dist > radius_squared) {
			// if the squared distance is larger than the squared
			// the bounding check is ok
			return true;
		}
		else {
			// if distance is less than the radius, then the bounding box check fails
			return false;
		}
	}
	template< typename nodeType, typename inputtype, int k >
	void  kdTreeSearch<nodeType, inputtype, k>::SearchForMultiFeature(const inputtype& search_point,
		pointfeature::PointEulerDistanceFeature& search_feature,
		const size_t& search_point_index,
		const std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		int depth) {
		// if the root is empty, immediately return
		if (root == NULL) {
			std::cout << "error, this node is empty! \n";
			return;
		}
		/*
		std::cout << "Index " << root->index << " reached \n";
		std::cout << "Position X: " << root->point[VECTOR_X] << ", Y: " << root->point[VECTOR_Y] << "\n";
		*/
		// calculate the feature dimension
		int cd = depth % k;
		// if the root is not empty, determine witch child is promising
		//std::cout << " dimension is: " << cd << "\n";
		bool isleftchild = false;
		if (search_point[cd] <= root->point[cd]) {
			isleftchild = true;
			
		}
		else {
			
			isleftchild = false;
		}
		// determine whether the promising side has any child node
		if (isleftchild) {
			if (root->left == NULL) {
				// if the node is empty then do nothing
			}
			else {
				// if the node is not empty, repeat the search
				//std::cout << "go to left branch \n";
				SearchForMultiFeature(search_point,
					search_feature,
					search_point_index,
					root->left,
					depth + 1);
			}
		}
		else {
			// same with the leftchild
			if (root->right == NULL) {
				// if the node is empty then do nothing
			}
			else {
				//std::cout << "go to right branch \n";
				SearchForMultiFeature(search_point,
					search_feature,
					search_point_index,
					root->right,
					depth + 1);
			}
		}
		// update the mimimum distance of this point with the current node (square distance)
		float temp_x = root->point[VECTOR_X] - search_point[VECTOR_X];
		float temp_y = root->point[VECTOR_Y] - search_point[VECTOR_Y];
		float temp_distance = temp_x * temp_x + temp_y * temp_y;

		if ((temp_distance <= search_feature.Maxdistance) && (search_point_index != root->index)) {
			// here update the buffer
			search_feature(temp_distance, root->index);
		}// update distance

		// check whether the not promising side has potiential solutions
		// check the boundary
		if (SimpleBoundingBoxCheck(root, search_point, search_feature.Maxdistance, cd)) {
			// if the bounding box check passes the bounding box check, then there is not need to check the other side
			// do nothing
			//std::cout << "bounding box check passes, no need to traverse the other tree \n";
		}
		else {
			// if the bounding box check failes check the other side of the child
			if (isleftchild) {
				
				// if the promising branch is left and is not empty, then check the right
				if (root->right != NULL) {
					//std::cout << "bounding box check fails, traverse left tree \n";
					SearchForMultiFeature(search_point,
						search_feature,
						search_point_index,
						root->right,
						depth + 1);
				}
			}
			else {
				//
				if (root->left != NULL) {
					//std::cout << "bounding box check fails, traverse right tree \n";
					SearchForMultiFeature(search_point,
						search_feature,
						search_point_index,
						root->left,
						depth + 1);
				}
			}
		}
		return;
	}
	template< typename nodeType, typename inputtype, int k >
	void  kdTreeSearch<nodeType, inputtype, k>::SearchForNearestPoint(const inputtype& search_point,
		pointfeature::PointEulerDistanceFeature& search_feature,
		const size_t& search_point_index) {
		SearchForMultiFeature(search_point,search_feature,search_point_index, kdroot_,0);
	}

	template<typename nodeType, typename inputtype, int k>
	void kdTreeSearch<nodeType, inputtype, k>::BatchSearchForNearestPoint(const std::vector<inputtype>& search_point_list, 
		std::vector<pointfeature::PointEulerDistanceFeature>& search_feature_list, 
		const size_t begin_index,
		const size_t number_of_points)
	{
		for (size_t i = begin_index; i < number_of_points + begin_index; i++) {
			SearchForNearestPoint(search_point_list[i], search_feature_list[i], i);
			// take the sqrt as the distance feature
			search_feature_list[i].Mindistance = sqrt(search_feature_list[i].Mindistance);
			search_feature_list[i].Maxdistance = sqrt(search_feature_list[i].Maxdistance);
			for (int j = 0; j < 4; j++) {
				search_feature_list[i].Feature[j] = sqrt(search_feature_list[i].Feature[j]);
			}
		}
	}

	template< typename nodeType, typename inputtype, int k >
	void  kdTreeSearch<nodeType, inputtype, k>::DisplayTree(std::unique_ptr<kdTree::Node<nodeType, k>>& root) {

		if (root == NULL) {
			return;
		}
		std::cout << "node index: " << root->index << "\n";
		std::cout << "node position: ";
		for (int i = 0; i < k; i++) {
			std::cout << root->point[i] << ", ";
		}
		std::cout << "\n";
		DisplayTree(root->left);
		DisplayTree(root->right);


	}
	template< typename nodeType, typename inputtype, int k >
	void kdTreeSearch<nodeType, inputtype, k>::SearchForSingleNearestPoint(const inputtype& search_point,
		const size_t& search_point_index,
		nodeType& minimum_distance,
		size_t& index, // the position of the nearest point in the point list
		const std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		int depth) {
		// if the root is empty, immediately return
		if (root == NULL) {
			std::cout << "error, this node is empty! \n";
			return;
		}
		// calculate the feature dimension
		int cd = depth % k;
		// if the root is not empty, determine witch child is promising
		bool isleftchild = false;
		if (search_point[cd] <= root->point[cd]) {
			isleftchild = true;
			//std::cout << "left branch \n";
		}
		else {
			//std::cout << "right branch \n";
			isleftchild = false;
		}
		// determine whether the promising side has any child node
		if (isleftchild) {
			if (root->left == NULL) {
				// if the node is empty then do nothing
			}
			else {
				// if the node is not empty, repeat the search
				SearchForSingleNearestPoint(search_point,
					search_point_index,
					minimum_distance,
					index,
					root->left,
					depth + 1);
			}
		}
		else {
			// same with the leftchild
			if (root->right == NULL) {
				// if the node is empty then do nothing
			}
			else {
				SearchForSingleNearestPoint(search_point,
					search_point_index,
					minimum_distance,
					index,
					root->right,
					depth + 1);
			}
		}
		// update the mimimum distance of this point with the current node (square distance)
		float temp_x = root->point[VECTOR_X] - search_point[VECTOR_X];
		float temp_y = root->point[VECTOR_Y] - search_point[VECTOR_Y];
		float temp_distance = temp_x * temp_x + temp_y * temp_y;

		if ((temp_distance <= minimum_distance) && (search_point_index != root->index)) {
			index = root->index;// update index
			minimum_distance = temp_distance;
			// std::cout << "reach the same index! \n";
		}// update distance
		// check whether the not promising side has potiential solutions
		// check the boundary
		if (SimpleBoundingBoxCheck(root, search_point, minimum_distance, cd)) {
			// if the bounding box check passes the bounding box check, then there is not need to check the other side
			// do nothing
		}
		else {
			// if the bounding box check failes check the other side of the child
			if (isleftchild) {
				// if the promising branch is left and is not empty, then check the right
				if (root->right != NULL) {
					SearchForSingleNearestPoint(search_point,
						search_point_index,
						minimum_distance,
						index,
						root->right,
						depth + 1);
				}
			}
			else {
				//
				if (root->left != NULL) {
					SearchForSingleNearestPoint(search_point,
						search_point_index,
						minimum_distance,
						index,
						root->left,
						depth + 1);
				}
			}
		}
		return;
	}
	template<typename nodeType, typename inputtype, int k>
	void kdTreeSearch<nodeType, inputtype, k>::SearchForSingleNearestPoint(const inputtype & search_point,
				const size_t & search_point_index, 
				nodeType & minimum_distance,
				size_t& index)
	{
		SearchForSingleNearestPoint(search_point, search_point_index, minimum_distance,
			index, // the position of the nearest point in the point list
			kdroot_,
			0);
	}
}