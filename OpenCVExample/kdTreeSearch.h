#pragma once
#include "pointFeature.h"
using namespace cv;
namespace kdTree{
  // definition of a node in kd tree
  template< typename pointType, int dimension > 
  struct Node {
		size_t index{ 0 }; //index of this point at the original point list
		pointType point[dimension];
		std::unique_ptr<Node> left;  // pointer to the left tree
		std::unique_ptr<Node> right; // pointer to the right tree
  };

  template< typename nodeType, typename inputtype, int k >
  struct std::unique_ptr<kdTree::Node<nodeType,k>> newNode(const inputtype& point, size_t point_index) {
		std::unique_ptr<kdTree::Node<nodeType, k>> temp{ new kdTree::Node<nodeType,k> };
		// load the points
		for (int i = 0; i < k; i++) {
			temp->point[i] = point[i];
		}
		// initialize pointers
		temp->left = NULL;
		temp->right = NULL;
		temp->index = point_index;
		// transfer the ownership of the unique ptr to the return value
		return std::move(temp);
  }

  template< typename nodeType, typename inputtype, int k >
  class kdTreeSearch
  {
  public:
	kdTreeSearch(std::vector<inputtype>& pointlist);// initial depth

	// inset one node to the kd tree (recursive)
	void InsertRec(std::unique_ptr<kdTree::Node<nodeType, k>>& root, 
				   const inputtype& point, 
				   size_t point_index, 
				   int depth);	

	// insert point list to the tree using quick median algorithm
	void BatchInsertRecstd(std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		std::vector<inputtype>& pointlist,
		size_t index_begin,
		size_t index_end,
		int depth);

	bool SimpleBoundingBoxCheck(const std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		const inputtype& center,
		const nodeType& radius_squared,
		const int feature_dimension);

	// depth search for the nearest n neighbour
	void SearchForMultiFeature(const inputtype& search_point,
		pointfeature::PointEulerDistanceFeature& search_feature,
		const size_t& search_point_index,
		const std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		int depth);
	
	void SearchForNearestPoint(const inputtype& search_point,
		pointfeature::PointEulerDistanceFeature& search_feature,
		const size_t& search_point_index);

	void BatchSearchForNearestPoint(const std::vector< inputtype>& search_point_list,
		std::vector<pointfeature::PointEulerDistanceFeature>& search_feature_list,
		const size_t begin_index,
		const size_t number_of_points);

	void SearchForSingleNearestPoint(const inputtype& search_point,
		const size_t& search_point_index,
		nodeType& minimum_distance,
		size_t& index, // the position of the nearest point in the point list
		const std::unique_ptr<kdTree::Node<nodeType, k>>& root,
		int depth);

	void SearchForSingleNearestPoint(const inputtype& search_point,
		const size_t& search_point_index,
		nodeType& minimum_distance,
		size_t& index);

	void DisplayTree(std::unique_ptr<kdTree::Node<nodeType, k>>& root);

  	~kdTreeSearch();
	std::unique_ptr<kdTree::Node<nodeType,k>> kdroot_; // the pointer to the root node 
	std::vector< kdTree::Node<nodeType, k>*> treeaccesslist_;
  private:


  };
}