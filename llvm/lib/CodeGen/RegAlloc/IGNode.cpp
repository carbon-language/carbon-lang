//===-- IGNode.cpp --------------------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
// 
// This file implements an Interference graph node for coloring-based register
// allocation.
// 
//===----------------------------------------------------------------------===//

#include "IGNode.h"
#include <algorithm>
#include <iostream>

//-----------------------------------------------------------------------------
// Sets this IGNode on stack and reduce the degree of neighbors  
//-----------------------------------------------------------------------------

void IGNode::pushOnStack() {
  OnStack = true; 
  int neighs = AdjList.size();

  if (neighs < 0) {
    std::cerr << "\nAdj List size = " << neighs;
    assert(0 && "Invalid adj list size");
  }

  for (int i=0; i < neighs; i++)
    AdjList[i]->decCurDegree();
}
 
//-----------------------------------------------------------------------------
// Deletes an adjacency node. IGNodes are deleted when coalescing merges
// two IGNodes together.
//-----------------------------------------------------------------------------

void IGNode::delAdjIGNode(const IGNode *Node) {
  std::vector<IGNode *>::iterator It=find(AdjList.begin(), AdjList.end(), Node);
  assert(It != AdjList.end() && "The node must be there!");
  AdjList.erase(It);
}

//-----------------------------------------------------------------------------
// Get the number of unique neighbors if these two nodes are merged
//-----------------------------------------------------------------------------

unsigned
IGNode::getCombinedDegree(const IGNode* otherNode) const {
  std::vector<IGNode*> nbrs(AdjList);
  nbrs.insert(nbrs.end(), otherNode->AdjList.begin(), otherNode->AdjList.end());
  sort(nbrs.begin(), nbrs.end());
  std::vector<IGNode*>::iterator new_end = unique(nbrs.begin(), nbrs.end());
  return new_end - nbrs.begin();
}
