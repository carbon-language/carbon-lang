/* Title:   IGNode.h                      -*- C++ -*-
   Author:  Ruchira Sasanka
   Date:    July 25, 01
   Purpose: Represents a node in an interference graph. 
   Notes:

   For efficiency, the AdjList is updated only once - ie. we can add but not
   remove nodes from AdjList. 

   The removal of nodes from IG is simulated by decrementing the CurDegree.
   If this node is put on stack (that is removed from IG), the CurDegree of all
   the neighbors are decremented and this node is marked OnSack. Hence
   the effective neighbors in the AdjList are the ones that do not have the
   OnStack flag set (therefore, they are in the IG).

   The methods that modify/use the CurDegree Must be called only
   after all modifications to the IG are over (i.e., all neighbors are fixed).

   The vector representation is the most efficient one for adj list.
   Though nodes are removed when coalsing is done, we access it in sequence
   for far many times when coloring (colorNode()).

*/

#ifndef IG_NODE_H
#define IG_NODE_H

#include "llvm/CodeGen/LiveRange.h"
class RegClass;

//----------------------------------------------------------------------------
// Class IGNode
//
// Represents a node in an interference graph.
//----------------------------------------------------------------------------

class IGNode {
  const unsigned Index;         // index within IGNodeList 
  bool OnStack;                 // this has been pushed on to stack for coloring
  std::vector<IGNode *> AdjList;// adjacency list for this live range

  int CurDegree;     
  //
  // set by InterferenceGraph::setCurDegreeOfIGNodes() after calculating
  // all adjacency lists.
  // Decremented when a neighbor is pushed on to the stack. 
  // After that, never incremented/set again nor used.

  LiveRange *const ParentLR;
public:

  IGNode(LiveRange *LR, unsigned index) : Index(index), ParentLR(LR) {
    OnStack = false;
    CurDegree = -1;
    ParentLR->setUserIGNode(this);
  }

  inline unsigned int getIndex() const { return Index; }

  // adjLists must be updated only once.  However, the CurDegree can be changed
  //
  inline void addAdjIGNode(IGNode *AdjNode) { AdjList.push_back(AdjNode);  } 

  inline IGNode *getAdjIGNode(unsigned ind) const 
    { assert ( ind < AdjList.size()); return AdjList[ind]; }

  // delete a node in AdjList - node must be in the list
  // should not be called often
  //
  void delAdjIGNode(const IGNode *Node); 

  inline unsigned getNumOfNeighbors() const { return AdjList.size(); }

  // Get the number of unique neighbors if these two nodes are merged
  unsigned getCombinedDegree(const IGNode* otherNode) const;

  inline bool isOnStack() const { return OnStack; }

  // remove form IG and pushes on to stack (reduce the degree of neighbors)
  //
  void pushOnStack(); 

  // CurDegree is the effective number of neighbors when neighbors are
  // pushed on to the stack during the coloring phase. Must be called
  // after all modifications to the IG are over (i.e., all neighbors are
  // fixed).
  //
  inline void setCurDegree() {
    assert(CurDegree == -1);
    CurDegree = AdjList.size();
  }

  inline int getCurDegree() const { return CurDegree; }

  // called when a neigh is pushed on to stack
  //
  inline void decCurDegree() { assert(CurDegree > 0); --CurDegree; }


  // The following methods call the methods in ParentLR
  // They are added to this class for convenience
  // If many of these are called within a single scope,
  // consider calling the methods directly on LR

  inline void setRegClass(RegClass *RC) { ParentLR->setRegClass(RC);  }

  inline RegClass *getRegClass() const { return ParentLR->getRegClass(); }

  inline bool hasColor() const { return ParentLR->hasColor();  }

  inline unsigned int getColor() const { return ParentLR->getColor();  }

  inline void setColor(unsigned Col) { ParentLR->setColor(Col);  }

  inline void markForSpill() { ParentLR->markForSpill(); }

  inline void markForSaveAcrossCalls() { ParentLR->markForSaveAcrossCalls();  }

  inline unsigned int isCallInterference() const 
  { return ParentLR->isCallInterference(); } 

  inline LiveRange *getParentLR() const { return ParentLR; }
};

#endif
