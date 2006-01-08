//===- llvm/Analysis/ET-Forest.h - ET-Forest implementation -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was written by Daniel Berlin from code written by Pavel Nejedy, and
// is distributed under the University of Illinois Open Source License. See
// LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the following classes:
//  1. ETNode:  A node in the ET forest.
//  2. ETOccurrence: An occurrence of the node in the splay tree
//  storing the DFS path information.
//
//  The ET-forest structure is described in:
//    D. D. Sleator and R. E. Tarjan. A data structure for dynamic trees.
//    J.  G'omput. System Sci., 26(3):362 381, 1983.
//
// Basically, the ET-Forest is storing the dominator tree (ETNode),
// and a splay tree containing the depth first path information for
// those nodes (ETOccurrence).  This enables us to answer queries
// about domination (DominatedBySlow), and ancestry (NCA) in
// logarithmic time, and perform updates to the information in
// logarithmic time.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ETFOREST_H
#define LLVM_ANALYSIS_ETFOREST_H

#include <cassert>
#include <cstdlib>

namespace llvm {
class ETNode;

/// ETOccurrence - An occurrence for a node in the et tree
///
/// The et occurrence tree is really storing the sequences you get from
/// doing a DFS over the ETNode's.  It is stored as a modified splay
/// tree.
/// ET occurrences can occur at multiple places in the ordering depending
/// on how many ET nodes have it as their father.  To handle
/// this, they are separate from the nodes.
///
class ETOccurrence {
public:
  ETOccurrence(ETNode *n): OccFor(n), Parent(NULL), Left(NULL), Right(NULL),
    Depth(0), Min(0), MinOccurrence(this) {};

  void setParent(ETOccurrence *n) {
    Parent = n;
  }

  // Add D to our current depth
  void setDepthAdd(int d) {
    Min += d;
    Depth += d;
  }
  
  // Reset our depth to D
  void setDepth(int d) {
    Min += d - Depth;
    Depth = d;
  }

  // Set Left to N
  void setLeft(ETOccurrence *n) {
    assert(n != this && "Trying to set our left to ourselves");
    Left = n;
    if (n)
      n->setParent(this);
  }
  
  // Set Right to N
  void setRight(ETOccurrence *n) {
    assert(n != this && "Trying to set our right to ourselves");
    Right = n;
    if (n)
      n->setParent(this);
  }
  
  // Splay us to the root of the tree
  void Splay(void);

  // Recompute the minimum occurrence for this occurrence.
  void recomputeMin(void) {
    ETOccurrence *themin = Left;
    
    // The min may be our Right, too.
    if (!themin || (Right && themin->Min > Right->Min))
      themin = Right;
    
    if (themin && themin->Min < 0) {
      Min = themin->Min + Depth;
      MinOccurrence = themin->MinOccurrence;
    } else {
      Min = Depth;
      MinOccurrence = this;
    }
  }
 private:
  friend class ETNode;

    // Node we represent
  ETNode *OccFor;

  // Parent in the splay tree
  ETOccurrence *Parent;

  // Left Son in the splay tree
  ETOccurrence *Left;

  // Right Son in the splay tree
  ETOccurrence *Right;

  // Depth of the node is the sum of the depth on the path to the
  // root.
  int Depth;

  // Subtree occurrence's minimum depth
  int Min;

  // Subtree occurrence with minimum depth
  ETOccurrence *MinOccurrence;
};


class ETNode {
public:
  ETNode(void *d) : data(d), Father(NULL), Left(NULL),
                    Right(NULL), Son(NULL), ParentOcc(NULL) {   
    RightmostOcc = new ETOccurrence(this);
  };

  // This does *not* maintain the tree structure.
  // If you want to remove a node from the forest structure, use
  // removeFromForest()
  ~ETNode() {
    delete RightmostOcc;
  }

  void removeFromForest() {
    // Split us away from all our sons.
    while (Son)
      Son->Split();
    
    // And then split us away from our father.
    if (Father)
      Father->Split();
  }

  // Split us away from our parents and children, so that we can be
  // reparented. NB: setFather WILL NOT DO WHAT YOU WANT IF YOU DO NOT
  // SPLIT US FIRST.
  void Split();

  // Set our parent node to the passed in node
  void setFather(ETNode *);
  
  // Nearest Common Ancestor of two et nodes.
  ETNode *NCA(ETNode *);
  
  // Return true if we are below the passed in node in the forest.
  bool Below(ETNode *);
  /*
   Given a dominator tree, we can determine whether one thing
   dominates another in constant time by using two DFS numbers:
  
   1. The number for when we visit a node on the way down the tree
   2. The number for when we visit a node on the way back up the tree
  
   You can view these as bounds for the range of dfs numbers the
   nodes in the subtree of the dominator tree rooted at that node
   will contain.
  
   The dominator tree is always a simple acyclic tree, so there are
   only three possible relations two nodes in the dominator tree have
   to each other:
  
   1. Node A is above Node B (and thus, Node A dominates node B)
  
        A
        |
        C
       / \ 
      B   D
  
  
   In the above case, DFS_Number_In of A will be <= DFS_Number_In of
   B, and DFS_Number_Out of A will be >= DFS_Number_Out of B.  This is
   because we must hit A in the dominator tree *before* B on the walk
   down, and we will hit A *after* B on the walk back up
  
   2. Node A is below node B (and thus, node B dominates node B)
       
        B
        |
        A
       / \ 
      C   D
  
   In the above case, DFS_Number_In of A will be >= DFS_Number_In of
   B, and DFS_Number_Out of A will be <= DFS_Number_Out of B.
  
   This is because we must hit A in the dominator tree *after* B on
   the walk down, and we will hit A *before* B on the walk back up
  
   3. Node A and B are siblings (and thus, neither dominates the other)
  
        C
        |
        D
       / \                        
      A   B
  
   In the above case, DFS_Number_In of A will *always* be <=
   DFS_Number_In of B, and DFS_Number_Out of A will *always* be <=
   DFS_Number_Out of B.  This is because we will always finish the dfs
   walk of one of the subtrees before the other, and thus, the dfs
   numbers for one subtree can't intersect with the range of dfs
   numbers for the other subtree.  If you swap A and B's position in
   the dominator tree, the comparison changes direction, but the point
   is that both comparisons will always go the same way if there is no
   dominance relationship.
  
   Thus, it is sufficient to write
  
   A_Dominates_B(node A, node B) {
      return DFS_Number_In(A) <= DFS_Number_In(B) &&
             DFS_Number_Out(A) >= DFS_Number_Out(B);
   }
  
   A_Dominated_by_B(node A, node B) {
     return DFS_Number_In(A) >= DFS_Number_In(A) &&
            DFS_Number_Out(A) <= DFS_Number_Out(B);
   }
  */
  bool DominatedBy(ETNode *other) const {
    return this->DFSNumIn >= other->DFSNumIn &&
           this->DFSNumOut <= other->DFSNumOut;
  }
  
  // This method is slower, but doesn't require the DFS numbers to
  // be up to date.
  bool DominatedBySlow(ETNode *other) {
    return this->Below(other);
  }

  void assignDFSNumber(int &num) {
    DFSNumIn = num++;
    
    if (Son) {
      Son->assignDFSNumber(num);
      for (ETNode *son = Son->Right; son != Son; son = son->Right)
        son->assignDFSNumber(num);
    }
    DFSNumOut = num++;
  }
  
  bool hasFather() const {
    return Father != NULL;
  }

  // Do not let people play around with fathers.
  const ETNode *getFather() const {
    return Father;
  }

  template <typename T>
  T *getData() const {
    return static_cast<T*>(data);
  }
  
  unsigned getDFSNumIn() const {
    return DFSNumIn;
  }
  
  unsigned getDFSNumOut() const {
    return DFSNumOut;
  }

 private:
  // Data represented by the node
  void *data;

  // DFS Numbers
  unsigned DFSNumIn, DFSNumOut;

  // Father
  ETNode *Father;

  // Brothers.  Node, this ends up being a circularly linked list.
  // Thus, if you want to get all the brothers, you need to stop when
  // you hit node == this again.
  ETNode *Left, *Right;

  // First Son
  ETNode *Son;

  // Rightmost occurrence for this node
  ETOccurrence *RightmostOcc;

  // Parent occurrence for this node
  ETOccurrence *ParentOcc;
};
}  // end llvm namespace

#endif
