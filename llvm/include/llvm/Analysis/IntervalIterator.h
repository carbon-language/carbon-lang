//===- IntervalIterator.h - Interval Iterator Declaration --------*- C++ -*--=//
//
// This file defines an iterator that enumerates the intervals in a control flow
// graph of some sort.  This iterator is parametric, allowing iterator over the
// following types of graphs:
// 
//  TODO
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERVAL_ITERATOR_H
#define LLVM_INTERVAL_ITERATOR_H

#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Method.h"
#include "llvm/CFG.h"
#include <stack>
#include <set>
#include <algorithm>

namespace cfg {

// TODO: Provide an interval iterator that codifies the internals of 
// IntervalPartition.  Inside, it would have a stack of Interval*'s, and would
// walk the interval partition in depth first order.  IntervalPartition would
// then be a client of this iterator.  The iterator should work on Method*,
// const Method*, IntervalPartition*, and const IntervalPartition*.
//


// getNodeHeader - Given a source graph node and the source graph, return the 
// BasicBlock that is the header node.  This is the opposite of
// getSourceGraphNode.
//
inline BasicBlock *getNodeHeader(BasicBlock *BB) { return BB; }
inline BasicBlock *getNodeHeader(Interval *I) { return I->getHeaderNode(); }

// getSourceGraphNode - Given a BasicBlock and the source graph, return the 
// source graph node that corresponds to the BasicBlock.  This is the opposite
// of getNodeHeader.
//
inline BasicBlock *getSourceGraphNode(Method *, BasicBlock *BB) {
  return BB; 
}
inline Interval *getSourceGraphNode(IntervalPartition *IP, BasicBlock *BB) { 
  return IP->getBlockInterval(BB);
}

// addNodeToInterval - This method exists to assist the generic ProcessNode
// with the task of adding a node to the new interval, depending on the 
// type of the source node.  In the case of a CFG source graph (BasicBlock 
// case), the BasicBlock itself is added to the interval.
//
inline void addNodeToInterval(Interval *Int, BasicBlock *BB){
  Int->Nodes.push_back(BB);
}

// addNodeToInterval - This method exists to assist the generic ProcessNode
// with the task of adding a node to the new interval, depending on the 
// type of the source node.  In the case of a CFG source graph (BasicBlock 
// case), the BasicBlock itself is added to the interval.  In the case of
// an IntervalPartition source graph (Interval case), all of the member
// BasicBlocks are added to the interval.
//
inline void addNodeToInterval(Interval *Int, Interval *I) {
  // Add all of the nodes in I as new nodes in Int.
  copy(I->Nodes.begin(), I->Nodes.end(), back_inserter(Int->Nodes));
}


template<class NodeTy, class OrigContainer_t>
class IntervalIterator {
  stack<pair<Interval, typename Interval::succ_iterator> > IntStack;
  set<BasicBlock*> Visited;
  OrigContainer_t *OrigContainer;
public:
  typedef BasicBlock* _BB;

  typedef IntervalIterator<NodeTy, OrigContainer_t> _Self;
  typedef forward_iterator_tag iterator_category;
 
  IntervalIterator() {} // End iterator, empty stack
  IntervalIterator(Method *M) {
    OrigContainer = M;
    if (!ProcessInterval(M->getBasicBlocks().front())) {
      assert(0 && "ProcessInterval should never fail for first interval!");
    }
  }

  IntervalIterator(IntervalPartition &IP) {
    OrigContainer = &IP;
    if (!ProcessInterval(IP.getRootInterval())) {
      assert(0 && "ProcessInterval should never fail for first interval!");
    }
  }

  inline bool operator==(const _Self& x) const { return IntStack == x.IntStack; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }

  inline Interval &operator*() const { return IntStack.top(); }
  inline Interval *operator->() const { return &(operator*()); }

  inline _Self& operator++() {  // Preincrement
    do {
      // All of the intervals on the stack have been visited.  Try visiting their
      // successors now.
      Interval           &CurInt = IntStack.top().first;
      Interval::iterator &SuccIt = IntStack.top().second,End = succ_end(&CurInt);

      for (; SuccIt != End; ++SuccIt)    // Loop over all interval successors
	if (ProcessInterval(*SuccIt))    // Found a new interval!
	  return *this;                  // Use it!

      // We ran out of successors for this interval... pop off the stack
      IntStack.pop();
    } while (!IntStack.empty());

    return *this; 
  }
  inline _Self operator++(int) { // Postincrement
    _Self tmp = *this; ++*this; return tmp; 
  }

private:
  // ProcessInterval - This method is used during the construction of the 
  // interval graph.  It walks through the source graph, recursively creating
  // an interval per invokation until the entire graph is covered.  This uses
  // the ProcessNode method to add all of the nodes to the interval.
  //
  // This method is templated because it may operate on two different source
  // graphs: a basic block graph, or a preexisting interval graph.
  //
  bool ProcessInterval(NodeTy *Node) {
    BasicBlock *Header = getNodeHeader(Node);
    if (Visited.count(Header)) return false;

    Interval Int(Header);
    Visited.insert(Header);   // The header has now been visited!

    // Check all of our successors to see if they are in the interval...
    for (typename NodeTy::succ_iterator I = succ_begin(Node), E = succ_end(Node);
	 I != E; ++I)
      ProcessNode(&Int, getSourceGraphNode(OrigContainer, *I));

    IntStack.push(make_pair(Int, succ_begin(&Int)));
    return true;
  }
  
  // ProcessNode - This method is called by ProcessInterval to add nodes to the
  // interval being constructed, and it is also called recursively as it walks
  // the source graph.  A node is added to the current interval only if all of
  // its predecessors are already in the graph.  This also takes care of keeping
  // the successor set of an interval up to date.
  //
  // This method is templated because it may operate on two different source
  // graphs: a basic block graph, or a preexisting interval graph.
  //
  void ProcessNode(Interval *Int, NodeTy *Node) {
    assert(Int && "Null interval == bad!");
    assert(Node && "Null Node == bad!");
  
    BasicBlock *NodeHeader = getNodeHeader(Node);

    if (Visited.count(NodeHeader)) {     // Node already been visited?
      if (Int->contains(NodeHeader)) {   // Already in this interval...
	return;
      } else {                           // In another interval, add as successor
	if (!Int->isSuccessor(NodeHeader)) // Add only if not already in set
	  Int->Successors.push_back(NodeHeader);
      }
    } else {                             // Otherwise, not in interval yet
      for (typename NodeTy::pred_iterator I = pred_begin(Node), 
	                                  E = pred_end(Node); I != E; ++I) {
	if (!Int->contains(*I)) {        // If pred not in interval, we can't be
	  if (!Int->isSuccessor(NodeHeader)) // Add only if not already in set
	    Int->Successors.push_back(NodeHeader);
	  return;                        // See you later
	}
      }

      // If we get here, then all of the predecessors of BB are in the interval
      // already.  In this case, we must add BB to the interval!
      addNodeToInterval(Int, Node);
      Visited.insert(NodeHeader);     // The node has now been visited!
    
      if (Int->isSuccessor(NodeHeader)) {
	// If we were in the successor list from before... remove from succ list
	Int->Successors.erase(remove(Int->Successors.begin(),
				     Int->Successors.end(), NodeHeader), 
			      Int->Successors.end());
      }
    
      // Now that we have discovered that Node is in the interval, perhaps some
      // of its successors are as well?
      for (typename NodeTy::succ_iterator It = succ_begin(Node), 
	     End = succ_end(Node); It != End; ++It)
	ProcessNode(Int, getSourceGraphNode(OrigContainer, *It));
    }
  }
};

typedef IntervalIterator<BasicBlock, Method> method_interval_iterator;
typedef IntervalIterator<Interval, IntervalPartition> interval_part_interval_iterator;


inline method_interval_iterator intervals_begin(Method *M) {
  return method_interval_iterator(M);
}
inline method_interval_iterator intervals_end(Method *M) {
  return method_interval_iterator();
}

inline interval_part_interval_iterator intervals_begin(IntervalPartition &IP) {
  return interval_part_interval_iterator(IP);
}

inline interval_part_interval_iterator intervals_end(IntervalPartition &IP) {
  return interval_part_interval_iterator();
}

}    // End namespace cfg

#endif
