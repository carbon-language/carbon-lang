//===- llvm/Analysis/Intervals.h - Interval partition Calculation-*- C++ -*--=//
//
// This file contains the declaration of the cfg::IntervalPartition class, which
// calculates and represents the interval partition of a method, or a
// preexisting interval partition.
//
// In this way, the interval partition may be used to reduce a flow graph down
// to its degenerate single node interval partition (unless it is irreducible).
//
// TODO: The IntervalPartition class should take a bool parameter that tells
// whether it should add the "tails" of an interval to an interval itself or if
// they should be represented as distinct intervals.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERVALS_H
#define LLVM_INTERVALS_H

#include "llvm/Method.h"
#include "llvm/CFG.h"
#include <vector>
#include <map>
#include <stack>
#include <set>
#include <algorithm>

class Method;
class BasicBlock;

namespace cfg {

class IntervalPartition;

//===----------------------------------------------------------------------===//
//
// Interval Class - An Interval is a set of nodes defined such that every node
// in the interval has all of its predecessors in the interval (except for the
// header)
//
class Interval {
  friend class IntervalPartition;

  // HeaderNode - The header BasicBlock, which dominates all BasicBlocks in this
  // interval.  Also, any loops in this interval must go through the HeaderNode.
  //
  BasicBlock *HeaderNode;
public:
  typedef vector<BasicBlock*>::iterator succ_iterator;
  typedef vector<BasicBlock*>::iterator pred_iterator;
  typedef vector<BasicBlock*>::iterator node_iterator;

  inline BasicBlock *getHeaderNode() const { return HeaderNode; }

  // Nodes - The basic blocks in this interval.
  //
  vector<BasicBlock*> Nodes;

  // Successors - List of BasicBlocks that are reachable directly from nodes in
  // this interval, but are not in the interval themselves.
  // These nodes neccesarily must be header nodes for other intervals.
  //
  vector<BasicBlock*> Successors;

  // Predecessors - List of BasicBlocks that have this Interval's header block
  // as one of their successors.
  //
  vector<BasicBlock*> Predecessors;

  // contains - Find out if a basic block is in this interval
  inline bool contains(BasicBlock *BB) const {
    return find(Nodes.begin(), Nodes.end(), BB) != Nodes.end();
  }

  // isSuccessor - find out if a basic block is a successor of this Interval
  inline bool isSuccessor(BasicBlock *BB) const {
    return find(Successors.begin(), Successors.end(), BB) != Successors.end();
  }

  // isLoop - Find out if there is a back edge in this interval...
  bool isLoop() const;

  //private:           // Only accessable by IntervalPartition class
  inline Interval(BasicBlock *Header) : HeaderNode(Header) {
    Nodes.push_back(Header);
  }
};


// succ_begin/succ_end - define global functions so that Intervals may be used
// just like BasicBlocks can with the succ_* functions, and *::succ_iterator.
//
inline Interval::succ_iterator succ_begin(Interval *I) { 
  return I->Successors.begin();
}
inline Interval::succ_iterator succ_end(Interval *I) { 
  return I->Successors.end();
}

// pred_begin/pred_end - define global functions so that Intervals may be used
// just like BasicBlocks can with the pred_* functions, and *::pred_iterator.
//
inline Interval::pred_iterator pred_begin(Interval *I) { 
  return I->Predecessors.begin();
}
inline Interval::pred_iterator pred_end(Interval *I) { 
  return I->Predecessors.end();
}


//===----------------------------------------------------------------------===//
//                             IntervalIterator
//
// TODO: Provide an interval iterator that codifies the internals of 
// IntervalPartition.  Inside, it would have a stack of Interval*'s, and would
// walk the interval partition in depth first order.  IntervalPartition would
// then be a client of this iterator.  The iterator should work on Method*,
// const Method*, IntervalPartition*, and const IntervalPartition*.
//

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


//===----------------------------------------------------------------------===//
//
// IntervalPartition - This class builds and holds an "interval partition" for
// a method.  This partition divides the control flow graph into a set of
// maximal intervals, as defined with the properties above.  Intuitively, a
// BasicBlock is a (possibly nonexistent) loop with a "tail" of non looping
// nodes following it.
//
class IntervalPartition {
  typedef map<BasicBlock*, Interval*> IntervalMapTy;
  IntervalMapTy IntervalMap;

  typedef vector<Interval*> IntervalListTy;
  IntervalListTy IntervalList;
  Interval *RootInterval;

public:
  typedef IntervalListTy::iterator iterator;

public:
  // IntervalPartition ctor - Build the partition for the specified method
  IntervalPartition(Method *M);

  // IntervalPartition ctor - Build a reduced interval partition from an
  // existing interval graph.  This takes an additional boolean parameter to
  // distinguish it from a copy constructor.  Always pass in false for now.
  //
  IntervalPartition(IntervalPartition &I, bool);

  // Destructor - Free memory
  ~IntervalPartition();

  // getRootInterval() - Return the root interval that contains the starting
  // block of the method.
  inline Interval *getRootInterval() { return RootInterval; }

  // isDegeneratePartition() - Returns true if the interval partition contains
  // a single interval, and thus cannot be simplified anymore.
  bool isDegeneratePartition() { return size() == 1; }

  // TODO: isIrreducible - look for triangle graph.

  // getBlockInterval - Return the interval that a basic block exists in.
  inline Interval *getBlockInterval(BasicBlock *BB) {
    IntervalMapTy::iterator I = IntervalMap.find(BB);
    return I != IntervalMap.end() ? I->second : 0;
  }

  // Iterators to iterate over all of the intervals in the method
  inline iterator begin() { return IntervalList.begin(); }
  inline iterator end()   { return IntervalList.end(); }
  inline unsigned size()  { return IntervalList.size(); }

private:
  // ProcessInterval - This method is used during the construction of the 
  // interval graph.  It walks through the source graph, recursively creating
  // an interval per invokation until the entire graph is covered.  This uses
  // the ProcessNode method to add all of the nodes to the interval.
  //
  // This method is templated because it may operate on two different source
  // graphs: a basic block graph, or a preexisting interval graph.
  //
  template<class NodeTy, class OrigContainer>
  void ProcessInterval(NodeTy *Node, OrigContainer *OC);

  // ProcessNode - This method is called by ProcessInterval to add nodes to the
  // interval being constructed, and it is also called recursively as it walks
  // the source graph.  A node is added to the current interval only if all of
  // its predecessors are already in the graph.  This also takes care of keeping
  // the successor set of an interval up to date.
  //
  // This method is templated because it may operate on two different source
  // graphs: a basic block graph, or a preexisting interval graph.
  //
  template<class NodeTy, class OrigContainer>
  void ProcessNode(Interval *Int, NodeTy *Node, OrigContainer *OC);

  // addNodeToInterval - This method exists to assist the generic ProcessNode
  // with the task of adding a node to the new interval, depending on the 
  // type of the source node.  In the case of a CFG source graph (BasicBlock 
  // case), the BasicBlock itself is added to the interval.  In the case of
  // an IntervalPartition source graph (Interval case), all of the member
  // BasicBlocks are added to the interval.
  //
  inline void addNodeToInterval(Interval *Int, Interval *I);
  inline void addNodeToInterval(Interval *Int, BasicBlock *BB);

  // updatePredecessors - Interval generation only sets the successor fields of
  // the interval data structures.  After interval generation is complete,
  // run through all of the intervals and propogate successor info as
  // predecessor info.
  //
  void updatePredecessors(Interval *Int);
};



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




typedef IntervalIterator<BasicBlock, Method> method_interval_iterator;


method_interval_iterator intervals_begin(Method *M) {
  return method_interval_iterator(M);
}
method_interval_iterator intervals_end(Method *M) {
  return method_interval_iterator();
}

}    // End namespace cfg

#endif
