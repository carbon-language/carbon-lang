//===- Intervals.cpp - Interval partition Calculation ------------*- C++ -*--=//
//
// This file contains the declaration of the cfg::IntervalPartition class, which
// calculates and represent the interval partition of a method.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Intervals.h"
#include "llvm/Method.h"
#include "llvm/BasicBlock.h"
#include "llvm/CFG.h"

using namespace cfg;

//===----------------------------------------------------------------------===//
// Interval Implementation
//===----------------------------------------------------------------------===//

// isLoop - Find out if there is a back edge in this interval...
//
bool Interval::isLoop() const {
  // There is a loop in this interval iff one of the predecessors of the header
  // node lives in the interval.
  for (BasicBlock::pred_iterator I = pred_begin(HeaderNode), 
                                 E = pred_end(HeaderNode); I != E; ++I) {
    if (contains(*I)) return true;
  }
  return false;
}


//===----------------------------------------------------------------------===//
// IntervalPartition Implementation
//===----------------------------------------------------------------------===//

template <class T> static inline void deleter(T *Ptr) { delete Ptr; }

// Destructor - Free memory
IntervalPartition::~IntervalPartition() {
  for_each(begin(), end(), deleter<cfg::Interval>);
}


// getNodeHeader - Given a source graph node and the source graph, return the 
// BasicBlock that is the header node.  This is the opposite of
// getSourceGraphNode.
//
inline static BasicBlock *getNodeHeader(BasicBlock *BB) { return BB; }
inline static BasicBlock *getNodeHeader(Interval *I) { return I->HeaderNode; }


// getSourceGraphNode - Given a BasicBlock and the source graph, return the 
// source graph node that corresponds to the BasicBlock.  This is the opposite
// of getNodeHeader.
//
inline static BasicBlock *getSourceGraphNode(Method *, BasicBlock *BB) {
  return BB; 
}
inline static Interval *getSourceGraphNode(IntervalPartition *IP, 
					   BasicBlock *BB) { 
  return IP->getBlockInterval(BB);
}


// addNodeToInterval - This method exists to assist the generic ProcessNode
// with the task of adding a node to the new interval, depending on the 
// type of the source node.  In the case of a CFG source graph (BasicBlock 
// case), the BasicBlock itself is added to the interval.
//
inline void IntervalPartition::addNodeToInterval(Interval *Int, BasicBlock *BB){
  Int->Nodes.push_back(BB);
  IntervalMap.insert(make_pair(BB, Int));
}

// addNodeToInterval - This method exists to assist the generic ProcessNode
// with the task of adding a node to the new interval, depending on the 
// type of the source node.  In the case of a CFG source graph (BasicBlock 
// case), the BasicBlock itself is added to the interval.  In the case of
// an IntervalPartition source graph (Interval case), all of the member
// BasicBlocks are added to the interval.
//
inline void IntervalPartition::addNodeToInterval(Interval *Int, Interval *I) {
  // Add all of the nodes in I as new nodes in Int.
  copy(I->Nodes.begin(), I->Nodes.end(), back_inserter(Int->Nodes));

  // Add mappings for all of the basic blocks in I to the IntervalPartition
  for (Interval::node_iterator It = I->Nodes.begin(), End = I->Nodes.end();
       It != End; ++It)
    IntervalMap.insert(make_pair(*It, Int));
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
template<class NodeTy, class OrigContainer>
void IntervalPartition::ProcessNode(Interval *Int, 
				    NodeTy *Node, OrigContainer *OC) {
  assert(Int && "Null interval == bad!");
  assert(Node && "Null Node == bad!");
  
  BasicBlock *NodeHeader = getNodeHeader(Node);
  Interval *CurInt = getBlockInterval(NodeHeader);
  if (CurInt == Int) {                  // Already in this interval...
    return;
  } else if (CurInt != 0) {             // In another interval, add as successor
    if (!Int->isSuccessor(NodeHeader))  // Add only if not already in set
      Int->Successors.push_back(NodeHeader);
  } else {                              // Otherwise, not in interval yet
    for (typename NodeTy::pred_iterator I = pred_begin(Node), 
                                        E = pred_end(Node); I != E; ++I) {
      if (!Int->contains(*I)) {         // If pred not in interval, we can't be
	if (!Int->isSuccessor(NodeHeader)) // Add only if not already in set
	  Int->Successors.push_back(NodeHeader);
	return;                         // See you later
      }
    }
    
    // If we get here, then all of the predecessors of BB are in the interval
    // already.  In this case, we must add BB to the interval!
    addNodeToInterval(Int, Node);
    
    if (Int->isSuccessor(NodeHeader)) {
      // If we were in the successor list from before... remove from succ list
      Int->Successors.erase(remove(Int->Successors.begin(),
				   Int->Successors.end(), NodeHeader), 
			    Int->Successors.end());
    }
    
    // Now that we have discovered that Node is in the interval, perhaps some of
    // its successors are as well?
    for (typename NodeTy::succ_iterator It = succ_begin(Node), 
                                       End = succ_end(Node); It != End; ++It)
      ProcessNode(Int, getSourceGraphNode(OC, *It), OC);
  }
}


// ProcessInterval - This method is used during the construction of the 
// interval graph.  It walks through the source graph, recursively creating
// an interval per invokation until the entire graph is covered.  This uses
// the ProcessNode method to add all of the nodes to the interval.
//
// This method is templated because it may operate on two different source
// graphs: a basic block graph, or a preexisting interval graph.
//
template<class NodeTy, class OrigContainer>
void IntervalPartition::ProcessInterval(NodeTy *Node, OrigContainer *OC) {
  BasicBlock *Header = getNodeHeader(Node);
  if (getBlockInterval(Header)) return;  // Interval already constructed?

  // Create a new interval and add the interval to our current set
  Interval *Int = new Interval(Header);
  IntervalList.push_back(Int);
  IntervalMap.insert(make_pair(Header, Int));

  // Check all of our successors to see if they are in the interval...
  for (typename NodeTy::succ_iterator I = succ_begin(Node), E = succ_end(Node); 
       I != E; ++I)
    ProcessNode(Int, getSourceGraphNode(OC, *I), OC);

  // Build all of the successor intervals of this interval now...
  for(Interval::succ_iterator I = Int->Successors.begin(), 
                              E = Int->Successors.end(); I != E; ++I) {
    ProcessInterval(getSourceGraphNode(OC, *I), OC);
  }
}



// updatePredecessors - Interval generation only sets the successor fields of
// the interval data structures.  After interval generation is complete,
// run through all of the intervals and propogate successor info as
// predecessor info.
//
void IntervalPartition::updatePredecessors(cfg::Interval *Int) {
  BasicBlock *Header = Int->HeaderNode;
  for (Interval::succ_iterator I = Int->Successors.begin(), 
	                       E = Int->Successors.end(); I != E; ++I)
    getBlockInterval(*I)->Predecessors.push_back(Header);
}



// IntervalPartition ctor - Build the first level interval partition for the
// specified method...
//
IntervalPartition::IntervalPartition(Method *M) {
  BasicBlock *MethodStart = M->getBasicBlocks().front();
  assert(MethodStart && "Cannot operate on prototypes!");

  ProcessInterval(MethodStart, M);
  RootInterval = getBlockInterval(MethodStart);

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for(iterator I = begin(), E = end(); I != E; ++I)
    updatePredecessors(*I);
}


// IntervalPartition ctor - Build a reduced interval partition from an
// existing interval graph.  This takes an additional boolean parameter to
// distinguish it from a copy constructor.  Always pass in false for now.
//
IntervalPartition::IntervalPartition(IntervalPartition &I, bool) {
  Interval *MethodStart = I.getRootInterval();
  assert(MethodStart && "Cannot operate on empty IntervalPartitions!");

  ProcessInterval(MethodStart, &I);
  RootInterval = getBlockInterval(*MethodStart->Nodes.begin());

  // Now that we know all of the successor information, propogate this to the
  // predecessors for each block...
  for(iterator I = begin(), E = end(); I != E; ++I)
    updatePredecessors(*I);
}
