//===-- SchedGraphCommon.h - Scheduling Base Graph --------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// A common graph class that is based on the SSA graph. It includes
// extra dependencies that are caused by machine resources.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDGRAPHCOMMON_H
#define LLVM_CODEGEN_SCHEDGRAPHCOMMON_H

#include "llvm/Value.h"
#include "Support/iterator"
#include <vector>

namespace llvm {

class SchedGraphEdge;
class SchedGraphNode;

/******************** Exported Data Types and Constants ********************/

typedef int ResourceId;
const ResourceId InvalidRID        = -1;
const ResourceId MachineCCRegsRID  = -2; // use +ve numbers for actual regs
const ResourceId MachineIntRegsRID = -3; // use +ve numbers for actual regs
const ResourceId MachineFPRegsRID  = -4; // use +ve numbers for actual regs


//*********************** Public Class Declarations ************************/
class SchedGraphNodeCommon {
protected:
  unsigned ID;
  std::vector<SchedGraphEdge*> inEdges;
  std::vector<SchedGraphEdge*> outEdges;
  int latency;
  int origIndexInBB;            // original position of instr in BB

public:
  typedef std::vector<SchedGraphEdge*>::iterator iterator;
  typedef std::vector<SchedGraphEdge*>::const_iterator const_iterator;
  typedef std::vector<SchedGraphEdge*>::reverse_iterator reverse_iterator;
  typedef std::vector<SchedGraphEdge*>::const_reverse_iterator const_reverse_iterator;
  
  // Accessor methods
  unsigned getNodeId() const { return ID; }
  int getLatency() const { return latency; }
  unsigned getNumInEdges() const { return inEdges.size(); }
  unsigned getNumOutEdges() const { return outEdges.size(); }
  int getOrigIndexInBB() const { return origIndexInBB; }

  // Iterators
  iterator beginInEdges() { return inEdges.begin(); }
  iterator endInEdges()	 { return inEdges.end(); }
  iterator beginOutEdges() { return outEdges.begin(); }
  iterator endOutEdges() { return outEdges.end(); }
  
  const_iterator beginInEdges() const { return inEdges.begin(); }
  const_iterator endInEdges() const { return inEdges.end(); }
  const_iterator beginOutEdges() const { return outEdges.begin(); }
  const_iterator endOutEdges() const { return outEdges.end(); }

  void dump(int indent=0) const;

  // Debugging support
  virtual void print(std::ostream &os) const = 0;
  
protected:
  friend class SchedGraphCommon;
  friend class SchedGraphEdge;		// give access for adding edges
  
  
  // disable default constructor and provide a ctor for single-block graphs
  SchedGraphNodeCommon();	// DO NOT IMPLEMENT
  
  inline SchedGraphNodeCommon(unsigned Id, int index) : ID(Id), latency(0), 
							origIndexInBB(index) {}
  inline SchedGraphNodeCommon(unsigned Id, int late, int index) : ID(Id), latency(late), origIndexInBB(index) {}
  
  virtual ~SchedGraphNodeCommon();
  
  //Functions to add and remove edges
  inline void addInEdge(SchedGraphEdge* edge) { inEdges.push_back(edge); }
  inline void addOutEdge(SchedGraphEdge* edge) { outEdges.push_back(edge); }
  void removeInEdge(const SchedGraphEdge* edge);
  void removeOutEdge(const SchedGraphEdge* edge);
  
};

// ostream << operator for SchedGraphNode class
inline std::ostream &operator<<(std::ostream &os, 
				const SchedGraphNodeCommon &node) {
  node.print(os);
  return os;
}




//
// SchedGraphEdge - Edge class to represent dependencies
//
class SchedGraphEdge {
public:
  enum SchedGraphEdgeDepType {
    CtrlDep, MemoryDep, ValueDep, MachineRegister, MachineResource
  };
  enum DataDepOrderType {
    TrueDep = 0x1, AntiDep=0x2, OutputDep=0x4, NonDataDep=0x8
  };
  
protected:
  SchedGraphNodeCommon*	src;
  SchedGraphNodeCommon*	sink;
  SchedGraphEdgeDepType depType;
  unsigned int depOrderType;
  int minDelay; // cached latency (assumes fixed target arch)
  int iteDiff;
  
  union {
    const Value* val;
    int          machineRegNum;
    ResourceId   resourceId;
  };

public:	
  // For all constructors, if minDelay is unspecified, minDelay is
  // set to _src->getLatency().
  
  // constructor for CtrlDep or MemoryDep edges, selected by 3rd argument
  SchedGraphEdge(SchedGraphNodeCommon* _src, SchedGraphNodeCommon* _sink,
		 SchedGraphEdgeDepType _depType, unsigned int _depOrderType,
		 int _minDelay = -1);
  
  // constructor for explicit value dependence (may be true/anti/output)
  SchedGraphEdge(SchedGraphNodeCommon* _src, SchedGraphNodeCommon* _sink,
		 const Value* _val, unsigned int _depOrderType,
		 int _minDelay = -1);
  
  // constructor for machine register dependence
  SchedGraphEdge(SchedGraphNodeCommon* _src,SchedGraphNodeCommon* _sink,
		 unsigned int _regNum, unsigned int _depOrderType,
		 int _minDelay = -1);
  
  // constructor for any other machine resource dependences.
  // DataDepOrderType is always NonDataDep.  It it not an argument to
  // avoid overloading ambiguity with previous constructor.
  SchedGraphEdge(SchedGraphNodeCommon* _src, SchedGraphNodeCommon* _sink,
		 ResourceId _resourceId, int _minDelay = -1);
  
  ~SchedGraphEdge() {}
  
  SchedGraphNodeCommon*	getSrc() const { return src; }
  SchedGraphNodeCommon*	getSink() const { return sink; }
  int getMinDelay() const { return minDelay; }
  SchedGraphEdgeDepType getDepType() const { return depType; }
  
  const Value* getValue() const {
    assert(depType == ValueDep); return val;
  }

  int getMachineReg() const {
    assert(depType == MachineRegister); return machineRegNum;
  }

  int getResourceId() const {
    assert(depType == MachineResource); return resourceId;
  }

  void setIteDiff(int _iteDiff) {
    iteDiff = _iteDiff;
  }

  int getIteDiff() {
    return iteDiff;
  }
  
public:
  // Debugging support
  void print(std::ostream &os) const;
  void dump(int indent=0) const;
    
private:
  // disable default ctor
  SchedGraphEdge();	// DO NOT IMPLEMENT
};

// ostream << operator for SchedGraphNode class
inline std::ostream &operator<<(std::ostream &os, const SchedGraphEdge &edge) {
  edge.print(os);
  return os;
}

class SchedGraphCommon {
  
protected:
  SchedGraphNodeCommon* graphRoot;     // the root and leaf are not inserted
  SchedGraphNodeCommon* graphLeaf;     //  in the hash_map (see getNumNodes())

public:
  //
  // Accessor methods
  //
  SchedGraphNodeCommon* getRoot() const { return graphRoot; }
  SchedGraphNodeCommon* getLeaf() const { return graphLeaf; } 
 
  //
  // Delete nodes or edges from the graph.
  // 
  void eraseNode(SchedGraphNodeCommon* node);
  void eraseIncomingEdges(SchedGraphNodeCommon* node, bool addDummyEdges = true);
  void eraseOutgoingEdges(SchedGraphNodeCommon* node, bool addDummyEdges = true);
  void eraseIncidentEdges(SchedGraphNodeCommon* node, bool addDummyEdges = true);
  
  SchedGraphCommon() {}
  ~SchedGraphCommon();
};


//********************** Sched Graph Iterators *****************************/

// Ok to make it a template because it shd get instantiated at most twice:
// for <SchedGraphNode, SchedGraphNode::iterator> and
// for <const SchedGraphNode, SchedGraphNode::const_iterator>.
// 
template <class _NodeType, class _EdgeType, class _EdgeIter>
class SGPredIterator: public bidirectional_iterator<_NodeType, ptrdiff_t> {
protected:
  _EdgeIter oi;
public:
  typedef SGPredIterator<_NodeType, _EdgeType, _EdgeIter> _Self;
  
  inline SGPredIterator(_EdgeIter startEdge) : oi(startEdge) {}
  
  inline bool operator==(const _Self& x) const { return oi == x.oi; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }
  
  // operator*() differs for pred or succ iterator
  inline _NodeType* operator*() const { return (_NodeType*)(*oi)->getSrc(); }
  inline _NodeType* operator->() const { return operator*(); }
  
  inline _EdgeType* getEdge() const { return *(oi); }
  
  inline _Self &operator++() { ++oi; return *this; }    // Preincrement
  inline _Self operator++(int) {                      	// Postincrement
    _Self tmp(*this); ++*this; return tmp; 
  }
  
  inline _Self &operator--() { --oi; return *this; }    // Predecrement
  inline _Self operator--(int) {                       	// Postdecrement
    _Self tmp = *this; --*this; return tmp;
  }
};

template <class _NodeType, class _EdgeType, class _EdgeIter>
class SGSuccIterator : public bidirectional_iterator<_NodeType, ptrdiff_t> {
protected:
  _EdgeIter oi;
public:
  typedef SGSuccIterator<_NodeType, _EdgeType, _EdgeIter> _Self;
  
  inline SGSuccIterator(_EdgeIter startEdge) : oi(startEdge) {}
  
  inline bool operator==(const _Self& x) const { return oi == x.oi; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }
  
  inline _NodeType* operator*() const { return (_NodeType*)(*oi)->getSink(); }
  inline _NodeType* operator->() const { return operator*(); }
  
  inline _EdgeType* getEdge() const { return *(oi); }
  
  inline _Self &operator++() { ++oi; return *this; }    // Preincrement
  inline _Self operator++(int) {                      	// Postincrement
    _Self tmp(*this); ++*this; return tmp; 
  }
  
  inline _Self &operator--() { --oi; return *this; }    // Predecrement
  inline _Self operator--(int) {                       	// Postdecrement
    _Self tmp = *this; --*this; return tmp;
  }
};
} // End llvm namespace

#endif
