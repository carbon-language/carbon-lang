//===- DependenceGraph.h - Dependence graph for a function ------*- C++ -*-===//
//
// This file provides an explicit representation for the dependence graph
// of a function, with one node per instruction and one edge per dependence.
// Dependences include both data and control dependences.
// 
// Each dep. graph node (class DepGraphNode) keeps lists of incoming and
// outgoing dependence edges.
// 
// Each dep. graph edge (class Dependence) keeps a pointer to one end-point
// of the dependence.  This saves space and is important because dep. graphs
// can grow quickly.  It works just fine because the standard idiom is to
// start with a known node and enumerate the dependences to or from that node.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DEPENDENCEGRAPH_H
#define LLVM_ANALYSIS_DEPENDENCEGRAPH_H

#include "Support/hash_map"
#include <iosfwd>
#include <vector>
#include <utility>
#include <cassert>

class Instruction;
class Function;
class Dependence;
class DepGraphNode;
class DependenceGraph;


//----------------------------------------------------------------------------
// enum DependenceType: The standard data dependence types.
//----------------------------------------------------------------------------

enum DependenceType {
  NoDependence       = 0x0,
  TrueDependence     = 0x1,
  AntiDependence     = 0x2,
  OutputDependence   = 0x4,
  ControlDependence  = 0x8,         // from a terminator to some other instr.
  IncomingFlag       = 0x10         // is this an incoming or outgoing dep?
};


//----------------------------------------------------------------------------
// class Dependence:
// 
// A representation of a simple (non-loop-related) dependence.
//----------------------------------------------------------------------------

class Dependence {
  DepGraphNode*   toOrFromNode;
  unsigned char  depType;

public:
  Dependence(DepGraphNode* toOrFromN, DependenceType type, bool isIncoming)
    : toOrFromNode(toOrFromN),
      depType(type | (isIncoming? IncomingFlag : 0x0)) { }

  /* copy ctor*/ Dependence     (const Dependence& D)
    : toOrFromNode(D.toOrFromNode),
      depType(D.depType) { }

  bool operator==(const Dependence& D) const {
    return toOrFromNode == D.toOrFromNode && depType == D.depType;
  }

  /// Get information about the type of dependence.
  /// 
  unsigned getDepType() const {
    return depType;
  }

  /// Get source or sink depending on what type of node this is!
  /// 
  DepGraphNode*  getSrc() {
    assert(depType & IncomingFlag); return toOrFromNode;
  }
  const DepGraphNode*  getSrc() const {
    assert(depType & IncomingFlag); return toOrFromNode;
  }

  DepGraphNode*  getSink() {
    assert(! (depType & IncomingFlag)); return toOrFromNode;
  }
  const DepGraphNode*  getSink() const {
    assert(! (depType & IncomingFlag)); return toOrFromNode;
  }

  /// Debugging support methods
  /// 
  void print(std::ostream &O) const;

  // Default constructor: Do not use directly except for graph builder code
  // 
  /*ctor*/ Dependence() : toOrFromNode(NULL), depType(NoDependence) { }
};


#ifdef SUPPORTING_LOOP_DEPENDENCES
struct LoopDependence: public Dependence {
  DependenceDirection dir;
  int                 distance;
  short               level;
  LoopInfo*           enclosingLoop;
};
#endif


//----------------------------------------------------------------------------
// class DepGraphNode:
// 
// A representation of a single node in a dependence graph, corresponding
// to a single instruction.
//----------------------------------------------------------------------------

class DepGraphNode {
  Instruction*  instr;
  std::vector<Dependence>  inDeps;
  std::vector<Dependence>  outDeps;
  friend class DependenceGraph;
  
  typedef std::vector<Dependence>::      iterator       iterator;
  typedef std::vector<Dependence>::const_iterator const_iterator;

        iterator           inDepBegin()         { return inDeps.begin(); }
  const_iterator           inDepBegin()   const { return inDeps.begin(); }
        iterator           inDepEnd()           { return inDeps.end(); }
  const_iterator           inDepEnd()     const { return inDeps.end(); }
  
        iterator           outDepBegin()        { return outDeps.begin(); }
  const_iterator           outDepBegin()  const { return outDeps.begin(); }
        iterator           outDepEnd()          { return outDeps.end(); }
  const_iterator           outDepEnd()    const { return outDeps.end(); }

public:

  DepGraphNode(Instruction& I) : instr(&I) { }

        Instruction&       getInstr()           { return *instr; }
  const Instruction&       getInstr()     const { return *instr; }

  /// Debugging support methods
  /// 
  void print(std::ostream &O) const;
};


//----------------------------------------------------------------------------
// class DependenceGraph:
// 
// A representation of a dependence graph for a procedure.
// The primary query operation here is to look up a DepGraphNode for
// a particular instruction, and then use the in/out dependence iterators
// for the node.
//----------------------------------------------------------------------------

class DependenceGraph {
  DependenceGraph(const DependenceGraph&); // DO NOT IMPLEMENT
  void operator=(const DependenceGraph&);  // DO NOT IMPLEMENT

  typedef hash_map<Instruction*, DepGraphNode*> DepNodeMapType;
  typedef DepNodeMapType::      iterator       map_iterator;
  typedef DepNodeMapType::const_iterator const_map_iterator;

  DepNodeMapType depNodeMap;

  inline DepGraphNode* getNodeInternal(Instruction& inst,
                                       bool  createIfMissing = false) {
    map_iterator I = depNodeMap.find(&inst);
    if (I == depNodeMap.end())
      return (!createIfMissing)? NULL :
        depNodeMap.insert(
            std::make_pair(&inst, new DepGraphNode(inst))).first->second;
    else
      return I->second;
  }

public:
  typedef std::vector<Dependence>::      iterator       iterator;
  typedef std::vector<Dependence>::const_iterator const_iterator;

public:
  DependenceGraph() { }
  ~DependenceGraph();

  /// Get the graph node for an instruction.  There will be one if and
  /// only if there are any dependences incident on this instruction.
  /// If there is none, these methods will return NULL.
  /// 
  DepGraphNode* getNode(Instruction& inst, bool createIfMissing = false) {
    return getNodeInternal(inst, createIfMissing);
  }
  const DepGraphNode* getNode(const Instruction& inst) const {
    return const_cast<DependenceGraph*>(this)
      ->getNodeInternal(const_cast<Instruction&>(inst));
  }

  iterator inDepBegin(DepGraphNode& T) {
    return T.inDeps.begin();
  }
  const_iterator inDepBegin (const DepGraphNode& T) const {
    return T.inDeps.begin();
  }

  iterator inDepEnd(DepGraphNode& T) {
    return T.inDeps.end();
  }
  const_iterator inDepEnd(const DepGraphNode& T) const {
    return T.inDeps.end();
  }

  iterator outDepBegin(DepGraphNode& F) {
    return F.outDeps.begin();
  }
  const_iterator outDepBegin(const DepGraphNode& F) const {
    return F.outDeps.begin();
  }

  iterator outDepEnd(DepGraphNode& F) {
    return F.outDeps.end();
  }
  const_iterator outDepEnd(const DepGraphNode& F) const {
    return F.outDeps.end();
  }

  /// Debugging support methods
  /// 
  void print(const Function& func, std::ostream &O) const;

public:
  /// Functions for adding and modifying the dependence graph.
  /// These should to be used only by dependence analysis implementations.
  void AddSimpleDependence(Instruction& fromI,
                           Instruction& toI,
                           DependenceType depType) {
    DepGraphNode* fromNode = getNodeInternal(fromI, /*create*/ true);
    DepGraphNode* toNode   = getNodeInternal(toI,   /*create*/ true);
    fromNode->outDeps.push_back(Dependence(toNode, depType, false));
    toNode->  inDeps. push_back(Dependence(fromNode, depType, true));
  }

#ifdef SUPPORTING_LOOP_DEPENDENCES
  /// This interface is a placeholder to show what information is needed.
  /// It will probably change when it starts being used.
  void AddLoopDependence(Instruction&  fromI,
                         Instruction&  toI,
                         DependenceType      depType,
                         DependenceDirection dir,
                         int                 distance,
                         short               level,
                         LoopInfo*           enclosingLoop);
#endif // SUPPORTING_LOOP_DEPENDENCES
};

//===----------------------------------------------------------------------===//

#endif
