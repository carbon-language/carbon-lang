/* -*-C++-*-
 ****************************************************************************
 * File:
 *	SchedGraph.h
 * 
 * Purpose:
 *	Scheduling graph based on SSA graph plus extra dependence edges
 *	capturing dependences due to machine resources (machine registers,
 *	CC registers, and any others).
 * 
 * Strategy:
 *	This graph tries to leverage the SSA graph as much as possible,
 *	but captures the extra dependences through a common interface.
 * 
 * History:
 *	7/20/01	 -  Vikram Adve  -  Created
 ***************************************************************************/

#ifndef LLVM_CODEGEN_SCHEDGRAPH_H
#define LLVM_CODEGEN_SCHEDGRAPH_H

#include "llvm/CFGdecls.h"			// just for graph iterators
#include "llvm/Support/NonCopyable.h"
#include "llvm/Support/HashExtras.h"
#include <hash_map>

class Value;
class Instruction;
class BasicBlock;
class Method;
class TargetMachine;
class SchedGraphEdge; 
class SchedGraphNode; 
class SchedGraph; 
class NodeToRegRefMap;
class MachineInstr;

/******************** Exported Data Types and Constants ********************/

typedef int ResourceId;
const ResourceId InvalidResourceId = -1;


//*********************** Public Class Declarations ************************/

class SchedGraphEdge: public NonCopyable {
public:
  enum SchedGraphEdgeDepType {
    CtrlDep, MemoryDep, DefUseDep, MachineRegister, MachineResource
  };
  enum DataDepOrderType {
    TrueDep, AntiDep, OutputDep, NonDataDep
  };
  
protected:
  SchedGraphNode*	src;
  SchedGraphNode*	sink;
  SchedGraphEdgeDepType depType;
  DataDepOrderType	depOrderType;
  int			minDelay; // cached latency (assumes fixed target arch)
  
  union {
    Value*	val;
    int		machineRegNum;
    ResourceId  resourceId;
  };
  
public:	
  // For all constructors, if minDelay is unspecified, minDelay is
  // set to _src->getLatency().
  // constructor for CtrlDep or MemoryDep edges, selected by 3rd argument
  /*ctor*/		SchedGraphEdge(SchedGraphNode* _src,
				       SchedGraphNode* _sink,
				       SchedGraphEdgeDepType _depType,
				       DataDepOrderType _depOrderType =TrueDep,
				       int _minDelay = -1);
  
  // constructor for explicit def-use or memory def-use edge
  /*ctor*/		SchedGraphEdge(SchedGraphNode* _src,
				       SchedGraphNode* _sink,
				       Value*          _val,
				       DataDepOrderType _depOrderType =TrueDep,
				       int _minDelay = -1);
  
  // constructor for machine register dependence
  /*ctor*/		SchedGraphEdge(SchedGraphNode* _src,
				       SchedGraphNode* _sink,
				       unsigned int    _regNum,
				       DataDepOrderType _depOrderType =TrueDep,
				       int _minDelay = -1);
  
  // constructor for any other machine resource dependences.
  // DataDepOrderType is always NonDataDep.  It it not an argument to
  // avoid overloading ambiguity with previous constructor.
  /*ctor*/		SchedGraphEdge(SchedGraphNode* _src,
				       SchedGraphNode* _sink,
				       ResourceId      _resourceId,
				       int _minDelay = -1);
  
  /*dtor*/		~SchedGraphEdge();
  
  SchedGraphNode*	getSrc		() const { return src; }
  SchedGraphNode*	getSink		() const { return sink; }
  int			getMinDelay	() const { return minDelay; }
  SchedGraphEdgeDepType getDepType	() const { return depType; }
  
  const Value*		getValue	() const {
    assert(depType == DefUseDep || depType == MemoryDep); return val;
  }
  int			getMachineReg	() const {
    assert(depType == MachineRegister); return machineRegNum;
  }
  int			getResourceId	() const {
    assert(depType == MachineResource); return resourceId;
  }
  
public:
  // 
  // Debugging support
  // 
  friend ostream& operator<<(ostream& os, const SchedGraphEdge& edge);
  
  void		dump	(int indent=0) const;
    
private:
  // disable default ctor
  /*ctor*/		SchedGraphEdge();	// DO NOT IMPLEMENT
};



class SchedGraphNode: public NonCopyable {
private:
  unsigned int nodeId;
  const Instruction* instr;
  const MachineInstr* minstr;
  vector<SchedGraphEdge*> inEdges;
  vector<SchedGraphEdge*> outEdges;
  int latency;
  
public:
  typedef vector<SchedGraphEdge*>::      iterator	        iterator;
  typedef vector<SchedGraphEdge*>::const_iterator         const_iterator;
  typedef vector<SchedGraphEdge*>::      reverse_iterator reverse_iterator;
  typedef vector<SchedGraphEdge*>::const_reverse_iterator const_reverse_iterator;
  
public:
  //
  // Accessor methods
  // 
  unsigned int		getNodeId	() const { return nodeId; }
  const Instruction*	getInstr	() const { return instr; }
  const MachineInstr*	getMachineInstr	() const { return minstr; }
  int			getLatency	() const { return latency; }
  unsigned int		getNumInEdges	() const { return inEdges.size(); }
  unsigned int		getNumOutEdges	() const { return outEdges.size(); }
  bool			isDummyNode	() const { return (minstr == NULL); }
  
  //
  // Iterators
  // 
  iterator		beginInEdges	()	 { return inEdges.begin(); }
  iterator		endInEdges	()	 { return inEdges.end(); }
  iterator		beginOutEdges	()	 { return outEdges.begin(); }
  iterator		endOutEdges	()	 { return outEdges.end(); }
  
  const_iterator	beginInEdges	() const { return inEdges.begin(); }
  const_iterator	endInEdges	() const { return inEdges.end(); }
  const_iterator	beginOutEdges	() const { return outEdges.begin(); }
  const_iterator	endOutEdges	() const { return outEdges.end(); }
  
public:
  //
  // Debugging support
  // 
  friend ostream& operator<<(ostream& os, const SchedGraphNode& node);
  
  void		dump	(int indent=0) const;
  
private:
  friend class SchedGraph;		// give access for ctor and dtor
  friend class SchedGraphEdge;		// give access for adding edges
  
  void			addInEdge	(SchedGraphEdge* edge);
  void			addOutEdge	(SchedGraphEdge* edge);
  
  void			removeInEdge	(const SchedGraphEdge* edge);
  void			removeOutEdge	(const SchedGraphEdge* edge);
  
  // disable default constructor and provide a ctor for single-block graphs
  /*ctor*/		SchedGraphNode();	// DO NOT IMPLEMENT
  /*ctor*/		SchedGraphNode	(unsigned int _nodeId,
					 const Instruction* _instr,
					 const MachineInstr* _minstr,
					 const TargetMachine& _target);
  /*dtor*/		~SchedGraphNode	();
};



class SchedGraph :
  public NonCopyable,
  private hash_map<const MachineInstr*, SchedGraphNode*>
{
private:
  vector<const BasicBlock*> bbVec;	// basic blocks included in the graph
  SchedGraphNode* graphRoot;		// the root and leaf are not inserted
  SchedGraphNode* graphLeaf;		//  in the hash_map (see getNumNodes())
  
public:
  typedef hash_map<const MachineInstr*, SchedGraphNode*>::iterator iterator;
  typedef hash_map<const MachineInstr*, SchedGraphNode*>::const_iterator const_iterator;
  
public:
  //
  // Accessor methods
  //
  const vector<const BasicBlock*>& getBasicBlocks() const { return bbVec; }
  const unsigned int		   getNumNodes()    const { return size()+2; }
  SchedGraphNode*		   getRoot()	    const { return graphRoot; }
  SchedGraphNode*		   getLeaf()	    const { return graphLeaf; }
  
  SchedGraphNode* getGraphNodeForInstr(const MachineInstr* minstr) const {
    const_iterator onePair = this->find(minstr);
    return (onePair != this->end())? (*onePair).second : NULL;
  }
  
  //
  // Delete nodes or edges from the graph.
  // 
  void		eraseNode		(SchedGraphNode* node);
  
  void		eraseIncomingEdges	(SchedGraphNode* node,
					 bool addDummyEdges = true);
  
  void		eraseOutgoingEdges	(SchedGraphNode* node,
					 bool addDummyEdges = true);
  
  void		eraseIncidentEdges	(SchedGraphNode* node,
					 bool addDummyEdges = true);
  
  //
  // Unordered iterators.
  // Return values is pair<const MachineIntr*,SchedGraphNode*>.
  //
  iterator	begin()	{
    return hash_map<const MachineInstr*, SchedGraphNode*>::begin();
  }
  iterator	end() {
    return hash_map<const MachineInstr*, SchedGraphNode*>::end();
  }
  const_iterator begin() const {
    return hash_map<const MachineInstr*, SchedGraphNode*>::begin();
  }
  const_iterator end()	const {
    return hash_map<const MachineInstr*, SchedGraphNode*>::end();
  }
  
  //
  // Ordered iterators.
  // Return values is pair<const MachineIntr*,SchedGraphNode*>.
  //
  // void			   postord_init();
  // postorder_iterator	   postord_begin();
  // postorder_iterator	   postord_end();
  // const_postorder_iterator postord_begin() const;
  // const_postorder_iterator postord_end() const;
  
  //
  // Debugging support
  // 
  void		dump	() const;
  
private:
  friend class SchedGraphSet;		// give access to ctor
  
  // disable default constructor and provide a ctor for single-block graphs
  /*ctor*/	SchedGraph		();	// DO NOT IMPLEMENT
  /*ctor*/	SchedGraph		(const BasicBlock* bb,
					 const TargetMachine& target);
  /*dtor*/	~SchedGraph		();
  
  inline void	noteGraphNodeForInstr	(const MachineInstr* minstr,
					 SchedGraphNode* node)
  {
    assert((*this)[minstr] == NULL);
    (*this)[minstr] = node;
  }

  //
  // Graph builder
  //
  void		buildGraph		(const TargetMachine& target);
  
  void		addEdgesForInstruction	(SchedGraphNode* node,
					 NodeToRegRefMap& regToRefVecMap,
					 const TargetMachine& target);
  
  void		addCDEdges		(const TerminatorInst* term,
					 const TargetMachine& target);
  
  void		addMemEdges	     (const vector<const Instruction*>& memVec,
				      const TargetMachine& target);
  
  void		addMachineRegEdges	(NodeToRegRefMap& regToRefVecMap,
					 const TargetMachine& target);
  
  void		addSSAEdge		(SchedGraphNode* node,
					 Value* val,
					 const TargetMachine& target);
  
  void		addDummyEdges		();
};


class SchedGraphSet :  
  public NonCopyable,
  private hash_map<const BasicBlock*, SchedGraph*>
{
private:
  const Method* method;
  
public:
  typedef hash_map<const BasicBlock*, SchedGraph*>::iterator iterator;
  typedef hash_map<const BasicBlock*, SchedGraph*>::const_iterator const_iterator;
  
public:
  /*ctor*/	SchedGraphSet		(const Method* _method,
					 const TargetMachine& target);
  /*dtor*/	~SchedGraphSet		();
  
  //
  // Accessors
  //
  SchedGraph*	getGraphForBasicBlock	(const BasicBlock* bb) const {
    const_iterator onePair = this->find(bb);
    return (onePair != this->end())? (*onePair).second : NULL;
  }
  
  //
  // Iterators
  //
  iterator	begin()	{
    return hash_map<const BasicBlock*, SchedGraph*>::begin();
  }
  iterator	end() {
    return hash_map<const BasicBlock*, SchedGraph*>::end();
  }
  const_iterator begin() const {
    return hash_map<const BasicBlock*, SchedGraph*>::begin();
  }
  const_iterator end()	const {
    return hash_map<const BasicBlock*, SchedGraph*>::end();
  }
  
  //
  // Debugging support
  // 
  void		dump	() const;
  
private:
  inline void	noteGraphForBlock(const BasicBlock* bb, SchedGraph* graph) {
    assert((*this)[bb] == NULL);
    (*this)[bb] = graph;
  }

  //
  // Graph builder
  //
  void		buildGraphsForMethod	(const Method *method,
					 const TargetMachine& target);
};


//********************** Sched Graph Iterators *****************************/

// Ok to make it a template because it shd get instantiated at most twice:
// for <SchedGraphNode, SchedGraphNode::iterator> and
// for <const SchedGraphNode, SchedGraphNode::const_iterator>.
// 
template <class _NodeType, class _EdgeType, class _EdgeIter>
class PredIterator: public std::bidirectional_iterator<_NodeType, ptrdiff_t> {
protected:
  _EdgeIter oi;
public:
  typedef PredIterator<_NodeType, _EdgeType, _EdgeIter> _Self;
  
  inline PredIterator(_EdgeIter startEdge) : oi(startEdge) {}
  
  inline bool operator==(const _Self& x) const { return oi == x.oi; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }
  
  // operator*() differs for pred or succ iterator
  inline _NodeType* operator*() const { return (*oi)->getSrc(); }
  inline	 _NodeType* operator->() const { return operator*(); }
  
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
class SuccIterator: public std::bidirectional_iterator<_NodeType, ptrdiff_t> {
protected:
  _EdgeIter oi;
public:
  typedef SuccIterator<_NodeType, _EdgeType, _EdgeIter> _Self;
  
  inline SuccIterator(_EdgeIter startEdge) : oi(startEdge) {}
  
  inline bool operator==(const _Self& x) const { return oi == x.oi; }
  inline bool operator!=(const _Self& x) const { return !operator==(x); }
  
  inline _NodeType* operator*() const { return (*oi)->getSink(); }
  inline	 _NodeType* operator->() const { return operator*(); }
  
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

// 
// sg_pred_iterator
// sg_pred_const_iterator
//
typedef PredIterator<SchedGraphNode, SchedGraphEdge, SchedGraphNode::iterator>
    sg_pred_iterator;
typedef PredIterator<const SchedGraphNode, const SchedGraphEdge,SchedGraphNode::const_iterator>
    sg_pred_const_iterator;

inline sg_pred_iterator       pred_begin(      SchedGraphNode *N) {
  return sg_pred_iterator(N->beginInEdges());
}
inline sg_pred_iterator       pred_end(        SchedGraphNode *N) {
  return sg_pred_iterator(N->endInEdges());
}
inline sg_pred_const_iterator pred_begin(const SchedGraphNode *N) {
  return sg_pred_const_iterator(N->beginInEdges());
}
inline sg_pred_const_iterator pred_end(  const SchedGraphNode *N) {
  return sg_pred_const_iterator(N->endInEdges());
}


// 
// sg_succ_iterator
// sg_succ_const_iterator
//
typedef SuccIterator<SchedGraphNode, SchedGraphEdge, SchedGraphNode::iterator>
    sg_succ_iterator;
typedef SuccIterator<const SchedGraphNode, const SchedGraphEdge,SchedGraphNode::const_iterator>
    sg_succ_const_iterator;

inline sg_succ_iterator       succ_begin(      SchedGraphNode *N) {
  return sg_succ_iterator(N->beginOutEdges());
}
inline sg_succ_iterator       succ_end(        SchedGraphNode *N) {
  return sg_succ_iterator(N->endOutEdges());
}
inline sg_succ_const_iterator succ_begin(const SchedGraphNode *N) {
  return sg_succ_const_iterator(N->beginOutEdges());
}
inline sg_succ_const_iterator succ_end(  const SchedGraphNode *N) {
  return sg_succ_const_iterator(N->endOutEdges());
}

// 
// po_iterator
// po_const_iterator
//
typedef cfg::POIterator<SchedGraphNode, sg_succ_iterator> sg_po_iterator;
typedef cfg::POIterator<const SchedGraphNode, 
		        sg_succ_const_iterator> sg_po_const_iterator;


//************************ External Functions *****************************/


ostream& operator<<(ostream& os, const SchedGraphEdge& edge);

ostream& operator<<(ostream& os, const SchedGraphNode& node);


/***************************************************************************/

#endif
