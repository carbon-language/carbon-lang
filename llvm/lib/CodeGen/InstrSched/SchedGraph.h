//===-- SchedGraph.h - Scheduling Graph -------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This is a scheduling graph based on SSA graph plus extra dependence edges
// capturing dependences due to machine resources (machine registers, CC
// registers, and any others).
// 
// This graph tries to leverage the SSA graph as much as possible, but captures
// the extra dependences through a common interface.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDGRAPH_H
#define LLVM_CODEGEN_SCHEDGRAPH_H

#include "llvm/CodeGen/SchedGraphCommon.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Transforms/Scalar.h"
#include "Support/hash_map"
#include "Support/GraphTraits.h"

namespace llvm {

class RegToRefVecMap;
class ValueToDefVecMap;
class RefVec;

class SchedGraphNode : public SchedGraphNodeCommon {

  MachineBasicBlock *MBB;
  const MachineInstr *MI;


  SchedGraphNode(unsigned nodeId, MachineBasicBlock *mbb, int indexInBB, 
		 const TargetMachine& Target);
  ~SchedGraphNode();

  friend class SchedGraph;		// give access for ctor and dtor
  friend class SchedGraphEdge;		// give access for adding edges

public:

  // Accessor methods
  const MachineInstr* getMachineInstr() const { return MI; }
  const MachineOpCode getOpcode() const { return MI->getOpcode(); }
  bool isDummyNode() const { return (MI == NULL); }
  MachineBasicBlock &getMachineBasicBlock() const { return *MBB; }

  void print(std::ostream &os) const;
};

class SchedGraph : public SchedGraphCommon {
  MachineBasicBlock &MBB;
  hash_map<const MachineInstr*, SchedGraphNode*> GraphMap;
  
public:
  typedef hash_map<const MachineInstr*, SchedGraphNode*>::const_iterator iterator;
  typedef hash_map<const MachineInstr*, SchedGraphNode*>::const_iterator const_iterator;
    
  MachineBasicBlock& getBasicBlock() const{return MBB;}
  const unsigned int getNumNodes() const { return GraphMap.size()+2; }
  SchedGraphNode* getGraphNodeForInstr(const MachineInstr* MI) const {
    const_iterator onePair = find(MI);
    return (onePair != end())? onePair->second : NULL;
  }
  
  // Debugging support
  void dump() const;
  
protected:
  SchedGraph(MachineBasicBlock& mbb, const TargetMachine& TM);
  ~SchedGraph();

  // Unordered iterators.
  // Return values is pair<const MachineIntr*,SchedGraphNode*>.
  //
  hash_map<const MachineInstr*, SchedGraphNode*>::const_iterator begin() const {
    return GraphMap.begin();
  }
  hash_map<const MachineInstr*, SchedGraphNode*>::const_iterator end() const {
    return GraphMap.end();
  }
 
  unsigned size() { return GraphMap.size(); }
  iterator find(const MachineInstr *MI) const { return GraphMap.find(MI); }
  
  SchedGraphNode *&operator[](const MachineInstr *MI) {
    return GraphMap[MI];
  }
  
private:
  friend class SchedGraphSet;		// give access to ctor
    
  inline void	noteGraphNodeForInstr	(const MachineInstr* minstr,
					 SchedGraphNode* node) {
    assert((*this)[minstr] == NULL);
    (*this)[minstr] = node;
  }

  //
  // Graph builder
  //
  void buildGraph(const TargetMachine& target);
  
  void  buildNodesForBB(const TargetMachine& target,MachineBasicBlock &MBB,
			std::vector<SchedGraphNode*>& memNV,
			std::vector<SchedGraphNode*>& callNV,
			RegToRefVecMap& regToRefVecMap,
			ValueToDefVecMap& valueToDefVecMap);

  
  void findDefUseInfoAtInstr(const TargetMachine& target, SchedGraphNode* node,
			     std::vector<SchedGraphNode*>& memNV,
			     std::vector<SchedGraphNode*>& callNV,
			     RegToRefVecMap& regToRefVecMap,
			     ValueToDefVecMap& valueToDefVecMap);
                                         
  void addEdgesForInstruction(const MachineInstr& minstr,
			      const ValueToDefVecMap& valueToDefVecMap,
			      const TargetMachine& target);
  
  void addCDEdges(const TerminatorInst* term, const TargetMachine& target);
  
  void addMemEdges(const std::vector<SchedGraphNode*>& memNod,
		   const TargetMachine& target);
  
  void addCallCCEdges(const std::vector<SchedGraphNode*>& memNod,
		      MachineBasicBlock& bbMvec,
		      const TargetMachine& target);

  void addCallDepEdges(const std::vector<SchedGraphNode*>& callNV,
		       const TargetMachine& target);
  
  void addMachineRegEdges(RegToRefVecMap& regToRefVecMap,
			  const TargetMachine& target);
  
  void addEdgesForValue(SchedGraphNode* refNode, const RefVec& defVec,
			const Value* defValue, bool  refNodeIsDef,
			bool  refNodeIsDefAndUse,
			const TargetMachine& target);

  void addDummyEdges();

};



class SchedGraphSet {
  const Function* function;
  std::vector<SchedGraph*> Graphs;

  // Graph builder
  void buildGraphsForMethod(const Function *F, const TargetMachine& target);

  inline void addGraph(SchedGraph* graph) {
    assert(graph != NULL);
    Graphs.push_back(graph);
  }  

public:
  SchedGraphSet(const Function *function, const TargetMachine& target);
  ~SchedGraphSet();
  
  //iterators
  typedef std::vector<SchedGraph*>::const_iterator iterator;
  typedef std::vector<SchedGraph*>::const_iterator const_iterator;

  std::vector<SchedGraph*>::const_iterator begin() const { return Graphs.begin(); }
  std::vector<SchedGraph*>::const_iterator end() const { return Graphs.end(); }

  // Debugging support
  void dump() const;
};




// 
// sg_pred_iterator
// sg_pred_const_iterator
//
typedef SGPredIterator<SchedGraphNode, SchedGraphEdge, SchedGraphNode::iterator>
    sg_pred_iterator;
typedef SGPredIterator<const SchedGraphNode, const SchedGraphEdge,SchedGraphNode::const_iterator>
    sg_pred_const_iterator;

inline sg_pred_iterator pred_begin(SchedGraphNode *N) {
  return sg_pred_iterator(N->beginInEdges());
}
inline sg_pred_iterator pred_end(SchedGraphNode *N) {
  return sg_pred_iterator(N->endInEdges());
}
inline sg_pred_const_iterator pred_begin(const SchedGraphNode *N) {
  return sg_pred_const_iterator(N->beginInEdges());
}
inline sg_pred_const_iterator pred_end(const SchedGraphNode *N) {
  return sg_pred_const_iterator(N->endInEdges());
}


// 
// sg_succ_iterator
// sg_succ_const_iterator
//
typedef SGSuccIterator<SchedGraphNode, SchedGraphEdge, SchedGraphNode::iterator>
    sg_succ_iterator;
typedef SGSuccIterator<const SchedGraphNode, const SchedGraphEdge,SchedGraphNode::const_iterator>
    sg_succ_const_iterator;

inline sg_succ_iterator succ_begin(SchedGraphNode *N) {
  return sg_succ_iterator(N->beginOutEdges());
}
inline sg_succ_iterator succ_end(SchedGraphNode *N) {
  return sg_succ_iterator(N->endOutEdges());
}
inline sg_succ_const_iterator succ_begin(const SchedGraphNode *N) {
  return sg_succ_const_iterator(N->beginOutEdges());
}
inline sg_succ_const_iterator succ_end(const SchedGraphNode *N) {
  return sg_succ_const_iterator(N->endOutEdges());
}

// Provide specializations of GraphTraits to be able to use graph iterators on
// the scheduling graph!
//
template <> struct GraphTraits<SchedGraph*> {
  typedef SchedGraphNode NodeType;
  typedef sg_succ_iterator ChildIteratorType;

  static inline NodeType *getEntryNode(SchedGraph *SG) { return (NodeType*)SG->getRoot(); }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return succ_begin(N); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return succ_end(N);
  }
};

template <> struct GraphTraits<const SchedGraph*> {
  typedef const SchedGraphNode NodeType;
  typedef sg_succ_const_iterator ChildIteratorType;

  static inline NodeType *getEntryNode(const SchedGraph *SG) {
    return (NodeType*)SG->getRoot();
  }
  static inline ChildIteratorType child_begin(NodeType *N) { 
    return succ_begin(N); 
  }
  static inline ChildIteratorType child_end(NodeType *N) { 
    return succ_end(N);
  }
};

} // End llvm namespace

#endif
