//===- ModuloSchedGraph.h - Modulo Scheduling Graph and Set -*- C++ -*-----===//
// 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULO_SCHED_GRAPH_H
#define LLVM_MODULO_SCHED_GRAPH_H

#include "llvm/Instruction.h"
#include "llvm/CodeGen/SchedGraphCommon.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/BasicBlock.h"
#include "llvm/Function.h"
#include "Support/hash_map"
#include <vector>


class ModuloSchedGraphNode : public SchedGraphNodeCommon {

  const Instruction *Inst;  //Node's Instruction
  unsigned Earliest;        //ASAP, or earliest time to be scheduled
  unsigned Latest;          //ALAP, or latested time to be scheduled
  unsigned Depth;           //Max Distance from node to the root
  unsigned Height;          //Max Distance from node to leaf
  unsigned Mobility;        //MOB, number of time slots it can be scheduled
  const TargetMachine &Target; //Target information.

public:
  ModuloSchedGraphNode(unsigned ID, int index, const Instruction *inst, 
		       const TargetMachine &target);
  
  void print(std::ostream &os) const;
  const Instruction* getInst() { return Inst; }
  unsigned getEarliest() { return Earliest; }
  unsigned getLatest() { return Latest; }
  unsigned getDepth() { return Depth; }
  unsigned getHeight() { return Height; }
  unsigned getMobility() { return Mobility; }
  
  void setEarliest(unsigned early) { Earliest = early; }
  void setLatest(unsigned late) { Latest = late; }
  void setDepth(unsigned depth) { Depth = depth; }
  void setHeight(unsigned height) { Height = height; }
  void setMobility(unsigned mob) { Mobility = mob; }


};

class ModuloSchedGraph : public SchedGraphCommon {
  
  const BasicBlock *BB; //The Basic block this graph represents
  const TargetMachine &Target;
  hash_map<const Instruction*, ModuloSchedGraphNode*> GraphMap;

  void buildNodesForBB();

public:
  typedef hash_map<const Instruction*, 
		   ModuloSchedGraphNode*>::iterator iterator;
  typedef hash_map<const Instruction*, 
		   ModuloSchedGraphNode*>::const_iterator const_iterator;


  ModuloSchedGraph(const BasicBlock *bb, const TargetMachine &targ);

  const BasicBlock* getBB() { return BB; }
  void setBB(BasicBlock *bb) { BB = bb; }
  unsigned size() { return GraphMap.size(); }
  void addNode(const Instruction *I, ModuloSchedGraphNode *node);
  void ASAP(); //Calculate earliest schedule time for all nodes in graph.
  void ALAP(); //Calculate latest schedule time for all nodes in graph.
  void MOB(); //Calculate mobility for all nodes in the graph.
  void ComputeDepth(); //Compute depth of each node in graph
  void ComputeHeight(); //Computer height of each node in graph
  void addDepEdges(); //Add Dependencies
  iterator find(const Instruction *I) { return GraphMap.find(I); }
};


class ModuloSchedGraphSet {
  
  const Function *function; //Function this set of graphs represent.
  std::vector<ModuloSchedGraph*> Graphs;

public:
  typedef std::vector<ModuloSchedGraph*>::iterator iterator;
  typedef std::vector<ModuloSchedGraph*>::const_iterator const_iterator;
 
  iterator begin() { return Graphs.begin(); }
  iterator end() { return Graphs.end(); }
 
  ModuloSchedGraphSet(const Function *func, const TargetMachine &target);
  ~ModuloSchedGraphSet();

  void addGraph(ModuloSchedGraph *graph);
  void dump() const;


};

#endif
