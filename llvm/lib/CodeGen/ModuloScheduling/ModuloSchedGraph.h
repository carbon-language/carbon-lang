//===- ModuloSchedGraph.h - Modulo Scheduling Graph and Set -*- C++ -*-----===//
//
// This header defines the primative classes that make up a data structure
// graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MODULO_SCHED_GRAPH_H
#define LLVM_CODEGEN_MODULO_SCHED_GRAPH_H

#include "llvm/Instruction.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "Support/GraphTraits.h"
#include "Support/hash_map"
#include "../InstrSched/SchedGraphCommon.h"
#include <iostream>

//===----------------------------------------------------------------------===//
// ModuloSchedGraphNode - Implement a data structure based on the
// SchedGraphNodeCommon this class stores informtion needed later to order the
// nodes in modulo scheduling
//
class ModuloSchedGraphNode:public SchedGraphNodeCommon {
private:
  // the corresponding instruction 
  const Instruction *inst;

  // whether this node's property(ASAP,ALAP, ...) has been computed
  bool propertyComputed;

  // ASAP: the earliest time the node could be scheduled
  // ALAP: the latest time the node couldbe scheduled
  // depth: the depth of the node
  // height: the height of the node
  // mov: the mobility function, computed as ALAP - ASAP
  // scheTime: the scheduled time if this node has been scheduled 
  // earlyStart: the earliest time to be tried to schedule the node
  // lateStart: the latest time to be tried to schedule the node
  int ASAP, ALAP, depth, height, mov;
  int schTime;
  int earlyStart, lateStart;

public:

  //get the instruction
  const Instruction *getInst() const {
    return inst;
  }
  //get the instruction op-code name
  const char *getInstOpcodeName() const {
    return inst->getOpcodeName();
  }
  //get the instruction op-code
  const unsigned getInstOpcode() const {
    return inst->getOpcode();
  }

  //return whether the node is NULL
  bool isNullNode() const {
    return (inst == NULL);
  }
  //return whether the property of the node has been computed
  bool getPropertyComputed() {
    return propertyComputed;
  }
  //set the propertyComputed
  void setPropertyComputed(bool _propertyComputed) {
    propertyComputed = _propertyComputed;
  }
  
  //get the corresponding property
  int getASAP() {
    return ASAP;
  }
  int getALAP() {
    return ALAP;
  }
  int getMov() {
    return mov;
  }
  int getDepth() {
    return depth;
  }
  int getHeight() {
    return height;
  }
  int getSchTime() {
    return schTime;
  }
  int getEarlyStart() {
    return earlyStart;
  }
  int getLateStart() {
    return lateStart;
  }
  void setEarlyStart(int _earlyStart) {
    earlyStart = _earlyStart;
  }
  void setLateStart(int _lateStart) {
    lateStart = _lateStart;
  }
  void setSchTime(int _time) {
    schTime = _time;
  }

private:
  friend class ModuloSchedGraph;
  friend class SchedGraphNode;

  //constructor:
  //nodeId: the node id, unique within the each BasicBlock
  //_bb: which BasicBlock the corresponding instruction belongs to 
  //_inst: the corresponding instruction
  //indexInBB: the corresponding instruction's index in the BasicBlock
  //target: the targetMachine
  ModuloSchedGraphNode(unsigned int _nodeId,
                       const BasicBlock * _bb,
                       const Instruction * _inst,
                       int indexInBB, const TargetMachine &target);

  
  friend std::ostream & operator<<(std::ostream & os,
                                   const ModuloSchedGraphNode & edge);

};

//FIXME: these two value should not be used
#define MAXNODE 100
#define MAXCC   100

//===----------------------------------------------------------------------===//
/// ModuloSchedGraph- the data structure to store dependence between nodes
/// it catches data dependence and constrol dependence
/// 
class ModuloSchedGraph :
  public SchedGraphCommon,
  protected hash_map<const Instruction*,ModuloSchedGraphNode*> {

private:

  BasicBlock* bb;
  
  //iteration Interval
  int MII;

  //target machine
  const TargetMachine & target;

  //the circuits in the dependence graph
  unsigned circuits[MAXCC][MAXNODE];

  //the order nodes
  std::vector<ModuloSchedGraphNode*> oNodes;

  typedef std::vector<ModuloSchedGraphNode*> NodeVec;

  //the function to compute properties
  void computeNodeASAP(const BasicBlock * in_bb);
  void computeNodeALAP(const BasicBlock * in_bb);
  void computeNodeMov(const BasicBlock *  in_bb);
  void computeNodeDepth(const BasicBlock * in_bb);
  void computeNodeHeight(const BasicBlock * in_bb);

  //the function to compute node property
  void computeNodeProperty(const BasicBlock * in_bb);

  //the function to sort nodes
  void orderNodes();

  //add the resource usage 
void addResourceUsage(std::vector<std::pair<int,int> > &, int);

  //debug functions:
  //dump circuits
  void dumpCircuits();
  //dump the input set of nodes
  void dumpSet(std::vector<ModuloSchedGraphNode*> set);
  //dump the input resource usage table  
  void dumpResourceUsage(std::vector<std::pair<int,int> > &);

public:
  //help functions

  //get the maxium the delay between two nodes
  SchedGraphEdge *getMaxDelayEdge(unsigned srcId, unsigned sinkId);

  //FIXME:
  //get the predessor Set of the set
  NodeVec predSet(NodeVec set, unsigned, unsigned);
  NodeVec predSet(NodeVec set);

  //get the predessor set of the node
  NodeVec predSet(ModuloSchedGraphNode *node, unsigned, unsigned);
  NodeVec predSet(ModuloSchedGraphNode *node);

  //get the successor set of the set
  NodeVec succSet(NodeVec set, unsigned, unsigned);
  NodeVec succSet(NodeVec set);

  //get the succssor set of the node
  NodeVec succSet(ModuloSchedGraphNode *node, unsigned, unsigned);
  NodeVec succSet(ModuloSchedGraphNode *node);

  //return the uniton of the two vectors
  NodeVec vectorUnion(NodeVec set1, NodeVec set2);
  
  //return the consjuction of the two vectors
  NodeVec vectorConj(NodeVec set1, NodeVec set2);

  //return all nodes in  set1 but not  set2
  NodeVec vectorSub(NodeVec set1, NodeVec set2);

  typedef hash_map<const Instruction*,ModuloSchedGraphNode*> map_base;

public:
  using map_base::iterator;
  using map_base::const_iterator;

public:

  //get target machine
  const TargetMachine & getTarget() {
    return target;
  }

  //get the basic block
  BasicBlock* getBasicBlock() const {
    return bb;
  }


  //get the iteration interval
  const int getMII() {
    return MII;
  }

  //get the ordered nodes
  const NodeVec & getONodes() {
    return oNodes;
  }

  //get the number of nodes (including the root and leaf)
  //note: actually root and leaf is not used
  const unsigned int getNumNodes() const {
    return size() + 2;
  }

  //return wether the BasicBlock 'bb' contains a loop
  bool isLoop(const BasicBlock *bb);

  //return the node for the input instruction
  ModuloSchedGraphNode *getGraphNodeForInst(const Instruction *inst) const {
    const_iterator onePair = this->find(inst);
    return (onePair != this->end()) ? (*onePair).second : NULL;
  }

  // Debugging support
  //dump the graph
  void dump() const;

  // dump the basicBlock
  void dump(const BasicBlock *bb);

  //dump the basicBlock into 'os' stream
  void dump(const BasicBlock *bb, std::ostream &os);

  //dump the node property
  void dumpNodeProperty() const;
  
private:
  friend class ModuloSchedGraphSet;     //give access to ctor

public:
  ModuloSchedGraph(BasicBlock * in_bb, 
		   const TargetMachine & in_target)
    :SchedGraphCommon(), bb(in_bb),target(in_target)
  {
    buildGraph(target);
  }

  ~ModuloSchedGraph() {
    for (const_iterator I = begin(); I != end(); ++I)
      delete I->second;
  }

  // Unorder iterators
  // return values are pair<const Instruction*, ModuloSchedGraphNode*>
  using map_base::begin;
  using map_base::end;

  void addHash(const Instruction *inst,
	       ModuloSchedGraphNode *node){
    
    assert((*this)[inst] == NULL);
    (*this)[inst] = node;
    
  }

  // Graph builder
  ModuloSchedGraphNode *getNode(const unsigned nodeId) const;

  // Build the graph from the basicBlock
  void buildGraph(const TargetMachine &target);

  // Build nodes for BasicBlock
  void buildNodesforBB(const TargetMachine &target,
                       const BasicBlock *bb);

  //find definitiona and use information for all nodes
  void findDefUseInfoAtInstr(const TargetMachine &target,
                             ModuloSchedGraphNode *node,
                             NodeVec &memNode,
                             RegToRefVecMap &regToRefVecMap,
                             ValueToDefVecMap &valueToDefVecMap);

  //add def-use edge
  void addDefUseEdges(const BasicBlock *bb);

  //add control dependence edges
  void addCDEdges(const BasicBlock *bb);

  //add memory dependence dges
  void addMemEdges(const BasicBlock *bb);

  //computer source restrictoin II
  int computeResII(const BasicBlock *bb);

  //computer recurrence II
  int computeRecII(const BasicBlock *bb);
};

//==================================-
// Graph set

class ModuloSchedGraphSet : public std::vector<ModuloSchedGraph*> {
private:
  const Function *method;

public:
  typedef std::vector<ModuloSchedGraph*> baseVector;
  using baseVector::iterator;
  using baseVector::const_iterator;

public:
  ModuloSchedGraphSet(const Function *function, const TargetMachine &target);
  ~ModuloSchedGraphSet();

  // Iterators
  using baseVector::begin;
  using baseVector::end;

  // Debugging support
  void dump() const;

private:
  void addGraph(ModuloSchedGraph *graph) {
    assert(graph != NULL);
    this->push_back(graph);
  }

  // Graph builder
  void buildGraphsForMethod(const Function *F,
                            const TargetMachine &target);
};

#endif
