// ModuloScheduling.h -------------------------------------------*- C++ -*-===//
//
// This header defines the the classes ModuloScheduling and 
// ModuloSchedulingSet's structure
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_CODEGEN_MODULOSCHEDULING_H
#define LLVM_CODEGEN_MODULOSCHEDULING_H

#include "ModuloSchedGraph.h"
#include <iostream>
#include <vector>

// for debug information selecton
enum ModuloSchedDebugLevel_t { 
  ModuloSchedDebugLevel_NoDebugInfo,
  ModuloSchedDebugLevel_PrintSchedule,
  ModuloSchedDebugLevel_PrintScheduleProcess,
};

class ModuloScheduling: NonCopyable {
private:

  typedef std::vector<ModuloSchedGraphNode*> NodeVec;
  typedef std::vector<std::vector<unsigned> > Resources;

  // The graph to feed in
  ModuloSchedGraph &graph;
  const TargetMachine &target;

  // The BasicBlock to be scheduled
  BasicBlock *bb;

  // Iteration Interval
  // FIXME: II may be a better name for its meaning
  unsigned II;

  // The vector containing the nodes which have been scheduled
  NodeVec nodeScheduled;

  // The remaining unscheduled nodes 
  const NodeVec &oNodes;

  // The machine resource table
  std::vector<std::vector<std::pair<int,int> > > resourceTable;

  ///the schedule( with many schedule stage)
  std::vector<std::vector<ModuloSchedGraphNode*> > schedule;

  ///the kernel(core) schedule(length = II)
  std::vector<std::vector<ModuloSchedGraphNode*> > coreSchedule;

  typedef BasicBlock::InstListType InstListType;
  typedef std::vector<std::vector<ModuloSchedGraphNode*> > vvNodeType;





public:

  ModuloScheduling(ModuloSchedGraph & _graph):
    graph(_graph), target(graph.getTarget()), oNodes(graph.getONodes())
  {
    II = graph.getMII();
    bb = (BasicBlock *) graph.getBasicBlocks()[0];
    instrScheduling();
  };

  ~ModuloScheduling() {};



  static bool 
  printSchedule() { 

    //return ModuloScheduling::DebugLevel >= DebugLevel_PrintSchedule; 
    return false;


  }
  static bool 
  printScheduleProcess() {
  
    //return DebugLevel >= DebugLevel_PrintScheduleProcess;
    return false;


  }

  // The method to compute schedule and instert epilogue and prologue
  void instrScheduling();

  // Debug functions:
  // Dump the schedule and core schedule
  void dumpScheduling();

  // Dump the input vector of nodes
  // sch: the input vector of nodes
  void dumpSchedule(std::vector<std::vector<ModuloSchedGraphNode*> > sch);

  // Dump the resource usage table
  void dumpResourceUsageTable();

  //*******************internal functions*******************************
private:
  //clear memory from the last round and initialize if necessary
  void clearInitMem(const TargetSchedInfo&);

  //compute schedule and coreSchedule with the current II
  bool computeSchedule();

  BasicBlock *getSuccBB(BasicBlock *);
  BasicBlock *getPredBB(BasicBlock *);
  void constructPrologue(BasicBlock *prologue);
  void constructKernel(BasicBlock *prologue,
                       BasicBlock *kernel,
                       BasicBlock *epilogue);
  void constructEpilogue(BasicBlock *epilogue, BasicBlock *succ_bb);

  // update the resource table at the startCycle
  // vec: the resouce usage
  // startCycle: the start cycle the resouce usage is
  void updateResourceTable(std::vector<std::vector<unsigned> > vec,
                           int startCycle);

  // un-do the update in the resource table in the startCycle
  // vec: the resouce usage
  // startCycle: the start cycle the resouce usage is
  void undoUpdateResourceTable(std::vector<std::vector<unsigned> > vec,
                               int startCycle);

  // return whether the resourcetable has negative element
  // this function is called after updateResouceTable() to determine whether a
  // node can be scheduled at certain cycle
  bool resourceTableNegative();

  // try to Schedule the node starting from start to end cycle(inclusive)
  // if it can be scheduled, put it in the schedule and update nodeScheduled
  // node: the node to be scheduled
  // start: start cycle
  // end : end cycle
  // nodeScheduled: a vector storing nodes which has been scheduled
  bool ScheduleNode(ModuloSchedGraphNode * node, unsigned start,
                    unsigned end, NodeVec &nodeScheduled);

  //each instruction has a memory of the latest clone instruction
  //the clone instruction can be get using getClone() 
  //this function clears the memory, i.e. getClone() after calling this function
  //returns null
  void clearCloneMemory();

  //this fuction make a clone of this input Instruction and update the clone
  //memory
  //inst: the instrution to be cloned
  Instruction *cloneInstSetMemory(Instruction *inst);

  //this function update each instrutions which uses ist as its operand
  //after update, each instruction will use ist's clone as its operand
  void updateUseWithClone(Instruction * ist);

};


class ModuloSchedulingSet:
NonCopyable {
private:

  //the graphSet to feed in
  ModuloSchedGraphSet & graphSet;

public:

  //constructor
  //Scheduling graph one by one
  ModuloSchedulingSet(ModuloSchedGraphSet _graphSet): graphSet(_graphSet) {
    for (unsigned i = 0; i < graphSet.size(); i++) {
      ModuloSchedGraph & graph = *(graphSet[i]);
      if (graph.isLoop())
        ModuloScheduling ModuloScheduling(graph);
    }
  };

  ~ModuloSchedulingSet() {};
};

#endif
