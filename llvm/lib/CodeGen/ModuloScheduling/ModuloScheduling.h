//===-- ModuloScheduling.h - Swing Modulo Scheduling------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// 
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULOSCHEDULING_H
#define LLVM_MODULOSCHEDULING_H

#include "MSchedGraph.h"
#include "MSSchedule.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include <set>

namespace llvm {
  

  //Struct to contain ModuloScheduling Specific Information for each node
  struct MSNodeAttributes {
    int ASAP; //Earliest time at which the opreation can be scheduled
    int ALAP; //Latest time at which the operation can be scheduled.
    int MOB;
    int depth;
    int height;
    MSNodeAttributes(int asap=-1, int alap=-1, int mob=-1, 
			     int d=-1, int h=-1) : ASAP(asap), ALAP(alap), 
						   MOB(mob), depth(d), 
						   height(h) {}
  };


  class ModuloSchedulingPass : public FunctionPass {
    const TargetMachine &target;

    //Map that holds node to node attribute information
    std::map<MSchedGraphNode*, MSNodeAttributes> nodeToAttributesMap;

    //Map to hold all reccurrences
    std::set<std::pair<int, std::vector<MSchedGraphNode*> > > recurrenceList;

    //Set of edges to ignore, stored as src node and index into vector of successors
    std::set<std::pair<MSchedGraphNode*, unsigned> > edgesToIgnore;
    
    //Vector containing the partial order
    std::vector<std::vector<MSchedGraphNode*> > partialOrder;

    //Vector containing the final node order
    std::vector<MSchedGraphNode*> FinalNodeOrder;

    //Schedule table, key is the cycle number and the vector is resource, node pairs
    MSSchedule schedule;

    //Current initiation interval
    int II;

    //Internal functions
    bool MachineBBisValid(const MachineBasicBlock *BI);
    int calculateResMII(const MachineBasicBlock *BI);
    int calculateRecMII(MSchedGraph *graph, int MII);
    void calculateNodeAttributes(MSchedGraph *graph, int MII);

    bool ignoreEdge(MSchedGraphNode *srcNode, MSchedGraphNode *destNode);


    int calculateASAP(MSchedGraphNode *node, int MII,MSchedGraphNode *destNode);
    int calculateALAP(MSchedGraphNode *node, int MII, int maxASAP, MSchedGraphNode *srcNode);

    int calculateHeight(MSchedGraphNode *node,MSchedGraphNode *srcNode);
    int calculateDepth(MSchedGraphNode *node, MSchedGraphNode *destNode);

    int findMaxASAP();
    void orderNodes();
    void findAllReccurrences(MSchedGraphNode *node, 
			     std::vector<MSchedGraphNode*> &visitedNodes, int II);
    void addReccurrence(std::vector<MSchedGraphNode*> &recurrence, int II, MSchedGraphNode*, MSchedGraphNode*);

    void computePartialOrder();
    void computeSchedule();
    bool scheduleNode(MSchedGraphNode *node, 
		      int start, int end);

    void predIntersect(std::vector<MSchedGraphNode*> &CurrentSet, std::vector<MSchedGraphNode*> &IntersectResult);
    void succIntersect(std::vector<MSchedGraphNode*> &CurrentSet, std::vector<MSchedGraphNode*> &IntersectResult);
    
    void reconstructLoop(const MachineBasicBlock*);
    
    //void saveValue(const MachineInstr*, const std::set<Value*>&, std::vector<Value*>*);

    void writePrologue(std::vector<MachineBasicBlock *> &prologues, MachineBasicBlock *origBB, std::vector<BasicBlock*> &llvm_prologues);

  public:
    ModuloSchedulingPass(TargetMachine &targ) : target(targ) {}
    virtual bool runOnFunction(Function &F);
  };

}


#endif
