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

    //Internal functions
    bool MachineBBisValid(const MachineBasicBlock *BI);
    int calculateResMII(const MachineBasicBlock *BI);
    void calculateNodeAttributes(MSchedGraph *graph, int MII);
    void calculateASAP(MSchedGraphNode *node, MSNodeAttributes &attributes, 
		       int MII,std::set<MSchedGraphNode*> &visitedNodes);
    void calculateALAP(MSchedGraphNode *node, MSNodeAttributes &attributes, int MII, 
		       int maxASAP, std::set<MSchedGraphNode*> &visitedNodes);
    void calculateHeight(MSchedGraphNode *node, 
			 MSNodeAttributes &attributes, std::set<MSchedGraphNode*> &visitedNodes);
    void calculateDepth(MSchedGraphNode *node, MSNodeAttributes &attributes, 
			std::set<MSchedGraphNode*> &visitedNodes);

    int findMaxASAP();
    void ModuloSchedulingPass::orderNodes();
    void findAllReccurrences(MSchedGraphNode *node, std::vector<MSchedGraphNode*> &visitedNodes);
  public:
    ModuloSchedulingPass(TargetMachine &targ) : target(targ) {}
    virtual bool runOnFunction(Function &F);
  };

}


#endif
