//===-- ModuloSchedulingSuperBlock.h -Swing Modulo Scheduling-----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//Swing Modulo Scheduling done on Superblocks ( entry, multiple exit,
//multiple basic block loops).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MODULOSCHEDULINGSB_H
#define LLVM_MODULOSCHEDULINGSB_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "MSSchedule.h"
#include "MSchedGraph.h"

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


  typedef std::vector<const MachineBasicBlock*> SuperBlock;

  class ModuloSchedulingSBPass : public FunctionPass {
    const TargetMachine &target;
    
    //Map to hold Value* defs
    std::map<const Value*, MachineInstr*> defMap;

    //Map to hold list of instructions associate to the induction var for each BB
    std::map<SuperBlock, std::map<const MachineInstr*, unsigned> > indVarInstrs;

    //Map to hold machine to  llvm instrs for each valid BB
    std::map<SuperBlock, std::map<MachineInstr*, Instruction*> > machineTollvm;
    
    //LLVM Instruction we know we can add TmpInstructions to its MCFI
    Instruction *defaultInst;

    //Map that holds node to node attribute information
    std::map<MSchedGraphNode*, MSNodeAttributes> nodeToAttributesMap;

    //Map to hold all reccurrences
    std::set<std::pair<int, std::vector<MSchedGraphNode*> > > recurrenceList;

    //Set of edges to ignore, stored as src node and index into vector of successors
    std::set<std::pair<MSchedGraphNode*, unsigned> > edgesToIgnore;

    //Vector containing the partial order
    std::vector<std::set<MSchedGraphNode*> > partialOrder;

    //Vector containing the final node order
    std::vector<MSchedGraphNode*> FinalNodeOrder;

    //Schedule table, key is the cycle number and the vector is resource, node pairs
    MSSchedule schedule;

    //Current initiation interval
    int II;
    
    //Internal Functions
    void FindSuperBlocks(Function &F, LoopInfo &LI, 
			 std::vector<std::vector<const MachineBasicBlock*> > &Worklist);
    bool MachineBBisValid(const MachineBasicBlock *B);
    bool CreateDefMap(std::vector<const MachineBasicBlock*> &SB);

  public:
    ModuloSchedulingSBPass(TargetMachine &targ) : target(targ) {}
      virtual bool runOnFunction(Function &F);
      virtual const char* getPassName() const { return "ModuloScheduling-SuperBlock"; }
      
      
      // getAnalysisUsage
      virtual void getAnalysisUsage(AnalysisUsage &AU) const {
	AU.addRequired<LoopInfo>();
	AU.addRequired<DependenceAnalyzer>();
      }
  };
}
#endif
