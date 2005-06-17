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
#include "DependenceAnalyzer.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
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

    //Map to hold Value* defs
    std::map<const Value*, MachineInstr*> defMap;

    //Map to hold list of instructions associate to the induction var for each BB
    std::map<const MachineBasicBlock*, std::map<const MachineInstr*, unsigned> > indVarInstrs;

    //Map to hold machine to  llvm instrs for each valid BB
    std::map<const MachineBasicBlock*, std::map<MachineInstr*, Instruction*> > machineTollvm;

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

    //Internal functions
    bool CreateDefMap(MachineBasicBlock *BI);
    bool MachineBBisValid(const MachineBasicBlock *BI);
    bool assocIndVar(Instruction *I, std::set<Instruction*> &indVar,
		     std::vector<Instruction*> &stack, BasicBlock *BB);
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
    void addSCC(std::vector<MSchedGraphNode*> &SCC, std::map<MSchedGraphNode*, MSchedGraphNode*> &newNodes);

    void findAllCircuits(MSchedGraph *MSG, int II);
    bool circuit(MSchedGraphNode *v, std::vector<MSchedGraphNode*> &stack,
		 std::set<MSchedGraphNode*> &blocked,
		 std::vector<MSchedGraphNode*> &SCC, MSchedGraphNode *s,
		 std::map<MSchedGraphNode*, std::set<MSchedGraphNode*> > &B, int II,
		 std::map<MSchedGraphNode*, MSchedGraphNode*> &newNodes);

    void unblock(MSchedGraphNode *u, std::set<MSchedGraphNode*> &blocked,
		 std::map<MSchedGraphNode*, std::set<MSchedGraphNode*> > &B);

    void addRecc(std::vector<MSchedGraphNode*> &stack, std::map<MSchedGraphNode*, MSchedGraphNode*> &newNodes);

    void searchPath(MSchedGraphNode *node, 
		    std::vector<MSchedGraphNode*> &path,
		    std::set<MSchedGraphNode*> &nodesToAdd,
		    std::set<MSchedGraphNode*> &new_reccurence);

    void pathToRecc(MSchedGraphNode *node,
		    std::vector<MSchedGraphNode*> &path,
		    std::set<MSchedGraphNode*> &poSet, std::set<MSchedGraphNode*> &lastNodes);

    void computePartialOrder();

    bool computeSchedule(const MachineBasicBlock *BB, MSchedGraph *MSG);
    bool scheduleNode(MSchedGraphNode *node, 
		      int start, int end);

    void predIntersect(std::set<MSchedGraphNode*> &CurrentSet, std::set<MSchedGraphNode*> &IntersectResult);
    void succIntersect(std::set<MSchedGraphNode*> &CurrentSet, std::set<MSchedGraphNode*> &IntersectResult);

    void reconstructLoop(MachineBasicBlock*);

    //void saveValue(const MachineInstr*, const std::set<Value*>&, std::vector<Value*>*);

    void fixBranches(std::vector<MachineBasicBlock *> &prologues, std::vector<BasicBlock*> &llvm_prologues, MachineBasicBlock *machineBB, BasicBlock *llvmBB, std::vector<MachineBasicBlock *> &epilogues, std::vector<BasicBlock*> &llvm_epilogues, MachineBasicBlock*);

    void writePrologues(std::vector<MachineBasicBlock *> &prologues, MachineBasicBlock *origBB, std::vector<BasicBlock*> &llvm_prologues, std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation);

    void writeEpilogues(std::vector<MachineBasicBlock *> &epilogues, const MachineBasicBlock *origBB, std::vector<BasicBlock*> &llvm_epilogues, std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave,std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation,  std::map<Value*, std::map<int, Value*> > &kernelPHIs);


    void writeKernel(BasicBlock *llvmBB, MachineBasicBlock *machineBB, std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, std::map<Value*, std::map<int, Value*> > &newValues, std::map<Value*, MachineBasicBlock*> &newValLocation, std::map<Value*, std::map<int, Value*> > &kernelPHIs);

    void removePHIs(const MachineBasicBlock* SB, std::vector<MachineBasicBlock*> &prologues, std::vector<MachineBasicBlock *> &epilogues, MachineBasicBlock *kernelBB, std::map<Value*, MachineBasicBlock*> &newValLocation);

    void connectedComponentSet(MSchedGraphNode *node, std::set<MSchedGraphNode*> &ccSet, std::set<MSchedGraphNode*> &lastNodes);

  public:
    ModuloSchedulingPass(TargetMachine &targ) : target(targ) {}
    virtual bool runOnFunction(Function &F);
    virtual const char* getPassName() const { return "ModuloScheduling"; }

    // getAnalysisUsage
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      /// HACK: We don't actually need loopinfo or scev, but we have
      /// to say we do so that the pass manager does not delete it
      /// before we run.
      AU.addRequired<LoopInfo>();
      AU.addRequired<ScalarEvolution>();
      
      AU.addRequired<DependenceAnalyzer>();
    }

  };

}


#endif
