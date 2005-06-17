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
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "MSScheduleSB.h"
#include "MSchedGraphSB.h"


namespace llvm {

  //Struct to contain ModuloScheduling Specific Information for each node
  struct MSNodeSBAttributes {
    int ASAP; //Earliest time at which the opreation can be scheduled
    int ALAP; //Latest time at which the operation can be scheduled.
    int MOB;
    int depth;
    int height;
    MSNodeSBAttributes(int asap=-1, int alap=-1, int mob=-1,
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
    std::map<MSchedGraphSBNode*, MSNodeSBAttributes> nodeToAttributesMap;

    //Map to hold all reccurrences
    std::set<std::pair<int, std::vector<MSchedGraphSBNode*> > > recurrenceList;

    //Set of edges to ignore, stored as src node and index into vector of successors
    std::set<std::pair<MSchedGraphSBNode*, unsigned> > edgesToIgnore;

    //Vector containing the partial order
    std::vector<std::set<MSchedGraphSBNode*> > partialOrder;

    //Vector containing the final node order
    std::vector<MSchedGraphSBNode*> FinalNodeOrder;

    //Schedule table, key is the cycle number and the vector is resource, node pairs
    MSScheduleSB schedule;

    //Current initiation interval
    int II;
    
    //Internal Functions
    void FindSuperBlocks(Function &F, LoopInfo &LI, 
			 std::vector<std::vector<const MachineBasicBlock*> > &Worklist);
    bool MachineBBisValid(const MachineBasicBlock *B,
			  std::map<const MachineInstr*, unsigned> &indexMap, 
			  unsigned &offset);
    bool CreateDefMap(std::vector<const MachineBasicBlock*> &SB);
    bool getIndVar(std::vector<const MachineBasicBlock*> &superBlock, 
		   std::map<BasicBlock*, MachineBasicBlock*> &bbMap,
		   std::map<const MachineInstr*, unsigned> &indexMap);
    bool assocIndVar(Instruction *I, std::set<Instruction*> &indVar,
		     std::vector<Instruction*> &stack, 
		     std::map<BasicBlock*, MachineBasicBlock*> &bbMap, 
		     const BasicBlock *first,
		     std::set<const BasicBlock*> &llvmSuperBlock);
    int calculateResMII(std::vector<const MachineBasicBlock*> &superBlock);
    int calculateRecMII(MSchedGraphSB *graph, int MII);
    void findAllCircuits(MSchedGraphSB *g, int II);
    void addRecc(std::vector<MSchedGraphSBNode*> &stack, 
		 std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes);
    bool circuit(MSchedGraphSBNode *v, std::vector<MSchedGraphSBNode*> &stack,
		 std::set<MSchedGraphSBNode*> &blocked, std::vector<MSchedGraphSBNode*> &SCC,
		 MSchedGraphSBNode *s, std::map<MSchedGraphSBNode*, 
		 std::set<MSchedGraphSBNode*> > &B,
		 int II, std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes);
    void unblock(MSchedGraphSBNode *u, std::set<MSchedGraphSBNode*> &blocked,
		 std::map<MSchedGraphSBNode*, std::set<MSchedGraphSBNode*> > &B);
    void addSCC(std::vector<MSchedGraphSBNode*> &SCC, std::map<MSchedGraphSBNode*, MSchedGraphSBNode*> &newNodes);
    void calculateNodeAttributes(MSchedGraphSB *graph, int MII);
    bool ignoreEdge(MSchedGraphSBNode *srcNode, MSchedGraphSBNode *destNode);
    int  calculateASAP(MSchedGraphSBNode *node, int MII, MSchedGraphSBNode *destNode);
    int calculateALAP(MSchedGraphSBNode *node, int MII,
		      int maxASAP, MSchedGraphSBNode *srcNode);
    int findMaxASAP();
    int calculateHeight(MSchedGraphSBNode *node,MSchedGraphSBNode *srcNode);
    int calculateDepth(MSchedGraphSBNode *node, MSchedGraphSBNode *destNode);
    void computePartialOrder();
    void connectedComponentSet(MSchedGraphSBNode *node, std::set<MSchedGraphSBNode*> &ccSet, 
			       std::set<MSchedGraphSBNode*> &lastNodes);
    void searchPath(MSchedGraphSBNode *node,
		    std::vector<MSchedGraphSBNode*> &path,
		    std::set<MSchedGraphSBNode*> &nodesToAdd,
		    std::set<MSchedGraphSBNode*> &new_reccurrence);
    void orderNodes();
    bool computeSchedule(std::vector<const MachineBasicBlock*> &BB, MSchedGraphSB *MSG);
    bool scheduleNode(MSchedGraphSBNode *node, int start, int end);
      void predIntersect(std::set<MSchedGraphSBNode*> &CurrentSet, std::set<MSchedGraphSBNode*> &IntersectResult);
    void succIntersect(std::set<MSchedGraphSBNode*> &CurrentSet, std::set<MSchedGraphSBNode*> &IntersectResult);
    void reconstructLoop(std::vector<const MachineBasicBlock*> &SB);
    void fixBranches(std::vector<std::vector<MachineBasicBlock*> > &prologues, 
		     std::vector<std::vector<BasicBlock*> > &llvm_prologues, 
		     std::vector<MachineBasicBlock*> &machineKernelBB, 
		     std::vector<BasicBlock*> &llvmKernelBB, 
		     std::vector<std::vector<MachineBasicBlock*> > &epilogues, 
		     std::vector<std::vector<BasicBlock*> > &llvm_epilogues, 
		     std::vector<const MachineBasicBlock*> &SB,
		     std::map<const MachineBasicBlock*, Value*> &sideExits);

    void writePrologues(std::vector<std::vector<MachineBasicBlock *> > &prologues, 
			std::vector<const MachineBasicBlock*> &origBB, 
			std::vector<std::vector<BasicBlock*> > &llvm_prologues, 
			std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, 
			std::map<Value*, std::map<int, Value*> > &newValues, 
			std::map<Value*, MachineBasicBlock*> &newValLocation);

    void writeKernel(std::vector<BasicBlock*> &llvmBB, std::vector<MachineBasicBlock*> &machineBB, 
		     std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave, 
		     std::map<Value*, std::map<int, Value*> > &newValues, 
		     std::map<Value*, MachineBasicBlock*> &newValLocation, 
		     std::map<Value*, std::map<int, Value*> > &kernelPHIs);

    void removePHIs(std::vector<const MachineBasicBlock*> &SB, 
		    std::vector<std::vector<MachineBasicBlock*> > &prologues, 
		    std::vector<std::vector<MachineBasicBlock*> > &epilogues, 
		    std::vector<MachineBasicBlock*> &kernelBB, 
		    std::map<Value*, MachineBasicBlock*> &newValLocation);
    
    void writeEpilogues(std::vector<std::vector<MachineBasicBlock*> > &epilogues, 
			std::vector<const MachineBasicBlock*> &origSB, 
			std::vector<std::vector<BasicBlock*> > &llvm_epilogues, 
			std::map<const Value*, std::pair<const MachineInstr*, int> > &valuesToSave,
			std::map<Value*, std::map<int, Value*> > &newValues,
			std::map<Value*, MachineBasicBlock*> &newValLocation, 
			std::map<Value*, std::map<int, Value*> > &kernelPHIs);
    
    void writeSideExits(std::vector<std::vector<MachineBasicBlock *> > &prologues, 
			std::vector<std::vector<BasicBlock*> > &llvm_prologues, 
			std::vector<std::vector<MachineBasicBlock *> > &epilogues, 
			std::vector<std::vector<BasicBlock*> > &llvm_epilogues, 
			std::map<const MachineBasicBlock*, Value*> &sideExits, 
			std::map<MachineBasicBlock*, std::vector<std::pair<MachineInstr*, int> > > &instrsMovedDown,
			std::vector<const MachineBasicBlock*> &SB, 
			std::vector<MachineBasicBlock*> &kernelMBBs,
			  std::map<MachineBasicBlock*, int> branchStage);

 public:
    ModuloSchedulingSBPass(TargetMachine &targ) : target(targ) {}
      virtual bool runOnFunction(Function &F);
      virtual const char* getPassName() const { return "ModuloScheduling-SuperBlock"; }
      
      
      // getAnalysisUsage
      virtual void getAnalysisUsage(AnalysisUsage &AU) const {
	/// HACK: We don't actually need scev, but we have
	/// to say we do so that the pass manager does not delete it
	/// before we run.
	AU.addRequired<LoopInfo>();
	AU.addRequired<ScalarEvolution>();
	AU.addRequired<DependenceAnalyzer>();
      }
  };
}
#endif
