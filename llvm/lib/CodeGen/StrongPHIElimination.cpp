//===- StrongPhiElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Owen Anderson and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates machine instruction PHI nodes by inserting copy
// instructions, using an intelligent copy-folding technique based on
// dominator information.  This is technique is derived from:
// 
//    Budimlic, et al. Fast copy coalescing and live-range identification.
//    In Proceedings of the ACM SIGPLAN 2002 Conference on Programming Language
//    Design and Implementation (Berlin, Germany, June 17 - 19, 2002).
//    PLDI '02. ACM, New York, NY, 25-32.
//    DOI= http://doi.acm.org/10.1145/512529.512534
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "strongphielim"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;


namespace {
  struct VISIBILITY_HIDDEN StrongPHIElimination : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid
    StrongPHIElimination() : MachineFunctionPass((intptr_t)&ID) {}

    bool runOnMachineFunction(MachineFunction &Fn);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addPreserved<LiveVariables>();
      AU.addPreservedID(PHIEliminationID);
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<LiveVariables>();
      AU.setPreservesAll();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    
    virtual void releaseMemory() {
      preorder.clear();
      maxpreorder.clear();
      
      waiting.clear();
    }

  private:
    struct DomForestNode {
    private:
      std::vector<DomForestNode*> children;
      MachineInstr* instr;
      
      void addChild(DomForestNode* DFN) { children.push_back(DFN); }
      
    public:
      typedef std::vector<DomForestNode*>::iterator iterator;
      
      DomForestNode(MachineInstr* MI, DomForestNode* parent) : instr(MI) {
        if (parent)
          parent->addChild(this);
      }
      
      ~DomForestNode() {
        for (iterator I = begin(), E = end(); I != E; ++I)
          delete *I;
      }
      
      inline MachineInstr* getInstr() { return instr; }
      
      inline DomForestNode::iterator begin() { return children.begin(); }
      inline DomForestNode::iterator end() { return children.end(); }
    };
    
    DenseMap<MachineBasicBlock*, unsigned> preorder;
    DenseMap<MachineBasicBlock*, unsigned> maxpreorder;
    
    DenseMap<MachineBasicBlock*, std::vector<MachineInstr*> > waiting;
    
    
    void computeDFS(MachineFunction& MF);
    
    std::vector<DomForestNode*>
      computeDomForest(SmallPtrSet<MachineInstr*, 8>& instrs);
    
  };

  char StrongPHIElimination::ID = 0;
  RegisterPass<StrongPHIElimination> X("strong-phi-node-elimination",
                  "Eliminate PHI nodes for register allocation, intelligently");
}

const PassInfo *llvm::StrongPHIEliminationID = X.getPassInfo();

/// computeDFS - Computes the DFS-in and DFS-out numbers of the dominator tree
/// of the given MachineFunction.  These numbers are then used in other parts
/// of the PHI elimination process.
void StrongPHIElimination::computeDFS(MachineFunction& MF) {
  SmallPtrSet<MachineDomTreeNode*, 8> frontier;
  SmallPtrSet<MachineDomTreeNode*, 8> visited;
  
  unsigned time = 0;
  
  MachineDominatorTree& DT = getAnalysis<MachineDominatorTree>();
  
  MachineDomTreeNode* node = DT.getRootNode();
  
  std::vector<MachineDomTreeNode*> worklist;
  worklist.push_back(node);
  
  while (!worklist.empty()) {
    MachineDomTreeNode* currNode = worklist.back();
    
    if (!frontier.count(currNode)) {
      frontier.insert(currNode);
      ++time;
      preorder.insert(std::make_pair(currNode->getBlock(), time));
    }
    
    bool inserted = false;
    for (MachineDomTreeNode::iterator I = node->begin(), E = node->end();
         I != E; ++I)
      if (!frontier.count(*I) && !visited.count(*I)) {
        worklist.push_back(*I);
        inserted = true;
        break;
      }
    
    if (!inserted) {
      frontier.erase(currNode);
      visited.insert(currNode);
      maxpreorder.insert(std::make_pair(currNode->getBlock(), time));
      
      worklist.pop_back();
    }
  }
}

class PreorderSorter {
private:
  DenseMap<MachineBasicBlock*, unsigned>& preorder;
  
public:
  PreorderSorter(DenseMap<MachineBasicBlock*, unsigned>& p) : preorder(p) { }
  
  bool operator()(MachineInstr* A, MachineInstr* B) {
    if (A == B)
      return false;
    
    if (preorder[A->getParent()] < preorder[B->getParent()])
      return true;
    else if (preorder[A->getParent()] > preorder[B->getParent()])
      return false;
    
    if (A->getOpcode() == TargetInstrInfo::PHI &&
        B->getOpcode() == TargetInstrInfo::PHI)
      return A < B;
    
    MachineInstr* begin = A->getParent()->begin();
    return std::distance(begin, A) < std::distance(begin, B);
  }
};

std::vector<StrongPHIElimination::DomForestNode*>
StrongPHIElimination::computeDomForest(SmallPtrSet<MachineInstr*, 8>& instrs) {
  DomForestNode* VirtualRoot = new DomForestNode(0, 0);
  maxpreorder.insert(std::make_pair((MachineBasicBlock*)0, ~0UL));
  
  std::vector<MachineInstr*> worklist;
  worklist.reserve(instrs.size());
  for (SmallPtrSet<MachineInstr*, 8>::iterator I = instrs.begin(),
       E = instrs.end(); I != E; ++I)
    worklist.push_back(*I);
  PreorderSorter PS(preorder);
  std::sort(worklist.begin(), worklist.end(), PS);
  
  DomForestNode* CurrentParent = VirtualRoot;
  std::vector<DomForestNode*> stack;
  stack.push_back(VirtualRoot);
  
  for (std::vector<MachineInstr*>::iterator I = worklist.begin(),
       E = worklist.end(); I != E; ++I) {
    while (preorder[(*I)->getParent()] >
           maxpreorder[CurrentParent->getInstr()->getParent()]) {
      stack.pop_back();
      CurrentParent = stack.back();
    }
    
    DomForestNode* child = new DomForestNode(*I, CurrentParent);
    stack.push_back(child);
    CurrentParent = child;
  }
  
  std::vector<DomForestNode*> ret;
  ret.insert(ret.end(), VirtualRoot->begin(), VirtualRoot->end());
  return ret;
}

bool StrongPHIElimination::runOnMachineFunction(MachineFunction &Fn) {
  computeDFS(Fn);
  
  
  return false;
}
