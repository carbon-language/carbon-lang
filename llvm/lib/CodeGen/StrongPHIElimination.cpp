//===- StrongPhiElimination.cpp - Eliminate PHI nodes by inserting copies -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
using namespace llvm;


namespace {
  struct VISIBILITY_HIDDEN StrongPHIElimination : public MachineFunctionPass {
    static char ID; // Pass identification, replacement for typeid
    StrongPHIElimination() : MachineFunctionPass((intptr_t)&ID) {}

    // Waiting stores, for each MBB, the set of copies that need to
    // be inserted into that MBB
    DenseMap<MachineBasicBlock*,
             std::map<unsigned, unsigned> > Waiting;
    
    // Stacks holds the renaming stack for each register
    std::map<unsigned, std::vector<unsigned> > Stacks;
    
    // Registers in UsedByAnother are PHI nodes that are themselves
    // used as operands to another another PHI node
    std::set<unsigned> UsedByAnother;
    
    // RenameSets are the sets of operands to a PHI (the defining instruction
    // of the key) that can be renamed without copies
    std::map<unsigned, std::set<unsigned> > RenameSets;

    // Store the DFS-in number of each block
    DenseMap<MachineBasicBlock*, unsigned> preorder;
    
    // Store the DFS-out number of each block
    DenseMap<MachineBasicBlock*, unsigned> maxpreorder;

    bool runOnMachineFunction(MachineFunction &Fn);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineDominatorTree>();
      AU.addRequired<LiveVariables>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }
    
    virtual void releaseMemory() {
      preorder.clear();
      maxpreorder.clear();
      
      Waiting.clear();
      Stacks.clear();
      UsedByAnother.clear();
      RenameSets.clear();
    }

  private:
    
    /// DomForestNode - Represents a node in the "dominator forest".  This is
    /// a forest in which the nodes represent registers and the edges
    /// represent a dominance relation in the block defining those registers.
    struct DomForestNode {
    private:
      // Store references to our children
      std::vector<DomForestNode*> children;
      // The register we represent
      unsigned reg;
      
      // Add another node as our child
      void addChild(DomForestNode* DFN) { children.push_back(DFN); }
      
    public:
      typedef std::vector<DomForestNode*>::iterator iterator;
      
      // Create a DomForestNode by providing the register it represents, and
      // the node to be its parent.  The virtual root node has register 0
      // and a null parent.
      DomForestNode(unsigned r, DomForestNode* parent) : reg(r) {
        if (parent)
          parent->addChild(this);
      }
      
      ~DomForestNode() {
        for (iterator I = begin(), E = end(); I != E; ++I)
          delete *I;
      }
      
      /// getReg - Return the regiser that this node represents
      inline unsigned getReg() { return reg; }
      
      // Provide iterator access to our children
      inline DomForestNode::iterator begin() { return children.begin(); }
      inline DomForestNode::iterator end() { return children.end(); }
    };
    
    void computeDFS(MachineFunction& MF);
    void processBlock(MachineBasicBlock* MBB);
    
    std::vector<DomForestNode*> computeDomForest(std::set<unsigned>& instrs);
    void processPHIUnion(MachineInstr* Inst,
                         std::set<unsigned>& PHIUnion,
                         std::vector<StrongPHIElimination::DomForestNode*>& DF,
                         std::vector<std::pair<unsigned, unsigned> >& locals);
    void ScheduleCopies(MachineBasicBlock* MBB, std::set<unsigned>& pushed);
    void InsertCopies(MachineBasicBlock* MBB, std::set<MachineBasicBlock*>& v);
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

/// PreorderSorter - a helper class that is used to sort registers
/// according to the preorder number of their defining blocks
class PreorderSorter {
private:
  DenseMap<MachineBasicBlock*, unsigned>& preorder;
  LiveVariables& LV;
  
public:
  PreorderSorter(DenseMap<MachineBasicBlock*, unsigned>& p,
                LiveVariables& L) : preorder(p), LV(L) { }
  
  bool operator()(unsigned A, unsigned B) {
    if (A == B)
      return false;
    
    MachineBasicBlock* ABlock = LV.getVarInfo(A).DefInst->getParent();
    MachineBasicBlock* BBlock = LV.getVarInfo(A).DefInst->getParent();
    
    if (preorder[ABlock] < preorder[BBlock])
      return true;
    else if (preorder[ABlock] > preorder[BBlock])
      return false;
    
    return false;
  }
};

/// computeDomForest - compute the subforest of the DomTree corresponding
/// to the defining blocks of the registers in question
std::vector<StrongPHIElimination::DomForestNode*>
StrongPHIElimination::computeDomForest(std::set<unsigned>& regs) {
  LiveVariables& LV = getAnalysis<LiveVariables>();
  
  // Begin by creating a virtual root node, since the actual results
  // may well be a forest.  Assume this node has maximum DFS-out number.
  DomForestNode* VirtualRoot = new DomForestNode(0, 0);
  maxpreorder.insert(std::make_pair((MachineBasicBlock*)0, ~0UL));
  
  // Populate a worklist with the registers
  std::vector<unsigned> worklist;
  worklist.reserve(regs.size());
  for (std::set<unsigned>::iterator I = regs.begin(), E = regs.end();
       I != E; ++I)
    worklist.push_back(*I);
  
  // Sort the registers by the DFS-in number of their defining block
  PreorderSorter PS(preorder, LV);
  std::sort(worklist.begin(), worklist.end(), PS);
  
  // Create a "current parent" stack, and put the virtual root on top of it
  DomForestNode* CurrentParent = VirtualRoot;
  std::vector<DomForestNode*> stack;
  stack.push_back(VirtualRoot);
  
  // Iterate over all the registers in the previously computed order
  for (std::vector<unsigned>::iterator I = worklist.begin(), E = worklist.end();
       I != E; ++I) {
    unsigned pre = preorder[LV.getVarInfo(*I).DefInst->getParent()];
    MachineBasicBlock* parentBlock = CurrentParent->getReg() ?
                 LV.getVarInfo(CurrentParent->getReg()).DefInst->getParent() :
                 0;
    
    // If the DFS-in number of the register is greater than the DFS-out number
    // of the current parent, repeatedly pop the parent stack until it isn't.
    while (pre > maxpreorder[parentBlock]) {
      stack.pop_back();
      CurrentParent = stack.back();
      
      parentBlock = CurrentParent->getReg() ?
                   LV.getVarInfo(CurrentParent->getReg()).DefInst->getParent() :
                   0;
    }
    
    // Now that we've found the appropriate parent, create a DomForestNode for
    // this register and attach it to the forest
    DomForestNode* child = new DomForestNode(*I, CurrentParent);
    
    // Push this new node on the "current parent" stack
    stack.push_back(child);
    CurrentParent = child;
  }
  
  // Return a vector containing the children of the virtual root node
  std::vector<DomForestNode*> ret;
  ret.insert(ret.end(), VirtualRoot->begin(), VirtualRoot->end());
  return ret;
}

/// isLiveIn - helper method that determines, from a VarInfo, if a register
/// is live into a block
static bool isLiveIn(LiveVariables::VarInfo& V, MachineBasicBlock* MBB) {
  if (V.AliveBlocks.test(MBB->getNumber()))
    return true;
  
  if (V.DefInst->getParent() != MBB &&
      V.UsedBlocks.test(MBB->getNumber()))
    return true;
  
  return false;
}

/// isLiveOut - help method that determines, from a VarInfo, if a register is
/// live out of a block.
static bool isLiveOut(LiveVariables::VarInfo& V, MachineBasicBlock* MBB) {
  if (MBB == V.DefInst->getParent() ||
      V.UsedBlocks.test(MBB->getNumber())) {
    for (std::vector<MachineInstr*>::iterator I = V.Kills.begin(), 
         E = V.Kills.end(); I != E; ++I)
      if ((*I)->getParent() == MBB)
        return false;
    
    return true;
  }
  
  return false;
}

/// isKillInst - helper method that determines, from a VarInfo, if an 
/// instruction kills a given register
static bool isKillInst(LiveVariables::VarInfo& V, MachineInstr* MI) {
  return std::find(V.Kills.begin(), V.Kills.end(), MI) != V.Kills.end();
}

/// interferes - checks for local interferences by scanning a block.  The only
/// trick parameter is 'mode' which tells it the relationship of the two
/// registers. 0 - defined in the same block, 1 - first properly dominates
/// second, 2 - second properly dominates first 
static bool interferes(LiveVariables::VarInfo& First,
                       LiveVariables::VarInfo& Second,
                       MachineBasicBlock* scan, unsigned mode) {
  MachineInstr* def = 0;
  MachineInstr* kill = 0;
  
  bool interference = false;
  
  // Wallk the block, checking for interferences
  for (MachineBasicBlock::iterator MBI = scan->begin(), MBE = scan->end();
       MBI != MBE; ++MBI) {
    MachineInstr* curr = MBI;
    
    // Same defining block...
    if (mode == 0) {
      if (curr == First.DefInst) {
        // If we find our first DefInst, save it
        if (!def) {
          def = curr;
        // If there's already an unkilled DefInst, then 
        // this is an interference
        } else if (!kill) {
          interference = true;
          break;
        // If there's a DefInst followed by a KillInst, then
        // they can't interfere
        } else {
          interference = false;
          break;
        }
      // Symmetric with the above
      } else if (curr == Second.DefInst ) {
        if (!def) {
          def = curr;
        } else if (!kill) {
          interference = true;
          break;
        } else {
          interference = false;
          break;
        }
      // Store KillInsts if they match up with the DefInst
      } else if (isKillInst(First, curr)) {
        if (def == First.DefInst) {
          kill = curr;
        } else if (isKillInst(Second, curr)) {
          if (def == Second.DefInst) {
            kill = curr;
          }
        }
      }
    // First properly dominates second...
    } else if (mode == 1) {
      if (curr == Second.DefInst) {
        // DefInst of second without kill of first is an interference
        if (!kill) {
          interference = true;
          break;
        // DefInst after a kill is a non-interference
        } else {
          interference = false;
          break;
        }
      // Save KillInsts of First
      } else if (isKillInst(First, curr)) {
        kill = curr;
      }
    // Symmetric with the above
    } else if (mode == 2) {
      if (curr == First.DefInst) {
        if (!kill) {
          interference = true;
          break;
        } else {
          interference = false;
          break;
        }
      } else if (isKillInst(Second, curr)) {
        kill = curr;
      }
    }
  }
  
  return interference;
}

/// processBlock - Eliminate PHIs in the given block
void StrongPHIElimination::processBlock(MachineBasicBlock* MBB) {
  LiveVariables& LV = getAnalysis<LiveVariables>();
  
  // Holds names that have been added to a set in any PHI within this block
  // before the current one.
  std::set<unsigned> ProcessedNames;
  
  MachineBasicBlock::iterator P = MBB->begin();
  while (P != MBB->end() && P->getOpcode() == TargetInstrInfo::PHI) {
    LiveVariables::VarInfo& PHIInfo = LV.getVarInfo(P->getOperand(0).getReg());

    unsigned DestReg = P->getOperand(0).getReg();

    // Hold the names that are currently in the candidate set.
    std::set<unsigned> PHIUnion;
    std::set<MachineBasicBlock*> UnionedBlocks;
  
    for (int i = P->getNumOperands() - 1; i >= 2; i-=2) {
      unsigned SrcReg = P->getOperand(i-1).getReg();
      LiveVariables::VarInfo& SrcInfo = LV.getVarInfo(SrcReg);
    
      // Check for trivial interferences
      if (isLiveIn(SrcInfo, P->getParent()) ||
          isLiveOut(PHIInfo, SrcInfo.DefInst->getParent()) ||
          ( PHIInfo.DefInst->getOpcode() == TargetInstrInfo::PHI &&
            isLiveIn(PHIInfo, SrcInfo.DefInst->getParent()) ) ||
          ProcessedNames.count(SrcReg) ||
          UnionedBlocks.count(SrcInfo.DefInst->getParent())) {
        
        // add a copy from a_i to p in Waiting[From[a_i]]
        MachineBasicBlock* From = P->getOperand(i).getMBB();
        Waiting[From].insert(std::make_pair(SrcReg, DestReg));
        UsedByAnother.insert(SrcReg);
      } else {
        PHIUnion.insert(SrcReg);
        UnionedBlocks.insert(SrcInfo.DefInst->getParent());
      }
    }
    
    std::vector<StrongPHIElimination::DomForestNode*> DF = 
                                                     computeDomForest(PHIUnion);
    
    // Walk DomForest to resolve interferences
    std::vector<std::pair<unsigned, unsigned> > localInterferences;
    processPHIUnion(P, PHIUnion, DF, localInterferences);
    
    // Check for local interferences
    for (std::vector<std::pair<unsigned, unsigned> >::iterator I =
        localInterferences.begin(), E = localInterferences.end(); I != E; ++I) {
      std::pair<unsigned, unsigned> p = *I;
      
      LiveVariables::VarInfo& FirstInfo = LV.getVarInfo(p.first);
      LiveVariables::VarInfo& SecondInfo = LV.getVarInfo(p.second);
      
      MachineDominatorTree& MDT = getAnalysis<MachineDominatorTree>();
      
      // Determine the block we need to scan and the relationship between
      // the two registers
      MachineBasicBlock* scan = 0;
      unsigned mode = 0;
      if (FirstInfo.DefInst->getParent() == SecondInfo.DefInst->getParent()) {
        scan = FirstInfo.DefInst->getParent();
        mode = 0; // Same block
      } else if (MDT.dominates(FirstInfo.DefInst->getParent(),
                             SecondInfo.DefInst->getParent())) {
        scan = SecondInfo.DefInst->getParent();
        mode = 1; // First dominates second
      } else {
        scan = FirstInfo.DefInst->getParent();
        mode = 2; // Second dominates first
      }
      
      // If there's an interference, we need to insert  copies
      if (interferes(FirstInfo, SecondInfo, scan, mode)) {
        // Insert copies for First
        for (int i = P->getNumOperands() - 1; i >= 2; i-=2) {
          if (P->getOperand(i-1).getReg() == p.first) {
            unsigned SrcReg = p.first;
            MachineBasicBlock* From = P->getOperand(i).getMBB();
            
            Waiting[From].insert(std::make_pair(SrcReg,
                                                P->getOperand(0).getReg()));
            UsedByAnother.insert(SrcReg);
            
            PHIUnion.erase(SrcReg);
          }
        }
      }
    }
    
    // Cache renaming information
    RenameSets.insert(std::make_pair(P->getOperand(0).getReg(), PHIUnion));
    
    ProcessedNames.insert(PHIUnion.begin(), PHIUnion.end());
    ++P;
  }
}

/// processPHIUnion - Take a set of candidate registers to be coallesced when
/// decomposing the PHI instruction.  Use the DominanceForest to remove the ones
/// that are known to interfere, and flag others that need to be checked for
/// local interferences.
void StrongPHIElimination::processPHIUnion(MachineInstr* Inst,
                                           std::set<unsigned>& PHIUnion,
                        std::vector<StrongPHIElimination::DomForestNode*>& DF,
                        std::vector<std::pair<unsigned, unsigned> >& locals) {
  
  std::vector<DomForestNode*> worklist(DF.begin(), DF.end());
  SmallPtrSet<DomForestNode*, 4> visited;
  
  LiveVariables& LV = getAnalysis<LiveVariables>();
  unsigned DestReg = Inst->getOperand(0).getReg();
  
  // DF walk on the DomForest
  while (!worklist.empty()) {
    DomForestNode* DFNode = worklist.back();
    
    LiveVariables::VarInfo& Info = LV.getVarInfo(DFNode->getReg());
    visited.insert(DFNode);
    
    bool inserted = false;
    for (DomForestNode::iterator CI = DFNode->begin(), CE = DFNode->end();
         CI != CE; ++CI) {
      DomForestNode* child = *CI;   
      LiveVariables::VarInfo& CInfo = LV.getVarInfo(child->getReg());
        
      if (isLiveOut(Info, CInfo.DefInst->getParent())) {
        // Insert copies for parent
        for (int i = Inst->getNumOperands() - 1; i >= 2; i-=2) {
          if (Inst->getOperand(i-1).getReg() == DFNode->getReg()) {
            unsigned SrcReg = DFNode->getReg();
            MachineBasicBlock* From = Inst->getOperand(i).getMBB();
            
            Waiting[From].insert(std::make_pair(SrcReg, DestReg));
            UsedByAnother.insert(SrcReg);
            
            PHIUnion.erase(SrcReg);
          }
        }
      } else if (isLiveIn(Info, CInfo.DefInst->getParent()) ||
                 Info.DefInst->getParent() == CInfo.DefInst->getParent()) {
        // Add (p, c) to possible local interferences
        locals.push_back(std::make_pair(DFNode->getReg(), child->getReg()));
      }
      
      if (!visited.count(child)) {
        worklist.push_back(child);
        inserted = true;
      }
    }
    
    if (!inserted) worklist.pop_back();
  }
}

/// ScheduleCopies - Insert copies into predecessor blocks, scheduling
/// them properly so as to avoid the 'lost copy' and the 'virtual swap'
/// problems.
///
/// Based on "Practical Improvements to the Construction and Destruction
/// of Static Single Assignment Form" by Briggs, et al.
void StrongPHIElimination::ScheduleCopies(MachineBasicBlock* MBB,
                                          std::set<unsigned>& pushed) {
  // FIXME: This function needs to update LiveVariables
  std::map<unsigned, unsigned>& copy_set= Waiting[MBB];
  
  std::map<unsigned, unsigned> worklist;
  std::map<unsigned, unsigned> map;
  
  // Setup worklist of initial copies
  for (std::map<unsigned, unsigned>::iterator I = copy_set.begin(),
       E = copy_set.end(); I != E; ) {
    map.insert(std::make_pair(I->first, I->first));
    map.insert(std::make_pair(I->second, I->second));
         
    if (!UsedByAnother.count(I->first)) {
      worklist.insert(*I);
      
      // Avoid iterator invalidation
      unsigned first = I->first;
      ++I;
      copy_set.erase(first);
    } else {
      ++I;
    }
  }
  
  LiveVariables& LV = getAnalysis<LiveVariables>();
  MachineFunction* MF = MBB->getParent();
  const TargetInstrInfo *TII = MF->getTarget().getInstrInfo();
  
  // Iterate over the worklist, inserting copies
  while (!worklist.empty() || !copy_set.empty()) {
    while (!worklist.empty()) {
      std::pair<unsigned, unsigned> curr = *worklist.begin();
      worklist.erase(curr.first);
      
      const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(curr.first);
      
      if (isLiveOut(LV.getVarInfo(curr.second), MBB)) {
        // Create a temporary
        unsigned t = MF->getRegInfo().createVirtualRegister(RC);
        
        // Insert copy from curr.second to a temporary at
        // the Phi defining curr.second
        LiveVariables::VarInfo VI = LV.getVarInfo(curr.second);
        MachineBasicBlock::iterator PI = VI.DefInst;
        TII->copyRegToReg(*VI.DefInst->getParent(), PI, t,
                          curr.second, RC, RC);
        
        // Push temporary on Stacks
        Stacks[curr.second].push_back(t);
        
        // Insert curr.second in pushed
        pushed.insert(curr.second);
      }
      
      // Insert copy from map[curr.first] to curr.second
      TII->copyRegToReg(*MBB, MBB->getFirstTerminator(), curr.second,
                        map[curr.first], RC, RC);
      map[curr.first] = curr.second;
      
      // If curr.first is a destination in copy_set...
      for (std::map<unsigned, unsigned>::iterator I = copy_set.begin(),
           E = copy_set.end(); I != E; )
        if (curr.first == I->second) {
          std::pair<unsigned, unsigned> temp = *I;
          
          // Avoid iterator invalidation
          ++I;
          copy_set.erase(temp.first);
          worklist.insert(temp);
          
          break;
        } else {
          ++I;
        }
    }
    
    if (!copy_set.empty()) {
      std::pair<unsigned, unsigned> curr = *copy_set.begin();
      copy_set.erase(curr.first);
      
      const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(curr.first);
      
      // Insert a copy from dest to a new temporary t at the end of b
      unsigned t = MF->getRegInfo().createVirtualRegister(RC);
      TII->copyRegToReg(*MBB, MBB->getFirstTerminator(), t,
                        curr.second, RC, RC);
      map[curr.second] = t;
      
      worklist.insert(curr);
    }
  }
}

/// InsertCopies - insert copies into MBB and all of its successors
void StrongPHIElimination::InsertCopies(MachineBasicBlock* MBB,
                                        std::set<MachineBasicBlock*>& visited) {
  visited.insert(MBB);
  
  std::set<unsigned> pushed;
  
  // Rewrite register uses from Stacks
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
      I != E; ++I)
    for (unsigned i = 0; i < I->getNumOperands(); ++i)
      if (I->getOperand(i).isRegister() &&
          Stacks[I->getOperand(i).getReg()].size()) {
        I->getOperand(i).setReg(Stacks[I->getOperand(i).getReg()].back());
      }
  
  // Schedule the copies for this block
  ScheduleCopies(MBB, pushed);
  
  // Recur to our successors
  for (GraphTraits<MachineBasicBlock*>::ChildIteratorType I = 
       GraphTraits<MachineBasicBlock*>::child_begin(MBB), E =
       GraphTraits<MachineBasicBlock*>::child_end(MBB); I != E; ++I)
    if (!visited.count(*I))
      InsertCopies(*I, visited);
  
  // As we exit this block, pop the names we pushed while processing it
  for (std::set<unsigned>::iterator I = pushed.begin(), 
       E = pushed.end(); I != E; ++I)
    Stacks[*I].pop_back();
}

bool StrongPHIElimination::runOnMachineFunction(MachineFunction &Fn) {
  // Compute DFS numbers of each block
  computeDFS(Fn);
  
  // Determine which phi node operands need copies
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    if (!I->empty() &&
        I->begin()->getOpcode() == TargetInstrInfo::PHI)
      processBlock(I);
  
  // Insert copies
  // FIXME: This process should probably preserve LiveVariables
  std::set<MachineBasicBlock*> visited;
  InsertCopies(Fn.begin(), visited);
  
  // Perform renaming
  typedef std::map<unsigned, std::set<unsigned> > RenameSetType;
  for (RenameSetType::iterator I = RenameSets.begin(), E = RenameSets.end();
       I != E; ++I)
    for (std::set<unsigned>::iterator SI = I->second.begin(),
         SE = I->second.end(); SI != SE; ++SI)
      Fn.getRegInfo().replaceRegWith(*SI, I->first);
  
  // FIXME: Insert last-minute copies
  
  // Remove PHIs
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    for (MachineBasicBlock::iterator BI = I->begin(), BE = I->end();
         BI != BE; ++BI)
      if (BI->getOpcode() == TargetInstrInfo::PHI)
        BI->eraseFromParent();
  
  return false;
}
