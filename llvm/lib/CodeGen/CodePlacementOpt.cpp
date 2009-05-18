//===-- CodePlacementOpt.cpp - Code Placement pass. -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass that optimize code placement and align loop
// headers to target specific alignment boundary.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "code-placement"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumHeaderAligned, "Number of loop header aligned");
STATISTIC(NumIntraElim,     "Number of intra loop branches eliminated");
STATISTIC(NumIntraMoved,    "Number of intra loop branches moved");

namespace {
  class CodePlacementOpt : public MachineFunctionPass {
    const MachineLoopInfo *MLI;
    const TargetInstrInfo *TII;
    const TargetLowering  *TLI;

    /// ChangedMBBs - BBs which are modified by OptimizeIntraLoopEdges.
    SmallPtrSet<MachineBasicBlock*, 8> ChangedMBBs;

    /// UncondJmpMBBs - A list of BBs which are in loops and end with
    /// unconditional branches.
    SmallVector<std::pair<MachineBasicBlock*,MachineBasicBlock*>, 4>
    UncondJmpMBBs;

    /// LoopHeaders - A list of BBs which are loop headers.
    SmallVector<MachineBasicBlock*, 4> LoopHeaders;

  public:
    static char ID;
    CodePlacementOpt() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const {
      return "Code Placement Optimizater";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineLoopInfo>();
      AU.addPreservedID(MachineDominatorsID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    bool OptimizeIntraLoopEdges();
    bool HeaderShouldBeAligned(MachineBasicBlock *MBB, MachineLoop *L,
                               SmallPtrSet<MachineBasicBlock*, 4> &DoNotAlign);
    bool AlignLoops(MachineFunction &MF);
  };

  char CodePlacementOpt::ID = 0;
} // end anonymous namespace

FunctionPass *llvm::createCodePlacementOptPass() {
  return new CodePlacementOpt();
}

/// OptimizeBackEdges - Place loop back edges to move unconditional branches
/// out of the loop.
///
///       A:
///       ...
///       <fallthrough to B>
///
///       B:  --> loop header
///       ...
///       jcc <cond> C, [exit]
///
///       C:
///       ...
///       jmp B
///
/// ==>
///
///       A:
///       ...
///       jmp B
///
///       C:  --> new loop header
///       ...
///       <fallthough to B>
///       
///       B:
///       ...
///       jcc <cond> C, [exit]
///
bool CodePlacementOpt::OptimizeIntraLoopEdges() {
  if (!TLI->shouldOptimizeCodePlacement())
    return false;

  bool Changed = false;
  for (unsigned i = 0, e = UncondJmpMBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = UncondJmpMBBs[i].first;
    MachineBasicBlock *SuccMBB = UncondJmpMBBs[i].second;
    MachineLoop *L = MLI->getLoopFor(MBB);
    assert(L && "BB is expected to be in a loop!");

    if (ChangedMBBs.count(MBB)) {
      // BB has been modified, re-analyze.
      MachineBasicBlock *TBB = 0, *FBB = 0;
      SmallVector<MachineOperand, 4> Cond;
      if (TII->AnalyzeBranch(*MBB, TBB, FBB, Cond) || !Cond.empty())
        continue;
      if (MLI->getLoopFor(TBB) != L || TBB->isLandingPad())
        continue;
      SuccMBB = TBB;
    } else {
      assert(MLI->getLoopFor(SuccMBB) == L &&
             "Successor is not in the same loop!");
    }

    if (MBB->isLayoutSuccessor(SuccMBB)) {
      // Successor is right after MBB, just eliminate the unconditional jmp.
      // Can this happen?
      TII->RemoveBranch(*MBB);
      ChangedMBBs.insert(MBB);
      ++NumIntraElim;
      Changed = true;
      continue;
    }

    // Now check if the predecessor is fallthrough from any BB. If there is,
    // that BB should be from outside the loop since edge will become a jmp.
    bool OkToMove = true;
    MachineBasicBlock *FtMBB = 0, *FtTBB = 0, *FtFBB = 0;
    SmallVector<MachineOperand, 4> FtCond;    
    for (MachineBasicBlock::pred_iterator PI = SuccMBB->pred_begin(),
           PE = SuccMBB->pred_end(); PI != PE; ++PI) {
      MachineBasicBlock *PredMBB = *PI;
      if (PredMBB->isLayoutSuccessor(SuccMBB)) {
        if (TII->AnalyzeBranch(*PredMBB, FtTBB, FtFBB, FtCond)) {
          OkToMove = false;
          break;
        }
        if (!FtTBB)
          FtTBB = SuccMBB;
        else if (!FtFBB) {
          assert(FtFBB != SuccMBB && "Unexpected control flow!");
          FtFBB = SuccMBB;
        }
        
        // A fallthrough.
        FtMBB = PredMBB;
        MachineLoop *PL = MLI->getLoopFor(PredMBB);
        if (PL && (PL == L || PL->getLoopDepth() >= L->getLoopDepth()))
          OkToMove = false;

        break;
      }
    }

    if (!OkToMove)
      continue;

    // Is it profitable? If SuccMBB can fallthrough itself, that can be changed
    // into a jmp.
    MachineBasicBlock *TBB = 0, *FBB = 0;
    SmallVector<MachineOperand, 4> Cond;
    if (TII->AnalyzeBranch(*SuccMBB, TBB, FBB, Cond))
      continue;
    if (!TBB && Cond.empty())
      TBB = next(MachineFunction::iterator(SuccMBB));
    else if (!FBB && !Cond.empty())
      FBB = next(MachineFunction::iterator(SuccMBB));

    // This calculate the cost of the transformation. Also, it finds the *only*
    // intra-loop edge if there is one.
    int Cost = 0;
    bool HasOneIntraSucc = true;
    MachineBasicBlock *IntraSucc = 0;
    for (MachineBasicBlock::succ_iterator SI = SuccMBB->succ_begin(),
           SE = SuccMBB->succ_end(); SI != SE; ++SI) {
      MachineBasicBlock *SSMBB = *SI;
      if (MLI->getLoopFor(SSMBB) == L) {
        if (!IntraSucc)
          IntraSucc = SSMBB;
        else
          HasOneIntraSucc = false;
      }

      if (SuccMBB->isLayoutSuccessor(SSMBB))
        // This will become a jmp.
        ++Cost;
      else if (MBB->isLayoutSuccessor(SSMBB)) {
        // One of the successor will become the new fallthrough.
        if (SSMBB == FBB) {
          FBB = 0;
          --Cost;
        } else if (!FBB && SSMBB == TBB && Cond.empty()) {
          TBB = 0;
          --Cost;
        } else if (!Cond.empty() && !TII->ReverseBranchCondition(Cond)) {
          assert(SSMBB == TBB);
          TBB = FBB;
          FBB = 0;
          --Cost;
        }
      }
    }
    if (Cost)
      continue;

    // Now, let's move the successor to below the BB to eliminate the jmp.
    SuccMBB->moveAfter(MBB);
    TII->RemoveBranch(*MBB);
    TII->RemoveBranch(*SuccMBB);
    if (TBB)
      TII->InsertBranch(*SuccMBB, TBB, FBB, Cond);
    ChangedMBBs.insert(MBB);
    ChangedMBBs.insert(SuccMBB);
    if (FtMBB) {
      TII->RemoveBranch(*FtMBB);
      TII->InsertBranch(*FtMBB, FtTBB, FtFBB, FtCond);
      ChangedMBBs.insert(FtMBB);
    }
    Changed = true;

    // If BB is the loop latch, we may have a new loop headr.
    if (MBB == L->getLoopLatch()) {
      assert(MLI->isLoopHeader(SuccMBB) &&
             "Only succ of loop latch is not the header?");
      if (HasOneIntraSucc && IntraSucc)
        std::replace(LoopHeaders.begin(),LoopHeaders.end(), SuccMBB, IntraSucc);
    }
  }

  ++NumIntraMoved;
  return Changed;
}

/// HeaderShouldBeAligned - Return true if the specified loop header block
/// should be aligned. For now, we will not align it if all the predcessors
/// (i.e. loop back edges) are laid out above the header. FIXME: Do not
/// align small loops.
bool
CodePlacementOpt::HeaderShouldBeAligned(MachineBasicBlock *MBB, MachineLoop *L,
                               SmallPtrSet<MachineBasicBlock*, 4> &DoNotAlign) {
  if (DoNotAlign.count(MBB))
    return false;

  bool BackEdgeBelow = false;
  for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
         PE = MBB->pred_end(); PI != PE; ++PI) {
    MachineBasicBlock *PredMBB = *PI;
    if (PredMBB == MBB || PredMBB->getNumber() > MBB->getNumber()) {
      BackEdgeBelow = true;
      break;
    }
  }

  if (!BackEdgeBelow)
    return false;

  // Ok, we are going to align this loop header. If it's an inner loop,
  // do not align its outer loop.
  MachineBasicBlock *PreHeader = L->getLoopPreheader();
  if (PreHeader) {
    MachineLoop *L = MLI->getLoopFor(PreHeader);
    if (L) {
      MachineBasicBlock *HeaderBlock = L->getHeader();
      HeaderBlock->setAlignment(0);
      DoNotAlign.insert(HeaderBlock);
    }
  }
  return true;
}

/// AlignLoops - Align loop headers to target preferred alignments.
///
bool CodePlacementOpt::AlignLoops(MachineFunction &MF) {
  const Function *F = MF.getFunction();
  if (F->hasFnAttr(Attribute::OptimizeForSize))
    return false;

  unsigned Align = TLI->getPrefLoopAlignment();
  if (!Align)
    return false;  // Don't care about loop alignment.

  // Make sure blocks are numbered in order
  MF.RenumberBlocks();

  bool Changed = false;
  SmallPtrSet<MachineBasicBlock*, 4> DoNotAlign;
  for (unsigned i = 0, e = LoopHeaders.size(); i != e; ++i) {
    MachineBasicBlock *HeaderMBB = LoopHeaders[i];
    MachineBasicBlock *PredMBB = prior(MachineFunction::iterator(HeaderMBB));
    MachineLoop *L = MLI->getLoopFor(HeaderMBB);
    if (L == MLI->getLoopFor(PredMBB))
      // If previously BB is in the same loop, don't align this BB. We want
      // to prevent adding noop's inside a loop.
      continue;
    if (HeaderShouldBeAligned(HeaderMBB, L, DoNotAlign)) {
      HeaderMBB->setAlignment(Align);
      Changed = true;
      ++NumHeaderAligned;
    }
  }

  return Changed;
}

bool CodePlacementOpt::runOnMachineFunction(MachineFunction &MF) {
  MLI = &getAnalysis<MachineLoopInfo>();
  if (MLI->empty())
    return false;  // No loops.

  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();

  // Analyze the BBs first and keep track of loop headers and BBs that
  // end with an unconditional jmp to another block in the same loop.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = I;
    if (MBB->isLandingPad())
      continue;
    MachineLoop *L = MLI->getLoopFor(MBB);
    if (!L)
      continue;
    if (MLI->isLoopHeader(MBB))
      LoopHeaders.push_back(MBB);

    MachineBasicBlock *TBB = 0, *FBB = 0;
    SmallVector<MachineOperand, 4> Cond;
    if (TII->AnalyzeBranch(*MBB, TBB, FBB, Cond) || !Cond.empty())
      continue;
    if (MLI->getLoopFor(TBB) == L && !TBB->isLandingPad())
      UncondJmpMBBs.push_back(std::make_pair(MBB, TBB));
  }

  bool Changed = OptimizeIntraLoopEdges();

  Changed |= AlignLoops(MF);

  ChangedMBBs.clear();
  UncondJmpMBBs.clear();
  LoopHeaders.clear();

  return Changed;
}
