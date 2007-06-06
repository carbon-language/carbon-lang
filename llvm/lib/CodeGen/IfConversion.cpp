//===-- IfConversion.cpp - Machine code if conversion pass. ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the machine instruction level if-conversion pass.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "ifcvt"
#include "llvm/Function.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSimple,    "Number of simple if-conversions performed");
STATISTIC(NumSimpleRev, "Number of simple (reversed) if-conversions performed");
STATISTIC(NumTriangle,  "Number of triangle if-conversions performed");
STATISTIC(NumDiamonds,  "Number of diamond if-conversions performed");
STATISTIC(NumIfConvBBs, "Number of if-converted blocks");

namespace {
  class IfConverter : public MachineFunctionPass {
    enum BBICKind {
      ICNotAnalyzed,   // BB has not been analyzed.
      ICReAnalyze,     // BB must be re-analyzed.
      ICNotClassfied,  // BB data valid, but not classified.
      ICSimple,        // BB is entry of an one split, no rejoin sub-CFG.
      ICSimpleFalse,   // Same as ICSimple, but on the false path.
      ICTriangle,      // BB is entry of a triangle sub-CFG.
      ICDiamond,       // BB is entry of a diamond sub-CFG.
      ICChild,         // BB is part of the sub-CFG that'll be predicated.
      ICDead           // BB has been converted and merged, it's now dead.
    };

    /// BBInfo - One per MachineBasicBlock, this is used to cache the result
    /// if-conversion feasibility analysis. This includes results from
    /// TargetInstrInfo::AnalyzeBranch() (i.e. TBB, FBB, and Cond), and its
    /// classification, and common tail block of its successors (if it's a
    /// diamond shape), its size, whether it's predicable, and whether any
    /// instruction can clobber the 'would-be' predicate.
    ///
    /// Kind            - Type of block. See BBICKind.
    /// NonPredSize     - Number of non-predicated instructions.
    /// IsAnalyzable    - True if AnalyzeBranch() returns false.
    /// ModifyPredicate - FIXME: Not used right now. True if BB would modify
    ///                   the predicate (e.g. has cmp, call, etc.)
    /// BB              - Corresponding MachineBasicBlock.
    /// TrueBB / FalseBB- See AnalyzeBranch().
    /// BrCond          - Conditions for end of block conditional branches.
    /// Predicate       - Predicate used in the BB.
    struct BBInfo {
      BBICKind Kind;
      unsigned NonPredSize;
      bool IsAnalyzable;
      bool ModifyPredicate;
      MachineBasicBlock *BB;
      MachineBasicBlock *TrueBB;
      MachineBasicBlock *FalseBB;
      MachineBasicBlock *TailBB;
      std::vector<MachineOperand> BrCond;
      std::vector<MachineOperand> Predicate;
      BBInfo() : Kind(ICNotAnalyzed), NonPredSize(0),
                 IsAnalyzable(false), ModifyPredicate(false),
                 BB(0), TrueBB(0), FalseBB(0), TailBB(0) {}
    };

    /// Roots - Basic blocks that do not have successors. These are the starting
    /// points of Graph traversal.
    std::vector<MachineBasicBlock*> Roots;

    /// BBAnalysis - Results of if-conversion feasibility analysis indexed by
    /// basic block number.
    std::vector<BBInfo> BBAnalysis;

    const TargetLowering *TLI;
    const TargetInstrInfo *TII;
    bool MadeChange;
  public:
    static char ID;
    IfConverter() : MachineFunctionPass((intptr_t)&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "If converter"; }

  private:
    bool ReverseBranchCondition(BBInfo &BBI);
    void StructuralAnalysis(MachineBasicBlock *BB);
    bool FeasibilityAnalysis(BBInfo &BBI,
                             std::vector<MachineOperand> &Cond,
                             bool IgnoreTerm = false);
    bool AttemptRestructuring(BBInfo &BBI);
    bool AnalyzeBlocks(MachineFunction &MF,
                       std::vector<BBInfo*> &Candidates);
    void ReTryPreds(MachineBasicBlock *BB);
    bool IfConvertSimple(BBInfo &BBI);
    bool IfConvertTriangle(BBInfo &BBI);
    bool IfConvertDiamond(BBInfo &BBI);
    void PredicateBlock(BBInfo &BBI,
                        std::vector<MachineOperand> &Cond,
                        bool IgnoreTerm = false);
    void MergeBlocks(BBInfo &TrueBBI, BBInfo &FalseBBI);

    // IfcvtCandidateCmp - Used to sort if-conversion candidates.
    static bool IfcvtCandidateCmp(BBInfo* C1, BBInfo* C2){
      // Favor diamond over triangle, etc.
      return (unsigned)C1->Kind < (unsigned)C2->Kind;
    }
  };
  char IfConverter::ID = 0;
}

FunctionPass *llvm::createIfConverterPass() { return new IfConverter(); }

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  DOUT << "\nIfcvt: function \'" << MF.getFunction()->getName() << "\'\n";

  MF.RenumberBlocks();
  BBAnalysis.resize(MF.getNumBlockIDs());

  // Look for root nodes, i.e. blocks without successors.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    if (I->succ_size() == 0)
      Roots.push_back(I);

  std::vector<BBInfo*> Candidates;
  MadeChange = false;
  while (true) {
    // Do an intial analysis for each basic block and finding all the potential
    // candidates to perform if-convesion.
    bool Change = AnalyzeBlocks(MF, Candidates);
    while (!Candidates.empty()) {
      BBInfo &BBI = *Candidates.back();
      Candidates.pop_back();

      bool RetVal = false;
      switch (BBI.Kind) {
      default: assert(false && "Unexpected!");
        break;
      case ICReAnalyze:
        // One or more of 'children' have been modified, abort!
      case ICDead:
        // Block has been already been if-converted, abort!
        break;
      case ICSimple:
      case ICSimpleFalse:
        DOUT << "Ifcvt (Simple" << (BBI.Kind == ICSimpleFalse ? " false" : "")
             << "): BB#" << BBI.BB->getNumber() << " ";
        RetVal = IfConvertSimple(BBI);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal)
          if (BBI.Kind == ICSimple) NumSimple++;
          else                      NumSimpleRev++;
       break;
      case ICTriangle:
        DOUT << "Ifcvt (Triangle): BB#" << BBI.BB->getNumber() << " ";
        RetVal = IfConvertTriangle(BBI);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) NumTriangle++;
        break;
      case ICDiamond:
        DOUT << "Ifcvt (Diamond): BB#" << BBI.BB->getNumber() << " ";
        RetVal = IfConvertDiamond(BBI);
        DOUT << (RetVal ? "succeeded!" : "failed!") << "\n";
        if (RetVal) NumDiamonds++;
        break;
      }
      Change |= RetVal;
    }

    if (!Change)
      break;
    MadeChange |= Change;
  }

  Roots.clear();
  BBAnalysis.clear();

  return MadeChange;
}

static MachineBasicBlock *findFalseBlock(MachineBasicBlock *BB,
                                         MachineBasicBlock *TrueBB) {
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         E = BB->succ_end(); SI != E; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    if (SuccBB != TrueBB)
      return SuccBB;
  }
  return NULL;
}

bool IfConverter::ReverseBranchCondition(BBInfo &BBI) {
  if (!TII->ReverseBranchCondition(BBI.BrCond)) {
    TII->RemoveBranch(*BBI.BB);
    TII->InsertBranch(*BBI.BB, BBI.FalseBB, BBI.TrueBB, BBI.BrCond);
    std::swap(BBI.TrueBB, BBI.FalseBB);
    return true;
  }
  return false;
}

/// StructuralAnalysis - Analyze the structure of the sub-CFG starting from
/// the specified block. Record its successors and whether it looks like an
/// if-conversion candidate.
void IfConverter::StructuralAnalysis(MachineBasicBlock *BB) {
  BBInfo &BBI = BBAnalysis[BB->getNumber()];

  if (BBI.Kind == ICReAnalyze) {
    BBI.BrCond.clear();
    BBI.TrueBB = BBI.FalseBB = NULL;
  } else {
    if (BBI.Kind != ICNotAnalyzed)
      return;  // Already analyzed.
    BBI.BB = BB;
    BBI.NonPredSize = std::distance(BB->begin(), BB->end());
  }

  // Look for 'root' of a simple (non-nested) triangle or diamond.
  BBI.Kind = ICNotClassfied;
  BBI.IsAnalyzable =
    !TII->AnalyzeBranch(*BB, BBI.TrueBB, BBI.FalseBB, BBI.BrCond);
  if (!BBI.IsAnalyzable || BBI.BrCond.size() == 0)
    return;
  // Do not ifcvt if either path is a back edge to the entry block.
  if (BBI.TrueBB == BB || BBI.FalseBB == BB)
    return;

  StructuralAnalysis(BBI.TrueBB);
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];

  // No false branch. This BB must end with a conditional branch and a
  // fallthrough.
  if (!BBI.FalseBB)
    BBI.FalseBB = findFalseBlock(BB, BBI.TrueBB);  
  assert(BBI.FalseBB && "Expected to find the fallthrough block!");

  StructuralAnalysis(BBI.FalseBB);
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  
  // Look for more opportunities to if-convert a triangle. Try to restructure
  // the CFG to form a triangle with the 'false' path.
  std::vector<MachineOperand> RevCond(BBI.BrCond);
  bool CanRevCond = !TII->ReverseBranchCondition(RevCond);
  if (FalseBBI.FalseBB) {
    if (TrueBBI.TrueBB && TrueBBI.TrueBB == BBI.FalseBB)
      return;
    std::vector<MachineOperand> Cond(BBI.BrCond);
    if (CanRevCond &&
        FalseBBI.TrueBB && FalseBBI.BB->pred_size() == 1 &&
        FeasibilityAnalysis(FalseBBI, RevCond, true)) {
      std::vector<MachineOperand> FalseCond(FalseBBI.BrCond);
      if (FalseBBI.TrueBB == BBI.TrueBB &&
          TII->SubsumesPredicate(FalseCond, BBI.BrCond)) {
        // Reverse 'true' and 'false' paths.
        ReverseBranchCondition(BBI);
        BBI.Kind = ICTriangle;
        FalseBBI.Kind = ICChild;
      } else if (FalseBBI.FalseBB == BBI.TrueBB &&
                 !TII->ReverseBranchCondition(FalseCond) &&
                 TII->SubsumesPredicate(FalseCond, BBI.BrCond)) {
        // Reverse 'false' block's 'true' and 'false' paths and then
        // reverse 'true' and 'false' paths.
        ReverseBranchCondition(FalseBBI);
        ReverseBranchCondition(BBI);
        BBI.Kind = ICTriangle;
        FalseBBI.Kind = ICChild;
      }
    }
  } else if (TrueBBI.TrueBB == FalseBBI.TrueBB && CanRevCond &&
             TrueBBI.BB->pred_size() == 1 &&
             FalseBBI.BB->pred_size() == 1 &&
             // Check the 'true' and 'false' blocks if either isn't ended with
             // a branch. If the block does not fallthrough to another block
             // then we need to add a branch to its successor.
             !(TrueBBI.ModifyPredicate &&
               !TrueBBI.TrueBB && TrueBBI.BB->succ_size()) &&
             !(FalseBBI.ModifyPredicate &&
               !FalseBBI.TrueBB && FalseBBI.BB->succ_size()) &&
             FeasibilityAnalysis(TrueBBI, BBI.BrCond) &&
             FeasibilityAnalysis(FalseBBI, RevCond)) {
    // Diamond:
    //   EBB
    //   / \_
    //  |   |
    // TBB FBB
    //   \ /
    //  TailBB
    // Note TailBB can be empty.
    BBI.Kind = ICDiamond;
    TrueBBI.Kind = FalseBBI.Kind = ICChild;
    BBI.TailBB = TrueBBI.TrueBB;
  } else {
    // FIXME: Consider duplicating if BB is small.
    bool TryTriangle = TrueBBI.TrueBB && TrueBBI.TrueBB == BBI.FalseBB &&
                       TrueBBI.BB->pred_size() == 1;
    bool TrySimple = TrueBBI.BrCond.size() == 0 && TrueBBI.BB->pred_size() == 1;
    if ((TryTriangle || TrySimple) &&
        FeasibilityAnalysis(TrueBBI, BBI.BrCond)) {
      if (TryTriangle) {
        // Triangle:
        //   EBB
        //   | \_
        //   |  |
        //   | TBB
        //   |  /
        //   FBB
        BBI.Kind = ICTriangle;
        TrueBBI.Kind = ICChild;
      } else {
        // Simple (split, no rejoin):
        //   EBB
        //   | \_
        //   |  |
        //   | TBB---> exit
        //   |    
        //   FBB
        BBI.Kind = ICSimple;
        TrueBBI.Kind = ICChild;
      }
    } else if (FalseBBI.BrCond.size() == 0 && FalseBBI.BB->pred_size() == 1) {
      // Try the other path...
      std::vector<MachineOperand> RevCond(BBI.BrCond);
      if (!TII->ReverseBranchCondition(RevCond) &&
          FeasibilityAnalysis(FalseBBI, RevCond)) {
        if (FalseBBI.TrueBB && FalseBBI.TrueBB == BBI.TrueBB &&
            FalseBBI.BB->pred_size() == 1) {
          // Reverse 'true' and 'false' paths.
          ReverseBranchCondition(BBI);
          BBI.Kind = ICTriangle;
          FalseBBI.Kind = ICChild;
        } else {
          BBI.Kind = ICSimpleFalse;
          FalseBBI.Kind = ICChild;
        }
      }
    }
  }
  return;
}

/// FeasibilityAnalysis - Determine if the block is predicable. In most
/// cases, that means all the instructions in the block has M_PREDICABLE flag.
/// Also checks if the block contains any instruction which can clobber a
/// predicate (e.g. condition code register). If so, the block is not
/// predicable unless it's the last instruction. If IgnoreTerm is true then
/// all the terminator instructions are skipped.
bool IfConverter::FeasibilityAnalysis(BBInfo &BBI,
                                      std::vector<MachineOperand> &Cond,
                                      bool IgnoreTerm) {
  // If the block is dead, or it is going to be the entry block of a sub-CFG
  // that will be if-converted, then it cannot be predicated.
  if (BBI.Kind != ICNotAnalyzed &&
      BBI.Kind != ICNotClassfied &&
      BBI.Kind != ICChild)
    return false;

  // Check predication threshold.
  if (BBI.NonPredSize == 0 || BBI.NonPredSize > TLI->getIfCvtBlockSizeLimit())
    return false;

  // If it is already predicated, check if its predicate subsumes the new
  // predicate.
  if (BBI.Predicate.size() && !TII->SubsumesPredicate(BBI.Predicate, Cond))
    return false;

  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    if (IgnoreTerm && TII->isTerminatorInstr(I->getOpcode()))
      continue;
    // TODO: check if instruction clobbers predicate.
    if (!I->isPredicable())
      return false;
  }

  return true;
}

/// AttemptRestructuring - Restructure the sub-CFG rooted in the given block to
/// expose more if-conversion opportunities. e.g.
///
///                cmp
///                b le BB1
///                /  \____
///               /        |
///             cmp        |
///             b eq BB1   |
///              /  \____  |
///             /        \ |
///                      BB1
///  ==>
///
///                cmp
///                b eq BB1
///                /  \____
///               /        |
///             cmp        |
///             b le BB1   |
///              /  \____  |
///             /        \ |
///                      BB1
bool IfConverter::AttemptRestructuring(BBInfo &BBI) {
  return false;
}

/// AnalyzeBlocks - Analyze all blocks and find entries for all if-conversion
/// candidates. It returns true if any CFG restructuring is done to expose more
/// if-conversion opportunities.
bool IfConverter::AnalyzeBlocks(MachineFunction &MF,
                                std::vector<BBInfo*> &Candidates) {
  bool Change = false;
  std::set<MachineBasicBlock*> Visited;
  for (unsigned i = 0, e = Roots.size(); i != e; ++i) {
    for (idf_ext_iterator<MachineBasicBlock*> I=idf_ext_begin(Roots[i],Visited),
           E = idf_ext_end(Roots[i], Visited); I != E; ++I) {
      MachineBasicBlock *BB = *I;
      StructuralAnalysis(BB);
      BBInfo &BBI = BBAnalysis[BB->getNumber()];
      switch (BBI.Kind) {
        case ICSimple:
        case ICSimpleFalse:
        case ICTriangle:
        case ICDiamond:
          Candidates.push_back(&BBI);
          break;
        default:
          Change |= AttemptRestructuring(BBI);
          break;
      }
    }
  }

  // Sort to favor more complex ifcvt scheme.
  std::stable_sort(Candidates.begin(), Candidates.end(), IfcvtCandidateCmp);

  return Change;
}

/// isNextBlock - Returns true either if ToBB the next block after BB or
/// that all the intervening blocks are empty.
static bool isNextBlock(MachineBasicBlock *BB, MachineBasicBlock *ToBB) {
  MachineFunction::iterator I = BB;
  MachineFunction::iterator TI = ToBB;
  MachineFunction::iterator E = BB->getParent()->end();
  while (++I != TI)
    if (I == E || !I->empty())
      return false;
  return true;
}

/// ReTryPreds - Invalidate predecessor BB info so it would be re-analyzed
/// to determine if it can be if-converted.
void IfConverter::ReTryPreds(MachineBasicBlock *BB) {
  for (MachineBasicBlock::pred_iterator PI = BB->pred_begin(),
         E = BB->pred_end(); PI != E; ++PI) {
    BBInfo &PBBI = BBAnalysis[(*PI)->getNumber()];
    PBBI.Kind = ICReAnalyze;
  }
}

/// InsertUncondBranch - Inserts an unconditional branch from BB to ToBB.
///
static void InsertUncondBranch(MachineBasicBlock *BB, MachineBasicBlock *ToBB,
                               const TargetInstrInfo *TII) {
  std::vector<MachineOperand> NoCond;
  TII->InsertBranch(*BB, ToBB, NULL, NoCond);
}

/// IfConvertSimple - If convert a simple (split, no rejoin) sub-CFG.
///
bool IfConverter::IfConvertSimple(BBInfo &BBI) {
  bool ReverseCond = BBI.Kind == ICSimpleFalse;

  BBI.Kind = ICNotClassfied;

  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  BBInfo *CvtBBI = &TrueBBI;
  BBInfo *NextBBI = &FalseBBI;

  std::vector<MachineOperand> Cond(BBI.BrCond);
  if (ReverseCond) {
    std::swap(CvtBBI, NextBBI);
    TII->ReverseBranchCondition(Cond);
  }

  PredicateBlock(*CvtBBI, Cond);
  // If the 'true' block ends without a branch, add a conditional branch
  // to its successor unless that happens to be the 'false' block.
  if (CvtBBI->IsAnalyzable && CvtBBI->TrueBB == NULL) {
    assert(CvtBBI->BB->succ_size() == 1 && "Unexpected!");
    MachineBasicBlock *SuccBB = *CvtBBI->BB->succ_begin();
    if (SuccBB != NextBBI->BB)
      TII->InsertBranch(*CvtBBI->BB, SuccBB, NULL, Cond);
  }

  // Merge converted block into entry block. Also add an unconditional branch
  // to the 'false' branch.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
  MergeBlocks(BBI, *CvtBBI);
  if (!isNextBlock(BBI.BB, NextBBI->BB))
    InsertUncondBranch(BBI.BB, NextBBI->BB, TII);
  std::copy(Cond.begin(), Cond.end(), std::back_inserter(BBI.Predicate));

  // Update block info. BB can be iteratively if-converted.
  BBI.Kind = ICReAnalyze;
  ReTryPreds(BBI.BB);
  CvtBBI->Kind = ICDead;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// IfConvertTriangle - If convert a triangle sub-CFG.
///
bool IfConverter::IfConvertTriangle(BBInfo &BBI) {
  BBI.Kind = ICNotClassfied;

  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];

  // Predicate the 'true' block after removing its branch.
  TrueBBI.NonPredSize -= TII->RemoveBranch(*BBI.TrueBB);
  PredicateBlock(TrueBBI, BBI.BrCond);

  // If 'true' block has a 'false' successor, add an exit branch to it.
  bool HasEarlyExit = TrueBBI.FalseBB != NULL;
  if (HasEarlyExit) {
    std::vector<MachineOperand> RevCond(TrueBBI.BrCond);
    if (TII->ReverseBranchCondition(RevCond))
      assert(false && "Unable to reverse branch condition!");
    TII->InsertBranch(*BBI.TrueBB, TrueBBI.FalseBB, NULL, RevCond);
  }

  // Join the 'true' and 'false' blocks if the 'false' block has no other
  // predecessors. Otherwise, add a unconditional branch from 'true' to 'false'.
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  bool FalseBBDead = false;
  if (!HasEarlyExit && FalseBBI.BB->pred_size() == 2) {
    MergeBlocks(TrueBBI, FalseBBI);
    FalseBBDead = true;
  } else if (!isNextBlock(TrueBBI.BB, FalseBBI.BB))
    InsertUncondBranch(TrueBBI.BB, FalseBBI.BB, TII);

  // Now merge the entry of the triangle with the true block.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
  MergeBlocks(BBI, TrueBBI);
  std::copy(BBI.BrCond.begin(), BBI.BrCond.end(),
            std::back_inserter(BBI.Predicate));

  // Update block info. BB can be iteratively if-converted.
  BBI.Kind = ICReAnalyze;
  ReTryPreds(BBI.BB);
  TrueBBI.Kind = ICDead;
  if (FalseBBDead)
    FalseBBI.Kind = ICDead;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// IfConvertDiamond - If convert a diamond sub-CFG.
///
bool IfConverter::IfConvertDiamond(BBInfo &BBI) {
  BBI.Kind = ICNotClassfied;

  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];

  SmallVector<MachineInstr*, 2> Dups;
  if (!BBI.TailBB) {
    // No common merge block. Check if the terminators (e.g. return) are
    // the same or predicable.
    MachineBasicBlock::iterator TT = BBI.TrueBB->getFirstTerminator();
    MachineBasicBlock::iterator FT = BBI.FalseBB->getFirstTerminator();
    while (TT != BBI.TrueBB->end() && FT != BBI.FalseBB->end()) {
      if (TT->isIdenticalTo(FT))
        Dups.push_back(TT);  // Will erase these later.
      else if (!TT->isPredicable() && !FT->isPredicable())
        return false; // Can't if-convert. Abort!
      ++TT;
      ++FT;
    }

    // One of the two pathes have more terminators, make sure they are
    // all predicable.
    while (TT != BBI.TrueBB->end()) {
      if (!TT->isPredicable()) {
        return false; // Can't if-convert. Abort!
      }
      ++TT;
    }
    while (FT != BBI.FalseBB->end()) {
      if (!FT->isPredicable()) {
        return false; // Can't if-convert. Abort!
      }
      ++FT;
    }
  }

  // Remove the duplicated instructions from the 'true' block.
  for (unsigned i = 0, e = Dups.size(); i != e; ++i) {
    Dups[i]->eraseFromParent();
    --TrueBBI.NonPredSize;
  }
    
  // Merge the 'true' and 'false' blocks by copying the instructions
  // from the 'false' block to the 'true' block. That is, unless the true
  // block would clobber the predicate, in that case, do the opposite.
  BBInfo *BBI1 = &TrueBBI;
  BBInfo *BBI2 = &FalseBBI;
  std::vector<MachineOperand> RevCond(BBI.BrCond);
  TII->ReverseBranchCondition(RevCond);
  std::vector<MachineOperand> *Cond1 = &BBI.BrCond;
  std::vector<MachineOperand> *Cond2 = &RevCond;
  // Check the 'true' and 'false' blocks if either isn't ended with a branch.
  // Either the block fallthrough to another block or it ends with a
  // return. If it's the former, add a branch to its successor.
  bool NeedBr1 = !BBI1->TrueBB && BBI1->BB->succ_size();
  bool NeedBr2 = !BBI2->TrueBB && BBI2->BB->succ_size(); 

  if ((TrueBBI.ModifyPredicate && !FalseBBI.ModifyPredicate) ||
      (!TrueBBI.ModifyPredicate && !FalseBBI.ModifyPredicate &&
       NeedBr1 && !NeedBr2)) {
    std::swap(BBI1, BBI2);
    std::swap(Cond1, Cond2);
    std::swap(NeedBr1, NeedBr2);
  }

  // Predicate the 'true' block after removing its branch.
  BBI1->NonPredSize -= TII->RemoveBranch(*BBI1->BB);
  PredicateBlock(*BBI1, *Cond1);

  // Add an early exit branch if needed.
  if (NeedBr1)
    TII->InsertBranch(*BBI1->BB, *BBI1->BB->succ_begin(), NULL, *Cond1);

  // Predicate the 'false' block.
  PredicateBlock(*BBI2, *Cond2, true);

  // Add an unconditional branch from 'false' to to 'false' successor if it
  // will not be the fallthrough block.
  if (NeedBr2 && !isNextBlock(BBI2->BB, *BBI2->BB->succ_begin()))
    InsertUncondBranch(BBI2->BB, *BBI2->BB->succ_begin(), TII);

  // Keep them as two separate blocks if there is an early exit.
  if (!NeedBr1)
    MergeBlocks(*BBI1, *BBI2);

  // Remove the conditional branch from entry to the blocks.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);

  // Merge the combined block into the entry of the diamond.
  MergeBlocks(BBI, *BBI1);

  // 'True' and 'false' aren't combined, see if we need to add a unconditional
  // branch to the 'false' block.
  if (NeedBr1 && !isNextBlock(BBI.BB, BBI2->BB))
    InsertUncondBranch(BBI1->BB, BBI2->BB, TII);

  // If the if-converted block fallthrough or unconditionally branch into the
  // tail block, and the tail block does not have other predecessors, then
  // fold the tail block in as well.
  BBInfo *CvtBBI = NeedBr1 ? BBI2 : &BBI;
  if (BBI.TailBB &&
      BBI.TailBB->pred_size() == 1 && CvtBBI->BB->succ_size() == 1) {
    CvtBBI->NonPredSize -= TII->RemoveBranch(*CvtBBI->BB);
    BBInfo TailBBI = BBAnalysis[BBI.TailBB->getNumber()];
    MergeBlocks(*CvtBBI, TailBBI);
    TailBBI.Kind = ICDead;
  }

  // Update block info.
  TrueBBI.Kind = ICDead;
  FalseBBI.Kind = ICDead;

  // FIXME: Must maintain LiveIns.
  return true;
}

/// PredicateBlock - Predicate every instruction in the block with the specified
/// condition. If IgnoreTerm is true, skip over all terminator instructions.
void IfConverter::PredicateBlock(BBInfo &BBI,
                                 std::vector<MachineOperand> &Cond,
                                 bool IgnoreTerm) {
  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    if (IgnoreTerm && TII->isTerminatorInstr(I->getOpcode()))
      continue;
    if (TII->isPredicated(I))
      continue;
    if (!TII->PredicateInstruction(I, Cond)) {
      cerr << "Unable to predicate " << *I << "!\n";
      abort();
    }
  }

  BBI.NonPredSize = 0;
  NumIfConvBBs++;
}

/// TransferPreds - Transfer all the predecessors of FromBB to ToBB.
///
static void TransferPreds(MachineBasicBlock *ToBB, MachineBasicBlock *FromBB) {
  for (MachineBasicBlock::pred_iterator I = FromBB->pred_begin(),
         E = FromBB->pred_end(); I != E; ++I) {
    MachineBasicBlock *Pred = *I;
    Pred->removeSuccessor(FromBB);
    if (!Pred->isSuccessor(ToBB))
      Pred->addSuccessor(ToBB);
  }
}

/// TransferSuccs - Transfer all the successors of FromBB to ToBB.
///
static void TransferSuccs(MachineBasicBlock *ToBB, MachineBasicBlock *FromBB) {
  for (MachineBasicBlock::succ_iterator I = FromBB->succ_begin(),
         E = FromBB->succ_end(); I != E; ++I) {
    MachineBasicBlock *Succ = *I;
    FromBB->removeSuccessor(Succ);
    if (!ToBB->isSuccessor(Succ))
      ToBB->addSuccessor(Succ);
  }
}

/// MergeBlocks - Move all instructions from FromBB to the end of ToBB.
///
void IfConverter::MergeBlocks(BBInfo &ToBBI, BBInfo &FromBBI) {
  ToBBI.BB->splice(ToBBI.BB->end(),
                   FromBBI.BB, FromBBI.BB->begin(), FromBBI.BB->end());

  // If FromBBI is previously a successor, remove it from ToBBI's successor
  // list and update its TrueBB / FalseBB field if needed.
  if (ToBBI.BB->isSuccessor(FromBBI.BB))
    ToBBI.BB->removeSuccessor(FromBBI.BB);

  // Redirect all branches to FromBB to ToBB.
  std::vector<MachineBasicBlock *> Preds(FromBBI.BB->pred_begin(),
                                         FromBBI.BB->pred_end());
  for (unsigned i = 0, e = Preds.size(); i != e; ++i)
    Preds[i]->ReplaceUsesOfBlockWith(FromBBI.BB, ToBBI.BB);

  // Transfer preds / succs and update size.
  TransferPreds(ToBBI.BB, FromBBI.BB);
  TransferSuccs(ToBBI.BB, FromBBI.BB);
  ToBBI.NonPredSize += FromBBI.NonPredSize;
  FromBBI.NonPredSize = 0;
}
