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

#define DEBUG_TYPE "ifconversion"
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

STATISTIC(NumIfConvBBs, "Number of if-converted blocks");

namespace {
  class IfConverter : public MachineFunctionPass {
    enum BBICKind {
      ICNotAnalyzed,   // BB has not been analyzed.
      ICReAnalyze,     // BB must be re-analyzed.
      ICNotClassfied,  // BB data valid, but not classified.
      ICEarlyExit,     // BB is entry of an early-exit sub-CFG.
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
    /// isPredicable    - Is it predicable. (FIXME: Remove.)
    /// hasEarlyExit    - Ends with a return, indirect jump or br to jumptable.
    /// ModifyPredicate - FIXME: Not used right now. True if BB would modify
    ///                   the predicate (e.g. has cmp, call, etc.)
    /// BB              - Corresponding MachineBasicBlock.
    /// TrueBB / FalseBB- See AnalyzeBranch().
    /// BrCond          - Conditions for end of block conditional branches.
    /// Predicate       - Predicate used in the BB.
    struct BBInfo {
      BBICKind Kind;
      unsigned NonPredSize;
      bool isPredicable;
      bool hasEarlyExit;
      bool ModifyPredicate;
      MachineBasicBlock *BB;
      MachineBasicBlock *TrueBB;
      MachineBasicBlock *FalseBB;
      MachineBasicBlock *TailBB;
      std::vector<MachineOperand> BrCond;
      std::vector<MachineOperand> Predicate;
      BBInfo() : Kind(ICNotAnalyzed), NonPredSize(0), isPredicable(false),
                 hasEarlyExit(false), ModifyPredicate(false),
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
    void StructuralAnalysis(MachineBasicBlock *BB);
    void FeasibilityAnalysis(BBInfo &BBI,
                             std::vector<MachineOperand> &Cond);
    bool AttemptRestructuring(BBInfo &BBI);
    bool AnalyzeBlocks(MachineFunction &MF,
                       std::vector<BBInfo*> &Candidates);
    void ReTryPreds(MachineBasicBlock *BB);
    bool IfConvertEarlyExit(BBInfo &BBI);
    bool IfConvertTriangle(BBInfo &BBI);
    bool IfConvertDiamond(BBInfo &BBI);
    void PredicateBlock(BBInfo &BBI,
                        std::vector<MachineOperand> &Cond,
                        bool IgnoreTerm = false);
    void MergeBlocks(BBInfo &TrueBBI, BBInfo &FalseBBI);
  };
  char IfConverter::ID = 0;
}

FunctionPass *llvm::createIfConverterPass() { return new IfConverter(); }

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TLI = MF.getTarget().getTargetLowering();
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  MF.RenumberBlocks();
  unsigned NumBBs = MF.getNumBlockIDs();
  BBAnalysis.resize(NumBBs);

  // Look for root nodes, i.e. blocks without successors.
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    if (I->succ_size() == 0)
      Roots.push_back(I);

  std::vector<BBInfo*> Candidates;
  MadeChange = false;
  while (true) {
    bool Change = false;

    // Do an intial analysis for each basic block and finding all the potential
    // candidates to perform if-convesion.
    Change |= AnalyzeBlocks(MF, Candidates);
    while (!Candidates.empty()) {
      BBInfo &BBI = *Candidates.back();
      Candidates.pop_back();
      switch (BBI.Kind) {
      default: assert(false && "Unexpected!");
        break;
      case ICEarlyExit:
        Change |= IfConvertEarlyExit(BBI);
        break;
      case ICTriangle:
        Change |= IfConvertTriangle(BBI);
        break;
      case ICDiamond:
        Change |= IfConvertDiamond(BBI);
        break;
      }
    }

    MadeChange |= Change;
    if (!Change)
      break;
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

/// StructuralAnalysis - Analyze the structure of the sub-CFG starting from
/// the specified block. Record its successors and whether it looks like an
/// if-conversion candidate.
void IfConverter::StructuralAnalysis(MachineBasicBlock *BB) {
  BBInfo &BBI = BBAnalysis[BB->getNumber()];

  if (BBI.Kind != ICReAnalyze) {
    if (BBI.Kind != ICNotAnalyzed)
      return;  // Already analyzed.
    BBI.BB = BB;
    BBI.NonPredSize = std::distance(BB->begin(), BB->end());
  }

  // Look for 'root' of a simple (non-nested) triangle or diamond.
  BBI.Kind = ICNotClassfied;
  bool CanAnalyze = !TII->AnalyzeBranch(*BB, BBI.TrueBB, BBI.FalseBB,
                                        BBI.BrCond);
  // Does it end with a return, indirect jump, or jumptable branch?
  BBI.hasEarlyExit = TII->BlockHasNoFallThrough(*BB) && !BBI.TrueBB;
  if (!CanAnalyze || !BBI.TrueBB || BBI.BrCond.size() == 0)
    return;

  // Not a candidate if 'true' block is going to be if-converted.
  StructuralAnalysis(BBI.TrueBB);
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  if (TrueBBI.Kind != ICNotClassfied)
    return;

  // TODO: Only handle very simple cases for now.
  if (TrueBBI.FalseBB || TrueBBI.BrCond.size())
    return;

  // No false branch. This BB must end with a conditional branch and a
  // fallthrough.
  if (!BBI.FalseBB)
    BBI.FalseBB = findFalseBlock(BB, BBI.TrueBB);  
  assert(BBI.FalseBB && "Expected to find the fallthrough block!");

  // Not a candidate if 'false' block is going to be if-converted.
  StructuralAnalysis(BBI.FalseBB);
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  if (FalseBBI.Kind != ICNotClassfied)
    return;

  // TODO: Only handle very simple cases for now.
  if (FalseBBI.FalseBB || FalseBBI.BrCond.size())
    return;

  unsigned TrueNumPreds  = BBI.TrueBB->pred_size();
  unsigned FalseNumPreds = BBI.FalseBB->pred_size();
  if ((TrueBBI.hasEarlyExit && TrueNumPreds <= 1) &&
      !(FalseBBI.hasEarlyExit && FalseNumPreds <=1)) {
    BBI.Kind = ICEarlyExit;
    TrueBBI.Kind = ICChild;
  } else if (!(TrueBBI.hasEarlyExit && TrueNumPreds <= 1) &&
             (FalseBBI.hasEarlyExit && FalseNumPreds <=1)) {
    BBI.Kind = ICEarlyExit;
    FalseBBI.Kind = ICChild;
  } else if (TrueBBI.TrueBB && TrueBBI.TrueBB == BBI.FalseBB) {
    // Triangle:
    //   EBB
    //   | \_
    //   |  |
    //   | TBB
    //   |  /
    //   FBB
    BBI.Kind = ICTriangle;
    TrueBBI.Kind = FalseBBI.Kind = ICChild;
  } else if (TrueBBI.TrueBB == FalseBBI.TrueBB &&
             TrueNumPreds <= 1 && FalseNumPreds <= 1) {
    // Diamond:
    //   EBB
    //   / \_
    //  |   |
    // TBB FBB
    //   \ /
    //  TailBB
    // Note MBB can be empty in case both TBB and FBB are return blocks.
    BBI.Kind = ICDiamond;
    TrueBBI.Kind = FalseBBI.Kind = ICChild;
    BBI.TailBB = TrueBBI.TrueBB;
  }
  return;
}

/// FeasibilityAnalysis - Determine if the block is predicable. In most
/// cases, that means all the instructions in the block has M_PREDICABLE flag.
/// Also checks if the block contains any instruction which can clobber a
/// predicate (e.g. condition code register). If so, the block is not
/// predicable unless it's the last instruction. Note, this function assumes
/// all the terminator instructions can be converted or deleted so it ignore
/// them.
void IfConverter::FeasibilityAnalysis(BBInfo &BBI,
                                      std::vector<MachineOperand> &Cond) {
  if (BBI.NonPredSize == 0 || BBI.NonPredSize > TLI->getIfCvtBlockSizeLimit())
    return;

  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    // TODO: check if instruction clobbers predicate.
    if (TII->isTerminatorInstr(I->getOpcode()))
      break;
    if (!I->isPredicable())
      return;
  }

  if (BBI.Predicate.size() && !TII->SubsumesPredicate(BBI.Predicate, Cond))
    return;

  BBI.isPredicable = true;
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
        case ICEarlyExit:
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

  return Change;
}

/// TransferPreds - Transfer all the predecessors of FromBB to ToBB.
///
static void TransferPreds(MachineBasicBlock *ToBB, MachineBasicBlock *FromBB) {
   std::vector<MachineBasicBlock*> Preds(FromBB->pred_begin(),
                                         FromBB->pred_end());
    for (unsigned i = 0, e = Preds.size(); i != e; ++i) {
      MachineBasicBlock *Pred = Preds[i];
      Pred->removeSuccessor(FromBB);
      if (!Pred->isSuccessor(ToBB))
        Pred->addSuccessor(ToBB);
    }
}

/// TransferSuccs - Transfer all the successors of FromBB to ToBB.
///
static void TransferSuccs(MachineBasicBlock *ToBB, MachineBasicBlock *FromBB) {
   std::vector<MachineBasicBlock*> Succs(FromBB->succ_begin(),
                                         FromBB->succ_end());
    for (unsigned i = 0, e = Succs.size(); i != e; ++i) {
      MachineBasicBlock *Succ = Succs[i];
      FromBB->removeSuccessor(Succ);
      if (!ToBB->isSuccessor(Succ))
        ToBB->addSuccessor(Succ);
    }
}

/// isNextBlock - Returns true if ToBB the next basic block after BB.
///
static bool isNextBlock(MachineBasicBlock *BB, MachineBasicBlock *ToBB) {
  MachineFunction::iterator Fallthrough = BB;
  return MachineFunction::iterator(ToBB) == ++Fallthrough;
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

/// IfConvertEarlyExit - If convert a early exit sub-CFG.
///
bool IfConverter::IfConvertEarlyExit(BBInfo &BBI) {
  BBI.Kind = ICNotClassfied;

  BBInfo &TrueBBI  = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  BBInfo *CvtBBI = &TrueBBI;
  BBInfo *NextBBI = &FalseBBI;

  bool ReserveCond = false;
  if (TrueBBI.Kind != ICChild) {
    std::swap(CvtBBI, NextBBI);
    ReserveCond = true;
  }

  std::vector<MachineOperand> NewCond(BBI.BrCond);
  if (ReserveCond)
    TII->ReverseBranchCondition(NewCond);
  FeasibilityAnalysis(*CvtBBI, NewCond);
  if (!CvtBBI->isPredicable)
    return false;

  PredicateBlock(*CvtBBI, NewCond);

  // Merge converted block into entry block. Also convert the end of the
  // block conditional branch (to the non-converted block) into an
  // unconditional one.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
  MergeBlocks(BBI, *CvtBBI);
  if (!isNextBlock(BBI.BB, NextBBI->BB))
    InsertUncondBranch(BBI.BB, NextBBI->BB, TII);
  std::copy(NewCond.begin(), NewCond.end(), std::back_inserter(BBI.Predicate));

  // Update block info. BB can be iteratively if-converted.
  BBI.Kind = ICNotAnalyzed;
  BBI.TrueBB = BBI.FalseBB = NULL;
  BBI.BrCond.clear();
  TII->AnalyzeBranch(*BBI.BB, BBI.TrueBB, BBI.FalseBB, BBI.BrCond);
  ReTryPreds(BBI.BB);
  CvtBBI->Kind = ICDead;

  // FIXME: Must maintain LiveIns.
  NumIfConvBBs++;
  return true;
}

/// IfConvertTriangle - If convert a triangle sub-CFG.
///
bool IfConverter::IfConvertTriangle(BBInfo &BBI) {
  BBI.Kind = ICNotClassfied;

  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  FeasibilityAnalysis(TrueBBI, BBI.BrCond);
  if (!TrueBBI.isPredicable)
    return false;

  // Predicate the 'true' block after removing its branch.
  TrueBBI.NonPredSize -= TII->RemoveBranch(*BBI.TrueBB);
  PredicateBlock(TrueBBI, BBI.BrCond);

  // Join the 'true' and 'false' blocks by copying the instructions
  // from the 'false' block to the 'true' block.
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  MergeBlocks(TrueBBI, FalseBBI);

  // Now merge the entry of the triangle with the true block.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);
  MergeBlocks(BBI, TrueBBI);
  std::copy(BBI.BrCond.begin(), BBI.BrCond.end(),
            std::back_inserter(BBI.Predicate));

  // Update block info. BB can be iteratively if-converted.
  BBI.Kind = ICNotClassfied;
  BBI.TrueBB = BBI.FalseBB = NULL;
  BBI.BrCond.clear();
  TII->AnalyzeBranch(*BBI.BB, BBI.TrueBB, BBI.FalseBB, BBI.BrCond);
  ReTryPreds(BBI.BB);
  TrueBBI.Kind = ICDead;

  // FIXME: Must maintain LiveIns.
  NumIfConvBBs++;
  return true;
}

/// IfConvertDiamond - If convert a diamond sub-CFG.
///
bool IfConverter::IfConvertDiamond(BBInfo &BBI) {
  BBI.Kind = ICNotClassfied;

  bool TrueNeedBr;
  bool FalseNeedBr;
  BBInfo &TrueBBI = BBAnalysis[BBI.TrueBB->getNumber()];
  BBInfo &FalseBBI = BBAnalysis[BBI.FalseBB->getNumber()];
  FeasibilityAnalysis(TrueBBI, BBI.BrCond);
  std::vector<MachineOperand> RevCond(BBI.BrCond);
  TII->ReverseBranchCondition(RevCond);
  FeasibilityAnalysis(FalseBBI, RevCond);

  SmallVector<MachineInstr*, 2> Dups;
  bool Proceed = TrueBBI.isPredicable && FalseBBI.isPredicable;
  if (Proceed) {
    // Check the 'true' and 'false' blocks if either isn't ended with a branch.
    // Either the block fallthrough to another block or it ends with a
    // return. If it's the former, add a branch to its successor.
    TrueNeedBr  = !TrueBBI.TrueBB && BBI.TrueBB->succ_size();
    FalseNeedBr = !FalseBBI.TrueBB && BBI.FalseBB->succ_size();
    if (TrueNeedBr && TrueBBI.ModifyPredicate) {
      TrueBBI.isPredicable = false;
      Proceed = false;
    }
    if (FalseNeedBr && FalseBBI.ModifyPredicate) {
      FalseBBI.isPredicable = false;
      Proceed = false;
    }

    if (Proceed) {
      if (!BBI.TailBB) {
        // No common merge block. Check if the terminators (e.g. return) are
        // the same or predicable.
        MachineBasicBlock::iterator TT = BBI.TrueBB->getFirstTerminator();
        MachineBasicBlock::iterator FT = BBI.FalseBB->getFirstTerminator();
        while (TT != BBI.TrueBB->end() && FT != BBI.FalseBB->end()) {
          if (TT->isIdenticalTo(FT))
            Dups.push_back(TT);  // Will erase these later.
          else if (!TT->isPredicable() && !FT->isPredicable()) {
            Proceed = false;
            break; // Can't if-convert. Abort!
          }
          ++TT;
          ++FT;
        }

        // One of the two pathes have more terminators, make sure they are
        // all predicable.
        while (Proceed && TT != BBI.TrueBB->end())
          if (!TT->isPredicable()) {
            Proceed = false;
            break; // Can't if-convert. Abort!
          }
        while (Proceed && FT != BBI.FalseBB->end())
          if (!FT->isPredicable()) {
            Proceed = false;
            break; // Can't if-convert. Abort!
          }
      }
    }
  }

  if (!Proceed)
    return false;

  // Remove the duplicated instructions from the 'true' block.
  for (unsigned i = 0, e = Dups.size(); i != e; ++i) {
    Dups[i]->eraseFromParent();
    --TrueBBI.NonPredSize;
  }
    
  // Predicate the 'true' block after removing its branch.
  TrueBBI.NonPredSize -= TII->RemoveBranch(*BBI.TrueBB);
  PredicateBlock(TrueBBI, BBI.BrCond);

  // Predicate the 'false' block.
  PredicateBlock(FalseBBI, RevCond, true);

  // Merge the 'true' and 'false' blocks by copying the instructions
  // from the 'false' block to the 'true' block. That is, unless the true
  // block would clobber the predicate, in that case, do the opposite.
  BBInfo *CvtBBI;
  if (!TrueBBI.ModifyPredicate) {
    // Add a conditional branch from 'true' to 'true' successor if needed.
    if (TrueNeedBr)
      TII->InsertBranch(*BBI.TrueBB, *BBI.TrueBB->succ_begin(), NULL,
                        BBI.BrCond);
    // Add an unconditional branch from 'false' to to 'false' successor if it
    // will not be the fallthrough block.
    if (FalseNeedBr &&
        !isNextBlock(BBI.BB, *BBI.FalseBB->succ_begin()))
      InsertUncondBranch(BBI.FalseBB, *BBI.FalseBB->succ_begin(), TII);
    MergeBlocks(TrueBBI, FalseBBI);
    CvtBBI = &TrueBBI;
  } else {
    // Add a conditional branch from 'false' to 'false' successor if needed.
    if (FalseNeedBr)
      TII->InsertBranch(*BBI.FalseBB, *BBI.FalseBB->succ_begin(), NULL,
                        RevCond);
    // Add an unconditional branch from 'true' to to 'true' successor if it
    // will not be the fallthrough block.
    if (TrueNeedBr &&
        !isNextBlock(BBI.BB, *BBI.TrueBB->succ_begin()))
      InsertUncondBranch(BBI.TrueBB, *BBI.TrueBB->succ_begin(), TII);
    MergeBlocks(FalseBBI, TrueBBI);
    CvtBBI = &FalseBBI;
  }

  // Remove the conditional branch from entry to the blocks.
  BBI.NonPredSize -= TII->RemoveBranch(*BBI.BB);

  bool OkToIfcvt = true;
  // Merge the combined block into the entry of the diamond if the entry
  // block is its only predecessor. Otherwise, insert an unconditional
  // branch from entry to the if-converted block.
  if (CvtBBI->BB->pred_size() == 1) {
    MergeBlocks(BBI, *CvtBBI);
    CvtBBI = &BBI;
    OkToIfcvt = false;
  } else
    InsertUncondBranch(BBI.BB, CvtBBI->BB, TII);

  // If the if-converted block fallthrough or unconditionally branch into the
  // tail block, and the tail block does not have other predecessors, then
  // fold the tail block in as well.
  if (BBI.TailBB &&
      BBI.TailBB->pred_size() == 1 && CvtBBI->BB->succ_size() == 1) {
    CvtBBI->NonPredSize -= TII->RemoveBranch(*CvtBBI->BB);
    BBInfo TailBBI = BBAnalysis[BBI.TailBB->getNumber()];
    MergeBlocks(*CvtBBI, TailBBI);
    TailBBI.Kind = ICDead;
  }

  // Update block info. BB may be iteratively if-converted.
  if (OkToIfcvt) {
    BBI.Kind = ICNotClassfied;
    BBI.TrueBB = BBI.FalseBB = NULL;
    BBI.BrCond.clear();
    TII->AnalyzeBranch(*BBI.BB, BBI.TrueBB, BBI.FalseBB, BBI.BrCond);
    ReTryPreds(BBI.BB);
  }
  TrueBBI.Kind = ICDead;
  FalseBBI.Kind = ICDead;

  // FIXME: Must maintain LiveIns.
  NumIfConvBBs += 2;
  return true;
}

/// PredicateBlock - Predicate every instruction in the block with the specified
/// condition. If IgnoreTerm is true, skip over all terminator instructions.
void IfConverter::PredicateBlock(BBInfo &BBI,
                                 std::vector<MachineOperand> &Cond,
                                 bool IgnoreTerm) {
  for (MachineBasicBlock::iterator I = BBI.BB->begin(), E = BBI.BB->end();
       I != E; ++I) {
    MachineInstr *MI = I;
    if (IgnoreTerm && TII->isTerminatorInstr(MI->getOpcode()))
      continue;
    if (TII->isPredicated(MI))
      continue;
    if (!TII->PredicateInstruction(MI, Cond)) {
      cerr << "Unable to predication " << *I << "!\n";
      abort();
    }
  }

  BBI.NonPredSize = 0;
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

  // Transfer preds / succs and update size.
  TransferPreds(ToBBI.BB, FromBBI.BB);
  TransferSuccs(ToBBI.BB, FromBBI.BB);
  ToBBI.NonPredSize += FromBBI.NonPredSize;
  FromBBI.NonPredSize = 0;
}
