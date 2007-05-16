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
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumIfConvBBs, "Number of if-converted blocks");

namespace {
  class IfConverter : public MachineFunctionPass {
    enum BBICKind {
      ICInvalid,       // BB data invalid.
      ICNotClassfied,  // BB data valid, but not classified.
      ICTriangle,      // BB is part of a triangle sub-CFG.
      ICDiamond,       // BB is part of a diamond sub-CFG.
      ICTriangleEntry, // BB is entry of a triangle sub-CFG.
      ICDiamondEntry   // BB is entry of a diamond sub-CFG.
    };

    /// BBInfo - One per MachineBasicBlock, this is used to cache the result
    /// if-conversion feasibility analysis. This includes results from
    /// TargetInstrInfo::AnalyzeBranch() (i.e. TBB, FBB, and Cond), and its
    /// classification, and common merge block of its successors (if it's a
    /// diamond shape).
    struct BBInfo {
      BBICKind Kind;
      MachineBasicBlock *EBB;
      MachineBasicBlock *TBB;
      MachineBasicBlock *FBB;
      MachineBasicBlock *CMBB;
      std::vector<MachineOperand> Cond;
      BBInfo() : Kind(ICInvalid), EBB(0), TBB(0), FBB(0), CMBB(0) {}
    };

    /// BBAnalysis - Results of if-conversion feasibility analysis indexed by
    /// basic block number.
    std::vector<BBInfo> BBAnalysis;

    const TargetInstrInfo *TII;
    bool MadeChange;
  public:
    static char ID;
    IfConverter() : MachineFunctionPass((intptr_t)&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    virtual const char *getPassName() const { return "If converter"; }

  private:
    void AnalyzeBlock(MachineBasicBlock *BB);
    void InitialFunctionAnalysis(MachineFunction &MF,
                                 std::vector<int> &Candidates);
    bool IfConvertDiamond(BBInfo &BBI);
    bool IfConvertTriangle(BBInfo &BBI);
    bool isBlockPredicable(MachineBasicBlock *BB,
                             bool IgnoreTerm = false) const;
    void PredicateBlock(MachineBasicBlock *BB,
                        std::vector<MachineOperand> &Cond,
                        bool IgnoreTerm = false);
    void MergeBlocks(MachineBasicBlock *TBB, MachineBasicBlock *FBB);
  };
  char IfConverter::ID = 0;
}

FunctionPass *llvm::createIfConverterPass() { return new IfConverter(); }

bool IfConverter::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  if (!TII) return false;

  MadeChange = false;

  MF.RenumberBlocks();
  unsigned NumBBs = MF.getNumBlockIDs();
  BBAnalysis.resize(NumBBs);

  std::vector<int> Candidates;
  // Do an intial analysis for each basic block and finding all the potential
  // candidates to perform if-convesion.
  InitialFunctionAnalysis(MF, Candidates);

  for (unsigned i = 0, e = Candidates.size(); i != e; ++i) {
    BBInfo &BBI = BBAnalysis[i];
    switch (BBI.Kind) {
    default: assert(false && "Unexpected!");
      break;
    case ICTriangleEntry:
      MadeChange |= IfConvertTriangle(BBI);
      break;
    case ICDiamondEntry:
      MadeChange |= IfConvertDiamond(BBI);
      break;
    }
  }
  return MadeChange;
}

static MachineBasicBlock *findFalseBlock(MachineBasicBlock *BB,
                                         MachineBasicBlock *TBB) {
  for (MachineBasicBlock::succ_iterator SI = BB->succ_begin(),
         E = BB->succ_end(); SI != E; ++SI) {
    MachineBasicBlock *SuccBB = *SI;
    if (SuccBB != TBB)
      return SuccBB;
  }
  return NULL;
}

void IfConverter::AnalyzeBlock(MachineBasicBlock *BB) {
  BBInfo &BBI = BBAnalysis[BB->getNumber()];

  if (BBI.Kind != ICInvalid)
    return;  // Always analyzed.
  BBI.EBB = BB;

  // Look for 'root' of a simple (non-nested) triangle or diamond.
  BBI.Kind = ICNotClassfied;
  if (TII->AnalyzeBranch(*BB, BBI.TBB, BBI.FBB, BBI.Cond)
      || !BBI.TBB || BBI.Cond.size() == 0)
    return;
  AnalyzeBlock(BBI.TBB);
  BBInfo &TBBI = BBAnalysis[BBI.TBB->getNumber()];
  if (TBBI.Kind != ICNotClassfied)
    return;
  
  if (!BBI.FBB)
    BBI.FBB = findFalseBlock(BB, BBI.TBB);  
  assert(BBI.FBB && "Expected to find the fallthrough block!");

  AnalyzeBlock(BBI.FBB);
  BBInfo &FBBI = BBAnalysis[BBI.FBB->getNumber()];
  if (FBBI.Kind != ICNotClassfied)
    return;

  // TODO: Only handle very simple cases for now.
  if (TBBI.FBB || FBBI.FBB || TBBI.Cond.size() > 1 || FBBI.Cond.size() > 1)
    return;

  if (TBBI.TBB && TBBI.TBB == BBI.FBB) {
    // Triangle:
    //   EBB
    //   | \_
    //   |  |
    //   | TBB
    //   |  /
    //   FBB
    BBI.Kind = ICTriangleEntry;
    TBBI.Kind = FBBI.Kind = ICTriangle;
  } else if (TBBI.TBB == FBBI.TBB) {
    // Diamond:
    //   EBB
    //   / \_
    //  |   |
    // TBB FBB
    //   \ /
    //   MBB
    // Note MBB can be empty in case both TBB and FBB are return blocks.
    BBI.Kind = ICDiamondEntry;
    TBBI.Kind = FBBI.Kind = ICDiamond;
    BBI.CMBB = TBBI.TBB;
  }
  return;
}

void IfConverter::InitialFunctionAnalysis(MachineFunction &MF,
                                          std::vector<int> &Candidates) {
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *BB = I;
    AnalyzeBlock(BB);
    BBInfo &BBI = BBAnalysis[BB->getNumber()];
    if (BBI.Kind == ICTriangleEntry || BBI.Kind == ICDiamondEntry)
      Candidates.push_back(BB->getNumber());
  }
}

bool IfConverter::IfConvertTriangle(BBInfo &BBI) {
  if (isBlockPredicable(BBI.TBB, true)) {
    // Predicate the 'true' block after removing its branch.
    TII->RemoveBranch(*BBI.TBB);
    PredicateBlock(BBI.TBB, BBI.Cond);

    // Join the 'true' and 'false' blocks by copying the instructions
    // from the 'false' block to the 'true' block.
    MergeBlocks(BBI.TBB, BBI.FBB);

    // Adjust entry block, it should have but a single unconditional
    // branch.
    BBI.EBB->removeSuccessor(BBI.FBB);
    TII->RemoveBranch(*BBI.EBB);
    std::vector<MachineOperand> NoCond;
    TII->InsertBranch(*BBI.EBB, BBI.TBB, NULL, NoCond);

    // FIXME: Must maintain LiveIns.
    NumIfConvBBs++;
    return true;
  }
  return false;
}

bool IfConverter::IfConvertDiamond(BBInfo &BBI) {
  if (isBlockPredicable(BBI.TBB, true) &&
      isBlockPredicable(BBI.FBB, true)) {
    std::vector<MachineInstr*> Dups;
    if (!BBI.CMBB) {
      // No common merge block. Check if the terminators (e.g. return) are
      // the same or predicable.
      MachineBasicBlock::iterator TT = BBI.TBB->getFirstTerminator();
      MachineBasicBlock::iterator FT = BBI.FBB->getFirstTerminator();
      while (TT != BBI.TBB->end() && FT != BBI.FBB->end()) {
        if (TT->isIdenticalTo(FT))
          Dups.push_back(TT);  // Will erase these later.
        else if (!TT->isPredicable() && !FT->isPredicable())
          return false; // Can't if-convert. Abort!
        ++TT;
        ++FT;
      }

      while (TT != BBI.TBB->end())
        if (!TT->isPredicable())
          return false; // Can't if-convert. Abort!
      while (FT != BBI.FBB->end())
        if (!FT->isPredicable())
          return false; // Can't if-convert. Abort!
    }

    // Remove the duplicated instructions from the 'true' block.
    for (unsigned i = 0, e = Dups.size(); i != e; ++i)
      Dups[i]->eraseFromParent();
    
    // Predicate the 'true' block after removing its branch.
    TII->RemoveBranch(*BBI.TBB);
    PredicateBlock(BBI.TBB, BBI.Cond);

    // Predicate the 'false' block.
    std::vector<MachineOperand> NewCond(BBI.Cond);
    TII->ReverseBranchCondition(NewCond);
    PredicateBlock(BBI.FBB, NewCond, true);

    // Join the 'true' and 'false' blocks by copying the instructions
    // from the 'false' block to the 'true' block.
    MergeBlocks(BBI.TBB, BBI.FBB);

    // Adjust entry block, it should have but a single unconditional
    // branch .
    BBI.EBB->removeSuccessor(BBI.FBB);
    TII->RemoveBranch(*BBI.EBB);
    std::vector<MachineOperand> NoCond;
    TII->InsertBranch(*BBI.EBB, BBI.TBB, NULL, NoCond);

    // FIXME: Must maintain LiveIns.
    NumIfConvBBs += 2;
    return true;
  }
  return false;
}

/// isBlockPredicable - Returns true if the block is predicable. In most
/// cases, that means all the instructions in the block has M_PREDICABLE flag.
/// If IgnoreTerm is true, assume all the terminator instructions can be 
/// converted or deleted.
bool IfConverter::isBlockPredicable(MachineBasicBlock *BB,
                                      bool IgnoreTerm) const {
  for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
       I != E; ++I) {
    if (IgnoreTerm && TII->isTerminatorInstr(I->getOpcode()))
      continue;
    if (!I->isPredicable())
      return false;
  }
  return true;
}

/// PredicateBlock - Predicate every instruction in the block with the specified
/// condition. If IgnoreTerm is true, skip over all terminator instructions.
void IfConverter::PredicateBlock(MachineBasicBlock *BB,
                                 std::vector<MachineOperand> &Cond,
                                 bool IgnoreTerm) {
  for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end();
       I != E; ++I) {
    if (IgnoreTerm && TII->isTerminatorInstr(I->getOpcode()))
      continue;
    TII->PredicateInstruction(&*I, Cond);
  }
}

/// MergeBlocks - Move all instructions from FBB to the end of TBB.
///
void IfConverter::MergeBlocks(MachineBasicBlock *TBB, MachineBasicBlock *FBB) {
  TBB->splice(TBB->end(), FBB, FBB->begin(), FBB->end());
}
