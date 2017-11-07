//===--- BinaryBasicBlock.cpp - Interface for assembly-level basic block --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "BinaryBasicBlock.h"
#include "BinaryContext.h"
#include "BinaryFunction.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include <limits>
#include <string>

#undef  DEBUG_TYPE
#define DEBUG_TYPE "bolt"

namespace llvm {
namespace bolt {

constexpr uint32_t BinaryBasicBlock::INVALID_OFFSET;

bool operator<(const BinaryBasicBlock &LHS, const BinaryBasicBlock &RHS) {
  return LHS.Index < RHS.Index;
}

void BinaryBasicBlock::adjustNumPseudos(const MCInst &Inst, int Sign) {
  auto &BC = Function->getBinaryContext();
  if (BC.MII->get(Inst.getOpcode()).isPseudo())
    NumPseudos += Sign;
}

BinaryBasicBlock::iterator BinaryBasicBlock::getFirstNonPseudo() {
  const auto &BC = Function->getBinaryContext();
  for (auto II = Instructions.begin(), E = Instructions.end(); II != E; ++II) {
    if (!BC.MII->get(II->getOpcode()).isPseudo())
      return II;
  }
  return end();
}

BinaryBasicBlock::reverse_iterator BinaryBasicBlock::getLastNonPseudo() {
  const auto &BC = Function->getBinaryContext();
  for (auto RII = Instructions.rbegin(), E = Instructions.rend();
       RII != E; ++RII) {
    if (!BC.MII->get(RII->getOpcode()).isPseudo())
      return RII;
  }
  return rend();
}

bool BinaryBasicBlock::validateSuccessorInvariants() {
  const auto *Inst = getLastNonPseudoInstr();
  const auto *JT = Inst ? Function->getJumpTable(*Inst) : nullptr;
  auto &BC = Function->getBinaryContext();
  bool Valid = true;

  if (JT) {
    // Note: for now we assume that successors do not reference labels from
    // any overlapping jump tables.  We only look at the entries for the jump
    // table that is referenced at the last instruction.
    const auto Range = JT->getEntriesForAddress(BC.MIA->getJumpTable(*Inst));
    const std::vector<const MCSymbol *> Entries(&JT->Entries[Range.first],
                                                &JT->Entries[Range.second]);
    std::set<const MCSymbol *> UniqueSyms(Entries.begin(), Entries.end());
    for (auto *Succ : Successors) {
      auto Itr = UniqueSyms.find(Succ->getLabel());
      if (Itr != UniqueSyms.end()) {
        UniqueSyms.erase(Itr);
      } else  {
        // Work on the assumption that jump table blocks don't
        // have a conditional successor.
        Valid = false;
      }
    }
    // If there are any leftover entries in the jump table, they
    // must be one of the function end labels.
    for (auto *Sym : UniqueSyms) {
      Valid &= (Sym == Function->getFunctionEndLabel() ||
                Sym == Function->getFunctionColdEndLabel());
    }
  } else {
    const MCSymbol *TBB = nullptr;
    const MCSymbol *FBB = nullptr;
    MCInst *CondBranch = nullptr;
    MCInst *UncondBranch = nullptr;

    if (analyzeBranch(TBB, FBB, CondBranch, UncondBranch)) {
      switch (Successors.size()) {
      case 0:
        Valid = !CondBranch && !UncondBranch;
        break;
      case 1:
        Valid = !CondBranch ||
          (CondBranch &&
           !Function->getBasicBlockForLabel(BC.MIA->getTargetSymbol(*CondBranch)));
        break;
      case 2:
        Valid =
          (CondBranch &&
           (TBB == getConditionalSuccessor(true)->getLabel() &&
            ((!UncondBranch && !FBB) ||
             (UncondBranch &&
              FBB == getConditionalSuccessor(false)->getLabel()))));
        break;
      }
    }
  }
  if (!Valid) {
    errs() << "BOLT-WARNING: CFG invalid in " << *getFunction() << " @ "
           << getName() << "\n";
    if (JT) {
      errs() << "Jump Table instruction addr = 0x"
             << Twine::utohexstr(BC.MIA->getJumpTable(*Inst)) << "\n";
      JT->print(errs());
    }
    dump();
  }
  return Valid;
}

BinaryBasicBlock *BinaryBasicBlock::getSuccessor(const MCSymbol *Label) const {
  if (!Label && succ_size() == 1)
    return *succ_begin();

  for (BinaryBasicBlock *BB : successors()) {
    if (BB->getLabel() == Label)
      return BB;
  }

  return nullptr;
}

BinaryBasicBlock *BinaryBasicBlock::getLandingPad(const MCSymbol *Label) const {
  for (BinaryBasicBlock *BB : landing_pads()) {
    if (BB->getLabel() == Label)
      return BB;
  }

  return nullptr;
}

int32_t BinaryBasicBlock::getCFIStateAtInstr(const MCInst *Instr) const {
  assert(
      getFunction()->getState() >= BinaryFunction::State::CFG &&
      "can only calculate CFI state when function is in or past the CFG state");

  const auto &FDEProgram = getFunction()->getFDEProgram();

  // Find the last CFI preceding Instr in this basic block.
  const MCInst *LastCFI = nullptr;
  bool InstrSeen = (Instr == nullptr);
  for (auto RII = Instructions.rbegin(), E = Instructions.rend();
       RII != E; ++RII) {
    if (!InstrSeen) {
      InstrSeen = (&*RII == Instr);
      continue;
    }
    if (Function->getBinaryContext().MIA->isCFI(*RII)) {
      LastCFI = &*RII;
      break;
    }
  }

  assert(InstrSeen && "instruction expected in basic block");

  // CFI state is the same as at basic block entry point.
  if (!LastCFI)
    return getCFIState();

  // Fold all RememberState/RestoreState sequences, such as for:
  //
  //   [ CFI #(K-1) ]
  //   RememberState (#K)
  //     ....
  //   RestoreState
  //   RememberState
  //     ....
  //   RestoreState
  //   [ GNU_args_size ]
  //   RememberState
  //     ....
  //   RestoreState   <- LastCFI
  //
  // we return K - the most efficient state to (re-)generate.
  int64_t State = LastCFI->getOperand(0).getImm();
  while (State >= 0 &&
         FDEProgram[State].getOperation() == MCCFIInstruction::OpRestoreState) {
    int32_t Depth = 1;
    --State;
    assert(State >= 0 && "first CFI cannot be RestoreState");
    while (Depth && State >= 0) {
      const auto &CFIInstr = FDEProgram[State];
      if (CFIInstr.getOperation() == MCCFIInstruction::OpRestoreState) {
        ++Depth;
      } else if (CFIInstr.getOperation() == MCCFIInstruction::OpRememberState) {
        --Depth;
      }
      --State;
    }
    assert(Depth == 0 && "unbalanced RememberState/RestoreState stack");

    // Skip any GNU_args_size.
    while (State >= 0 &&
           FDEProgram[State].getOperation() == MCCFIInstruction::OpGnuArgsSize){
      --State;
    }
  }

  assert((State + 1 >= 0) && "miscalculated CFI state");
  return State + 1;
}

void BinaryBasicBlock::addSuccessor(BinaryBasicBlock *Succ,
                                    uint64_t Count,
                                    uint64_t MispredictedCount) {
  Successors.push_back(Succ);
  BranchInfo.push_back({Count, MispredictedCount});
  Succ->Predecessors.push_back(this);
}

void BinaryBasicBlock::replaceSuccessor(BinaryBasicBlock *Succ,
                                        BinaryBasicBlock *NewSucc,
                                        uint64_t Count,
                                        uint64_t MispredictedCount) {
  Succ->removePredecessor(this);
  auto I = succ_begin();
  auto BI = BranchInfo.begin();
  for (; I != succ_end(); ++I) {
    assert(BI != BranchInfo.end() && "missing BranchInfo entry");
    if (*I == Succ)
      break;
    ++BI;
  }
  assert(I != succ_end() && "no such successor!");

  *I = NewSucc;
  *BI = BinaryBranchInfo{Count, MispredictedCount};
  NewSucc->addPredecessor(this);
}

void BinaryBasicBlock::removeSuccessor(BinaryBasicBlock *Succ) {
  Succ->removePredecessor(this);
  auto I = succ_begin();
  auto BI = BranchInfo.begin();
  for (; I != succ_end(); ++I) {
    assert(BI != BranchInfo.end() && "missing BranchInfo entry");
    if (*I == Succ)
      break;
    ++BI;
  }
  assert(I != succ_end() && "no such successor!");

  Successors.erase(I);
  BranchInfo.erase(BI);
}

void BinaryBasicBlock::addPredecessor(BinaryBasicBlock *Pred) {
  Predecessors.push_back(Pred);
}

void BinaryBasicBlock::removePredecessor(BinaryBasicBlock *Pred) {
  auto I = std::find(pred_begin(), pred_end(), Pred);
  assert(I != pred_end() && "Pred is not a predecessor of this block!");
  Predecessors.erase(I);
}

void BinaryBasicBlock::removeDuplicateConditionalSuccessor(MCInst *CondBranch) {
  assert(succ_size() == 2 && Successors[0] == Successors[1] &&
         "conditional successors expected");

  auto *Succ = Successors[0];
  const auto CondBI = BranchInfo[0];
  const auto UncondBI = BranchInfo[1];

  eraseInstruction(CondBranch);

  Successors.clear();
  BranchInfo.clear();

  Successors.push_back(Succ);

  uint64_t Count = COUNT_NO_PROFILE;
  if (CondBI.Count != COUNT_NO_PROFILE && UncondBI.Count != COUNT_NO_PROFILE)
    Count = CondBI.Count + UncondBI.Count;
  BranchInfo.push_back({Count, 0});
}

bool BinaryBasicBlock::analyzeBranch(const MCSymbol *&TBB,
                                     const MCSymbol *&FBB,
                                     MCInst *&CondBranch,
                                     MCInst *&UncondBranch) {
  auto &MIA = Function->getBinaryContext().MIA;
  return MIA->analyzeBranch(Instructions, TBB, FBB, CondBranch, UncondBranch);
}

MCInst *BinaryBasicBlock::getTerminatorBefore(MCInst *Pos) {
  auto &BC = Function->getBinaryContext();
  auto Itr = rbegin();
  bool Check = Pos ? false : true;
  MCInst *FirstTerminator{nullptr};
  while (Itr != rend()) {
    if (!Check) {
      if (&*Itr == Pos)
        Check = true;
      ++Itr;
      continue;
    }
    if (BC.MIA->isTerminator(*Itr))
      FirstTerminator = &*Itr;
    ++Itr;
  }
  return FirstTerminator;
}

bool BinaryBasicBlock::hasTerminatorAfter(MCInst *Pos) {
  auto &BC = Function->getBinaryContext();
  auto Itr = rbegin();
  while (Itr != rend()) {
    if (&*Itr == Pos)
      return false;
    if (BC.MIA->isTerminator(*Itr))
      return true;
    ++Itr;
  }
  return false;
}

bool BinaryBasicBlock::swapConditionalSuccessors() {
  if (succ_size() != 2)
    return false;

  std::swap(Successors[0], Successors[1]);
  std::swap(BranchInfo[0], BranchInfo[1]);
  return true;
}

void BinaryBasicBlock::addBranchInstruction(const BinaryBasicBlock *Successor) {
  assert(isSuccessor(Successor));
  auto &BC = Function->getBinaryContext();
  MCInst NewInst;
  BC.MIA->createUncondBranch(NewInst, Successor->getLabel(), BC.Ctx.get());
  Instructions.emplace_back(std::move(NewInst));
}

void BinaryBasicBlock::addTailCallInstruction(const MCSymbol *Target) {
  auto &BC = Function->getBinaryContext();
  MCInst NewInst;
  BC.MIA->createTailCall(NewInst, Target, BC.Ctx.get());
  Instructions.emplace_back(std::move(NewInst));
}

uint32_t BinaryBasicBlock::getNumCalls() const {
  uint32_t N{0};
  auto &BC = Function->getBinaryContext();
  for (auto &Instr : Instructions) {
    if (BC.MIA->isCall(Instr))
      ++N;
  }
  return N;
}

uint32_t BinaryBasicBlock::getNumPseudos() const {
#ifndef NDEBUG
  auto &BC = Function->getBinaryContext();
  uint32_t N = 0;
  for (auto &Instr : Instructions) {
    if (BC.MII->get(Instr.getOpcode()).isPseudo())
      ++N;
  }
  if (N != NumPseudos) {
    errs() << "BOLT-ERROR: instructions for basic block " << getName()
           << " in function " << *Function << ": calculated pseudos "
           << N << ", set pseudos " << NumPseudos << ", size " << size()
           << '\n';
    llvm_unreachable("pseudos mismatch");
  }
#endif
  return NumPseudos;
}

ErrorOr<std::pair<double, double>>
BinaryBasicBlock::getBranchStats(const BinaryBasicBlock *Succ) const {
  if (Function->hasValidProfile()) {
    uint64_t TotalCount = 0;
    uint64_t TotalMispreds = 0;
    for (const auto &BI : BranchInfo) {
      if (BI.Count != COUNT_NO_PROFILE) {
        TotalCount += BI.Count;
        TotalMispreds += BI.MispredictedCount;
      }
    }

    if (TotalCount > 0) {
      auto Itr = std::find(Successors.begin(), Successors.end(), Succ);
      assert(Itr != Successors.end());
      const auto &BI = BranchInfo[Itr - Successors.begin()];
      if (BI.Count && BI.Count != COUNT_NO_PROFILE) {
        if (TotalMispreds == 0) TotalMispreds = 1;
        return std::make_pair(double(BI.Count) / TotalCount,
                              double(BI.MispredictedCount) / TotalMispreds);
      }
    }
  }
  return make_error_code(llvm::errc::result_out_of_range);
}

void BinaryBasicBlock::dump() const {
  auto &BC = Function->getBinaryContext();
  if (Label) outs() << Label->getName() << ":\n";
  BC.printInstructions(outs(), Instructions.begin(), Instructions.end(),
                       getOffset());
  outs() << "preds:";
  for (auto itr = pred_begin(); itr != pred_end(); ++itr) {
    outs() << " " << (*itr)->getName();
  }
  outs() << "\nsuccs:";
  for (auto itr = succ_begin(); itr != succ_end(); ++itr) {
    outs() << " " << (*itr)->getName();
  }
  outs() << "\n";
}

uint64_t BinaryBasicBlock::estimateSize() const {
  return Function->getBinaryContext().computeCodeSize(begin(), end());
}

} // namespace bolt
} // namespace llvm
