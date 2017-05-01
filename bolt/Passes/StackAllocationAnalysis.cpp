//===--- Passes/StackAllocationAnalysis.cpp -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "StackAllocationAnalysis.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "saa"

namespace llvm {
namespace bolt {

void StackAllocationAnalysis::preflight() {
  DEBUG(dbgs() << "Starting StackAllocationAnalysis on \""
                << Func.getPrintName() << "\"\n");

  for (auto &BB : this->Func) {
    for (auto &Inst : BB) {
      MCPhysReg From, To;
      if (!BC.MIA->isPush(Inst) && (!BC.MIA->isRegToRegMove(Inst, From, To) ||
                                    To != BC.MIA->getStackPointer() ||
                                    From != BC.MIA->getFramePointer()) &&
          !BC.MII->get(Inst.getOpcode())
               .hasDefOfPhysReg(Inst, BC.MIA->getStackPointer(), *BC.MRI))
        continue;
      this->Expressions.push_back(&Inst);
      this->ExprToIdx[&Inst] = this->NumInstrs++;
    }
  }
}

BitVector
StackAllocationAnalysis::getStartingStateAtBB(const BinaryBasicBlock &BB) {
  return BitVector(this->NumInstrs, false);
}

BitVector
StackAllocationAnalysis::getStartingStateAtPoint(const MCInst &Point) {
  return BitVector(this->NumInstrs, false);
}

void StackAllocationAnalysis::doConfluence(BitVector &StateOut,
                                           const BitVector &StateIn) {
  StateOut |= StateIn;
}

BitVector StackAllocationAnalysis::doKill(const MCInst &Point,
                                          const BitVector &StateIn,
                                          int DeallocSize) {
  int64_t SPOffset = SPT.getStateAt(Point)->first;
  BitVector Next = StateIn;
  if (SPOffset == SPT.SUPERPOSITION || SPOffset == SPT.EMPTY)
    return Next;
  for (auto I = this->expr_begin(Next), E = this->expr_end(); I != E; ++I) {
    const MCInst *Instr = *I;
    int64_t InstrOffset = SPT.getStateAt(*Instr)->first;
    if (InstrOffset == SPT.SUPERPOSITION || InstrOffset == SPT.EMPTY)
      continue;
    if (InstrOffset < SPOffset) {
      Next.reset(I.getBitVectorIndex());
      DEBUG({
        dbgs() << "SAA FYI: Killed: ";
        Instr->dump();
        dbgs() << "by: ";
        Point.dump();
        dbgs() << "  (more info: Killed instr offset = " << InstrOffset
               << ". SPOffset = " << SPOffset
               << "; DeallocSize= " << DeallocSize << "\n";
      });
    }
  }
  return Next;
}

void StackAllocationAnalysis::doConfluenceWithLP(BitVector &StateOut,
                                                 const BitVector &StateIn,
                                                 const MCInst &Invoke) {
  BitVector NewIn = StateIn;
  for (const auto &Operand : Invoke) {
    if (Operand.isGnuArgsSize()) {
      auto ArgsSize = Operand.getGnuArgsSize();
      NewIn = doKill(Invoke, NewIn, ArgsSize);
    }
  }
  StateOut |= NewIn;
}

BitVector StackAllocationAnalysis::computeNext(const MCInst &Point,
                                               const BitVector &Cur) {
  const auto &MIA = BC.MIA;
  BitVector Next = Cur;
  if (int Sz = MIA->getPopSize(Point)) {
    Next = doKill(Point, Next, Sz);
    return Next;
  }
  if (MIA->isPush(Point)) {
    Next.set(this->ExprToIdx[&Point]);
    return Next;
  }

  MCPhysReg From, To;
  int64_t SPOffset, FPOffset;
  std::tie(SPOffset, FPOffset) = *SPT.getStateBefore(Point);
  if (MIA->isRegToRegMove(Point, From, To) && To == MIA->getStackPointer() &&
      From == MIA->getFramePointer()) {
    if (MIA->isLeave(Point))
      FPOffset += 8;
    if (SPOffset < FPOffset) {
      Next = doKill(Point, Next, FPOffset - SPOffset);
      return Next;
    }
    if (SPOffset > FPOffset) {
      Next.set(this->ExprToIdx[&Point]);
      return Next;
    }
  }
  if (BC.MII->get(Point.getOpcode())
          .hasDefOfPhysReg(Point, MIA->getStackPointer(), *BC.MRI)) {
    std::pair<MCPhysReg, int64_t> SP;
    if (SPOffset != SPT.EMPTY && SPOffset != SPT.SUPERPOSITION)
      SP = std::make_pair(MIA->getStackPointer(), SPOffset);
    else
      SP = std::make_pair(0, 0);
    std::pair<MCPhysReg, int64_t> FP;
    if (FPOffset != SPT.EMPTY && FPOffset != SPT.SUPERPOSITION)
      FP = std::make_pair(MIA->getFramePointer(), FPOffset);
    else
      FP = std::make_pair(0, 0);
    int64_t Output;
    if (!MIA->evaluateSimple(Point, Output, SP, FP))
      return Next;

    if (SPOffset < Output) {
      Next = doKill(Point, Next, Output - SPOffset);
      return Next;
    }
    if (SPOffset > Output) {
      Next.set(this->ExprToIdx[&Point]);
      return Next;
    }
  }
  return Next;
}

} // end namespace bolt
} // end namespace llvm
