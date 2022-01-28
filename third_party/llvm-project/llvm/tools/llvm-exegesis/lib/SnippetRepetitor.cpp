//===-- SnippetRepetitor.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <array>
#include <string>

#include "SnippetRepetitor.h"
#include "Target.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

namespace llvm {
namespace exegesis {
namespace {

class DuplicateSnippetRepetitor : public SnippetRepetitor {
public:
  using SnippetRepetitor::SnippetRepetitor;

  // Repeats the snippet until there are at least MinInstructions in the
  // resulting code.
  FillFunction Repeat(ArrayRef<MCInst> Instructions, unsigned MinInstructions,
                      unsigned LoopBodySize) const override {
    return [Instructions, MinInstructions](FunctionFiller &Filler) {
      auto Entry = Filler.getEntry();
      if (!Instructions.empty()) {
        // Add the whole snippet at least once.
        Entry.addInstructions(Instructions);
        for (unsigned I = Instructions.size(); I < MinInstructions; ++I) {
          Entry.addInstruction(Instructions[I % Instructions.size()]);
        }
      }
      Entry.addReturn();
    };
  }

  BitVector getReservedRegs() const override {
    // We're using no additional registers.
    return State.getRATC().emptyRegisters();
  }
};

class LoopSnippetRepetitor : public SnippetRepetitor {
public:
  explicit LoopSnippetRepetitor(const LLVMState &State)
      : SnippetRepetitor(State),
        LoopCounter(State.getExegesisTarget().getLoopCounterRegister(
            State.getTargetMachine().getTargetTriple())) {}

  // Loop over the snippet ceil(MinInstructions / Instructions.Size()) times.
  FillFunction Repeat(ArrayRef<MCInst> Instructions, unsigned MinInstructions,
                      unsigned LoopBodySize) const override {
    return [this, Instructions, MinInstructions,
            LoopBodySize](FunctionFiller &Filler) {
      const auto &ET = State.getExegesisTarget();
      auto Entry = Filler.getEntry();
      auto Loop = Filler.addBasicBlock();
      auto Exit = Filler.addBasicBlock();

      const unsigned LoopUnrollFactor =
          LoopBodySize <= Instructions.size()
              ? 1
              : divideCeil(LoopBodySize, Instructions.size());
      assert(LoopUnrollFactor >= 1 && "Should end up with at least 1 snippet.");

      // Set loop counter to the right value:
      const APInt LoopCount(
          32,
          divideCeil(MinInstructions, LoopUnrollFactor * Instructions.size()));
      assert(LoopCount.uge(1) && "Trip count should be at least 1.");
      for (const MCInst &Inst :
           ET.setRegTo(State.getSubtargetInfo(), LoopCounter, LoopCount))
        Entry.addInstruction(Inst);

      // Set up the loop basic block.
      Entry.MBB->addSuccessor(Loop.MBB, BranchProbability::getOne());
      Loop.MBB->addSuccessor(Loop.MBB, BranchProbability::getOne());
      // The live ins are: the loop counter, the registers that were setup by
      // the entry block, and entry block live ins.
      Loop.MBB->addLiveIn(LoopCounter);
      for (unsigned Reg : Filler.getRegistersSetUp())
        Loop.MBB->addLiveIn(Reg);
      for (const auto &LiveIn : Entry.MBB->liveins())
        Loop.MBB->addLiveIn(LiveIn);
      for (auto _ : seq(0U, LoopUnrollFactor)) {
        (void)_;
        Loop.addInstructions(Instructions);
      }
      ET.decrementLoopCounterAndJump(*Loop.MBB, *Loop.MBB,
                                     State.getInstrInfo());

      // Set up the exit basic block.
      Loop.MBB->addSuccessor(Exit.MBB, BranchProbability::getZero());
      Exit.addReturn();
    };
  }

  BitVector getReservedRegs() const override {
    // We're using a single loop counter, but we have to reserve all aliasing
    // registers.
    return State.getRATC().getRegister(LoopCounter).aliasedBits();
  }

private:
  const unsigned LoopCounter;
};

} // namespace

SnippetRepetitor::~SnippetRepetitor() {}

std::unique_ptr<const SnippetRepetitor>
SnippetRepetitor::Create(InstructionBenchmark::RepetitionModeE Mode,
                         const LLVMState &State) {
  switch (Mode) {
  case InstructionBenchmark::Duplicate:
    return std::make_unique<DuplicateSnippetRepetitor>(State);
  case InstructionBenchmark::Loop:
    return std::make_unique<LoopSnippetRepetitor>(State);
  case InstructionBenchmark::AggregateMin:
    break;
  }
  llvm_unreachable("Unknown RepetitionModeE enum");
}

} // namespace exegesis
} // namespace llvm
