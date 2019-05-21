//===--- Passes/RetpolineInsertion.cpp-------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements a pass that replaces indirect branches (calls and
// jumps) with calls to retpolines to protect against branch target injection
// attacks.
// A unique retpoline is created for each register holding the address of the
// callee, if the callee address is in memory %r11 is used if available to
// hold the address of the callee before calling the retpoline, otherwise an
// address pattern specific retpoline is called where the callee address is
// loaded inside the retpoline.
// The user can determine when to assume %r11 available using r11-availability
// option, by default %r11 is assumed not available.
// Adding lfence instruction to the body of the speculate code is enabled by
// default and can be controlled by the user using retpoline-lfence option.
//===----------------------------------------------------------------------===//
#include "RetpolineInsertion.h"
#include "RewriteInstance.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "bolt-retpoline"

using namespace llvm;
using namespace bolt;
namespace opts {

extern cl::OptionCategory BoltCategory;

llvm::cl::opt<bool>
InsertRetpolines("insert-retpolines",
  cl::desc("run retpoline insertion pass"),
  cl::init(false),
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

llvm::cl::opt<bool>
RetpolineLfence("retpoline-lfence",
  cl::desc("determine if lfence instruction should exist in the retpoline"),
  cl::init(true),
  cl::ZeroOrMore,
  cl::Hidden,
  cl::cat(BoltCategory));

cl::opt<RetpolineInsertion::AvailabilityOptions>
R11Availability("r11-availability",
  cl::desc("determine the availablity of r11 before indirect branches"),
  cl::init(RetpolineInsertion::AvailabilityOptions::NEVER),
  cl::values(
    clEnumValN(RetpolineInsertion::AvailabilityOptions::NEVER,
      "never", "r11 not available"),
    clEnumValN(RetpolineInsertion::AvailabilityOptions::ALWAYS,
      "always", "r11 avaialable before calls and jumps"),
    clEnumValN(RetpolineInsertion::AvailabilityOptions::ABI,
      "abi", "r11 avaialable before calls but not before jumps")),
  cl::ZeroOrMore,
  cl::cat(BoltCategory));

} // namespace opts

namespace llvm {
namespace bolt {

// Retpoline function structure:
// BB0: call BB2
// BB1: pause
//      lfence
//      jmp BB1
// BB2: mov %reg, (%rsp)
//      ret
// or
// BB2: push %r11
//      mov Address, %r11
//      mov %r11, 8(%rsp)
//      pop %r11
//      ret
BinaryFunction *createNewRetpoline(BinaryContext &BC,
                                   const std::string &RetpolineTag,
                                   const IndirectBranchInfo &BrInfo,
                                   bool R11Available) {
  auto &MIB = *BC.MIB;
  auto &Ctx = *BC.Ctx.get();
  DEBUG(dbgs() << "BOLT-DEBUG: Creating a new retpoline function["
               << RetpolineTag << "]\n");

  auto *NewRetpoline = BC.createInjectedBinaryFunction(RetpolineTag, true);
  std::vector<std::unique_ptr<BinaryBasicBlock>> NewBlocks(3);
  for (int I = 0; I < 3; I++) {
    auto Symbol =
        Ctx.createTempSymbol(Twine(RetpolineTag + "_BB" + to_string(I)), true);
    NewBlocks[I] = NewRetpoline->createBasicBlock(
        BinaryBasicBlock::INVALID_OFFSET, Symbol);
    NewBlocks[I].get()->setCFIState(0);
  }

  auto &BB0 = *NewBlocks[0].get();
  auto &BB1 = *NewBlocks[1].get();
  auto &BB2 = *NewBlocks[2].get();

  BB0.addSuccessor(&BB2, 0, 0);
  BB1.addSuccessor(&BB1, 0, 0);

  // Build BB0
  MCInst DirectCall;
  MIB.createDirectCall(DirectCall, BB2.getLabel(), &Ctx);
  BB0.addInstruction(DirectCall);

  // Build BB1
  MCInst Pause;
  MIB.createPause(Pause);
  BB1.addInstruction(Pause);

  if (opts::RetpolineLfence) {
    MCInst Lfence;
    MIB.createLfence(Lfence);
    BB1.addInstruction(Lfence);
  }

  std::vector<MCInst> Seq;
  MIB.createShortJmp(Seq, BB1.getLabel(), &Ctx);
  BB1.addInstructions(Seq.begin(), Seq.end());

  // Build BB2
  if (BrInfo.isMem()) {
    if (R11Available) {
      MCInst StoreToStack;
      MIB.createSaveToStack(StoreToStack, MIB.getStackPointer(), 0,
                            MIB.getX86R11(), 8);
      BB2.addInstruction(StoreToStack);
    } else {
      MCInst PushR11;
      MIB.createPushRegister(PushR11, MIB.getX86R11(), 8);
      BB2.addInstruction(PushR11);

      MCInst LoadCalleeAddrs;
      const auto &MemRef = BrInfo.Memory;
      MIB.createLoad(LoadCalleeAddrs, MemRef.BaseRegNum, MemRef.ScaleValue,
                     MemRef.IndexRegNum, MemRef.DispValue, MemRef.DispExpr,
                     MemRef.SegRegNum, MIB.getX86R11(), 8);

      BB2.addInstruction(LoadCalleeAddrs);

      MCInst StoreToStack;
      MIB.createSaveToStack(StoreToStack, MIB.getStackPointer(), 8,
                            MIB.getX86R11(), 8);
      BB2.addInstruction(StoreToStack);

      MCInst PopR11;
      MIB.createPopRegister(PopR11, MIB.getX86R11(), 8);
      BB2.addInstruction(PopR11);
    }
  } else if (BrInfo.isReg()) {
    MCInst StoreToStack;
    MIB.createSaveToStack(StoreToStack, MIB.getStackPointer(), 0,
                          BrInfo.BranchReg, 8);
    BB2.addInstruction(StoreToStack);
  } else {
    llvm_unreachable("not expected");
  }

  // return
  MCInst Return;
  MIB.createReturn(Return);
  BB2.addInstruction(Return);
  NewRetpoline->insertBasicBlocks(nullptr, move(NewBlocks),
                                  /* UpdateLayout */ true,
                                  /* UpdateCFIState */ false);

  NewRetpoline->updateState(BinaryFunction::State::CFG_Finalized);
  return NewRetpoline;
}

std::string createRetpolineFunctionTag(BinaryContext &BC,
                                       const IndirectBranchInfo &BrInfo,
                                       bool R11Available) {
  if (BrInfo.isReg())
    return "__retpoline_r" + to_string(BrInfo.BranchReg) + "_";

  // Memory Branch
  if (R11Available)
    return "__retpoline_r11";

  std::string Tag = "__retpoline_mem_";

  const auto &MemRef = BrInfo.Memory;

  std::string DispExprStr;
  if (MemRef.DispExpr) {
    llvm::raw_string_ostream Ostream(DispExprStr);
    MemRef.DispExpr->print(Ostream, BC.AsmInfo.get());
    Ostream.flush();
  }

  Tag += MemRef.BaseRegNum != BC.MIB->getNoRegister()
             ? "r" + to_string(MemRef.BaseRegNum)
             : "";

  Tag +=
      MemRef.DispExpr ? "+" + DispExprStr : "+" + to_string(MemRef.DispValue);

  Tag += MemRef.IndexRegNum != BC.MIB->getNoRegister()
             ? "+" + to_string(MemRef.ScaleValue) + "*" +
                   to_string(MemRef.IndexRegNum)
             : "";

  Tag += MemRef.SegRegNum != BC.MIB->getNoRegister()
             ? "_seg_" + to_string(MemRef.SegRegNum)
             : "";

  return Tag;
}

BinaryFunction *RetpolineInsertion::getOrCreateRetpoline(
    BinaryContext &BC, const IndirectBranchInfo &BrInfo, bool R11Available) {
  const auto RetpolineTag =
      createRetpolineFunctionTag(BC, BrInfo, R11Available);

  if (CreatedRetpolines.count(RetpolineTag))
    return CreatedRetpolines[RetpolineTag];

  return CreatedRetpolines[RetpolineTag] =
             createNewRetpoline(BC, RetpolineTag, BrInfo, R11Available);
}

void createBranchReplacement(BinaryContext &BC,
                                const IndirectBranchInfo &BrInfo,
                                bool R11Available,
                                std::vector<MCInst> &Replacement,
                                const MCSymbol *RetpolineSymbol) {
  auto &MIB = *BC.MIB;
  // Load the branch address in r11 if available
  if (BrInfo.isMem() && R11Available) {
    const auto &MemRef = BrInfo.Memory;
    MCInst LoadCalleeAddrs;
    MIB.createLoad(LoadCalleeAddrs, MemRef.BaseRegNum, MemRef.ScaleValue,
                   MemRef.IndexRegNum, MemRef.DispValue, MemRef.DispExpr,
                   MemRef.SegRegNum, MIB.getX86R11(), 8);
    Replacement.push_back(LoadCalleeAddrs);
  }

  // Call the retpoline
  MCInst RetpolineCall;
  if (BrInfo.isJump() || BrInfo.isTailCall())
    MIB.createTailCall(RetpolineCall, RetpolineSymbol, BC.Ctx.get());
  else
    MIB.createDirectCall(RetpolineCall, RetpolineSymbol, BC.Ctx.get());

  Replacement.push_back(RetpolineCall);
}

IndirectBranchInfo::IndirectBranchInfo(MCInst &Inst, MCPlusBuilder &MIB) {
  IsCall = MIB.isCall(Inst);
  IsTailCall = MIB.isTailCall(Inst);

  if (MIB.isBranchOnMem(Inst)) {
    IsMem = true;
    if (!MIB.evaluateX86MemoryOperand(Inst, &Memory.BaseRegNum,
                                      &Memory.ScaleValue,
                                      &Memory.IndexRegNum, &Memory.DispValue,
                                      &Memory.SegRegNum, &Memory.DispExpr)) {
      llvm_unreachable("not expected");
    }
  } else if (MIB.isBranchOnReg(Inst)) {
    assert(MCPlus::getNumPrimeOperands(Inst) == 1 && "expect 1 operand");
    BranchReg = Inst.getOperand(0).getReg();
  } else {
    llvm_unreachable("unexpected instruction");
  }
}

void RetpolineInsertion::runOnFunctions(BinaryContext &BC) {
  if (!opts::InsertRetpolines)
    return;

  assert(BC.isX86() &&
         "retpoline insertion not supported for target architecture");

  assert(BC.HasRelocations && "retpoline mode not supported in non-reloc");

  auto &MIB = *BC.MIB;
  uint32_t RetpolinedBranches = 0;
  for (auto &It : BC.getBinaryFunctions()) {
    auto &Function = It.second;
    for (auto &BB : Function) {
      for (auto It = BB.begin(); It != BB.end(); ++It) {
        auto &Inst = *It;

        if (!MIB.isIndirectCall(Inst) && !MIB.isIndirectBranch(Inst))
          continue;

        IndirectBranchInfo BrInfo(Inst, MIB);
        bool R11Available = false;
        BinaryFunction *TargetRetpoline;
        std::vector<MCInst> Replacement;

        // Determine if r11 is available before this instruction
        if (BrInfo.isMem()) {
          if(MIB.hasAnnotation(Inst, "PLTCall"))
            R11Available= true;
          else if (opts::R11Availability == AvailabilityOptions::ALWAYS)
            R11Available = true;
          else if (opts::R11Availability == AvailabilityOptions::ABI)
            R11Available = BrInfo.isCall();
        }

        // If the instruction addressing pattern uses rsp and the retpoline
        // loads the callee address then displacement needs to be updated
        if (BrInfo.isMem() && !R11Available) {
          auto &MemRef = BrInfo.Memory;
          auto Addend = (BrInfo.isJump() || BrInfo.isTailCall()) ? 8 : 16;
          if (MemRef.BaseRegNum == MIB.getStackPointer()) {
            MemRef.DispValue += Addend;
          }
          if (MemRef.IndexRegNum == MIB.getStackPointer())
            MemRef.DispValue += Addend * MemRef.ScaleValue;
        }

        TargetRetpoline = getOrCreateRetpoline(BC, BrInfo, R11Available);

        createBranchReplacement(BC, BrInfo, R11Available, Replacement,
                                   TargetRetpoline->getSymbol());

        It = BB.replaceInstruction(It, Replacement.begin(), Replacement.end());
        RetpolinedBranches++;
      }
    }
  }
  outs() << "BOLT-INFO: The number of created retpoline functions is : "
         << CreatedRetpolines.size()
         << "\nBOLT-INFO: The number of retpolined branches is : " << RetpolinedBranches
         << "\n";
}

} // namespace bolt
} // namespace llvm
