//===- bolt/Passes/ValidateInternalCalls.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ValidateInternalCalls class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/ValidateInternalCalls.h"
#include "bolt/Passes/DataflowInfoManager.h"
#include "bolt/Passes/FrameAnalysis.h"
#include "llvm/MC/MCInstPrinter.h"

#define DEBUG_TYPE "bolt-internalcalls"

namespace llvm {
namespace bolt {

namespace {

// Helper used to extract the target basic block used in an internal call.
// Return nullptr if this is not an internal call target.
BinaryBasicBlock *getInternalCallTarget(BinaryFunction &Function,
                                        const MCInst &Inst) {
  const BinaryContext &BC = Function.getBinaryContext();
  if (!BC.MIB->isCall(Inst) || MCPlus::getNumPrimeOperands(Inst) != 1 ||
      !Inst.getOperand(0).isExpr())
    return nullptr;

  return Function.getBasicBlockForLabel(BC.MIB->getTargetSymbol(Inst));
}

// A special StackPointerTracking that considers internal calls
class StackPointerTrackingForInternalCalls
    : public StackPointerTrackingBase<StackPointerTrackingForInternalCalls> {
  friend class DataflowAnalysis<StackPointerTrackingForInternalCalls,
                                std::pair<int, int>>;

  Optional<unsigned> AnnotationIndex;

protected:
  // We change the starting state to only consider the first block as an
  // entry point, otherwise the analysis won't converge (there will be two valid
  // stack offsets, one for an external call and another for an internal call).
  std::pair<int, int> getStartingStateAtBB(const BinaryBasicBlock &BB) {
    if (&BB == &*Func.begin())
      return std::make_pair(-8, getEmpty());
    return std::make_pair(getEmpty(), getEmpty());
  }

  // Here we decrement SP for internal calls too, in addition to the regular
  // StackPointerTracking processing.
  std::pair<int, int> computeNext(const MCInst &Point,
                                  const std::pair<int, int> &Cur) {
    std::pair<int, int> Res = StackPointerTrackingBase<
        StackPointerTrackingForInternalCalls>::computeNext(Point, Cur);
    if (Res.first == StackPointerTracking::SUPERPOSITION ||
        Res.first == StackPointerTracking::EMPTY)
      return Res;

    if (BC.MIB->isReturn(Point)) {
      Res.first += 8;
      return Res;
    }

    BinaryBasicBlock *Target = getInternalCallTarget(Func, Point);
    if (!Target)
      return Res;

    Res.first -= 8;
    return Res;
  }

  StringRef getAnnotationName() const {
    return StringRef("StackPointerTrackingForInternalCalls");
  }

public:
  StackPointerTrackingForInternalCalls(BinaryFunction &BF)
      : StackPointerTrackingBase<StackPointerTrackingForInternalCalls>(BF) {}

  void run() {
    StackPointerTrackingBase<StackPointerTrackingForInternalCalls>::run();
  }
};

} // end anonymous namespace

bool ValidateInternalCalls::fixCFGForPIC(BinaryFunction &Function) const {
  const BinaryContext &BC = Function.getBinaryContext();
  for (BinaryBasicBlock &BB : Function) {
    for (auto II = BB.begin(); II != BB.end(); ++II) {
      MCInst &Inst = *II;
      BinaryBasicBlock *Target = getInternalCallTarget(Function, Inst);
      if (!Target || BC.MIB->hasAnnotation(Inst, getProcessedICTag()))
        continue;

      BC.MIB->addAnnotation(Inst, getProcessedICTag(), 0U);
      InstructionListType MovedInsts = BB.splitInstructions(&Inst);
      if (!MovedInsts.empty()) {
        // Split this block at the call instruction. Create an unreachable
        // block.
        std::vector<std::unique_ptr<BinaryBasicBlock>> NewBBs;
        NewBBs.emplace_back(Function.createBasicBlock(0));
        NewBBs.back()->addInstructions(MovedInsts.begin(), MovedInsts.end());
        BB.moveAllSuccessorsTo(NewBBs.back().get());
        Function.insertBasicBlocks(&BB, std::move(NewBBs));
      }
      // Update successors
      BB.removeAllSuccessors();
      BB.addSuccessor(Target, BB.getExecutionCount(), 0ULL);
      return true;
    }
  }
  return false;
}

bool ValidateInternalCalls::fixCFGForIC(BinaryFunction &Function) const {
  const BinaryContext &BC = Function.getBinaryContext();
  // Track SP value
  StackPointerTrackingForInternalCalls SPTIC(Function);
  SPTIC.run();

  // Track instructions reaching a given point of the CFG to answer
  // "There is a path from entry to point A that contains instruction B"
  ReachingInsns<false> RI(Function);
  RI.run();

  // We use the InsnToBB map that DataflowInfoManager provides us
  DataflowInfoManager Info(Function, nullptr, nullptr);

  bool Updated = false;

  auto processReturns = [&](BinaryBasicBlock &BB, MCInst &Return) {
    // Check all reaching internal calls
    for (auto I = RI.expr_begin(Return), E = RI.expr_end(); I != E; ++I) {
      MCInst &ReachingInst = **I;
      if (!getInternalCallTarget(Function, ReachingInst) ||
          BC.MIB->hasAnnotation(ReachingInst, getProcessedICTag()))
        continue;

      // Stack pointer matching
      int SPAtCall = SPTIC.getStateAt(ReachingInst)->first;
      int SPAtRet = SPTIC.getStateAt(Return)->first;
      if (SPAtCall != StackPointerTracking::SUPERPOSITION &&
          SPAtRet != StackPointerTracking::SUPERPOSITION &&
          SPAtCall != SPAtRet - 8)
        continue;

      Updated = true;

      // Mark this call as processed, so we don't try to analyze it as a
      // PIC-computation internal call.
      BC.MIB->addAnnotation(ReachingInst, getProcessedICTag(), 0U);

      // Connect this block with the returning block of the caller
      BinaryBasicBlock *CallerBlock = Info.getInsnToBBMap()[&ReachingInst];
      BinaryBasicBlock *ReturnDestBlock =
          Function.getBasicBlockAfter(CallerBlock);
      BB.addSuccessor(ReturnDestBlock, BB.getExecutionCount(), 0);
    }
  };

  // This will connect blocks terminated with RETs to their respective
  // internal caller return block. A note here: this is overly conservative
  // because in nested calls, or unrelated calls, it will create edges
  // connecting RETs to potentially unrelated internal calls. This is safe
  // and if this causes a problem to recover the stack offsets properly, we
  // will fail later.
  for (BinaryBasicBlock &BB : Function) {
    for (MCInst &Inst : BB) {
      if (!BC.MIB->isReturn(Inst))
        continue;

      processReturns(BB, Inst);
    }
  }
  return Updated;
}

bool ValidateInternalCalls::hasTailCallsInRange(
    BinaryFunction &Function) const {
  const BinaryContext &BC = Function.getBinaryContext();
  for (BinaryBasicBlock &BB : Function)
    for (MCInst &Inst : BB)
      if (BC.MIB->isTailCall(Inst))
        return true;
  return false;
}

bool ValidateInternalCalls::analyzeFunction(BinaryFunction &Function) const {
  while (fixCFGForPIC(Function)) {
  }
  clearAnnotations(Function);
  while (fixCFGForIC(Function)) {
  }

  BinaryContext &BC = Function.getBinaryContext();
  RegAnalysis RA = RegAnalysis(BC, nullptr, nullptr);
  RA.setConservativeStrategy(RegAnalysis::ConservativeStrategy::CLOBBERS_NONE);
  bool HasTailCalls = hasTailCallsInRange(Function);

  for (BinaryBasicBlock &BB : Function) {
    for (MCInst &Inst : BB) {
      BinaryBasicBlock *Target = getInternalCallTarget(Function, Inst);
      if (!Target || BC.MIB->hasAnnotation(Inst, getProcessedICTag()))
        continue;

      if (HasTailCalls) {
        LLVM_DEBUG(dbgs() << Function
                          << " has tail calls and internal calls.\n");
        return false;
      }

      FrameIndexEntry FIE;
      int32_t SrcImm = 0;
      MCPhysReg Reg = 0;
      int64_t StackOffset = 0;
      bool IsIndexed = false;
      MCInst *TargetInst = ProgramPoint::getFirstPointAt(*Target).getInst();
      if (!BC.MIB->isStackAccess(*TargetInst, FIE.IsLoad, FIE.IsStore,
                                 FIE.IsStoreFromReg, Reg, SrcImm,
                                 FIE.StackPtrReg, StackOffset, FIE.Size,
                                 FIE.IsSimple, IsIndexed)) {
        LLVM_DEBUG({
          dbgs() << "Frame analysis failed - not simple: " << Function << "\n";
          Function.dump();
        });
        return false;
      }
      if (!FIE.IsLoad || FIE.StackPtrReg != BC.MIB->getStackPointer() ||
          StackOffset != 0) {
        LLVM_DEBUG({
          dbgs() << "Target instruction does not fetch return address - not "
                    "simple: "
                 << Function << "\n";
          Function.dump();
        });
        return false;
      }
      // Now track how the return address is used by tracking uses of Reg
      ReachingDefOrUse</*Def=*/false> RU =
          ReachingDefOrUse<false>(RA, Function, Reg);
      RU.run();

      int64_t Offset = static_cast<int64_t>(Target->getInputOffset());
      bool UseDetected = false;
      for (auto I = RU.expr_begin(*RU.getStateBefore(*TargetInst)),
                E = RU.expr_end();
           I != E; ++I) {
        MCInst &Use = **I;
        BitVector UsedRegs = BitVector(BC.MRI->getNumRegs(), false);
        BC.MIB->getTouchedRegs(Use, UsedRegs);
        if (!UsedRegs[Reg])
          continue;
        UseDetected = true;
        int64_t Output;
        std::pair<MCPhysReg, int64_t> Input1 = std::make_pair(Reg, 0);
        std::pair<MCPhysReg, int64_t> Input2 = std::make_pair(0, 0);
        if (!BC.MIB->evaluateSimple(Use, Output, Input1, Input2)) {
          LLVM_DEBUG(dbgs() << "Evaluate simple failed.\n");
          return false;
        }
        if (Offset + Output < 0 ||
            Offset + Output > static_cast<int64_t>(Function.getSize())) {
          LLVM_DEBUG({
            dbgs() << "Detected out-of-range PIC reference in " << Function
                   << "\nReturn address load: ";
            BC.InstPrinter->printInst(TargetInst, 0, "", *BC.STI, dbgs());
            dbgs() << "\nUse: ";
            BC.InstPrinter->printInst(&Use, 0, "", *BC.STI, dbgs());
            dbgs() << "\n";
            Function.dump();
          });
          return false;
        }
        LLVM_DEBUG({
          dbgs() << "Validated access: ";
          BC.InstPrinter->printInst(&Use, 0, "", *BC.STI, dbgs());
          dbgs() << "\n";
        });
      }
      if (!UseDetected) {
        LLVM_DEBUG(dbgs() << "No use detected.\n");
        return false;
      }
    }
  }
  return true;
}

void ValidateInternalCalls::runOnFunctions(BinaryContext &BC) {
  if (!BC.isX86())
    return;

  // Look for functions that need validation. This should be pretty rare.
  std::set<BinaryFunction *> NeedsValidation;
  for (auto &BFI : BC.getBinaryFunctions()) {
    BinaryFunction &Function = BFI.second;
    for (BinaryBasicBlock &BB : Function) {
      for (MCInst &Inst : BB) {
        if (getInternalCallTarget(Function, Inst)) {
          NeedsValidation.insert(&Function);
          Function.setSimple(false);
          break;
        }
      }
    }
  }

  // Skip validation for non-relocation mode
  if (!BC.HasRelocations)
    return;

  // Since few functions need validation, we can work with our most expensive
  // algorithms here. Fix the CFG treating internal calls as unconditional
  // jumps. This optimistically assumes this call is a PIC trick to get the PC
  // value, so it is not really a call, but a jump. If we find that it's not the
  // case, we mark this function as non-simple and stop processing it.
  std::set<BinaryFunction *> Invalid;
  for (BinaryFunction *Function : NeedsValidation) {
    LLVM_DEBUG(dbgs() << "Validating " << *Function << "\n");
    if (!analyzeFunction(*Function))
      Invalid.insert(Function);
    clearAnnotations(*Function);
  }

  if (!Invalid.empty()) {
    errs() << "BOLT-WARNING: will skip the following function(s) as unsupported"
              " internal calls were detected:\n";
    for (BinaryFunction *Function : Invalid) {
      errs() << "              " << *Function << "\n";
      Function->setIgnored();
    }
  }
}

} // namespace bolt
} // namespace llvm
