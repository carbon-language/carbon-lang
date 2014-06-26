//=== ScopDetectionDiagnostic.cpp - Error diagnostics --------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small set of diagnostic helper classes to encapsulate any errors occurred
// during the detection of Scops.
//
// The ScopDetection defines a set of error classes (via Statistic variables)
// that groups a number of individual errors into a group, e.g. non-affinity
// related errors.
// On error we generate an object that carries enough additional information
// to diagnose the error and generate a helpful error message.
//
//===----------------------------------------------------------------------===//
#include "polly/ScopDetectionDiagnostic.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#include "llvm/Analysis/RegionInfo.h"

#define DEBUG_TYPE "polly-detect"
#include "llvm/Support/Debug.h"

#include <string>

#define BADSCOP_STAT(NAME, DESC)                                               \
  STATISTIC(Bad##NAME##ForScop, "Number of bad regions for Scop: " DESC)

BADSCOP_STAT(CFG, "CFG too complex");
BADSCOP_STAT(IndVar, "Non canonical induction variable in loop");
BADSCOP_STAT(IndEdge, "Found invalid region entering edges");
BADSCOP_STAT(LoopBound, "Loop bounds can not be computed");
BADSCOP_STAT(FuncCall, "Function call with side effects appeared");
BADSCOP_STAT(AffFunc, "Expression not affine");
BADSCOP_STAT(Alias, "Found base address alias");
BADSCOP_STAT(SimpleLoop, "Loop not in -loop-simplify form");
BADSCOP_STAT(Other, "Others");

namespace polly {
/// @brief Small string conversion via raw_string_stream.
template <typename T> std::string operator+(Twine LHS, const T &RHS) {
  std::string Buf;
  raw_string_ostream fmt(Buf);
  fmt << RHS;
  fmt.flush();

  return LHS.concat(Buf).str();
}

void getDebugLocation(const Region *R, unsigned &LineBegin, unsigned &LineEnd,
                      std::string &FileName) {
  LineBegin = -1;
  LineEnd = 0;

  for (const BasicBlock *BB : R->blocks())
    for (const Instruction &Inst : *BB) {
      DebugLoc DL = Inst.getDebugLoc();
      if (DL.isUnknown())
        continue;

      DIScope Scope(DL.getScope(Inst.getContext()));

      if (FileName.empty())
        FileName = Scope.getFilename();

      unsigned NewLine = DL.getLine();

      LineBegin = std::min(LineBegin, NewLine);
      LineEnd = std::max(LineEnd, NewLine);
    }
}

void emitRejectionRemarks(const llvm::Function &F, const RejectLog &Log) {
  LLVMContext &Ctx = F.getContext();

  const Region *R = Log.region();
  const BasicBlock *Entry = R->getEntry();
  DebugLoc DL = Entry->getTerminator()->getDebugLoc();

  emitOptimizationRemarkMissed(
      Ctx, DEBUG_TYPE, F, DL,
      "The following errors keep this region from being a Scop.");
  for (RejectReasonPtr RR : Log) {
    const DebugLoc &Loc = RR->getDebugLoc();
    if (!Loc.isUnknown())
      emitOptimizationRemarkMissed(Ctx, DEBUG_TYPE, F, Loc,
                                   RR->getEndUserMessage());
  }
}

void emitValidRemarks(const llvm::Function &F, const Region *R) {
  LLVMContext &Ctx = F.getContext();

  const BasicBlock *Entry = R->getEntry();
  const BasicBlock *Exit = R->getExit();

  const DebugLoc &Begin = Entry->getFirstNonPHIOrDbg()->getDebugLoc();
  const DebugLoc &End = Exit->getFirstNonPHIOrDbg()->getDebugLoc();

  emitOptimizationRemark(Ctx, DEBUG_TYPE, F, Begin,
                         "A valid Scop begins here.");
  emitOptimizationRemark(Ctx, DEBUG_TYPE, F, End, "A valid Scop ends here.");
}

//===----------------------------------------------------------------------===//
// RejectReason.

const llvm::DebugLoc &RejectReason::getDebugLoc() const {
  // Allocate an empty DebugLoc and return it a reference to it.
  return *(std::make_shared<DebugLoc>().get());
}

//===----------------------------------------------------------------------===//
// ReportCFG.

ReportCFG::ReportCFG() { ++BadCFGForScop; }

//===----------------------------------------------------------------------===//
// ReportNonBranchTerminator.

std::string ReportNonBranchTerminator::getMessage() const {
  return ("Non branch instruction terminates BB: " + BB->getName()).str();
}

const DebugLoc &ReportNonBranchTerminator::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportCondition.

std::string ReportCondition::getMessage() const {
  return ("Not well structured condition at BB: " + BB->getName()).str();
}

const DebugLoc &ReportCondition::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportAffFunc.

ReportAffFunc::ReportAffFunc(const Instruction *Inst)
    : RejectReason(), Inst(Inst) {
  ++BadAffFuncForScop;
}

//===----------------------------------------------------------------------===//
// ReportUndefCond.

std::string ReportUndefCond::getMessage() const {
  return ("Condition based on 'undef' value in BB: " + BB->getName()).str();
}

//===----------------------------------------------------------------------===//
// ReportInvalidCond.

std::string ReportInvalidCond::getMessage() const {
  return ("Condition in BB '" + BB->getName()).str() +
         "' neither constant nor an icmp instruction";
}

//===----------------------------------------------------------------------===//
// ReportUndefOperand.

std::string ReportUndefOperand::getMessage() const {
  return ("undef operand in branch at BB: " + BB->getName()).str();
}

//===----------------------------------------------------------------------===//
// ReportNonAffBranch.

std::string ReportNonAffBranch::getMessage() const {
  return ("Non affine branch in BB '" + BB->getName()).str() + "' with LHS: " +
         *LHS + " and RHS: " + *RHS;
}

//===----------------------------------------------------------------------===//
// ReportNoBasePtr.

std::string ReportNoBasePtr::getMessage() const { return "No base pointer"; }

//===----------------------------------------------------------------------===//
// ReportUndefBasePtr.

std::string ReportUndefBasePtr::getMessage() const {
  return "Undefined base pointer";
}

//===----------------------------------------------------------------------===//
// ReportVariantBasePtr.

std::string ReportVariantBasePtr::getMessage() const {
  return "Base address not invariant in current region:" + *BaseValue;
}

//===----------------------------------------------------------------------===//
// ReportNonAffineAccess.

std::string ReportNonAffineAccess::getMessage() const {
  return "Non affine access function: " + *AccessFunction;
}

//===----------------------------------------------------------------------===//
// ReportIndVar.

ReportIndVar::ReportIndVar() : RejectReason() { ++BadIndVarForScop; }

//===----------------------------------------------------------------------===//
// ReportPhiNodeRefInRegion.

ReportPhiNodeRefInRegion::ReportPhiNodeRefInRegion(Instruction *Inst)
    : ReportIndVar(), Inst(Inst) {}

std::string ReportPhiNodeRefInRegion::getMessage() const {
  return "SCEV of PHI node refers to SSA names in region: " + *Inst;
}

const DebugLoc &ReportPhiNodeRefInRegion::getDebugLoc() const {
  return Inst->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportNonCanonicalPhiNode.

ReportNonCanonicalPhiNode::ReportNonCanonicalPhiNode(Instruction *Inst)
    : ReportIndVar(), Inst(Inst) {}

std::string ReportNonCanonicalPhiNode::getMessage() const {
  return "Non canonical PHI node: " + *Inst;
}

const DebugLoc &ReportNonCanonicalPhiNode::getDebugLoc() const {
  return Inst->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportLoopHeader.

ReportLoopHeader::ReportLoopHeader(Loop *L) : ReportIndVar(), L(L) {}

std::string ReportLoopHeader::getMessage() const {
  return ("No canonical IV at loop header: " + L->getHeader()->getName()).str();
}

const DebugLoc &ReportLoopHeader::getDebugLoc() const {
  BasicBlock *BB = L->getHeader();
  return BB->getTerminator()->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportIndEdge.

ReportIndEdge::ReportIndEdge(BasicBlock *BB) : RejectReason(), BB(BB) {
  ++BadIndEdgeForScop;
}

std::string ReportIndEdge::getMessage() const {
  return "Region has invalid entering edges!";
}

const DebugLoc &ReportIndEdge::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportLoopBound.

ReportLoopBound::ReportLoopBound(Loop *L, const SCEV *LoopCount)
    : RejectReason(), L(L), LoopCount(LoopCount) {
  ++BadLoopBoundForScop;
}

std::string ReportLoopBound::getMessage() const {
  return "Non affine loop bound '" + *LoopCount + "' in loop: " +
         L->getHeader()->getName();
}

const DebugLoc &ReportLoopBound::getDebugLoc() const {
  const BasicBlock *BB = L->getHeader();
  return BB->getTerminator()->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportFuncCall.

ReportFuncCall::ReportFuncCall(Instruction *Inst) : RejectReason(), Inst(Inst) {
  ++BadFuncCallForScop;
}

std::string ReportFuncCall::getMessage() const {
  return "Call instruction: " + *Inst;
}

const DebugLoc &ReportFuncCall::getDebugLoc() const {
  return Inst->getDebugLoc();
}

std::string ReportFuncCall::getEndUserMessage() const {
  return "This function call cannot be handeled. "
         "Try to inline it.";
}

//===----------------------------------------------------------------------===//
// ReportAlias.

ReportAlias::ReportAlias(Instruction *Inst, AliasSet *AS)
    : RejectReason(), AS(AS), Inst(Inst) {
  ++BadAliasForScop;
}

std::string ReportAlias::formatInvalidAlias(AliasSet &AS) const {
  std::string Message;
  raw_string_ostream OS(Message);

  OS << "Possible aliasing: ";

  std::vector<Value *> Pointers;

  for (const auto &I : AS)
    Pointers.push_back(I.getValue());

  std::sort(Pointers.begin(), Pointers.end());

  for (std::vector<Value *>::iterator PI = Pointers.begin(),
                                      PE = Pointers.end();
       ;) {
    Value *V = *PI;

    if (V->getName().size() == 0)
      OS << "\"" << *V << "\"";
    else
      OS << "\"" << V->getName() << "\"";

    ++PI;

    if (PI != PE)
      OS << ", ";
    else
      break;
  }

  return OS.str();
}

std::string ReportAlias::getMessage() const { return formatInvalidAlias(*AS); }

const DebugLoc &ReportAlias::getDebugLoc() const { return Inst->getDebugLoc(); }

//===----------------------------------------------------------------------===//
// ReportSimpleLoop.

ReportSimpleLoop::ReportSimpleLoop() : RejectReason() {
  ++BadSimpleLoopForScop;
}

std::string ReportSimpleLoop::getMessage() const {
  return "Loop not in simplify form is invalid!";
}

//===----------------------------------------------------------------------===//
// ReportOther.

std::string ReportOther::getMessage() const { return "Unknown reject reason"; }

ReportOther::ReportOther() : RejectReason() { ++BadOtherForScop; }

//===----------------------------------------------------------------------===//
// ReportIntToPtr.
ReportIntToPtr::ReportIntToPtr(Instruction *BaseValue)
    : ReportOther(), BaseValue(BaseValue) {}

std::string ReportIntToPtr::getMessage() const {
  return "Find bad intToptr prt: " + *BaseValue;
}

const DebugLoc &ReportIntToPtr::getDebugLoc() const {
  return BaseValue->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportAlloca.

ReportAlloca::ReportAlloca(Instruction *Inst) : ReportOther(), Inst(Inst) {}

std::string ReportAlloca::getMessage() const {
  return "Alloca instruction: " + *Inst;
}

const DebugLoc &ReportAlloca::getDebugLoc() const {
  return Inst->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportUnknownInst.

ReportUnknownInst::ReportUnknownInst(Instruction *Inst)
    : ReportOther(), Inst(Inst) {}

std::string ReportUnknownInst::getMessage() const {
  return "Unknown instruction: " + *Inst;
}

const DebugLoc &ReportUnknownInst::getDebugLoc() const {
  return Inst->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportPHIinExit.

ReportPHIinExit::ReportPHIinExit(Instruction *Inst)
    : ReportOther(), Inst(Inst) {}

std::string ReportPHIinExit::getMessage() const {
  return "PHI node in exit BB";
}

const DebugLoc &ReportPHIinExit::getDebugLoc() const {
  return Inst->getDebugLoc();
}

//===----------------------------------------------------------------------===//
// ReportEntry.
ReportEntry::ReportEntry(BasicBlock *BB) : ReportOther(), BB(BB) {}

std::string ReportEntry::getMessage() const {
  return "Region containing entry block of function is invalid!";
}

const DebugLoc &ReportEntry::getDebugLoc() const {
  return BB->getTerminator()->getDebugLoc();
}
} // namespace polly
