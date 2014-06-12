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
#include "llvm/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/DebugInfo.h"

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

//===----------------------------------------------------------------------===//
// ReportCFG.

ReportCFG::ReportCFG() { ++BadCFGForScop; }

std::string ReportNonBranchTerminator::getMessage() const {
  return ("Non branch instruction terminates BB: " + BB->getName()).str();
}

std::string ReportCondition::getMessage() const {
  return ("Not well structured condition at BB: " + BB->getName()).str();
}

ReportAffFunc::ReportAffFunc() { ++BadAffFuncForScop; }

std::string ReportUndefCond::getMessage() const {
  return ("Condition based on 'undef' value in BB: " + BB->getName()).str();
}

std::string ReportInvalidCond::getMessage() const {
  return ("Condition in BB '" + BB->getName()).str() +
         "' neither constant nor an icmp instruction";
}

std::string ReportUndefOperand::getMessage() const {
  return ("undef operand in branch at BB: " + BB->getName()).str();
}

std::string ReportNonAffBranch::getMessage() const {
  return ("Non affine branch in BB '" + BB->getName()).str() + "' with LHS: " +
         *LHS + " and RHS: " + *RHS;
}

std::string ReportNoBasePtr::getMessage() const { return "No base pointer"; }

std::string ReportUndefBasePtr::getMessage() const {
  return "Undefined base pointer";
}

std::string ReportVariantBasePtr::getMessage() const {
  return "Base address not invariant in current region:" + *BaseValue;
}

std::string ReportNonAffineAccess::getMessage() const {
  return "Non affine access function: " + *AccessFunction;
}

ReportIndVar::ReportIndVar() { ++BadIndVarForScop; }

std::string ReportPhiNodeRefInRegion::getMessage() const {
  return "SCEV of PHI node refers to SSA names in region: " + *Inst;
}

std::string ReportNonCanonicalPhiNode::getMessage() const {
  return "Non canonical PHI node: " + *Inst;
}

std::string ReportLoopHeader::getMessage() const {
  return ("No canonical IV at loop header: " + L->getHeader()->getName()).str();
}

ReportIndEdge::ReportIndEdge() { ++BadIndEdgeForScop; }

std::string ReportIndEdge::getMessage() const {
  return "Region has invalid entering edges!";
}

ReportLoopBound::ReportLoopBound(Loop *L, const SCEV *LoopCount)
    : L(L), LoopCount(LoopCount) {
  ++BadLoopBoundForScop;
}

std::string ReportLoopBound::getMessage() const {
  return "Non affine loop bound '" + *LoopCount + "' in loop: " +
         L->getHeader()->getName();
}

ReportFuncCall::ReportFuncCall(Instruction *Inst) : Inst(Inst) {
  ++BadFuncCallForScop;
}

std::string ReportFuncCall::getMessage() const {
  return "Call instruction: " + *Inst;
}

ReportAlias::ReportAlias(AliasSet *AS) : AS(AS) { ++BadAliasForScop; }

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

ReportSimpleLoop::ReportSimpleLoop() { ++BadSimpleLoopForScop; }

std::string ReportSimpleLoop::getMessage() const {
  return "Loop not in simplify form is invalid!";
}

ReportOther::ReportOther() { ++BadOtherForScop; }

std::string ReportIntToPtr::getMessage() const {
  return "Find bad intToptr prt: " + *BaseValue;
}

std::string ReportAlloca::getMessage() const {
  return "Alloca instruction: " + *Inst;
}

std::string ReportUnknownInst::getMessage() const {
  return "Unknown instruction: " + *Inst;
}

std::string ReportPHIinExit::getMessage() const {
  return "PHI node in exit BB";
}

std::string ReportEntry::getMessage() const {
  return "Region containing entry block of function is invalid!";
}
} // namespace polly
