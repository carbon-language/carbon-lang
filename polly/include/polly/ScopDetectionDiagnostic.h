//=== ScopDetectionDiagnostic.h -- Diagnostic for ScopDetection -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Small set of diagnostic helper classes to encapsulate any errors occurred
// during the detection of Scops.
//
// The ScopDetection defines a set of error classes (via Statistic variables)
// that groups a number of individual errors into a group, e.g. non-affinity
// related errors.
// On error we generate an object that carries enough additional information
// to diagnose the error and generate a helpful error message.
//===----------------------------------------------------------------------===//
#ifndef POLLY_SCOP_DETECTION_DIAGNOSTIC_H
#define POLLY_SCOP_DETECTION_DIAGNOSTIC_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

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

//===----------------------------------------------------------------------===//
/// @brief Base class of all reject reasons found during Scop detection.
///
/// Subclasses of RejectReason should provide means to capture enough
/// diagnostic information to help clients figure out what and where something
/// went wrong in the Scop detection.
class RejectReason {
  //===--------------------------------------------------------------------===//
public:
  virtual ~RejectReason() {};

  /// @brief Generate a reasonable diagnostic message describing this error.
  ///
  /// @return A debug message representing this error.
  virtual std::string getMessage() const = 0;
};

//===----------------------------------------------------------------------===//
/// @brief Base class for CFG related reject reasons.
///
/// Scop candidates that violate structural restrictions can be grouped under
/// this reject reason class.
class ReportCFG : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportCFG() { ++BadCFGForScop; }
};

class ReportNonBranchTerminator : public ReportCFG {
  BasicBlock *BB;

public:
  ReportNonBranchTerminator(BasicBlock *BB) : BB(BB) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("Non branch instruction terminates BB: " + BB->getName()).str();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a not well-structured condition within the CFG.
class ReportCondition : public ReportCFG {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportCondition(BasicBlock *BB) : BB(BB) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("Not well structured condition at BB: " + BB->getName()).str();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Base class for non-affine reject reasons.
///
/// Scop candidates that violate restrictions to affinity are reported under
/// this class.
class ReportAffFunc : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportAffFunc() { ++BadAffFuncForScop; }
};

//===----------------------------------------------------------------------===//
/// @brief Captures a condition that is based on an 'undef' value.
class ReportUndefCond : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportUndefCond(BasicBlock *BB) : BB(BB) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("Condition based on 'undef' value in BB: " + BB->getName()).str();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures an invalid condition
///
/// Conditions have to be either constants or icmp instructions.
class ReportInvalidCond : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportInvalidCond(BasicBlock *BB) : BB(BB) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("Condition in BB '" + BB->getName()).str() +
           "' neither constant nor an icmp instruction";
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures an undefined operand.
class ReportUndefOperand : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the undefined operand in.
  BasicBlock *BB;

public:
  ReportUndefOperand(BasicBlock *BB) : BB(BB) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("undef operand in branch at BB: " + BB->getName()).str();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non-affine branch.
class ReportNonAffBranch : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the non-affine branch in.
  BasicBlock *BB;

  /// @brief LHS & RHS of the failed condition.
  //@{
  const SCEV *LHS;
  const SCEV *RHS;
  //@}

public:
  ReportNonAffBranch(BasicBlock *BB, const SCEV *LHS, const SCEV *RHS)
      : BB(BB), LHS(LHS), RHS(RHS) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("Non affine branch in BB '" + BB->getName()).str() +
           "' with LHS: " + *LHS + " and RHS: " + *RHS;
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a missing base pointer.
class ReportNoBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//
public:
  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const { return "No base pointer"; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures an undefined base pointer.
class ReportUndefBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//
public:
  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const { return "Undefined base pointer"; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a base pointer that is not invariant in the region.
class ReportVariantBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The variant base pointer.
  Value *BaseValue;

public:
  ReportVariantBasePtr(Value *BaseValue) : BaseValue(BaseValue) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return "Base address not invariant in current region:" + *BaseValue;
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non-affine access function.
class ReportNonAffineAccess : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The non-affine access function.
  const SCEV *AccessFunction;

public:
  ReportNonAffineAccess(const SCEV *AccessFunction)
      : AccessFunction(AccessFunction) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return "Non affine access function: " + *AccessFunction;
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Base class for reject reasons related to induction variables.
///
//  ReportIndVar reject reasons are generated when the ScopDetection finds
/// errors in the induction variable(s) of the Scop candidate.
class ReportIndVar : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportIndVar() { ++BadIndVarForScop; }
};

//===----------------------------------------------------------------------===//
/// @brief Captures a phi node that refers to SSA names in the current region.
class ReportPhiNodeRefInRegion : public ReportIndVar {
  //===--------------------------------------------------------------------===//

  // The offending instruction.
  Instruction *Inst;

public:
  ReportPhiNodeRefInRegion(Instruction *Inst) : Inst(Inst) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return "SCEV of PHI node refers to SSA names in region: " + *Inst;
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non canonical phi node.
class ReportNonCanonicalPhiNode : public ReportIndVar {
  //===--------------------------------------------------------------------===//

  // The offending instruction.
  Instruction *Inst;

public:
  ReportNonCanonicalPhiNode(Instruction *Inst) : Inst(Inst) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return "Non canonical PHI node: " + *Inst;
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non canonical induction variable in the loop header.
class ReportLoopHeader : public ReportIndVar {
  //===--------------------------------------------------------------------===//

  // The offending loop.
  Loop *L;

public:
  ReportLoopHeader(Loop *L) : L(L) {};

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return ("No canonical IV at loop header: " + L->getHeader()->getName())
        .str();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a region with invalid entering edges.
class ReportIndEdge : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportIndEdge() { ++BadIndEdgeForScop; }

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return "Region has invalid entering edges!";
  }
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with non affine loop bounds.
class ReportLoopBound : public RejectReason {
  //===--------------------------------------------------------------------===//

  // The offending loop.
  Loop *L;

  // The non-affine loop bound.
  const SCEV *LoopCount;

public:
  ReportLoopBound(Loop *L, const SCEV *LoopCount) : L(L), LoopCount(LoopCount) {
    ++BadLoopBoundForScop;
  };

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const {
    return "Non affine loop bound '" + *LoopCount + "' in loop: " +
           L->getHeader()->getName();
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with non-side-effect-known function calls.
class ReportFuncCall : public RejectReason {
  //===--------------------------------------------------------------------===//

  // The offending call instruction.
  Instruction *Inst;

public:
  ReportFuncCall(Instruction *Inst) : Inst(Inst) {
    ++BadFuncCallForScop;
  };

  /// @name RejectReason interface
  //@{
  std::string getMessage() const { return "Call instruction: " + *Inst; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with aliasing.
class ReportAlias : public RejectReason {
  //===--------------------------------------------------------------------===//

  // The offending alias set.
  AliasSet *AS;

  /// @brief Format an invalid alias set.
  ///
  /// @param AS The invalid alias set to format.
  std::string formatInvalidAlias(AliasSet &AS) const {
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

public:
  ReportAlias(AliasSet *AS) : AS(AS) { ++BadAliasForScop; }

  /// @name RejectReason interface
  //@{
  std::string getMessage() const { return formatInvalidAlias(*AS); }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with non simplified loops.
class ReportSimpleLoop : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportSimpleLoop() { ++BadSimpleLoopForScop; }

  /// @name RejectReason interface
  //@{
  std::string getMessage() const {
    return "Loop not in simplify form is invalid!";
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Base class for otherwise ungrouped reject reasons.
class ReportOther : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportOther() { ++BadOtherForScop; }

  /// @name RejectReason interface
  //@{
  std::string getMessage() const { return "Unknown reject reason"; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with bad IntToPtr instructions.
class ReportIntToPtr : public ReportOther {
  //===--------------------------------------------------------------------===//

  // The offending base value.
  Value *BaseValue;

public:
  ReportIntToPtr(Value *BaseValue) : BaseValue(BaseValue) {};

  /// @name RejectReason interface
  //@{
  std::string getMessage() const {
    return "Find bad intToptr prt: " + *BaseValue;
  }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with alloca instructions.
class ReportAlloca : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportAlloca(Instruction *Inst) : Inst(Inst) {};

  /// @name RejectReason interface
  //@{
  std::string getMessage() const { return "Alloca instruction: " + *Inst; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with unknown instructions.
class ReportUnknownInst : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportUnknownInst(Instruction *Inst) : Inst(Inst) {};

  /// @name RejectReason interface
  //@{
  std::string getMessage() const { return "Unknown instruction: " + *Inst; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with phi nodes in exit BBs.
class ReportPHIinExit : public ReportOther {
  //===--------------------------------------------------------------------===//
public:
  /// @name RejectReason interface
  //@{
  std::string getMessage() const { return "PHI node in exit BB"; }
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with regions containing the function entry block.
class ReportEntry : public ReportOther {
  //===--------------------------------------------------------------------===//
public:
  /// @name RejectReason interface
  //@{
  std::string getMessage() const {
    return "Region containing entry block of function is invalid!";
  }
  //@}
};

} // namespace polly

#endif // POLLY_SCOP_DETECTION_DIAGNOSTIC_H
