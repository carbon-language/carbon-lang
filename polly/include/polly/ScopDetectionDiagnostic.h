//=== ScopDetectionDiagnostic.h -- Diagnostic for ScopDetection -*- C++ -*-===//
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
#ifndef POLLY_SCOP_DETECTION_DIAGNOSTIC_H
#define POLLY_SCOP_DETECTION_DIAGNOSTIC_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Casting.h"

#include <string>
#include <memory>

using namespace llvm;

namespace llvm {
class SCEV;
class BasicBlock;
class Value;
class Region;
}

namespace polly {

/// @brief Get the location of a region from the debug info.
///
/// @param R The region to get debug info for.
/// @param LineBegin The first line in the region.
/// @param LineEnd The last line in the region.
/// @param FileName The filename where the region was defined.
void getDebugLocation(const Region *R, unsigned &LineBegin, unsigned &LineEnd,
                      std::string &FileName);

class RejectLog;
/// @brief Emit optimization remarks about the rejected regions to the user.
///
/// This emits the content of the reject log as optimization remarks.
/// Remember to at least track failures (-polly-detect-track-failures).
/// @param F The function we emit remarks for.
/// @param Log The error log containing all messages being emitted as remark.
void emitRejectionRemarks(const llvm::Function &F, const RejectLog &Log);

/// @brief Emit diagnostic remarks for a valid Scop
///
/// @param F The function we emit remarks for
/// @param R The region that marks a valid Scop
void emitValidRemarks(const llvm::Function &F, const Region *R);

// Discriminator for LLVM-style RTTI (dyn_cast<> et al.)
enum RejectReasonKind {
  // CFG Category
  rrkCFG,
  rrkNonBranchTerminator,
  rrkCondition,
  rrkLastCFG,

  // Non-Affinity
  rrkAffFunc,
  rrkUndefCond,
  rrkInvalidCond,
  rrkUndefOperand,
  rrkNonAffBranch,
  rrkNoBasePtr,
  rrkUndefBasePtr,
  rrkVariantBasePtr,
  rrkNonAffineAccess,
  rrkLastAffFunc,

  // IndVar
  rrkIndVar,
  rrkPhiNodeRefInRegion,
  rrkNonCanonicalPhiNode,
  rrkLoopHeader,
  rrkLastIndVar,

  rrkIndEdge,

  rrkLoopBound,

  rrkFuncCall,

  rrkAlias,

  rrkSimpleLoop,

  // Other
  rrkOther,
  rrkIntToPtr,
  rrkAlloca,
  rrkUnknownInst,
  rrkPHIinExit,
  rrkEntry,
  rrkLastOther
};

//===----------------------------------------------------------------------===//
/// @brief Base class of all reject reasons found during Scop detection.
///
/// Subclasses of RejectReason should provide means to capture enough
/// diagnostic information to help clients figure out what and where something
/// went wrong in the Scop detection.
class RejectReason {
  //===--------------------------------------------------------------------===//
private:
  const RejectReasonKind Kind;

protected:
  static const DebugLoc Unknown;

public:
  RejectReasonKind getKind() const { return Kind; }

  RejectReason(RejectReasonKind K) : Kind(K) {}

  virtual ~RejectReason() {}

  /// @brief Generate a reasonable diagnostic message describing this error.
  ///
  /// @return A debug message representing this error.
  virtual std::string getMessage() const = 0;

  /// @brief Generate a message for the end-user describing this error.
  ///
  /// The message provided has to be suitable for the end-user. So it should
  /// not reference any LLVM internal data structures or terminology.
  /// Ideally, the message helps the end-user to increase the size of the
  /// regions amenable to Polly.
  ///
  /// @return A short message representing this error.
  virtual std::string getEndUserMessage() const {
    return "Unspecified error.";
  };

  /// @brief Get the source location of this error.
  ///
  /// @return The debug location for this error.
  virtual const llvm::DebugLoc &getDebugLoc() const;
};

typedef std::shared_ptr<RejectReason> RejectReasonPtr;

/// @brief Stores all errors that ocurred during the detection.
class RejectLog {
  Region *R;
  llvm::SmallVector<RejectReasonPtr, 1> ErrorReports;

public:
  explicit RejectLog(Region *R) : R(R) {}

  typedef llvm::SmallVector<RejectReasonPtr, 1>::const_iterator iterator;

  iterator begin() const { return ErrorReports.begin(); }
  iterator end() const { return ErrorReports.end(); }
  size_t size() const { return ErrorReports.size(); }

  /// @brief Returns true, if we store at least one error.
  ///
  /// @return true, if we store at least one error.
  bool hasErrors() const { return size() > 0; }

  const Region *region() const { return R; }
  void report(RejectReasonPtr Reject) { ErrorReports.push_back(Reject); }
};

/// @brief Store reject logs
class RejectLogsContainer {
  std::map<const Region *, RejectLog> Logs;

public:
  typedef std::map<const Region *, RejectLog>::iterator iterator;
  typedef std::map<const Region *, RejectLog>::const_iterator const_iterator;

  iterator begin() { return Logs.begin(); }
  iterator end() { return Logs.end(); }

  const_iterator begin() const { return Logs.begin(); }
  const_iterator end() const { return Logs.end(); }

  std::pair<iterator, bool>
  insert(const std::pair<const Region *, RejectLog> &New) {
    return Logs.insert(New);
  }

  std::map<const Region *, RejectLog>::mapped_type at(const Region *R) {
    return Logs.at(R);
  }

  void clear() { Logs.clear(); }

  size_t count(const Region *R) const { return Logs.count(R); }

  size_t size(const Region *R) const {
    if (!Logs.count(R))
      return 0;
    return Logs.at(R).size();
  }

  bool hasErrors(const Region *R) const {
    if (!Logs.count(R))
      return false;

    RejectLog Log = Logs.at(R);
    return Log.hasErrors();
  }

  bool hasErrors(Region *R) const { return hasErrors((const Region *)R); }
};

//===----------------------------------------------------------------------===//
/// @brief Base class for CFG related reject reasons.
///
/// Scop candidates that violate structural restrictions can be grouped under
/// this reject reason class.
class ReportCFG : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportCFG(const RejectReasonKind K);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non-branch terminator within a Scop candidate.
class ReportNonBranchTerminator : public ReportCFG {
  BasicBlock *BB;

public:
  ReportNonBranchTerminator(BasicBlock *BB)
      : ReportCFG(rrkNonBranchTerminator), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a not well-structured condition within the CFG.
class ReportCondition : public ReportCFG {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportCondition(BasicBlock *BB) : ReportCFG(rrkCondition), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Base class for non-affine reject reasons.
///
/// Scop candidates that violate restrictions to affinity are reported under
/// this class.
class ReportAffFunc : public RejectReason {
  //===--------------------------------------------------------------------===//

  // The instruction that caused non-affinity to occur.
  const Instruction *Inst;

public:
  ReportAffFunc(const RejectReasonKind K, const Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual const DebugLoc &getDebugLoc() const override {
    return Inst->getDebugLoc();
  };
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a condition that is based on an 'undef' value.
class ReportUndefCond : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the broken condition in.
  BasicBlock *BB;

public:
  ReportUndefCond(const Instruction *Inst, BasicBlock *BB)
      : ReportAffFunc(rrkUndefCond, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
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
  ReportInvalidCond(const Instruction *Inst, BasicBlock *BB)
      : ReportAffFunc(rrkInvalidCond, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures an undefined operand.
class ReportUndefOperand : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The BasicBlock we found the undefined operand in.
  BasicBlock *BB;

public:
  ReportUndefOperand(BasicBlock *BB, const Instruction *Inst)
      : ReportAffFunc(rrkUndefOperand, Inst), BB(BB) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
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
  ReportNonAffBranch(BasicBlock *BB, const SCEV *LHS, const SCEV *RHS,
                     const Instruction *Inst)
      : ReportAffFunc(rrkNonAffBranch, Inst), BB(BB), LHS(LHS), RHS(RHS) {}

  const SCEV *lhs() { return LHS; }
  const SCEV *rhs() { return RHS; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a missing base pointer.
class ReportNoBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//
public:
  ReportNoBasePtr(const Instruction *Inst)
      : ReportAffFunc(rrkNoBasePtr, Inst) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures an undefined base pointer.
class ReportUndefBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//
public:
  ReportUndefBasePtr(const Instruction *Inst)
      : ReportAffFunc(rrkUndefBasePtr, Inst) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a base pointer that is not invariant in the region.
class ReportVariantBasePtr : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The variant base pointer.
  Value *BaseValue;

public:
  ReportVariantBasePtr(Value *BaseValue, const Instruction *Inst)
      : ReportAffFunc(rrkVariantBasePtr, Inst), BaseValue(BaseValue) {}

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non-affine access function.
class ReportNonAffineAccess : public ReportAffFunc {
  //===--------------------------------------------------------------------===//

  // The non-affine access function.
  const SCEV *AccessFunction;

public:
  ReportNonAffineAccess(const SCEV *AccessFunction, const Instruction *Inst)
      : ReportAffFunc(rrkNonAffineAccess, Inst),
        AccessFunction(AccessFunction) {}

  const SCEV *get() { return AccessFunction; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
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
  ReportIndVar(const RejectReasonKind K);
};

//===----------------------------------------------------------------------===//
/// @brief Captures a phi node that refers to SSA names in the current region.
class ReportPhiNodeRefInRegion : public ReportIndVar {
  //===--------------------------------------------------------------------===//

  // The offending instruction.
  Instruction *Inst;

public:
  ReportPhiNodeRefInRegion(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non canonical phi node.
class ReportNonCanonicalPhiNode : public ReportIndVar {
  //===--------------------------------------------------------------------===//

  // The offending instruction.
  Instruction *Inst;

public:
  ReportNonCanonicalPhiNode(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a non canonical induction variable in the loop header.
class ReportLoopHeader : public ReportIndVar {
  //===--------------------------------------------------------------------===//

  // The offending loop.
  Loop *L;

public:
  ReportLoopHeader(Loop *L);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures a region with invalid entering edges.
class ReportIndEdge : public RejectReason {
  //===--------------------------------------------------------------------===//

  BasicBlock *BB;

public:
  ReportIndEdge(BasicBlock *BB);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
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
  ReportLoopBound(Loop *L, const SCEV *LoopCount);

  const SCEV *loopCount() { return LoopCount; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with non-side-effect-known function calls.
class ReportFuncCall : public RejectReason {
  //===--------------------------------------------------------------------===//

  // The offending call instruction.
  Instruction *Inst;

public:
  ReportFuncCall(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  virtual std::string getEndUserMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with aliasing.
class ReportAlias : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  typedef std::vector<const llvm::Value *> PointerSnapshotTy;

private:
  /// @brief Format an invalid alias set.
  ///
  /// @param AS The invalid alias set to format.
  std::string formatInvalidAlias() const;

  Instruction *Inst;

  // A snapshot of the llvm values that took part in the aliasing error.
  mutable PointerSnapshotTy Pointers;

public:
  ReportAlias(Instruction *Inst, AliasSet &AS);

  const PointerSnapshotTy &getPointers() const { return Pointers; }

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with non simplified loops.
class ReportSimpleLoop : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportSimpleLoop();

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Base class for otherwise ungrouped reject reasons.
class ReportOther : public RejectReason {
  //===--------------------------------------------------------------------===//
public:
  ReportOther(const RejectReasonKind K);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with bad IntToPtr instructions.
class ReportIntToPtr : public ReportOther {
  //===--------------------------------------------------------------------===//

  // The offending base value.
  Instruction *BaseValue;

public:
  ReportIntToPtr(Instruction *BaseValue);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with alloca instructions.
class ReportAlloca : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportAlloca(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with unknown instructions.
class ReportUnknownInst : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportUnknownInst(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with phi nodes in exit BBs.
class ReportPHIinExit : public ReportOther {
  //===--------------------------------------------------------------------===//
  Instruction *Inst;

public:
  ReportPHIinExit(Instruction *Inst);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

//===----------------------------------------------------------------------===//
/// @brief Captures errors with regions containing the function entry block.
class ReportEntry : public ReportOther {
  //===--------------------------------------------------------------------===//
  BasicBlock *BB;

public:
  ReportEntry(BasicBlock *BB);

  /// @name LLVM-RTTI interface
  //@{
  static bool classof(const RejectReason *RR);
  //@}

  /// @name RejectReason interface
  //@{
  virtual std::string getMessage() const override;
  virtual const DebugLoc &getDebugLoc() const override;
  //@}
};

} // namespace polly

#endif // POLLY_SCOP_DETECTION_DIAGNOSTIC_H
