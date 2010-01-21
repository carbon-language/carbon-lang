//===- LoopStrengthReduce.cpp - Strength Reduce IVs in Loops --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation analyzes and transforms the induction variables (and
// computations derived from them) into forms suitable for efficient execution
// on the target.
//
// This pass performs a strength reduction on array references inside loops that
// have as one or more of their components the loop induction variable, it
// rewrites expressions to take advantage of scaled-index addressing modes
// available on the target, and it performs a variety of other optimizations
// related to loop induction variables.
//
// Terminology note: this code has a lot of handling for "post-increment" or
// "post-inc" users. This is not talking about post-increment addressing modes;
// it is instead talking about code like this:
//
//   %i = phi [ 0, %entry ], [ %i.next, %latch ]
//   ...
//   %i.next = add %i, 1
//   %c = icmp eq %i.next, %n
//
// The SCEV for %i is {0,+,1}<%L>. The SCEV for %i.next is {1,+,1}<%L>, however
// it's useful to think about these as the same register, with some uses using
// the value of the register before the add and some using // it after. In this
// example, the icmp is a post-increment user, since it uses %i.next, which is
// the value of the induction variable after the increment. The other common
// case of post-increment users is users outside the loop.
//
// TODO: More sophistication in the way Formulae are generated.
//
// TODO: Handle multiple loops at a time.
//
// TODO: test/CodeGen/X86/full-lsr.ll should get full lsr. The problem is
//       that {0,+,1}<%bb> is getting picked first because all 7 uses can
//       use it, and while it's a pretty good solution, it means that LSR
//       doesn't look further to find an even better solution.
//
// TODO: Should TargetLowering::AddrMode::BaseGV be changed to a ConstantExpr
//       instead of a GlobalValue?
//
// TODO: When truncation is free, truncate ICmp users' operands to make it a
//       smaller encoding (on x86 at least).
//
// TODO: When a negated register is used by an add (such as in a list of
//       multiple base registers, or as the increment expression in an addrec),
//       we may not actually need both reg and (-1 * reg) in registers; the
//       negation can be implemented by using a sub instead of an add. The
//       lack of support for taking this into consideration when making
//       register pressure decisions is partly worked around by the "Special"
//       use kind.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loop-reduce"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include <algorithm>
using namespace llvm;

namespace {

// Constant strides come first which in turns are sorted by their absolute
// values. If absolute values are the same, then positive strides comes first.
// e.g.
// 4, -1, X, 1, 2 ==> 1, -1, 2, 4, X
struct StrideCompare {
  const ScalarEvolution &SE;
  explicit StrideCompare(const ScalarEvolution &se) : SE(se) {}

  bool operator()(const SCEV *const &LHS, const SCEV *const &RHS) const {
    const SCEVConstant *LHSC = dyn_cast<SCEVConstant>(LHS);
    const SCEVConstant *RHSC = dyn_cast<SCEVConstant>(RHS);
    if (LHSC && RHSC) {
      unsigned BitWidth = std::max(SE.getTypeSizeInBits(LHS->getType()),
                                   SE.getTypeSizeInBits(RHS->getType()));
      APInt  LV = LHSC->getValue()->getValue();
      APInt  RV = RHSC->getValue()->getValue();
      LV.sextOrTrunc(BitWidth);
      RV.sextOrTrunc(BitWidth);
      APInt ALV = LV.abs();
      APInt ARV = RV.abs();
      if (ALV == ARV) {
        if (LV != RV)
          return LV.sgt(RV);
      } else {
        return ALV.ult(ARV);
      }

      // If it's the same value but different type, sort by bit width so
      // that we emit larger induction variables before smaller
      // ones, letting the smaller be re-written in terms of larger ones.
      return SE.getTypeSizeInBits(RHS->getType()) <
             SE.getTypeSizeInBits(LHS->getType());
    }
    return LHSC && !RHSC;
  }
};

/// RegSortData - This class holds data which is used to order reuse
/// candidates.
class RegSortData {
public:
  /// Bits - This represents the set of LSRUses (by index) which reference a
  /// particular register.
  SmallBitVector Bits;

  /// MaxComplexity - This represents the greatest complexity (see the comments
  /// on Formula::getComplexity) seen with a particular register.
  uint32_t MaxComplexity;

  /// Index - This holds an arbitrary value used as a last-resort tie breaker
  /// to ensure deterministic behavior.
  unsigned Index;

  RegSortData() {}

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

void RegSortData::print(raw_ostream &OS) const {
  OS << "[NumUses=" << Bits.count()
     << ", MaxComplexity=";
  OS.write_hex(MaxComplexity);
  OS << ", Index=" << Index << ']';
}

void RegSortData::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

/// RegCount - This is a helper class to sort a given set of registers
/// according to associated RegSortData values.
class RegCount {
public:
  const SCEV *Reg;
  RegSortData Sort;

  RegCount(const SCEV *R, const RegSortData &RSD)
    : Reg(R), Sort(RSD) {}

  // Sort by count. Returning true means the register is preferred.
  bool operator<(const RegCount &Other) const {
    // Sort by the number of unique uses of this register.
    unsigned A = Sort.Bits.count();
    unsigned B = Other.Sort.Bits.count();
    if (A != B) return A > B;

    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Reg)) {
      const SCEVAddRecExpr *BR = dyn_cast<SCEVAddRecExpr>(Other.Reg);
      // AddRecs have higher priority than other things.
      if (!BR) return true;

      // Prefer affine values.
      if (AR->isAffine() != BR->isAffine())
        return AR->isAffine();

      const Loop *AL = AR->getLoop();
      const Loop *BL = BR->getLoop();
      if (AL != BL) {
        unsigned ADepth = AL->getLoopDepth();
        unsigned BDepth = BL->getLoopDepth();
        // Prefer a less deeply nested addrec.
        if (ADepth != BDepth)
          return ADepth < BDepth;

        // Different loops at the same depth; do something arbitrary.
        BasicBlock *AH = AL->getHeader();
        BasicBlock *BH = BL->getHeader();
        for (Function::iterator I = AH, E = AH->getParent()->end(); I != E; ++I)
          if (&*I == BH) return true;
        return false;
      }

      // Sort addrecs by stride.
      const SCEV *AStep = AR->getOperand(1);
      const SCEV *BStep = BR->getOperand(1);
      if (AStep != BStep) {
        if (const SCEVConstant *AC = dyn_cast<SCEVConstant>(AStep)) {
          const SCEVConstant *BC = dyn_cast<SCEVConstant>(BStep);
          if (!BC) return true;
          // Arbitrarily prefer wider registers.
          if (AC->getValue()->getValue().getBitWidth() !=
              BC->getValue()->getValue().getBitWidth())
            return AC->getValue()->getValue().getBitWidth() >
                   BC->getValue()->getValue().getBitWidth();
          // Ignore the sign bit, assuming that striding by a negative value
          // is just as easy as by a positive value.
          // Prefer the addrec with the lesser absolute value stride, as it
          // will allow uses to have simpler addressing modes.
          return AC->getValue()->getValue().abs()
            .ult(BC->getValue()->getValue().abs());
        }
      }

      // Then sort by the register which will permit the simplest uses.
      // This is a heuristic; currently we only track the most complex use as a
      // representative.
      if (Sort.MaxComplexity != Other.Sort.MaxComplexity)
        return Sort.MaxComplexity < Other.Sort.MaxComplexity;

      // Then sort them by their start values.
      const SCEV *AStart = AR->getStart();
      const SCEV *BStart = BR->getStart();
      if (AStart != BStart) {
        if (const SCEVConstant *AC = dyn_cast<SCEVConstant>(AStart)) {
          const SCEVConstant *BC = dyn_cast<SCEVConstant>(BStart);
          if (!BC) return true;
          // Arbitrarily prefer wider registers.
          if (AC->getValue()->getValue().getBitWidth() !=
              BC->getValue()->getValue().getBitWidth())
            return AC->getValue()->getValue().getBitWidth() >
                   BC->getValue()->getValue().getBitWidth();
          // Prefer positive over negative if the absolute values are the same.
          if (AC->getValue()->getValue().abs() ==
              BC->getValue()->getValue().abs())
            return AC->getValue()->getValue().isStrictlyPositive();
          // Prefer the addrec with the lesser absolute value start.
          return AC->getValue()->getValue().abs()
            .ult(BC->getValue()->getValue().abs());
        }
      }
    } else {
      // AddRecs have higher priority than other things.
      if (isa<SCEVAddRecExpr>(Other.Reg)) return false;
      // Sort by the register which will permit the simplest uses.
      // This is a heuristic; currently we only track the most complex use as a
      // representative.
      if (Sort.MaxComplexity != Other.Sort.MaxComplexity)
        return Sort.MaxComplexity < Other.Sort.MaxComplexity;
    }


    // Tie-breaker: the arbitrary index, to ensure a reliable ordering.
    return Sort.Index < Other.Sort.Index;
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

void RegCount::print(raw_ostream &OS) const {
  OS << *Reg << ':';
  Sort.print(OS);
}

void RegCount::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

/// Formula - This class holds information that describes a formula for
/// satisfying a use. It may include broken-out immediates and scaled registers.
struct Formula {
  /// AM - This is used to represent complex addressing, as well as other kinds
  /// of interesting uses.
  TargetLowering::AddrMode AM;

  /// BaseRegs - The list of "base" registers for this use. When this is
  /// non-empty, AM.HasBaseReg should be set to true.
  SmallVector<const SCEV *, 2> BaseRegs;

  /// ScaledReg - The 'scaled' register for this use. This should be non-null
  /// when AM.Scale is not zero.
  const SCEV *ScaledReg;

  Formula() : ScaledReg(0) {}

  unsigned getNumRegs() const;
  uint32_t getComplexity() const;
  const Type *getType() const;

  void InitialMatch(const SCEV *S, Loop *L,
                    ScalarEvolution &SE, DominatorTree &DT);

  /// referencesReg - Test if this formula references the given register.
  bool referencesReg(const SCEV *S) const {
    return S == ScaledReg ||
           std::find(BaseRegs.begin(), BaseRegs.end(), S) != BaseRegs.end();
  }

  bool operator==(const Formula &Other) const {
    return BaseRegs == Other.BaseRegs &&
           ScaledReg == Other.ScaledReg &&
           AM.Scale == Other.AM.Scale &&
           AM.BaseOffs == Other.AM.BaseOffs &&
           AM.BaseGV == Other.AM.BaseGV;
  }

  // This sorts at least partially based on host pointer values which are
  // not deterministic, so it is only usable for uniqification.
  bool operator<(const Formula &Other) const {
    if (BaseRegs != Other.BaseRegs)
      return BaseRegs < Other.BaseRegs;
    if (ScaledReg != Other.ScaledReg)
      return ScaledReg < Other.ScaledReg;
    if (AM.Scale != Other.AM.Scale)
      return AM.Scale < Other.AM.Scale;
    if (AM.BaseOffs != Other.AM.BaseOffs)
      return AM.BaseOffs < Other.AM.BaseOffs;
    if (AM.BaseGV != Other.AM.BaseGV)
      return AM.BaseGV < Other.AM.BaseGV;
    return false;
  }

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

/// getNumRegs - Return the total number of register operands used by this
/// formula. This does not include register uses implied by non-constant
/// addrec strides.
unsigned Formula::getNumRegs() const {
  return !!ScaledReg + BaseRegs.size();
}

/// getComplexity - Return an oversimplified value indicating the complexity
/// of this formula. This is used as a tie-breaker in choosing register
/// preferences.
uint32_t Formula::getComplexity() const {
  // Encode the information in a uint32_t so that comparing with operator<
  // will be interesting.
  return
    // Most significant, the number of registers. This saturates because we
    // need the bits, and because beyond a few hundred it doesn't really matter.
    (std::min(getNumRegs(), (1u<<15)-1) << 17) |
    // Having multiple base regs is worse than having a base reg and a scale.
    ((BaseRegs.size() > 1) << 16) |
    // Scale absolute value.
    ((AM.Scale != 0 ? (Log2_64(abs64(AM.Scale)) + 1) : 0u) << 9) |
    // Scale sign, which is less significant than the absolute value.
    ((AM.Scale < 0) << 8) |
    // Offset absolute value.
    ((AM.BaseOffs != 0 ? (Log2_64(abs64(AM.BaseOffs)) + 1) : 0u) << 1) |
    // If a GV is present, treat it like a maximal offset.
    ((AM.BaseGV ? ((1u<<7)-1) : 0) << 1) |
    // Offset sign, which is less significant than the absolute offset.
    ((AM.BaseOffs < 0) << 0);
}

/// getType - Return the type of this formula, if it has one, or null
/// otherwise. This type is meaningless except for the bit size.
const Type *Formula::getType() const {
  return !BaseRegs.empty() ? BaseRegs.front()->getType() :
         ScaledReg ? ScaledReg->getType() :
         AM.BaseGV ? AM.BaseGV->getType() :
         0;
}

namespace {

/// ComplexitySorter - A predicate which orders Formulae by the number of
/// registers they contain.
struct ComplexitySorter {
  bool operator()(const Formula &LHS, const Formula &RHS) const {
    unsigned L = LHS.getNumRegs();
    unsigned R = RHS.getNumRegs();
    if (L != R) return L < R;

    return LHS.getComplexity() < RHS.getComplexity();
  }
};

}

/// DoInitialMatch - Recurrsion helper for InitialMatch.
static void DoInitialMatch(const SCEV *S, Loop *L,
                           SmallVectorImpl<const SCEV *> &Good,
                           SmallVectorImpl<const SCEV *> &Bad,
                           ScalarEvolution &SE, DominatorTree &DT) {
  // Collect expressions which properly dominate the loop header.
  if (S->properlyDominates(L->getHeader(), &DT)) {
    Good.push_back(S);
    return;
  }

  // Look at add operands.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
         I != E; ++I)
      DoInitialMatch(*I, L, Good, Bad, SE, DT);
    return;
  }

  // Look at addrec operands.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (!AR->getStart()->isZero()) {
      DoInitialMatch(AR->getStart(), L, Good, Bad, SE, DT);
      DoInitialMatch(SE.getAddRecExpr(SE.getIntegerSCEV(0, AR->getType()),
                                      AR->getStepRecurrence(SE),
                                      AR->getLoop()),
                     L, Good, Bad, SE, DT);
      return;
    }
  }

  // Handle a multiplication by -1 (negation) if it didn't fold.
  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S))
    if (Mul->getOperand(0)->isAllOnesValue()) {
      SmallVector<const SCEV *, 4> Ops(Mul->op_begin()+1, Mul->op_end());
      const SCEV *NewMul = SE.getMulExpr(Ops);

      SmallVector<const SCEV *, 4> MyGood;
      SmallVector<const SCEV *, 4> MyBad;
      DoInitialMatch(NewMul, L, MyGood, MyBad, SE, DT);
      const SCEV *NegOne = SE.getSCEV(ConstantInt::getAllOnesValue(
        SE.getEffectiveSCEVType(NewMul->getType())));
      for (SmallVectorImpl<const SCEV *>::const_iterator I = MyGood.begin(),
           E = MyGood.end(); I != E; ++I)
        Good.push_back(SE.getMulExpr(NegOne, *I));
      for (SmallVectorImpl<const SCEV *>::const_iterator I = MyBad.begin(),
           E = MyBad.end(); I != E; ++I)
        Bad.push_back(SE.getMulExpr(NegOne, *I));
      return;
    }

  // Ok, we can't do anything interesting. Just stuff the whole thing into a
  // register and hope for the best.
  Bad.push_back(S);
}

/// InitialMatch - Incorporate loop-variant parts of S into this Formula,
/// attempting to keep all loop-invariant and loop-computable values in a
/// single base register.
void Formula::InitialMatch(const SCEV *S, Loop *L,
                           ScalarEvolution &SE, DominatorTree &DT) {
  SmallVector<const SCEV *, 4> Good;
  SmallVector<const SCEV *, 4> Bad;
  DoInitialMatch(S, L, Good, Bad, SE, DT);
  if (!Good.empty()) {
    BaseRegs.push_back(SE.getAddExpr(Good));
    AM.HasBaseReg = true;
  }
  if (!Bad.empty()) {
    BaseRegs.push_back(SE.getAddExpr(Bad));
    AM.HasBaseReg = true;
  }
}

void Formula::print(raw_ostream &OS) const {
  bool First = true;
  if (AM.BaseGV) {
    if (!First) OS << " + "; else First = false;
    WriteAsOperand(OS, AM.BaseGV, /*PrintType=*/false);
  }
  if (AM.BaseOffs != 0) {
    if (!First) OS << " + "; else First = false;
    OS << AM.BaseOffs;
  }
  for (SmallVectorImpl<const SCEV *>::const_iterator I = BaseRegs.begin(),
       E = BaseRegs.end(); I != E; ++I) {
    if (!First) OS << " + "; else First = false;
    OS << "reg(";
    OS << **I;
    OS << ")";
  }
  if (AM.Scale != 0) {
    if (!First) OS << " + "; else First = false;
    OS << AM.Scale << "*reg(";
    if (ScaledReg)
      OS << *ScaledReg;
    else
      OS << "<unknown>";
    OS << ")";
  }
}

void Formula::dump() const {
  print(errs()); errs() << '\n';
}

/// getSDiv - Return an expression for LHS /s RHS, if it can be determined,
/// or null otherwise. If IgnoreSignificantBits is true, expressions like
/// (X * Y) /s Y are simplified to Y, ignoring that the multiplication may
/// overflow, which is useful when the result will be used in a context where
/// the most significant bits are ignored.
static const SCEV *getSDiv(const SCEV *LHS, const SCEV *RHS,
                           ScalarEvolution &SE,
                           bool IgnoreSignificantBits = false) {
  // Handle the trivial case, which works for any SCEV type.
  if (LHS == RHS)
    return SE.getIntegerSCEV(1, LHS->getType());

  // Handle x /s -1 as x * -1, to give ScalarEvolution a chance to do some
  // folding.
  if (RHS->isAllOnesValue())
    return SE.getMulExpr(LHS, RHS);

  // Check for a division of a constant by a constant.
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(LHS)) {
    const SCEVConstant *RC = dyn_cast<SCEVConstant>(RHS);
    if (!RC)
      return 0;
    if (C->getValue()->getValue().srem(RC->getValue()->getValue()) != 0)
      return 0;
    return SE.getConstant(C->getValue()->getValue()
               .sdiv(RC->getValue()->getValue()));
  }

  // Distribute the sdiv over addrec operands.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LHS)) {
    const SCEV *Start = getSDiv(AR->getStart(), RHS, SE,
                                IgnoreSignificantBits);
    if (!Start) return 0;
    const SCEV *Step = getSDiv(AR->getStepRecurrence(SE), RHS, SE,
                               IgnoreSignificantBits);
    if (!Step) return 0;
    return SE.getAddRecExpr(Start, Step, AR->getLoop());
  }

  // Distribute the sdiv over add operands.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(LHS)) {
    SmallVector<const SCEV *, 8> Ops;
    for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
         I != E; ++I) {
      const SCEV *Op = getSDiv(*I, RHS, SE,
                               IgnoreSignificantBits);
      if (!Op) return 0;
      Ops.push_back(Op);
    }
    return SE.getAddExpr(Ops);
  }

  // Check for a multiply operand that we can pull RHS out of.
  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(LHS))
    if (IgnoreSignificantBits || Mul->hasNoSignedWrap()) {
      SmallVector<const SCEV *, 4> Ops;
      bool Found = false;
      for (SCEVMulExpr::op_iterator I = Mul->op_begin(), E = Mul->op_end();
           I != E; ++I) {
        if (!Found)
          if (const SCEV *Q = getSDiv(*I, RHS, SE, IgnoreSignificantBits)) {
            Ops.push_back(Q);
            Found = true;
            continue;
          }
        Ops.push_back(*I);
      }
      return Found ? SE.getMulExpr(Ops) : 0;
    }

  // Otherwise we don't know.
  return 0;
}

namespace {

/// LSRUse - This class holds the state that LSR keeps for each use in
/// IVUsers, as well as uses invented by LSR itself. It includes information
/// about what kinds of things can be folded into the user, information
/// about the user itself, and information about how the use may be satisfied.
/// TODO: Represent multiple users of the same expression in common?
class LSRUse {
  SmallSet<Formula, 8> FormulaeUniquifier;

public:
  /// KindType - An enum for a kind of use, indicating what types of
  /// scaled and immediate operands it might support.
  enum KindType {
    Basic,   ///< A normal use, with no folding.
    Special, ///< A special case of basic, allowing -1 scales.
    Address, ///< An address use; folding according to TargetLowering
    ICmpZero ///< An equality icmp with both operands folded into one.
    // TODO: Add a generic icmp too?
  };

  KindType Kind;
  const Type *AccessTy;
  Instruction *UserInst;
  Value *OperandValToReplace;

  /// PostIncLoop - If this user is to use the post-incremented value of an
  /// induction variable, this variable is non-null and holds the loop
  /// associated with the induction variable.
  const Loop *PostIncLoop;

  /// Formulae - A list of ways to build a value that can satisfy this user.
  /// After the list is populated, one of these is selected heuristically and
  /// used to formulate a replacement for OperandValToReplace in UserInst.
  SmallVector<Formula, 12> Formulae;

  LSRUse() : Kind(Basic), AccessTy(0),
             UserInst(0), OperandValToReplace(0), PostIncLoop(0) {}

  void InsertInitialFormula(const SCEV *S, Loop *L,
                            ScalarEvolution &SE, DominatorTree &DT);
  void InsertSupplementalFormula(const SCEV *S);

  bool InsertFormula(const Formula &F);

  void Rewrite(Loop *L, SCEVExpander &Rewriter,
               SmallVectorImpl<WeakVH> &DeadInsts,
               ScalarEvolution &SE, DominatorTree &DT,
               Pass *P) const;

  void print(raw_ostream &OS) const;
  void dump() const;

private:
  Value *Expand(BasicBlock::iterator IP,
                Loop *L, SCEVExpander &Rewriter,
                SmallVectorImpl<WeakVH> &DeadInsts,
                ScalarEvolution &SE, DominatorTree &DT) const;
};

}

/// ExtractImmediate - If S involves the addition of a constant integer value,
/// return that integer value, and mutate S to point to a new SCEV with that
/// value excluded.
static int64_t ExtractImmediate(const SCEV *&S, ScalarEvolution &SE) {
  if (const SCEVConstant *C = dyn_cast<SCEVConstant>(S)) {
    if (C->getValue()->getValue().getMinSignedBits() <= 64) {
      S = SE.getIntegerSCEV(0, C->getType());
      return C->getValue()->getSExtValue();
    }
  } else if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    SmallVector<const SCEV *, 8> NewOps(Add->op_begin(), Add->op_end());
    int64_t Result = ExtractImmediate(NewOps.front(), SE);
    S = SE.getAddExpr(NewOps);
    return Result;
  } else if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    SmallVector<const SCEV *, 8> NewOps(AR->op_begin(), AR->op_end());
    int64_t Result = ExtractImmediate(NewOps.front(), SE);
    S = SE.getAddRecExpr(NewOps, AR->getLoop());
    return Result;
  }
  return 0;
}

/// ExtractSymbol - If S involves the addition of a GlobalValue address,
/// return that symbol, and mutate S to point to a new SCEV with that
/// value excluded.
static GlobalValue *ExtractSymbol(const SCEV *&S, ScalarEvolution &SE) {
  if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
    if (GlobalValue *GV = dyn_cast<GlobalValue>(U->getValue())) {
      S = SE.getIntegerSCEV(0, GV->getType());
      return GV;
    }
  } else if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    SmallVector<const SCEV *, 8> NewOps(Add->op_begin(), Add->op_end());
    GlobalValue *Result = ExtractSymbol(NewOps.back(), SE);
    S = SE.getAddExpr(NewOps);
    return Result;
  } else if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    SmallVector<const SCEV *, 8> NewOps(AR->op_begin(), AR->op_end());
    GlobalValue *Result = ExtractSymbol(NewOps.front(), SE);
    S = SE.getAddRecExpr(NewOps, AR->getLoop());
    return Result;
  }
  return 0;
}

/// isLegalUse - Test whether the use described by AM is "legal", meaning
/// it can be completely folded into the user instruction at isel time.
/// This includes address-mode folding and special icmp tricks.
static bool isLegalUse(const TargetLowering::AddrMode &AM,
                       LSRUse::KindType Kind, const Type *AccessTy,
                       const TargetLowering *TLI) {
  switch (Kind) {
  case LSRUse::Address:
    // If we have low-level target information, ask the target if it can
    // completely fold this address.
    if (TLI) return TLI->isLegalAddressingMode(AM, AccessTy);

    // Otherwise, just guess that reg+reg addressing is legal.
    return !AM.BaseGV && AM.BaseOffs == 0 && AM.Scale <= 1;

  case LSRUse::ICmpZero:
    // There's not even a target hook for querying whether it would be legal
    // to fold a GV into an ICmp.
    if (AM.BaseGV)
      return false;

    // ICmp only has two operands; don't allow more than two non-trivial parts.
    if (AM.Scale != 0 && AM.HasBaseReg && AM.BaseOffs != 0)
      return false;

    // ICmp only supports no scale or a -1 scale, as we can "fold" a -1 scale
    // by putting the scaled register in the other operand of the icmp.
    if (AM.Scale != 0 && AM.Scale != -1)
      return false;

    // If we have low-level target information, ask the target if it can
    // fold an integer immediate on an icmp.
    if (AM.BaseOffs != 0) {
      if (TLI) return TLI->isLegalICmpImmediate(-AM.BaseOffs);
      return false;
    }

    return true;

  case LSRUse::Basic:
    // Only handle single-register values.
    return !AM.BaseGV && AM.Scale == 0 && AM.BaseOffs == 0;

  case LSRUse::Special:
    // Only handle -1 scales, or no scale.
    return AM.Scale == 0 || AM.Scale == -1;
  }

  return false;
}

static bool isAlwaysFoldable(const SCEV *S,
                             bool HasBaseReg,
                             LSRUse::KindType Kind, const Type *AccessTy,
                             const TargetLowering *TLI,
                             ScalarEvolution &SE) {
  // Fast-path: zero is always foldable.
  if (S->isZero()) return true;

  // Conservatively, create an address with an immediate and a
  // base and a scale.
  TargetLowering::AddrMode AM;
  AM.BaseOffs = ExtractImmediate(S, SE);
  AM.BaseGV = ExtractSymbol(S, SE);
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Kind == LSRUse::ICmpZero ? -1 : 1;

  // If there's anything else involved, it's not foldable.
  if (!S->isZero()) return false;

  return isLegalUse(AM, Kind, AccessTy, TLI);
}

/// InsertFormula - If the given formula has not yet been inserted, add it
/// to the list, and return true. Return false otherwise.
bool LSRUse::InsertFormula(const Formula &F) {
  Formula Copy = F;

  // Sort the base regs, to avoid adding the same solution twice with
  // the base regs in different orders. This uses host pointer values, but
  // it doesn't matter since it's only used for uniquifying.
  std::sort(Copy.BaseRegs.begin(), Copy.BaseRegs.end());

  DEBUG(for (SmallVectorImpl<const SCEV *>::const_iterator I =
             F.BaseRegs.begin(), E = F.BaseRegs.end(); I != E; ++I)
          assert(!(*I)->isZero() && "Zero allocated in a base register!");
        assert((!F.ScaledReg || !F.ScaledReg->isZero()) &&
               "Zero allocated in a scaled register!"));

  if (FormulaeUniquifier.insert(Copy)) {
    Formulae.push_back(F);
    return true;
  }

  return false;
}

void
LSRUse::InsertInitialFormula(const SCEV *S, Loop *L,
                             ScalarEvolution &SE, DominatorTree &DT) {
  Formula F;
  F.InitialMatch(S, L, SE, DT);
  bool Inserted = InsertFormula(F);
  assert(Inserted && "Initial formula already exists!"); (void)Inserted;
}

void
LSRUse::InsertSupplementalFormula(const SCEV *S) {
  Formula F;
  F.BaseRegs.push_back(S);
  F.AM.HasBaseReg = true;
  bool Inserted = InsertFormula(F);
  assert(Inserted && "Supplemental formula already exists!"); (void)Inserted;
}

/// getImmediateDominator - A handy utility for the specific DominatorTree
/// query that we need here.
///
static BasicBlock *getImmediateDominator(BasicBlock *BB, DominatorTree &DT) {
  DomTreeNode *Node = DT.getNode(BB);
  if (!Node) return 0;
  Node = Node->getIDom();
  if (!Node) return 0;
  return Node->getBlock();
}

Value *LSRUse::Expand(BasicBlock::iterator IP,
                      Loop *L, SCEVExpander &Rewriter,
                      SmallVectorImpl<WeakVH> &DeadInsts,
                      ScalarEvolution &SE, DominatorTree &DT) const {
  // Then, collect some instructions which we will remain dominated by when
  // expanding the replacement. These must be dominated by any operands that
  // will be required in the expansion.
  SmallVector<Instruction *, 4> Inputs;
  if (Instruction *I = dyn_cast<Instruction>(OperandValToReplace))
    Inputs.push_back(I);
  if (Kind == ICmpZero)
    if (Instruction *I =
          dyn_cast<Instruction>(cast<ICmpInst>(UserInst)->getOperand(1)))
      Inputs.push_back(I);
  if (PostIncLoop && !L->contains(UserInst))
    Inputs.push_back(L->getLoopLatch()->getTerminator());

  // Then, climb up the immediate dominator tree as far as we can go while
  // still being dominated by the input positions.
  for (;;) {
    bool AllDominate = true;
    Instruction *BetterPos = 0;
    BasicBlock *IDom = getImmediateDominator(IP->getParent(), DT);
    if (!IDom) break;
    Instruction *Tentative = IDom->getTerminator();
    for (SmallVectorImpl<Instruction *>::const_iterator I = Inputs.begin(),
         E = Inputs.end(); I != E; ++I) {
      Instruction *Inst = *I;
      if (Inst == Tentative || !DT.dominates(Inst, Tentative)) {
        AllDominate = false;
        break;
      }
      if (IDom == Inst->getParent() &&
          (!BetterPos || DT.dominates(BetterPos, Inst)))
        BetterPos = next(BasicBlock::iterator(Inst));
    }
    if (!AllDominate)
      break;
    if (BetterPos)
      IP = BetterPos;
    else
      IP = Tentative;
  }
  while (isa<PHINode>(IP)) ++IP;

  // The first formula in the list is the winner.
  const Formula &F = Formulae.front();

  // Inform the Rewriter if we have a post-increment use, so that it can
  // perform an advantageous expansion.
  Rewriter.setPostInc(PostIncLoop);

  // This is the type that the user actually needs.
  const Type *OpTy = OperandValToReplace->getType();
  // This will be the type that we'll initially expand to.
  const Type *Ty = F.getType();
  if (!Ty)
    // No type known; just expand directly to the ultimate type.
    Ty = OpTy;
  else if (SE.getEffectiveSCEVType(Ty) == SE.getEffectiveSCEVType(OpTy))
    // Expand directly to the ultimate type if it's the right size.
    Ty = OpTy;
  // This is the type to do integer arithmetic in.
  const Type *IntTy = SE.getEffectiveSCEVType(Ty);

  // Build up a list of operands to add together to form the full base.
  SmallVector<const SCEV *, 8> Ops;

  // Expand the BaseRegs portion.
  for (SmallVectorImpl<const SCEV *>::const_iterator I = F.BaseRegs.begin(),
       E = F.BaseRegs.end(); I != E; ++I) {
    const SCEV *Reg = *I;
    assert(!Reg->isZero() && "Zero allocated in a base register!");

    // If we're expanding for a post-inc user for the add-rec's loop, make the
    // post-inc adjustment.
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Reg))
      if (AR->getLoop() == PostIncLoop)
        Reg = SE.getAddExpr(Reg, AR->getStepRecurrence(SE));

    Ops.push_back(SE.getUnknown(Rewriter.expandCodeFor(Reg, 0, IP)));
  }

  // Expand the ScaledReg portion.
  Value *ICmpScaledV = 0;
  if (F.AM.Scale != 0) {
    const SCEV *ScaledS = F.ScaledReg;

    // If we're expanding for a post-inc user for the add-rec's loop, make the
    // post-inc adjustment.
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(ScaledS))
      if (AR->getLoop() == PostIncLoop)
        ScaledS = SE.getAddExpr(ScaledS, AR->getStepRecurrence(SE));

    if (Kind == ICmpZero) {
      // An interesting way of "folding" with an icmp is to use a negated
      // scale, which we'll implement by inserting it into the other operand
      // of the icmp.
      assert(F.AM.Scale == -1 &&
             "The only scale supported by ICmpZero uses is -1!");
      ICmpScaledV = Rewriter.expandCodeFor(ScaledS, 0, IP);
    } else {
      // Otherwise just expand the scaled register and an explicit scale,
      // which is expected to be matched as part of the address.
      ScaledS = SE.getUnknown(Rewriter.expandCodeFor(ScaledS, 0, IP));
      const Type *ScaledTy = SE.getEffectiveSCEVType(ScaledS->getType());
      ScaledS = SE.getMulExpr(ScaledS,
                              SE.getSCEV(ConstantInt::get(ScaledTy,
                                                          F.AM.Scale)));
      Ops.push_back(ScaledS);
    }
  }

  // Expand the immediate portions.
  if (F.AM.BaseGV)
    Ops.push_back(SE.getSCEV(F.AM.BaseGV));
  if (F.AM.BaseOffs != 0) {
    if (Kind == ICmpZero) {
      // The other interesting way of "folding" with an ICmpZero is to use a
      // negated immediate.
      if (!ICmpScaledV)
        ICmpScaledV = ConstantInt::get(IntTy, -F.AM.BaseOffs);
      else {
        Ops.push_back(SE.getUnknown(ICmpScaledV));
        ICmpScaledV = ConstantInt::get(IntTy, F.AM.BaseOffs);
      }
    } else {
      // Just add the immediate values. These again are expected to be matched
      // as part of the address.
      Ops.push_back(SE.getSCEV(ConstantInt::get(IntTy, F.AM.BaseOffs)));
    }
  }

  // Emit instructions summing all the operands.
  const SCEV *FullS = Ops.empty() ?
                      SE.getIntegerSCEV(0, IntTy) :
                      SE.getAddExpr(Ops);
  Value *FullV = Rewriter.expandCodeFor(FullS, Ty, IP);

  // We're done expanding now, so reset the rewriter.
  Rewriter.setPostInc(0);

  // An ICmpZero Formula represents an ICmp which we're handling as a
  // comparison against zero. Now that we've expanded an expression for that
  // form, update the ICmp's other operand.
  if (Kind == ICmpZero) {
    ICmpInst *CI = cast<ICmpInst>(UserInst);
    DeadInsts.push_back(CI->getOperand(1));
    assert(!F.AM.BaseGV && "ICmp does not support folding a global value and "
                           "a scale at the same time!");
    if (F.AM.Scale == -1) {
      if (ICmpScaledV->getType() != OpTy) {
        Instruction *Cast =
          CastInst::Create(CastInst::getCastOpcode(ICmpScaledV, false,
                                                   OpTy, false),
                           ICmpScaledV, OpTy, "tmp", CI);
        ICmpScaledV = Cast;
      }
      CI->setOperand(1, ICmpScaledV);
    } else {
      assert(F.AM.Scale == 0 &&
             "ICmp does not support folding a global value and "
             "a scale at the same time!");
      Constant *C = ConstantInt::getSigned(SE.getEffectiveSCEVType(OpTy),
                                           -(uint64_t)F.AM.BaseOffs);
      if (C->getType() != OpTy)
        C = ConstantExpr::getCast(CastInst::getCastOpcode(C, false,
                                                          OpTy, false),
                                  C, OpTy);

      CI->setOperand(1, C);
    }
  }

  return FullV;
}

/// Rewrite - Emit instructions for the leading candidate expression for this
/// LSRUse (this is called "expanding"), and update the UserInst to reference
/// the newly expanded value.
void LSRUse::Rewrite(Loop *L, SCEVExpander &Rewriter,
                     SmallVectorImpl<WeakVH> &DeadInsts,
                     ScalarEvolution &SE, DominatorTree &DT,
                     Pass *P) const {
  const Type *OpTy = OperandValToReplace->getType();

  // First, find an insertion point that dominates UserInst. For PHI nodes,
  // find the nearest block which dominates all the relevant uses.
  if (PHINode *PN = dyn_cast<PHINode>(UserInst)) {
    DenseMap<BasicBlock *, Value *> Inserted;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) == OperandValToReplace) {
        BasicBlock *BB = PN->getIncomingBlock(i);

        // If this is a critical edge, split the edge so that we do not insert
        // the code on all predecessor/successor paths.  We do this unless this
        // is the canonical backedge for this loop, which complicates post-inc
        // users.
        if (e != 1 && BB->getTerminator()->getNumSuccessors() > 1 &&
            !isa<IndirectBrInst>(BB->getTerminator()) &&
            (PN->getParent() != L->getHeader() || !L->contains(BB))) {
          // Split the critical edge.
          BasicBlock *NewBB = SplitCriticalEdge(BB, PN->getParent(), P);

          // If PN is outside of the loop and BB is in the loop, we want to
          // move the block to be immediately before the PHI block, not
          // immediately after BB.
          if (L->contains(BB) && !L->contains(PN))
            NewBB->moveBefore(PN->getParent());

          // Splitting the edge can reduce the number of PHI entries we have.
          e = PN->getNumIncomingValues();
          BB = NewBB;
          i = PN->getBasicBlockIndex(BB);
        }

        std::pair<DenseMap<BasicBlock *, Value *>::iterator, bool> Pair =
          Inserted.insert(std::make_pair(BB, static_cast<Value *>(0)));
        if (!Pair.second)
          PN->setIncomingValue(i, Pair.first->second);
        else {
          Value *FullV = Expand(BB->getTerminator(),
                                L, Rewriter, DeadInsts, SE, DT);

          // If this is reuse-by-noop-cast, insert the noop cast.
          if (FullV->getType() != OpTy)
            FullV =
              CastInst::Create(CastInst::getCastOpcode(FullV, false,
                                                       OpTy, false),
                               FullV, OperandValToReplace->getType(),
                               "tmp", BB->getTerminator());

          PN->setIncomingValue(i, FullV);
          Pair.first->second = FullV;
        }
      }
  } else {
    Value *FullV = Expand(UserInst, L, Rewriter, DeadInsts, SE, DT);

    // If this is reuse-by-noop-cast, insert the noop cast.
    if (FullV->getType() != OpTy) {
      Instruction *Cast =
        CastInst::Create(CastInst::getCastOpcode(FullV, false, OpTy, false),
                         FullV, OpTy, "tmp", UserInst);
      FullV = Cast;
    }

    // Update the user.
    UserInst->replaceUsesOfWith(OperandValToReplace, FullV);
  }

  DeadInsts.push_back(OperandValToReplace);
}

void LSRUse::print(raw_ostream &OS) const {
  OS << "LSR Use: Kind=";
  switch (Kind) {
  case Basic:    OS << "Basic"; break;
  case Special:  OS << "Special"; break;
  case ICmpZero: OS << "ICmpZero"; break;
  case Address:
    OS << "Address of ";
    if (isa<PointerType>(AccessTy))
      OS << "pointer"; // the full pointer type could be really verbose
    else
      OS << *AccessTy;
  }

  OS << ", UserInst=";
  // Store is common and interesting enough to be worth special-casing.
  if (StoreInst *Store = dyn_cast<StoreInst>(UserInst)) {
    OS << "store ";
    WriteAsOperand(OS, Store->getOperand(0), /*PrintType=*/false);
  } else if (UserInst->getType()->isVoidTy())
    OS << UserInst->getOpcodeName();
  else
    WriteAsOperand(OS, UserInst, /*PrintType=*/false);

  OS << ", OperandValToReplace=";
  WriteAsOperand(OS, OperandValToReplace, /*PrintType=*/false);

  if (PostIncLoop) {
    OS << ", PostIncLoop=";
    WriteAsOperand(OS, PostIncLoop->getHeader(), /*PrintType=*/false);
  }
}

void LSRUse::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

/// Score - This class is used to measure and compare candidate formulae.
class Score {
  unsigned NumRegs;
  unsigned NumPhis;
  unsigned NumIVMuls;
  unsigned NumBaseAdds;
  unsigned NumImms;

public:
  Score()
    : NumRegs(0), NumPhis(0), NumIVMuls(0), NumBaseAdds(0), NumImms(0) {}

  void RateInitial(SmallVector<LSRUse, 16> const &Uses, const Loop *L,
                   ScalarEvolution &SE);

  void Rate(const SCEV *Reg, const SmallBitVector &Bits,
            const SmallVector<LSRUse, 16> &Uses, const Loop *L,
            ScalarEvolution &SE);

  bool operator<(const Score &Other) const;

  void print_details(raw_ostream &OS, const SCEV *Reg,
                     const SmallPtrSet<const SCEV *, 8> &Regs) const;

  void print(raw_ostream &OS) const;
  void dump() const;

private:
  void RateRegister(const SCEV *Reg, SmallPtrSet<const SCEV *, 8> &Regs,
                    const Loop *L);
  void RateFormula(const Formula &F, SmallPtrSet<const SCEV *, 8> &Regs,
                   const Loop *L);

  void Loose();
};

}

/// RateRegister - Tally up interesting quantities from the given register.
void Score::RateRegister(const SCEV *Reg,
                         SmallPtrSet<const SCEV *, 8> &Regs,
                         const Loop *L) {
  if (Regs.insert(Reg))
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Reg)) {
      NumPhis += AR->getLoop() == L;

      // Add the step value register, if it needs one.
      if (!AR->isAffine() || !isa<SCEVConstant>(AR->getOperand(1)))
        RateRegister(AR->getOperand(1), Regs, L);
    }
}

void Score::RateFormula(const Formula &F,
                        SmallPtrSet<const SCEV *, 8> &Regs,
                        const Loop *L) {
  // Tally up the registers.
  if (F.ScaledReg)
    RateRegister(F.ScaledReg, Regs, L);
  for (SmallVectorImpl<const SCEV *>::const_iterator I = F.BaseRegs.begin(),
       E = F.BaseRegs.end(); I != E; ++I) {
    const SCEV *BaseReg = *I;
    RateRegister(BaseReg, Regs, L);

    NumIVMuls += isa<SCEVMulExpr>(BaseReg) &&
                 BaseReg->hasComputableLoopEvolution(L);
  }

  if (F.BaseRegs.size() > 1)
    NumBaseAdds += F.BaseRegs.size() - 1;

  // Tally up the non-zero immediates.
  if (F.AM.BaseGV || F.AM.BaseOffs != 0)
    ++NumImms;
}

/// Loose - Set this score to a loosing value.
void Score::Loose() {
  NumRegs = ~0u;
  NumPhis = ~0u;
  NumIVMuls = ~0u;
  NumBaseAdds = ~0u;
  NumImms = ~0u;
}

/// RateInitial - Compute a score for the initial "fully reduced" solution.
void Score::RateInitial(SmallVector<LSRUse, 16> const &Uses, const Loop *L,
                        ScalarEvolution &SE) {
  SmallPtrSet<const SCEV *, 8> Regs;
  for (SmallVectorImpl<LSRUse>::const_iterator I = Uses.begin(),
       E = Uses.end(); I != E; ++I)
    RateFormula(I->Formulae.front(), Regs, L);
  NumRegs += Regs.size();

  DEBUG(print_details(dbgs(), 0, Regs));
}

/// Rate - Compute a score for the solution where the reuse associated with
/// putting Reg in a register is selected.
void Score::Rate(const SCEV *Reg, const SmallBitVector &Bits,
                 const SmallVector<LSRUse, 16> &Uses, const Loop *L,
                 ScalarEvolution &SE) {
  SmallPtrSet<const SCEV *, 8> Regs;
  for (size_t i = 0, e = Uses.size(); i != e; ++i) {
    const LSRUse &LU = Uses[i];

    const Formula *BestFormula = 0;
    if (i >= Bits.size() || !Bits.test(i))
      // This use doesn't use the current register. Just go with the current
      // leading candidate formula.
      BestFormula = &LU.Formulae.front();
    else
      // Find the best formula for this use that uses the current register.
      for (SmallVectorImpl<Formula>::const_iterator I = LU.Formulae.begin(),
           E = LU.Formulae.end(); I != E; ++I) {
        const Formula &F = *I;
        if (F.referencesReg(Reg) &&
            (!BestFormula || ComplexitySorter()(F, *BestFormula)))
          BestFormula = &F;
      }

    // If we didn't find *any* forumlae, because earlier we eliminated some
    // in greedy fashion, skip the current register's reuse opportunity.
    if (!BestFormula) {
      DEBUG(dbgs() << "Reuse with reg " << *Reg
                   << " wouldn't help any users.\n");
      Loose();
      return;
    }

    // For an in-loop post-inc user, don't allow multiple base registers,
    // because that would require an awkward in-loop add after the increment.
    if (LU.PostIncLoop && LU.PostIncLoop->contains(LU.UserInst) &&
        BestFormula->BaseRegs.size() > 1) {
      DEBUG(dbgs() << "Reuse with reg " << *Reg
                   << " would require an in-loop post-inc add: ";
            BestFormula->dump());
      Loose();
      return;
    }

    RateFormula(*BestFormula, Regs, L);
  }
  NumRegs += Regs.size();

  DEBUG(print_details(dbgs(), Reg, Regs));
}

/// operator< - Choose the better score.
bool Score::operator<(const Score &Other) const {
  if (NumRegs != Other.NumRegs)
    return NumRegs < Other.NumRegs;
  if (NumPhis != Other.NumPhis)
    return NumPhis < Other.NumPhis;
  if (NumIVMuls != Other.NumIVMuls)
    return NumIVMuls < Other.NumIVMuls;
  if (NumBaseAdds != Other.NumBaseAdds)
    return NumBaseAdds < Other.NumBaseAdds;
  return NumImms < Other.NumImms;
}

void Score::print_details(raw_ostream &OS,
                          const SCEV *Reg,
                          const SmallPtrSet<const SCEV *, 8> &Regs) const {
  if (Reg) OS << "Reuse with reg " << *Reg << " would require ";
  else     OS << "The initial solution would require ";
  print(OS);
  OS << ". Regs:";
  for (SmallPtrSet<const SCEV *, 8>::const_iterator I = Regs.begin(),
       E = Regs.end(); I != E; ++I)
    OS << ' ' << **I;
  OS << '\n';
}

void Score::print(raw_ostream &OS) const {
  OS << NumRegs << " reg" << (NumRegs == 1 ? "" : "s");
  if (NumPhis != 0)
    OS << ", including " << NumPhis << " PHI" << (NumPhis == 1 ? "" : "s");
  if (NumIVMuls != 0)
    OS << ", plus " << NumIVMuls << " IV mul" << (NumIVMuls == 1 ? "" : "s");
  if (NumBaseAdds != 0)
    OS << ", plus " << NumBaseAdds << " base add"
       << (NumBaseAdds == 1 ? "" : "s");
  if (NumImms != 0)
    OS << ", plus " << NumImms << " imm" << (NumImms == 1 ? "" : "s");
}

void Score::dump() const {
  print(errs()); errs() << '\n';
}

/// isAddressUse - Returns true if the specified instruction is using the
/// specified value as an address.
static bool isAddressUse(Instruction *Inst, Value *OperandVal) {
  bool isAddress = isa<LoadInst>(Inst);
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
    if (SI->getOperand(1) == OperandVal)
      isAddress = true;
  } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
    // Addressing modes can also be folded into prefetches and a variety
    // of intrinsics.
    switch (II->getIntrinsicID()) {
      default: break;
      case Intrinsic::prefetch:
      case Intrinsic::x86_sse2_loadu_dq:
      case Intrinsic::x86_sse2_loadu_pd:
      case Intrinsic::x86_sse_loadu_ps:
      case Intrinsic::x86_sse_storeu_ps:
      case Intrinsic::x86_sse2_storeu_pd:
      case Intrinsic::x86_sse2_storeu_dq:
      case Intrinsic::x86_sse2_storel_dq:
        if (II->getOperand(1) == OperandVal)
          isAddress = true;
        break;
    }
  }
  return isAddress;
}

/// getAccessType - Return the type of the memory being accessed.
static const Type *getAccessType(const Instruction *Inst) {
  const Type *AccessTy = Inst->getType();
  if (const StoreInst *SI = dyn_cast<StoreInst>(Inst))
    AccessTy = SI->getOperand(0)->getType();
  else if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst)) {
    // Addressing modes can also be folded into prefetches and a variety
    // of intrinsics.
    switch (II->getIntrinsicID()) {
    default: break;
    case Intrinsic::x86_sse_storeu_ps:
    case Intrinsic::x86_sse2_storeu_pd:
    case Intrinsic::x86_sse2_storeu_dq:
    case Intrinsic::x86_sse2_storel_dq:
      AccessTy = II->getOperand(1)->getType();
      break;
    }
  }
  return AccessTy;
}

/// DeleteTriviallyDeadInstructions - If any of the instructions is the
/// specified set are trivially dead, delete them and see if this makes any of
/// their operands subsequently dead.
static bool
DeleteTriviallyDeadInstructions(SmallVectorImpl<WeakVH> &DeadInsts) {
  bool Changed = false;

  while (!DeadInsts.empty()) {
    Instruction *I = dyn_cast_or_null<Instruction>(DeadInsts.pop_back_val());

    if (I == 0 || !isInstructionTriviallyDead(I))
      continue;

    for (User::op_iterator OI = I->op_begin(), E = I->op_end(); OI != E; ++OI)
      if (Instruction *U = dyn_cast<Instruction>(*OI)) {
        *OI = 0;
        if (U->use_empty())
          DeadInsts.push_back(U);
      }

    I->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

namespace {

/// LSRInstance - This class holds state for the main loop strength
/// reduction logic.
class LSRInstance {
  IVUsers &IU;
  ScalarEvolution &SE;
  DominatorTree &DT;
  const TargetLowering *const TLI;
  Loop *const L;
  bool Changed;

  /// CurrentArbitraryRegIndex - To ensure a deterministic ordering, assign an
  /// arbitrary index value to each register as a sort tie breaker.
  unsigned CurrentArbitraryRegIndex;

  /// Factors - Interesting factors between use strides.
  SmallSetVector<int64_t, 4> Factors;

  /// Types - Interesting use types, to facilitate truncation reuse.
  SmallSetVector<const Type *, 4> Types;

  /// Uses - The list of interesting uses.
  SmallVector<LSRUse, 16> Uses;

  // TODO: Reorganize these data structures.
  typedef DenseMap<const SCEV *, RegSortData> RegUsesTy;
  RegUsesTy RegUses;
  SmallVector<const SCEV *, 16> RegSequence;

  void OptimizeShadowIV();
  bool FindIVUserForCond(ICmpInst *Cond, IVStrideUse *&CondUse,
                         const SCEV* &CondStride);
  ICmpInst *OptimizeMax(ICmpInst *Cond, IVStrideUse* &CondUse);
  bool StrideMightBeShared(const SCEV* Stride);
  bool OptimizeLoopTermCond(Instruction *&IVIncInsertPos);

  LSRUse &getNewUse() {
    Uses.push_back(LSRUse());
    return Uses.back();
  }

  void CountRegister(const SCEV *Reg, uint32_t Complexity, size_t LUIdx);
  void CountRegisters(const Formula &F, size_t LUIdx);

  void GenerateSymbolicOffsetReuse(LSRUse &LU, unsigned LUIdx,
                                   Formula Base);
  void GenerateICmpZeroScaledReuse(LSRUse &LU, unsigned LUIdx,
                                   Formula Base);
  void GenerateFormulaeFromReplacedBaseReg(LSRUse &LU,
                                           unsigned LUIdx,
                                           const Formula &Base, unsigned i,
                                           const SmallVectorImpl<const SCEV *>
                                             &AddOps);
  void GenerateReassociationReuse(LSRUse &LU, unsigned LUIdx,
                                  Formula Base);
  void GenerateCombinationReuse(LSRUse &LU, unsigned LUIdx,
                                Formula Base);
  void GenerateScaledReuse(LSRUse &LU, unsigned LUIdx,
                           Formula Base);
  void GenerateTruncateReuse(LSRUse &LU, unsigned LUIdx,
                             Formula Base);

  void GenerateConstantOffsetReuse();

  void GenerateAllReuseFormulae();

  void GenerateLoopInvariantRegisterUses();

public:
  LSRInstance(const TargetLowering *tli, Loop *l, Pass *P);

  bool getChanged() const { return Changed; }

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

/// OptimizeShadowIV - If IV is used in a int-to-float cast
/// inside the loop then try to eliminate the cast opeation.
void LSRInstance::OptimizeShadowIV() {
  const SCEV *BackedgeTakenCount = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    return;

  for (size_t StrideIdx = 0, e = IU.StrideOrder.size();
       StrideIdx != e; ++StrideIdx) {
    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU.IVUsesByStride.find(IU.StrideOrder[StrideIdx]);
    assert(SI != IU.IVUsesByStride.end() && "Stride doesn't exist!");
    if (!isa<SCEVConstant>(SI->first))
      continue;

    for (ilist<IVStrideUse>::iterator UI = SI->second->Users.begin(),
           E = SI->second->Users.end(); UI != E; /* empty */) {
      ilist<IVStrideUse>::iterator CandidateUI = UI;
      ++UI;
      Instruction *ShadowUse = CandidateUI->getUser();
      const Type *DestTy = NULL;

      /* If shadow use is a int->float cast then insert a second IV
         to eliminate this cast.

           for (unsigned i = 0; i < n; ++i)
             foo((double)i);

         is transformed into

           double d = 0.0;
           for (unsigned i = 0; i < n; ++i, ++d)
             foo(d);
      */
      if (UIToFPInst *UCast = dyn_cast<UIToFPInst>(CandidateUI->getUser()))
        DestTy = UCast->getDestTy();
      else if (SIToFPInst *SCast = dyn_cast<SIToFPInst>(CandidateUI->getUser()))
        DestTy = SCast->getDestTy();
      if (!DestTy) continue;

      if (TLI) {
        // If target does not support DestTy natively then do not apply
        // this transformation.
        EVT DVT = TLI->getValueType(DestTy);
        if (!TLI->isTypeLegal(DVT)) continue;
      }

      PHINode *PH = dyn_cast<PHINode>(ShadowUse->getOperand(0));
      if (!PH) continue;
      if (PH->getNumIncomingValues() != 2) continue;

      const Type *SrcTy = PH->getType();
      int Mantissa = DestTy->getFPMantissaWidth();
      if (Mantissa == -1) continue;
      if ((int)SE.getTypeSizeInBits(SrcTy) > Mantissa)
        continue;

      unsigned Entry, Latch;
      if (PH->getIncomingBlock(0) == L->getLoopPreheader()) {
        Entry = 0;
        Latch = 1;
      } else {
        Entry = 1;
        Latch = 0;
      }

      ConstantInt *Init = dyn_cast<ConstantInt>(PH->getIncomingValue(Entry));
      if (!Init) continue;
      Constant *NewInit = ConstantFP::get(DestTy, Init->getZExtValue());

      BinaryOperator *Incr =
        dyn_cast<BinaryOperator>(PH->getIncomingValue(Latch));
      if (!Incr) continue;
      if (Incr->getOpcode() != Instruction::Add
          && Incr->getOpcode() != Instruction::Sub)
        continue;

      /* Initialize new IV, double d = 0.0 in above example. */
      ConstantInt *C = NULL;
      if (Incr->getOperand(0) == PH)
        C = dyn_cast<ConstantInt>(Incr->getOperand(1));
      else if (Incr->getOperand(1) == PH)
        C = dyn_cast<ConstantInt>(Incr->getOperand(0));
      else
        continue;

      if (!C) continue;

      // Ignore negative constants, as the code below doesn't handle them
      // correctly. TODO: Remove this restriction.
      if (!C->getValue().isStrictlyPositive()) continue;

      /* Add new PHINode. */
      PHINode *NewPH = PHINode::Create(DestTy, "IV.S.", PH);

      /* create new increment. '++d' in above example. */
      Constant *CFP = ConstantFP::get(DestTy, C->getZExtValue());
      BinaryOperator *NewIncr =
        BinaryOperator::Create(Incr->getOpcode() == Instruction::Add ?
                                 Instruction::FAdd : Instruction::FSub,
                               NewPH, CFP, "IV.S.next.", Incr);

      NewPH->addIncoming(NewInit, PH->getIncomingBlock(Entry));
      NewPH->addIncoming(NewIncr, PH->getIncomingBlock(Latch));

      /* Remove cast operation */
      ShadowUse->replaceAllUsesWith(NewPH);
      ShadowUse->eraseFromParent();
      break;
    }
  }
}

/// FindIVUserForCond - If Cond has an operand that is an expression of an IV,
/// set the IV user and stride information and return true, otherwise return
/// false.
bool LSRInstance::FindIVUserForCond(ICmpInst *Cond,
                                    IVStrideUse *&CondUse,
                                    const SCEV* &CondStride) {
  for (unsigned StrideIdx = 0, e = IU.StrideOrder.size();
       StrideIdx != e && !CondUse; ++StrideIdx) {
    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU.IVUsesByStride.find(IU.StrideOrder[StrideIdx]);
    assert(SI != IU.IVUsesByStride.end() && "Stride doesn't exist!");

    for (ilist<IVStrideUse>::iterator UI = SI->second->Users.begin(),
         E = SI->second->Users.end(); UI != E; ++UI)
      if (UI->getUser() == Cond) {
        // NOTE: we could handle setcc instructions with multiple uses here, but
        // InstCombine does it as well for simple uses, it's not clear that it
        // occurs enough in real life to handle.
        CondUse = UI;
        CondStride = SI->first;
        return true;
      }
  }
  return false;
}

/// OptimizeMax - Rewrite the loop's terminating condition if it uses
/// a max computation.
///
/// This is a narrow solution to a specific, but acute, problem. For loops
/// like this:
///
///   i = 0;
///   do {
///     p[i] = 0.0;
///   } while (++i < n);
///
/// the trip count isn't just 'n', because 'n' might not be positive. And
/// unfortunately this can come up even for loops where the user didn't use
/// a C do-while loop. For example, seemingly well-behaved top-test loops
/// will commonly be lowered like this:
//
///   if (n > 0) {
///     i = 0;
///     do {
///       p[i] = 0.0;
///     } while (++i < n);
///   }
///
/// and then it's possible for subsequent optimization to obscure the if
/// test in such a way that indvars can't find it.
///
/// When indvars can't find the if test in loops like this, it creates a
/// max expression, which allows it to give the loop a canonical
/// induction variable:
///
///   i = 0;
///   max = n < 1 ? 1 : n;
///   do {
///     p[i] = 0.0;
///   } while (++i != max);
///
/// Canonical induction variables are necessary because the loop passes
/// are designed around them. The most obvious example of this is the
/// LoopInfo analysis, which doesn't remember trip count values. It
/// expects to be able to rediscover the trip count each time it is
/// needed, and it does this using a simple analysis that only succeeds if
/// the loop has a canonical induction variable.
///
/// However, when it comes time to generate code, the maximum operation
/// can be quite costly, especially if it's inside of an outer loop.
///
/// This function solves this problem by detecting this type of loop and
/// rewriting their conditions from ICMP_NE back to ICMP_SLT, and deleting
/// the instructions for the maximum computation.
///
ICmpInst *LSRInstance::OptimizeMax(ICmpInst *Cond, IVStrideUse* &CondUse) {
  // Check that the loop matches the pattern we're looking for.
  if (Cond->getPredicate() != CmpInst::ICMP_EQ &&
      Cond->getPredicate() != CmpInst::ICMP_NE)
    return Cond;

  SelectInst *Sel = dyn_cast<SelectInst>(Cond->getOperand(1));
  if (!Sel || !Sel->hasOneUse()) return Cond;

  const SCEV *BackedgeTakenCount = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(BackedgeTakenCount))
    return Cond;
  const SCEV *One = SE.getIntegerSCEV(1, BackedgeTakenCount->getType());

  // Add one to the backedge-taken count to get the trip count.
  const SCEV *IterationCount = SE.getAddExpr(BackedgeTakenCount, One);

  // Check for a max calculation that matches the pattern.
  if (!isa<SCEVSMaxExpr>(IterationCount) && !isa<SCEVUMaxExpr>(IterationCount))
    return Cond;
  const SCEVNAryExpr *Max = cast<SCEVNAryExpr>(IterationCount);
  if (Max != SE.getSCEV(Sel)) return Cond;

  // To handle a max with more than two operands, this optimization would
  // require additional checking and setup.
  if (Max->getNumOperands() != 2)
    return Cond;

  const SCEV *MaxLHS = Max->getOperand(0);
  const SCEV *MaxRHS = Max->getOperand(1);
  if (!MaxLHS || MaxLHS != One) return Cond;
  // Check the relevant induction variable for conformance to
  // the pattern.
  const SCEV *IV = SE.getSCEV(Cond->getOperand(0));
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(IV);
  if (!AR || !AR->isAffine() ||
      AR->getStart() != One ||
      AR->getStepRecurrence(SE) != One)
    return Cond;

  assert(AR->getLoop() == L &&
         "Loop condition operand is an addrec in a different loop!");

  // Check the right operand of the select, and remember it, as it will
  // be used in the new comparison instruction.
  Value *NewRHS = 0;
  if (SE.getSCEV(Sel->getOperand(1)) == MaxRHS)
    NewRHS = Sel->getOperand(1);
  else if (SE.getSCEV(Sel->getOperand(2)) == MaxRHS)
    NewRHS = Sel->getOperand(2);
  if (!NewRHS) return Cond;

  // Determine the new comparison opcode. It may be signed or unsigned,
  // and the original comparison may be either equality or inequality.
  CmpInst::Predicate Pred =
    isa<SCEVSMaxExpr>(Max) ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT;
  if (Cond->getPredicate() == CmpInst::ICMP_EQ)
    Pred = CmpInst::getInversePredicate(Pred);

  // Ok, everything looks ok to change the condition into an SLT or SGE and
  // delete the max calculation.
  ICmpInst *NewCond =
    new ICmpInst(Cond, Pred, Cond->getOperand(0), NewRHS, "scmp");

  // Delete the max calculation instructions.
  Cond->replaceAllUsesWith(NewCond);
  CondUse->setUser(NewCond);
  Instruction *Cmp = cast<Instruction>(Sel->getOperand(0));
  Cond->eraseFromParent();
  Sel->eraseFromParent();
  if (Cmp->use_empty())
    Cmp->eraseFromParent();
  return NewCond;
}

bool LSRInstance::StrideMightBeShared(const SCEV* Stride) {
  int64_t SInt = cast<SCEVConstant>(Stride)->getValue()->getSExtValue();
  for (unsigned i = 0, e = IU.StrideOrder.size(); i != e; ++i) {
    std::map<const SCEV *, IVUsersOfOneStride *>::iterator SI =
      IU.IVUsesByStride.find(IU.StrideOrder[i]);
    const SCEV *Share = SI->first;
    if (!isa<SCEVConstant>(SI->first) || Share == Stride)
      continue;
    int64_t SSInt = cast<SCEVConstant>(Share)->getValue()->getSExtValue();
    if (SSInt == SInt)
      return true; // This can definitely be reused.
    if (unsigned(abs64(SSInt)) < SInt || (SSInt % SInt) != 0)
      continue;
    int64_t Scale = SSInt / SInt;

    // This AM will be used for conservative queries. At this point in the
    // process we don't know which users will have a base reg, immediate,
    // etc., so we conservatively assume that it may not, making more
    // strides valid, thus erring on the side of assuming that there
    // might be sharing.
    TargetLowering::AddrMode AM;
    AM.Scale = Scale;

    // Any pre-inc iv use?
    IVUsersOfOneStride &StrideUses = *IU.IVUsesByStride[Share];
    for (ilist<IVStrideUse>::iterator I = StrideUses.Users.begin(),
           E = StrideUses.Users.end(); I != E; ++I) {
      bool isAddress = isAddressUse(I->getUser(), I->getOperandValToReplace());
      if (!I->isUseOfPostIncrementedValue() &&
          isLegalUse(AM, isAddress ? LSRUse::Address : LSRUse::Basic,
                     isAddress ? getAccessType(I->getUser()) : 0,
                     TLI))
        return true;
    }
  }
  return false;
}

/// OptimizeLoopTermCond - Change loop terminating condition to use the
/// postinc iv when possible.
bool
LSRInstance::OptimizeLoopTermCond(Instruction *&IVIncInsertPos) {
  SmallPtrSet<Instruction *, 4> PostIncs;

  BasicBlock *LatchBlock = L->getLoopLatch();
  SmallVector<BasicBlock*, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);

  for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
    BasicBlock *ExitingBlock = ExitingBlocks[i];

    // Get the terminating condition for the loop if possible.  If we
    // can, we want to change it to use a post-incremented version of its
    // induction variable, to allow coalescing the live ranges for the IV into
    // one register value.

    BranchInst *TermBr = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
    if (!TermBr)
      continue;
    // FIXME: Overly conservative, termination condition could be an 'or' etc..
    if (TermBr->isUnconditional() || !isa<ICmpInst>(TermBr->getCondition()))
      continue;

    // Search IVUsesByStride to find Cond's IVUse if there is one.
    IVStrideUse *CondUse = 0;
    const SCEV *CondStride = 0;
    ICmpInst *Cond = cast<ICmpInst>(TermBr->getCondition());
    if (!FindIVUserForCond(Cond, CondUse, CondStride))
      continue;

    // If the trip count is computed in terms of a max (due to ScalarEvolution
    // being unable to find a sufficient guard, for example), change the loop
    // comparison to use SLT or ULT instead of NE.
    // One consequence of doing this now is that it disrupts the count-down
    // optimization. That's not always a bad thing though, because in such
    // cases it may still be worthwhile to avoid a max.
    Cond = OptimizeMax(Cond, CondUse);

    // If this exiting block is the latch block, and the condition has only
    // one use inside the loop (the branch), use the post-incremented value
    // of the induction variable
    if (ExitingBlock != LatchBlock) {
      // If this exiting block dominates the latch block, it may also use
      // the post-inc value if it won't be shared with other uses.
      // Check for dominance.
      if (!DT.dominates(ExitingBlock, LatchBlock))
        continue;
      // Check for sharing within the same stride.
      bool SameStrideSharing = false;
      IVUsersOfOneStride &StrideUses = *IU.IVUsesByStride[CondStride];
      for (ilist<IVStrideUse>::iterator I = StrideUses.Users.begin(),
             E = StrideUses.Users.end(); I != E; ++I) {
        if (I->getUser() == Cond)
          continue;
        if (!I->isUseOfPostIncrementedValue()) {
          SameStrideSharing = true;
          break;
        }
      }
      if (SameStrideSharing)
        continue;
      // Check for sharing from a different stride.
      if (isa<SCEVConstant>(CondStride) && StrideMightBeShared(CondStride))
        continue;
    }
    if (!Cond->hasOneUse()) {
      bool HasOneUseInLoop = true;
      for (Value::use_iterator UI = Cond->use_begin(), UE = Cond->use_end();
           UI != UE; ++UI) {
        Instruction *U = cast<Instruction>(*UI);
        if (U == TermBr)
          continue;
        if (L->contains(U)) {
          HasOneUseInLoop = false;
          break;
        }
      }
      if (!HasOneUseInLoop)
        continue;
    }

    DEBUG(dbgs() << "  Change loop exiting icmp to use postinc iv: "
                 << *Cond << '\n');

    // It's possible for the setcc instruction to be anywhere in the loop, and
    // possible for it to have multiple users.  If it is not immediately before
    // the exiting block branch, move it.
    if (&*++BasicBlock::iterator(Cond) != TermBr) {
      if (Cond->hasOneUse()) {   // Condition has a single use, just move it.
        Cond->moveBefore(TermBr);
      } else {
        // Otherwise, clone the terminating condition and insert into the
        // loopend.
        ICmpInst *OldCond = Cond;
        Cond = cast<ICmpInst>(Cond->clone());
        Cond->setName(L->getHeader()->getName() + ".termcond");
        ExitingBlock->getInstList().insert(TermBr, Cond);

        // Clone the IVUse, as the old use still exists!
        IU.IVUsesByStride[CondStride]->addUser(CondUse->getOffset(), Cond,
                                             CondUse->getOperandValToReplace());
        CondUse = &IU.IVUsesByStride[CondStride]->Users.back();
        TermBr->replaceUsesOfWith(OldCond, Cond);
      }
    }

    // If we get to here, we know that we can transform the setcc instruction to
    // use the post-incremented version of the IV, allowing us to coalesce the
    // live ranges for the IV correctly.
    CondUse->setOffset(SE.getMinusSCEV(CondUse->getOffset(), CondStride));
    CondUse->setIsUseOfPostIncrementedValue(true);
    Changed = true;

    PostIncs.insert(Cond);
  }

  // Determine an insertion point for the loop induction variable increment. It
  // must dominate all the post-inc comparisons we just set up, and it must
  // dominate the loop latch edge.
  IVIncInsertPos = L->getLoopLatch()->getTerminator();
  for (SmallPtrSet<Instruction *, 4>::iterator I = PostIncs.begin(),
       E = PostIncs.end(); I != E; ++I) {
    BasicBlock *BB =
      DT.findNearestCommonDominator(IVIncInsertPos->getParent(),
                                    (*I)->getParent());
    if (BB == (*I)->getParent())
      IVIncInsertPos = *I;
    else if (BB != IVIncInsertPos->getParent())
      IVIncInsertPos = BB->getTerminator();
  }

  return Changed;
}

/// CountRegisters - Note the given register.
void LSRInstance::CountRegister(const SCEV *Reg, uint32_t Complexity,
                                size_t LUIdx) {
  std::pair<RegUsesTy::iterator, bool> Pair =
    RegUses.insert(std::make_pair(Reg, RegSortData()));
  RegSortData &BV = Pair.first->second;
  if (Pair.second) {
    BV.Index = CurrentArbitraryRegIndex++;
    BV.MaxComplexity = Complexity;
    RegSequence.push_back(Reg);
  } else {
    BV.MaxComplexity = std::max(BV.MaxComplexity, Complexity);
  }
  BV.Bits.resize(std::max(BV.Bits.size(), LUIdx + 1));
  BV.Bits.set(LUIdx);
}

/// CountRegisters - Note which registers are used by the given formula,
/// updating RegUses.
void LSRInstance::CountRegisters(const Formula &F, size_t LUIdx) {
  uint32_t Complexity = F.getComplexity();
  if (F.ScaledReg)
    CountRegister(F.ScaledReg, Complexity, LUIdx);
  for (SmallVectorImpl<const SCEV *>::const_iterator I = F.BaseRegs.begin(),
       E = F.BaseRegs.end(); I != E; ++I)
    CountRegister(*I, Complexity, LUIdx);
}

/// GenerateSymbolicOffsetReuse - Generate reuse formulae using symbolic
/// offsets.
void LSRInstance::GenerateSymbolicOffsetReuse(LSRUse &LU, unsigned LUIdx,
                                              Formula Base) {
  // We can't add a symbolic offset if the address already contains one.
  if (Base.AM.BaseGV) return;

  for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i) {
    const SCEV *G = Base.BaseRegs[i];
    GlobalValue *GV = ExtractSymbol(G, SE);
    if (G->isZero())
      continue;
    Formula F = Base;
    F.AM.BaseGV = GV;
    if (!isLegalUse(F.AM, LU.Kind, LU.AccessTy, TLI))
      continue;
    F.BaseRegs[i] = G;
    if (LU.InsertFormula(F))
      CountRegisters(LU.Formulae.back(), LUIdx);
  }
}

/// GenerateICmpZeroScaledReuse - For ICmpZero, check to see if we can scale up
/// the comparison. For example, x == y -> x*c == y*c.
void LSRInstance::GenerateICmpZeroScaledReuse(LSRUse &LU, unsigned LUIdx,
                                              Formula Base) {
  if (LU.Kind != LSRUse::ICmpZero) return;

  // Determine the integer type for the base formula.
  const Type *IntTy = Base.getType();
  if (!IntTy) return;
  if (SE.getTypeSizeInBits(IntTy) > 64) return;
  IntTy = SE.getEffectiveSCEVType(IntTy);

  assert(!Base.AM.BaseGV && "ICmpZero use is not legal!");

  // Check each interesting stride.
  for (SmallSetVector<int64_t, 4>::const_iterator
       I = Factors.begin(), E = Factors.end(); I != E; ++I) {
    int64_t Factor = *I;
    Formula F = Base;

    // Check that the multiplication doesn't overflow.
    F.AM.BaseOffs = (uint64_t)F.AM.BaseOffs * Factor;
    if ((int64_t)F.AM.BaseOffs / Factor != F.AM.BaseOffs)
      continue;

    // Check that this scale is legal.
    if (!isLegalUse(F.AM, LU.Kind, LU.AccessTy, TLI))
      continue;

    const SCEV *FactorS = SE.getSCEV(ConstantInt::get(IntTy, Factor));

    // Check that multiplying with each base register doesn't overflow.
    for (size_t i = 0, e = F.BaseRegs.size(); i != e; ++i) {
      F.BaseRegs[i] = SE.getMulExpr(F.BaseRegs[i], FactorS);
      if (getSDiv(F.BaseRegs[i], FactorS, SE) != Base.BaseRegs[i])
        goto next;
    }

    // Check that multiplying with the scaled register doesn't overflow.
    if (F.ScaledReg) {
      F.ScaledReg = SE.getMulExpr(F.ScaledReg, FactorS);
      if (getSDiv(F.ScaledReg, FactorS, SE) != Base.ScaledReg)
        continue;
    }

    // If we make it here and it's legal, add it.
    if (LU.InsertFormula(F))
      CountRegisters(LU.Formulae.back(), LUIdx);
  next:;
  }
}

/// GenerateFormulaeFromReplacedBaseReg - If removing base register with
/// index i from the BaseRegs list and adding the registers in AddOps
/// to the address forms an interesting formula, pursue it.
void
LSRInstance::GenerateFormulaeFromReplacedBaseReg(
                                             LSRUse &LU,
                                             unsigned LUIdx,
                                             const Formula &Base, unsigned i,
                                             const SmallVectorImpl<const SCEV *>
                                               &AddOps) {
  if (AddOps.empty()) return;

  Formula F = Base;
  std::swap(F.BaseRegs[i], F.BaseRegs.back());
  F.BaseRegs.pop_back();
  for (SmallVectorImpl<const SCEV *>::const_iterator I = AddOps.begin(),
       E = AddOps.end(); I != E; ++I)
    F.BaseRegs.push_back(*I);
  F.AM.HasBaseReg = !F.BaseRegs.empty();
  if (LU.InsertFormula(F)) {
    CountRegisters(LU.Formulae.back(), LUIdx);
    // Recurse.
    GenerateReassociationReuse(LU, LUIdx, LU.Formulae.back());
  }
}

/// GenerateReassociationReuse - Split out subexpressions from adds and
/// the bases of addrecs.
void LSRInstance::GenerateReassociationReuse(LSRUse &LU, unsigned LUIdx,
                                             Formula Base) {
  SmallVector<const SCEV *, 8> AddOps;
  for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i) {
    const SCEV *BaseReg = Base.BaseRegs[i];
    if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(BaseReg)) {
      for (SCEVAddExpr::op_iterator J = Add->op_begin(), JE = Add->op_end();
           J != JE; ++J) {
        // Don't pull a constant into a register if the constant could be
        // folded into an immediate field.
        if (isAlwaysFoldable(*J, true, LU.Kind, LU.AccessTy, TLI, SE)) continue;
        SmallVector<const SCEV *, 8> InnerAddOps;
        for (SCEVAddExpr::op_iterator K = Add->op_begin(); K != JE; ++K)
          if (K != J)
            InnerAddOps.push_back(*K);
        // Splitting a 2-operand add both ways is redundant. Pruning this
        // now saves compile time.
        if (InnerAddOps.size() < 2 && next(J) == JE)
          continue;
        AddOps.push_back(*J);
        const SCEV *InnerAdd = SE.getAddExpr(InnerAddOps);
        AddOps.push_back(InnerAdd);
        GenerateFormulaeFromReplacedBaseReg(LU, LUIdx, Base, i, AddOps);
        AddOps.clear();
      }
    } else if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(BaseReg)) {
      if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(AR->getStart())) {
        for (SCEVAddExpr::op_iterator J = Add->op_begin(), JE = Add->op_end();
             J != JE; ++J) {
          // Don't pull a constant into a register if the constant could be
          // folded into an immediate field.
          if (isAlwaysFoldable(*J, true, LU.Kind, LU.AccessTy, TLI, SE))
            continue;
          SmallVector<const SCEV *, 8> InnerAddOps;
          for (SCEVAddExpr::op_iterator K = Add->op_begin(); K != JE; ++K)
            if (K != J)
              InnerAddOps.push_back(*K);
          AddOps.push_back(*J);
          const SCEV *InnerAdd = SE.getAddExpr(InnerAddOps);
          AddOps.push_back(SE.getAddRecExpr(InnerAdd,
                                            AR->getStepRecurrence(SE),
                                            AR->getLoop()));
          GenerateFormulaeFromReplacedBaseReg(LU, LUIdx, Base, i, AddOps);
          AddOps.clear();
        }
      } else if (!isAlwaysFoldable(AR->getStart(), Base.BaseRegs.size() > 1,
                                   LU.Kind, LU.AccessTy,
                                   TLI, SE)) {
        AddOps.push_back(AR->getStart());
        AddOps.push_back(SE.getAddRecExpr(SE.getIntegerSCEV(0,
                                                            BaseReg->getType()),
                                          AR->getStepRecurrence(SE),
                                          AR->getLoop()));
        GenerateFormulaeFromReplacedBaseReg(LU, LUIdx, Base, i, AddOps);
        AddOps.clear();
      }
    }
  }
}

/// GenerateCombinationReuse - Generate a formula consisting of all of the
/// loop-dominating registers added into a single register.
void LSRInstance::GenerateCombinationReuse(LSRUse &LU, unsigned LUIdx,
                                           Formula Base) {
  if (Base.BaseRegs.size() <= 1) return;

  Formula F = Base;
  F.BaseRegs.clear();
  SmallVector<const SCEV *, 4> Ops;
  for (SmallVectorImpl<const SCEV *>::const_iterator
       I = Base.BaseRegs.begin(), E = Base.BaseRegs.end(); I != E; ++I) {
    const SCEV *BaseReg = *I;
    if (BaseReg->properlyDominates(L->getHeader(), &DT) &&
        !BaseReg->hasComputableLoopEvolution(L))
      Ops.push_back(BaseReg);
    else
      F.BaseRegs.push_back(BaseReg);
  }
  if (Ops.size() > 1) {
    F.BaseRegs.push_back(SE.getAddExpr(Ops));
    if (LU.InsertFormula(F))
      CountRegisters(LU.Formulae.back(), LUIdx);
  }
}

/// GenerateScaledReuse - Generate stride factor reuse formulae by making
/// use of scaled-offset address modes, for example.
void LSRInstance::GenerateScaledReuse(LSRUse &LU, unsigned LUIdx,
                                      Formula Base) {
  // Determine the integer type for the base formula.
  const Type *IntTy = Base.getType();
  if (!IntTy) return;
  IntTy = SE.getEffectiveSCEVType(IntTy);

  // Check each interesting stride.
  for (SmallSetVector<int64_t, 4>::const_iterator
       I = Factors.begin(), E = Factors.end(); I != E; ++I) {
    int64_t Factor = *I;

    // If this Formula already has a scaled register, we can't add another one.
    if (Base.AM.Scale != 0)
      continue;
    Formula F = Base;
    F.AM.Scale = Factor;
    // Check whether this scale is going to be legal.
    if (!isLegalUse(F.AM, LU.Kind, LU.AccessTy, TLI)) {
      // As a special-case, handle special out-of-loop Basic users specially.
      // TODO: Reconsider this special case.
      if (LU.Kind == LSRUse::Basic &&
          isLegalUse(F.AM, LSRUse::Special, LU.AccessTy, TLI) &&
          !L->contains(LU.UserInst))
        LU.Kind = LSRUse::Special;
      else
        continue;
    }
    // For each addrec base reg, apply the scale, if possible.
    for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i)
      if (const SCEVAddRecExpr *AR =
            dyn_cast<SCEVAddRecExpr>(Base.BaseRegs[i])) {
        const SCEV *FactorS = SE.getSCEV(ConstantInt::get(IntTy, Factor));
        // Divide out the factor, ignoring high bits, since we'll be
        // scaling the value back up in the end.
        if (const SCEV *Quotient = getSDiv(AR, FactorS, SE, true)) {
          // TODO: This could be optimized to avoid all the copying.
          Formula NewF = F;
          NewF.ScaledReg = Quotient;
          std::swap(NewF.BaseRegs[i], NewF.BaseRegs.back());
          NewF.BaseRegs.pop_back();
          NewF.AM.HasBaseReg = !NewF.BaseRegs.empty();
          if (LU.InsertFormula(NewF))
            CountRegisters(LU.Formulae.back(), LUIdx);
        }
      }
  }
}

/// GenerateTruncateReuse - Generate reuse formulae from different IV types.
void LSRInstance::GenerateTruncateReuse(LSRUse &LU, unsigned LUIdx,
                                        Formula Base) {
  // This requires TargetLowering to tell us which truncates are free.
  if (!TLI) return;

  // Don't attempt to truncate symbolic values.
  if (Base.AM.BaseGV) return;

  // Determine the integer type for the base formula.
  const Type *DstTy = Base.getType();
  if (!DstTy) return;
  DstTy = SE.getEffectiveSCEVType(DstTy);

  for (SmallSetVector<const Type *, 4>::const_iterator
       I = Types.begin(), E = Types.end(); I != E; ++I) {
    const Type *SrcTy = *I;
    if (SrcTy != DstTy && TLI->isTruncateFree(SrcTy, DstTy)) {
      Formula F = Base;
      if (F.ScaledReg) F.ScaledReg = SE.getAnyExtendExpr(F.ScaledReg, *I);
      for (SmallVectorImpl<const SCEV *>::iterator J = F.BaseRegs.begin(),
           JE = F.BaseRegs.end(); J != JE; ++J)
        *J = SE.getAnyExtendExpr(*J, SrcTy);
      if (LU.InsertFormula(F))
        CountRegisters(LU.Formulae.back(), LUIdx);
    }
  }
}

namespace {

/// WorkItem - Helper class for GenerateConstantOffsetReuse. It's used to
/// defer modifications so that the search phase doesn't have to worry about
/// the data structures moving underneath it.
struct WorkItem {
  LSRUse *LU;
  size_t LUIdx;
  int64_t Imm;
  const SCEV *OrigReg;

  WorkItem(LSRUse *U, size_t LI, int64_t I, const SCEV *R)
    : LU(U), LUIdx(LI), Imm(I), OrigReg(R) {}

  void print(raw_ostream &OS) const;
  void dump() const;
};

void WorkItem::print(raw_ostream &OS) const {
  OS << "in use ";
  LU->print(OS);
  OS << " (at index " << LUIdx << "), add offset " << Imm
     << " and compensate by adjusting refences to " << *OrigReg << "\n";
}

void WorkItem::dump() const {
  print(errs()); errs() << '\n';
}

}

/// GenerateConstantOffsetReuse - Look for registers which are a constant
/// distance apart and try to form reuse opportunities between them.
void LSRInstance::GenerateConstantOffsetReuse() {
  // Group the registers by their value without any added constant offset.
  typedef std::map<int64_t, const SCEV *> ImmMapTy;
  typedef DenseMap<const SCEV *, ImmMapTy> RegMapTy;
  RegMapTy Map;
  SmallVector<const SCEV *, 8> Sequence;
  for (SmallVectorImpl<const SCEV *>::iterator I = RegSequence.begin(),
       E = RegSequence.end(); I != E; ++I) {
    const SCEV *Reg = *I;
    int64_t Imm = ExtractImmediate(Reg, SE);
    std::pair<RegMapTy::iterator, bool> Pair =
      Map.insert(std::make_pair(Reg, ImmMapTy()));
    if (Pair.second)
      Sequence.push_back(Reg);
    Pair.first->second.insert(std::make_pair(Imm, *I));
  }

  // Insert an artificial expression at offset 0 (if there isn't one already),
  // as this may lead to more reuse opportunities.
  for (SmallVectorImpl<const SCEV *>::const_iterator I = Sequence.begin(),
       E = Sequence.end(); I != E; ++I)
    Map.find(*I)->second.insert(ImmMapTy::value_type(0, 0));

  // Now examine each set of registers with the same base value. Build up
  // a list of work to do and do the work in a separate step so that we're
  // not adding formulae and register counts while we're searching.
  SmallVector<WorkItem, 32> WorkItems;
  for (SmallVectorImpl<const SCEV *>::const_iterator I = Sequence.begin(),
       E = Sequence.end(); I != E; ++I) {
    const SCEV *Reg = *I;
    const ImmMapTy &Imms = Map.find(Reg)->second;
    // Examine each offset.
    for (ImmMapTy::const_iterator J = Imms.begin(), JE = Imms.end();
         J != JE; ++J) {
      const SCEV *OrigReg = J->second;
      // Skip the artifical register at offset 0.
      if (!OrigReg) continue;

      int64_t JImm = J->first;
      const SmallBitVector &Bits = RegUses.find(OrigReg)->second.Bits;

      // Examine each other offset associated with the same register. This is
      // quadradic in the number of registers with the same base, but it's
      // uncommon for this to be a large number.
      for (ImmMapTy::const_iterator M = Imms.begin(); M != JE; ++M) {
        if (M == J) continue;

        // Compute the difference between the two.
        int64_t Imm = (uint64_t)JImm - M->first;
        for (int LUIdx = Bits.find_first(); LUIdx != -1;
             LUIdx = Bits.find_next(LUIdx))
          // Make a memo of this use, offset, and register tuple.
          WorkItems.push_back(WorkItem(&Uses[LUIdx], LUIdx, Imm, OrigReg));
      }
    }
  }

  // Now iterate through the worklist and add new formulae.
  for (SmallVectorImpl<WorkItem>::const_iterator I = WorkItems.begin(),
       E = WorkItems.end(); I != E; ++I) {
    const WorkItem &WI = *I;
    LSRUse &LU = *WI.LU;
    size_t LUIdx = WI.LUIdx;
    int64_t Imm = WI.Imm;
    const SCEV *OrigReg = WI.OrigReg;

    const Type *IntTy = SE.getEffectiveSCEVType(OrigReg->getType());
    const SCEV *NegImmS = SE.getSCEV(ConstantInt::get(IntTy,
                                                      -(uint64_t)Imm));

    for (size_t L = 0, LE = LU.Formulae.size(); L != LE; ++L) {
      Formula F = LU.Formulae[L];
      // Use the immediate in the scaled register.
      if (F.ScaledReg == OrigReg) {
        int64_t Offs = (uint64_t)F.AM.BaseOffs +
                       Imm * (uint64_t)F.AM.Scale;
        // Don't create 50 + reg(-50).
        if (F.referencesReg(SE.getSCEV(
                   ConstantInt::get(IntTy, -(uint64_t)Offs))))
          continue;
        Formula NewF = F;
        NewF.AM.BaseOffs = Offs;
        if (!isLegalUse(NewF.AM, LU.Kind, LU.AccessTy, TLI))
          continue;
        const SCEV *Diff = SE.getAddExpr(NegImmS, NewF.ScaledReg);
        if (Diff->isZero()) continue;
        NewF.ScaledReg = Diff;
        if (LU.InsertFormula(NewF))
          CountRegisters(LU.Formulae.back(), LUIdx);
      }
      // Use the immediate in a base register.
      for (size_t N = 0, NE = F.BaseRegs.size(); N != NE; ++N) {
        const SCEV *BaseReg = F.BaseRegs[N];
        if (BaseReg != OrigReg)
          continue;
        Formula NewF = F;
        NewF.AM.BaseOffs = (uint64_t)NewF.AM.BaseOffs + Imm;
        if (!isLegalUse(NewF.AM, LU.Kind, LU.AccessTy, TLI))
          continue;
        const SCEV *Diff = SE.getAddExpr(NegImmS, BaseReg);
        if (Diff->isZero()) continue;
        // Don't create 50 + reg(-50).
        if (Diff ==
            SE.getSCEV(ConstantInt::get(IntTy,
                                        -(uint64_t)NewF.AM.BaseOffs)))
          continue;
        NewF.BaseRegs[N] = Diff;
        if (LU.InsertFormula(NewF))
          CountRegisters(LU.Formulae.back(), LUIdx);
      }
    }
  }
}

/// GenerateAllReuseFormulae - Generate formulae for each use.
void
LSRInstance::GenerateAllReuseFormulae() {
  SmallVector<Formula, 12> Save;
  for (size_t LUIdx = 0, NumUses = Uses.size(); LUIdx != NumUses; ++LUIdx) {
    LSRUse &LU = Uses[LUIdx];

    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateSymbolicOffsetReuse(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateICmpZeroScaledReuse(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateReassociationReuse(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateCombinationReuse(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateScaledReuse(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateTruncateReuse(LU, LUIdx, LU.Formulae[i]);
  }

  GenerateConstantOffsetReuse();
}

/// GenerateLoopInvariantRegisterUses - Check for other uses of loop-invariant
/// values which we're tracking. These other uses will pin these values in
/// registers, making them less profitable for elimination.
/// TODO: This currently misses non-constant addrec step registers.
/// TODO: Should this give more weight to users inside the loop?
void
LSRInstance::GenerateLoopInvariantRegisterUses() {
  for (size_t i = 0, e = RegSequence.size(); i != e; ++i)
    if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(RegSequence[i])) {
      const Value *V = U->getValue();
      if (const Instruction *Inst = dyn_cast<Instruction>(V))
        if (L->contains(Inst)) continue;
      for (Value::use_const_iterator UI = V->use_begin(), UE = V->use_end();
           UI != UE; ++UI) {
        const Instruction *UserInst = dyn_cast<Instruction>(*UI);
        // Ignore non-instructions.
        if (!UserInst)
          continue;
        // Ignore instructions in other functions (as can happen with
        // Constants).
        if (UserInst->getParent()->getParent() != L->getHeader()->getParent())
          continue;
        // Ignore instructions not dominated by the loop.
        const BasicBlock *UseBB = !isa<PHINode>(UserInst) ?
          UserInst->getParent() :
          cast<PHINode>(UserInst)->getIncomingBlock(
            PHINode::getIncomingValueNumForOperand(UI.getOperandNo()));
        if (!DT.dominates(L->getHeader(), UseBB))
          continue;
        // Ignore uses which are part of other SCEV expressions, to avoid
        // analyzing them multiple times.
        if (SE.isSCEVable(UserInst->getType()) &&
            !isa<SCEVUnknown>(SE.getSCEV(const_cast<Instruction *>(UserInst))))
          continue;
        // Ignore icmp instructions which are already being analyzed.
        if (const ICmpInst *ICI = dyn_cast<ICmpInst>(UserInst)) {
          unsigned OtherIdx = !UI.getOperandNo();
          Value *OtherOp = const_cast<Value *>(ICI->getOperand(OtherIdx));
          if (SE.getSCEV(OtherOp)->hasComputableLoopEvolution(L))
            continue;
        }

        LSRUse &LU = getNewUse();
        LU.UserInst = const_cast<Instruction *>(UserInst);
        LU.OperandValToReplace = UI.getUse();
        LU.InsertSupplementalFormula(U);
        CountRegisters(LU.Formulae.back(), Uses.size() - 1);
      }
    }
}

#ifndef NDEBUG

static void debug_winner(SmallVector<LSRUse, 16> const &Uses) {
  dbgs() << "LSR has selected formulae for each use:\n";
  for (SmallVectorImpl<LSRUse>::const_iterator I = Uses.begin(),
       E = Uses.end(); I != E; ++I) {
    const LSRUse &LU = *I;
    dbgs() << "  ";
    LU.print(dbgs());
    dbgs() << '\n';
    dbgs() << "    ";
    LU.Formulae.front().print(dbgs());
    dbgs() << "\n";
  }
}

#endif

LSRInstance::LSRInstance(const TargetLowering *tli, Loop *l, Pass *P)
  : IU(P->getAnalysis<IVUsers>()),
    SE(P->getAnalysis<ScalarEvolution>()),
    DT(P->getAnalysis<DominatorTree>()),
    TLI(tli), L(l), Changed(false), CurrentArbitraryRegIndex(0) {

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm()) return;

  // If there's no interesting work to be done, bail early.
  if (IU.IVUsesByStride.empty()) return;

  DEBUG(dbgs() << "\nLSR on loop ";
        WriteAsOperand(dbgs(), L->getHeader(), /*PrintType=*/false);
        dbgs() << ":\n");

  // Sort the StrideOrder so we process larger strides first.
  std::stable_sort(IU.StrideOrder.begin(), IU.StrideOrder.end(),
                   StrideCompare(SE));

  /// OptimizeShadowIV - If IV is used in a int-to-float cast
  /// inside the loop then try to eliminate the cast opeation.
  OptimizeShadowIV();

  // Change loop terminating condition to use the postinc iv when possible.
  Instruction *IVIncInsertPos;
  Changed |= OptimizeLoopTermCond(IVIncInsertPos);

  for (SmallVectorImpl<const SCEV *>::const_iterator SIter =
       IU.StrideOrder.begin(), SEnd = IU.StrideOrder.end();
       SIter != SEnd; ++SIter) {
    const SCEV *Stride = *SIter;

    // Collect interesting types.
    Types.insert(SE.getEffectiveSCEVType(Stride->getType()));

    // Collect interesting factors.
    for (SmallVectorImpl<const SCEV *>::const_iterator NewStrideIter =
         SIter + 1; NewStrideIter != SEnd; ++NewStrideIter) {
      const SCEV *OldStride = Stride;
      const SCEV *NewStride = *NewStrideIter;

      if (SE.getTypeSizeInBits(OldStride->getType()) !=
          SE.getTypeSizeInBits(NewStride->getType())) {
        if (SE.getTypeSizeInBits(OldStride->getType()) >
            SE.getTypeSizeInBits(NewStride->getType()))
          NewStride = SE.getSignExtendExpr(NewStride, OldStride->getType());
        else
          OldStride = SE.getSignExtendExpr(OldStride, NewStride->getType());
      }
      if (const SCEVConstant *Factor =
            dyn_cast_or_null<SCEVConstant>(getSDiv(NewStride, OldStride,
                                                   SE, true)))
        if (Factor->getValue()->getValue().getMinSignedBits() <= 64)
          Factors.insert(Factor->getValue()->getValue().getSExtValue());
    }

    std::map<const SCEV *, IVUsersOfOneStride *>::const_iterator SI =
      IU.IVUsesByStride.find(Stride);
    assert(SI != IU.IVUsesByStride.end() && "Stride doesn't exist!");
    for (ilist<IVStrideUse>::const_iterator UI = SI->second->Users.begin(),
         E = SI->second->Users.end(); UI != E; ++UI) {
      // Record the uses.
      LSRUse &LU = getNewUse();
      LU.UserInst = UI->getUser();
      LU.OperandValToReplace = UI->getOperandValToReplace();
      if (isAddressUse(LU.UserInst, LU.OperandValToReplace)) {
        LU.Kind = LSRUse::Address;
        LU.AccessTy = getAccessType(LU.UserInst);
      }
      if (UI->isUseOfPostIncrementedValue())
        LU.PostIncLoop = L;

      const SCEV *S = IU.getCanonicalExpr(*UI);

      // Equality (== and !=) ICmps are special. We can rewrite (i == N) as
      // (N - i == 0), and this allows (N - i) to be the expression that we
      // work with rather than just N or i, so we can consider the register
      // requirements for both N and i at the same time. Limiting this code
      // to equality icmps is not a problem because all interesting loops
      // use equality icmps, thanks to IndVarSimplify.
      if (ICmpInst *CI = dyn_cast<ICmpInst>(LU.UserInst))
        if (CI->isEquality()) {
          // Swap the operands if needed to put the OperandValToReplace on
          // the left, for consistency.
          Value *NV = CI->getOperand(1);
          if (NV == LU.OperandValToReplace) {
            CI->setOperand(1, CI->getOperand(0));
            CI->setOperand(0, NV);
          }

          // x == y  -->  x - y == 0
          const SCEV *N = SE.getSCEV(NV);
          if (N->isLoopInvariant(L)) {
            LU.Kind = LSRUse::ICmpZero;
            S = SE.getMinusSCEV(N, S);
          }

          // -1 and the negations of all interesting strides (except the
          // negation of -1) are now also interesting.
          for (size_t i = 0, e = Factors.size(); i != e; ++i)
            if (Factors[i] != -1)
              Factors.insert(-(uint64_t)Factors[i]);
          Factors.insert(-1);
        }

      // Ok, now enumerate all the different formulae we can find to compute
      // the value for this expression.
      LU.InsertInitialFormula(S, L, SE, DT);
      CountRegisters(LU.Formulae.back(), Uses.size() - 1);
    }
  }

  // If all uses use the same type, don't bother looking for truncation-based
  // reuse.
  if (Types.size() == 1)
    Types.clear();

  // Now use the reuse data to generate a bunch of interesting ways
  // to formulate the values needed for the uses.
  GenerateAllReuseFormulae();

  // If there are any uses of registers that we're tracking that have escaped
  // IVUsers' attention, add trivial uses for them, so that the register
  // voting process takes the into consideration.
  GenerateLoopInvariantRegisterUses();

  // Sort the formulae. TODO: This is redundantly sorted below.
  for (SmallVectorImpl<LSRUse>::iterator I = Uses.begin(), E = Uses.end();
       I != E; ++I) {
    LSRUse &LU = *I;
    std::stable_sort(LU.Formulae.begin(), LU.Formulae.end(),
                     ComplexitySorter());
  }

  // Ok, we've now collected all the uses and noted their register uses. The
  // next step is to start looking at register reuse possibilities.
  DEBUG(print(dbgs()); dbgs() << '\n');

  // Start by assuming we'll assign each use its own register. This is
  // sometimes called "full" strength reduction, or "superhero" mode.
  // Sometimes this is the best solution, but if there are opportunities for
  // reuse we may find a better solution.
  Score CurScore;
  CurScore.RateInitial(Uses, L, SE);

  // Create a sorted list of registers with those with the most uses appearing
  // earlier in the list. We'll visit them first, as they're the most likely
  // to represent profitable reuse opportunities.
  SmallVector<RegCount, 8> RegOrder;
  for (SmallVectorImpl<const SCEV *>::const_iterator I =
       RegSequence.begin(), E = RegSequence.end(); I != E; ++I)
    RegOrder.push_back(RegCount(*I, RegUses.find(*I)->second));
  std::stable_sort(RegOrder.begin(), RegOrder.end());

  // Visit each register. Determine which ones represent profitable reuse
  // opportunities and remember them.
  // TODO: Extract this code into a function.
  for (SmallVectorImpl<RegCount>::const_iterator I = RegOrder.begin(),
       E = RegOrder.end(); I != E; ++I) {
    const SCEV *Reg = I->Reg;
    const SmallBitVector &Bits = I->Sort.Bits;

    // Registers with only one use don't represent reuse opportunities, so
    // when we get there, we're done.
    if (Bits.count() <= 1) break;

    DEBUG(dbgs() << "Reg " << *Reg << ": ";
          I->Sort.print(dbgs());
          dbgs() << '\n');

    // Determine the total number of registers will be needed if we make use
    // of the reuse opportunity represented by the current register.
    Score NewScore;
    NewScore.Rate(Reg, Bits, Uses, L, SE);

    // Now decide whether this register's reuse opportunity is an overall win.
    // Currently the decision is heavily skewed for register pressure.
    if (!(NewScore < CurScore)) {
      continue;
    }

    // Ok, use this opportunity.
    DEBUG(dbgs() << "This candidate has been accepted.\n");
    CurScore = NewScore;

    // Now that we've selected a new reuse opportunity, delete formulae that
    // do not participate in that opportunity.
    for (int j = Bits.find_first(); j != -1; j = Bits.find_next(j)) {
      LSRUse &LU = Uses[j];
      for (unsigned k = 0, h = LU.Formulae.size(); k != h; ++k) {
        Formula &F = LU.Formulae[k];
        if (!F.referencesReg(Reg)) {
          std::swap(LU.Formulae[k], LU.Formulae.back());
          LU.Formulae.pop_back();
          --k; --h;
        }
      }
      // Also re-sort the list to put the formulae with the fewest registers
      // at the front.
      // TODO: Do this earlier, we don't need it each time.
      std::stable_sort(LU.Formulae.begin(), LU.Formulae.end(),
                       ComplexitySorter());
    }
  }

  // Ok, we've now made all our decisions. The first formula for each use
  // will be used.
  DEBUG(dbgs() << "Concluding, we need "; CurScore.print(dbgs());
        dbgs() << ".\n";
        debug_winner(Uses));

  // Free memory no longer needed.
  RegOrder.clear();
  Factors.clear();
  Types.clear();
  RegUses.clear();
  RegSequence.clear();

  // Keep track of instructions we may have made dead, so that
  // we can remove them after we are done working.
  SmallVector<WeakVH, 16> DeadInsts;

  SCEVExpander Rewriter(SE);
  Rewriter.disableCanonicalMode();
  Rewriter.setIVIncInsertPos(L, IVIncInsertPos);

  // Expand the new value definitions and update the users.
  for (SmallVectorImpl<LSRUse>::const_iterator I = Uses.begin(),
       E = Uses.end(); I != E; ++I) {
    // Formulae should be legal.
    DEBUG(for (SmallVectorImpl<Formula>::const_iterator J = I->Formulae.begin(),
               JE = I->Formulae.end(); J != JE; ++J)
            assert(isLegalUse(J->AM, I->Kind, I->AccessTy, TLI) &&
                   "Illegal formula generated!"));

    // Expand the new code and update the user.
    I->Rewrite(L, Rewriter, DeadInsts, SE, DT, P);
    Changed = true;
  }

  // Clean up after ourselves. This must be done before deleting any
  // instructions.
  Rewriter.clear();

  Changed |= DeleteTriviallyDeadInstructions(DeadInsts);
}

void LSRInstance::print(raw_ostream &OS) const {
  OS << "LSR has identified the following interesting factors and types: ";
  bool First = true;

  for (SmallSetVector<int64_t, 4>::const_iterator
       I = Factors.begin(), E = Factors.end(); I != E; ++I) {
    if (!First) OS << ", ";
    First = false;
    OS << '*' << *I;
  }

  for (SmallSetVector<const Type *, 4>::const_iterator
       I = Types.begin(), E = Types.end(); I != E; ++I) {
    if (!First) OS << ", ";
    First = false;
    OS << '(' << **I << ')';
  }
  OS << '\n';

  OS << "LSR is examining the following uses, and candidate formulae:\n";
  for (SmallVectorImpl<LSRUse>::const_iterator I = Uses.begin(),
       E = Uses.end(); I != E; ++I) {
    const LSRUse &LU = *I;
    dbgs() << "  ";
    LU.print(OS);
    OS << '\n';
    for (SmallVectorImpl<Formula>::const_iterator J = LU.Formulae.begin(),
         JE = LU.Formulae.end(); J != JE; ++J) {
      OS << "    ";
      J->print(OS);
      OS << "\n";
    }
  }
}

void LSRInstance::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

class LoopStrengthReduce : public LoopPass {
  /// TLI - Keep a pointer of a TargetLowering to consult for determining
  /// transformation profitability.
  const TargetLowering *const TLI;

public:
  static char ID; // Pass ID, replacement for typeid
  explicit LoopStrengthReduce(const TargetLowering *tli = NULL);

private:
  bool runOnLoop(Loop *L, LPPassManager &LPM);
  void getAnalysisUsage(AnalysisUsage &AU) const;
};

}

char LoopStrengthReduce::ID = 0;
static RegisterPass<LoopStrengthReduce>
X("loop-reduce", "Loop Strength Reduction");

Pass *llvm::createLoopStrengthReducePass(const TargetLowering *TLI) {
  return new LoopStrengthReduce(TLI);
}

LoopStrengthReduce::LoopStrengthReduce(const TargetLowering *tli)
  : LoopPass(&ID), TLI(tli) {}

void LoopStrengthReduce::getAnalysisUsage(AnalysisUsage &AU) const {
  // We split critical edges, so we change the CFG.  However, we do update
  // many analyses if they are around.
  AU.addPreservedID(LoopSimplifyID);
  AU.addPreserved<LoopInfo>();
  AU.addPreserved("domfrontier");

  AU.addRequiredID(LoopSimplifyID);
  AU.addRequired<DominatorTree>();
  AU.addPreserved<DominatorTree>();
  AU.addRequired<ScalarEvolution>();
  AU.addPreserved<ScalarEvolution>();
  AU.addRequired<IVUsers>();
  AU.addPreserved<IVUsers>();
}

bool LoopStrengthReduce::runOnLoop(Loop *L, LPPassManager & /*LPM*/) {
  bool Changed = false;

  // Run the main LSR transformation.
  Changed |= LSRInstance(TLI, L, this).getChanged();

  // At this point, it is worth checking to see if any recurrence PHIs are also
  // dead, so that we can remove them as well.
  Changed |= DeleteDeadPHIs(L->getHeader());

  return Changed;
}
