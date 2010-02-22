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
// TODO: More sophistication in the way Formulae are generated and filtered.
//
// TODO: Handle multiple loops at a time.
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLowering.h"
#include <algorithm>
using namespace llvm;

namespace {

/// RegSortData - This class holds data which is used to order reuse candidates.
class RegSortData {
public:
  /// UsedByIndices - This represents the set of LSRUse indices which reference
  /// a particular register.
  SmallBitVector UsedByIndices;

  RegSortData() {}

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

void RegSortData::print(raw_ostream &OS) const {
  OS << "[NumUses=" << UsedByIndices.count() << ']';
}

void RegSortData::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

/// RegUseTracker - Map register candidates to information about how they are
/// used.
class RegUseTracker {
  typedef DenseMap<const SCEV *, RegSortData> RegUsesTy;

  RegUsesTy RegUses;
  SmallVector<const SCEV *, 16> RegSequence;

public:
  void CountRegister(const SCEV *Reg, size_t LUIdx);

  bool isRegUsedByUsesOtherThan(const SCEV *Reg, size_t LUIdx) const;

  const SmallBitVector &getUsedByIndices(const SCEV *Reg) const;

  void clear();

  typedef SmallVectorImpl<const SCEV *>::iterator iterator;
  typedef SmallVectorImpl<const SCEV *>::const_iterator const_iterator;
  iterator begin() { return RegSequence.begin(); }
  iterator end()   { return RegSequence.end(); }
  const_iterator begin() const { return RegSequence.begin(); }
  const_iterator end() const   { return RegSequence.end(); }
};

}

void
RegUseTracker::CountRegister(const SCEV *Reg, size_t LUIdx) {
  std::pair<RegUsesTy::iterator, bool> Pair =
    RegUses.insert(std::make_pair(Reg, RegSortData()));
  RegSortData &RSD = Pair.first->second;
  if (Pair.second)
    RegSequence.push_back(Reg);
  RSD.UsedByIndices.resize(std::max(RSD.UsedByIndices.size(), LUIdx + 1));
  RSD.UsedByIndices.set(LUIdx);
}

bool
RegUseTracker::isRegUsedByUsesOtherThan(const SCEV *Reg, size_t LUIdx) const {
  if (!RegUses.count(Reg)) return false;
  const SmallBitVector &UsedByIndices =
    RegUses.find(Reg)->second.UsedByIndices;
  int i = UsedByIndices.find_first();
  if (i == -1) return false;
  if ((size_t)i != LUIdx) return true;
  return UsedByIndices.find_next(i) != -1;
}

const SmallBitVector &RegUseTracker::getUsedByIndices(const SCEV *Reg) const {
  RegUsesTy::const_iterator I = RegUses.find(Reg);
  assert(I != RegUses.end() && "Unknown register!");
  return I->second.UsedByIndices;
}

void RegUseTracker::clear() {
  RegUses.clear();
  RegSequence.clear();
}

namespace {

/// Formula - This class holds information that describes a formula for
/// computing satisfying a use. It may include broken-out immediates and scaled
/// registers.
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

  void InitialMatch(const SCEV *S, Loop *L,
                    ScalarEvolution &SE, DominatorTree &DT);

  unsigned getNumRegs() const;
  const Type *getType() const;

  bool referencesReg(const SCEV *S) const;
  bool hasRegsUsedByUsesOtherThan(size_t LUIdx,
                                  const RegUseTracker &RegUses) const;

  void print(raw_ostream &OS) const;
  void dump() const;
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
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S))
    if (!AR->getStart()->isZero()) {
      DoInitialMatch(AR->getStart(), L, Good, Bad, SE, DT);
      DoInitialMatch(SE.getAddRecExpr(SE.getIntegerSCEV(0, AR->getType()),
                                      AR->getStepRecurrence(SE),
                                      AR->getLoop()),
                     L, Good, Bad, SE, DT);
      return;
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

/// getNumRegs - Return the total number of register operands used by this
/// formula. This does not include register uses implied by non-constant
/// addrec strides.
unsigned Formula::getNumRegs() const {
  return !!ScaledReg + BaseRegs.size();
}

/// getType - Return the type of this formula, if it has one, or null
/// otherwise. This type is meaningless except for the bit size.
const Type *Formula::getType() const {
  return !BaseRegs.empty() ? BaseRegs.front()->getType() :
         ScaledReg ? ScaledReg->getType() :
         AM.BaseGV ? AM.BaseGV->getType() :
         0;
}

/// referencesReg - Test if this formula references the given register.
bool Formula::referencesReg(const SCEV *S) const {
  return S == ScaledReg ||
         std::find(BaseRegs.begin(), BaseRegs.end(), S) != BaseRegs.end();
}

/// hasRegsUsedByUsesOtherThan - Test whether this formula uses registers
/// which are used by uses other than the use with the given index.
bool Formula::hasRegsUsedByUsesOtherThan(size_t LUIdx,
                                         const RegUseTracker &RegUses) const {
  if (ScaledReg)
    if (RegUses.isRegUsedByUsesOtherThan(ScaledReg, LUIdx))
      return true;
  for (SmallVectorImpl<const SCEV *>::const_iterator I = BaseRegs.begin(),
       E = BaseRegs.end(); I != E; ++I)
    if (RegUses.isRegUsedByUsesOtherThan(*I, LUIdx))
      return true;
  return false;
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
    OS << "reg(" << **I << ')';
  }
  if (AM.Scale != 0) {
    if (!First) OS << " + "; else First = false;
    OS << AM.Scale << "*reg(";
    if (ScaledReg)
      OS << *ScaledReg;
    else
      OS << "<unknown>";
    OS << ')';
  }
}

void Formula::dump() const {
  print(errs()); errs() << '\n';
}

/// isAddRecSExtable - Return true if the given addrec can be sign-extended
/// without changing its value.
static bool isAddRecSExtable(const SCEVAddRecExpr *AR, ScalarEvolution &SE) {
  const Type *WideTy =
    IntegerType::get(SE.getContext(),
                     SE.getTypeSizeInBits(AR->getType()) + 1);
  return isa<SCEVAddRecExpr>(SE.getSignExtendExpr(AR, WideTy));
}

/// isAddSExtable - Return true if the given add can be sign-extended
/// without changing its value.
static bool isAddSExtable(const SCEVAddExpr *A, ScalarEvolution &SE) {
  const Type *WideTy =
    IntegerType::get(SE.getContext(),
                     SE.getTypeSizeInBits(A->getType()) + 1);
  return isa<SCEVAddExpr>(SE.getSignExtendExpr(A, WideTy));
}

/// isMulSExtable - Return true if the given add can be sign-extended
/// without changing its value.
static bool isMulSExtable(const SCEVMulExpr *A, ScalarEvolution &SE) {
  const Type *WideTy =
    IntegerType::get(SE.getContext(),
                     SE.getTypeSizeInBits(A->getType()) + 1);
  return isa<SCEVMulExpr>(SE.getSignExtendExpr(A, WideTy));
}

/// getExactSDiv - Return an expression for LHS /s RHS, if it can be determined
/// and if the remainder is known to be zero,  or null otherwise. If
/// IgnoreSignificantBits is true, expressions like (X * Y) /s Y are simplified
/// to Y, ignoring that the multiplication may overflow, which is useful when
/// the result will be used in a context where the most significant bits are
/// ignored.
static const SCEV *getExactSDiv(const SCEV *LHS, const SCEV *RHS,
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

  // Distribute the sdiv over addrec operands, if the addrec doesn't overflow.
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(LHS)) {
    if (IgnoreSignificantBits || isAddRecSExtable(AR, SE)) {
      const SCEV *Start = getExactSDiv(AR->getStart(), RHS, SE,
                                       IgnoreSignificantBits);
      if (!Start) return 0;
      const SCEV *Step = getExactSDiv(AR->getStepRecurrence(SE), RHS, SE,
                                      IgnoreSignificantBits);
      if (!Step) return 0;
      return SE.getAddRecExpr(Start, Step, AR->getLoop());
    }
  }

  // Distribute the sdiv over add operands, if the add doesn't overflow.
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(LHS)) {
    if (IgnoreSignificantBits || isAddSExtable(Add, SE)) {
      SmallVector<const SCEV *, 8> Ops;
      for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
           I != E; ++I) {
        const SCEV *Op = getExactSDiv(*I, RHS, SE,
                                      IgnoreSignificantBits);
        if (!Op) return 0;
        Ops.push_back(Op);
      }
      return SE.getAddExpr(Ops);
    }
  }

  // Check for a multiply operand that we can pull RHS out of.
  if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(LHS))
    if (IgnoreSignificantBits || isMulSExtable(Mul, SE)) {
      SmallVector<const SCEV *, 4> Ops;
      bool Found = false;
      for (SCEVMulExpr::op_iterator I = Mul->op_begin(), E = Mul->op_end();
           I != E; ++I) {
        if (!Found)
          if (const SCEV *Q = getExactSDiv(*I, RHS, SE,
                                           IgnoreSignificantBits)) {
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

  // All pointers have the same requirements, so canonicalize them to an
  // arbitrary pointer type to minimize variation.
  if (const PointerType *PTy = dyn_cast<PointerType>(AccessTy))
    AccessTy = PointerType::get(IntegerType::get(PTy->getContext(), 1),
                                PTy->getAddressSpace());

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

/// Cost - This class is used to measure and compare candidate formulae.
class Cost {
  /// TODO: Some of these could be merged. Also, a lexical ordering
  /// isn't always optimal.
  unsigned NumRegs;
  unsigned AddRecCost;
  unsigned NumIVMuls;
  unsigned NumBaseAdds;
  unsigned ImmCost;
  unsigned SetupCost;

public:
  Cost()
    : NumRegs(0), AddRecCost(0), NumIVMuls(0), NumBaseAdds(0), ImmCost(0),
      SetupCost(0) {}

  unsigned getNumRegs() const { return NumRegs; }

  bool operator<(const Cost &Other) const;

  void Loose();

  void RateFormula(const Formula &F,
                   SmallPtrSet<const SCEV *, 16> &Regs,
                   const DenseSet<const SCEV *> &VisitedRegs,
                   const Loop *L,
                   const SmallVectorImpl<int64_t> &Offsets,
                   ScalarEvolution &SE, DominatorTree &DT);

  void print(raw_ostream &OS) const;
  void dump() const;

private:
  void RateRegister(const SCEV *Reg,
                    SmallPtrSet<const SCEV *, 16> &Regs,
                    const Loop *L,
                    ScalarEvolution &SE, DominatorTree &DT);
  void RatePrimaryRegister(const SCEV *Reg,
                           SmallPtrSet<const SCEV *, 16> &Regs,
                           const Loop *L,
                           ScalarEvolution &SE, DominatorTree &DT);
};

}

/// RateRegister - Tally up interesting quantities from the given register.
void Cost::RateRegister(const SCEV *Reg,
                        SmallPtrSet<const SCEV *, 16> &Regs,
                        const Loop *L,
                        ScalarEvolution &SE, DominatorTree &DT) {
  if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Reg)) {
    if (AR->getLoop() == L)
      AddRecCost += 1; /// TODO: This should be a function of the stride.

    // If this is an addrec for a loop that's already been visited by LSR,
    // don't second-guess its addrec phi nodes. LSR isn't currently smart
    // enough to reason about more than one loop at a time. Consider these
    // registers free and leave them alone.
    else if (L->contains(AR->getLoop()) ||
             (!AR->getLoop()->contains(L) &&
              DT.dominates(L->getHeader(), AR->getLoop()->getHeader()))) {
      for (BasicBlock::iterator I = AR->getLoop()->getHeader()->begin();
           PHINode *PN = dyn_cast<PHINode>(I); ++I)
        if (SE.isSCEVable(PN->getType()) &&
            (SE.getEffectiveSCEVType(PN->getType()) ==
             SE.getEffectiveSCEVType(AR->getType())) &&
            SE.getSCEV(PN) == AR)
          return;

      // If this isn't one of the addrecs that the loop already has, it
      // would require a costly new phi and add. TODO: This isn't
      // precisely modeled right now.
      ++NumBaseAdds;
      if (!Regs.count(AR->getStart()))
        RateRegister(AR->getStart(), Regs, L, SE, DT);
    }

    // Add the step value register, if it needs one.
    // TODO: The non-affine case isn't precisely modeled here.
    if (!AR->isAffine() || !isa<SCEVConstant>(AR->getOperand(1)))
      if (!Regs.count(AR->getStart()))
        RateRegister(AR->getOperand(1), Regs, L, SE, DT);
  }
  ++NumRegs;

  // Rough heuristic; favor registers which don't require extra setup
  // instructions in the preheader.
  if (!isa<SCEVUnknown>(Reg) &&
      !isa<SCEVConstant>(Reg) &&
      !(isa<SCEVAddRecExpr>(Reg) &&
        (isa<SCEVUnknown>(cast<SCEVAddRecExpr>(Reg)->getStart()) ||
         isa<SCEVConstant>(cast<SCEVAddRecExpr>(Reg)->getStart()))))
    ++SetupCost;
}

/// RatePrimaryRegister - Record this register in the set. If we haven't seen it
/// before, rate it.
void Cost::RatePrimaryRegister(const SCEV *Reg,
                               SmallPtrSet<const SCEV *, 16> &Regs,
                               const Loop *L,
                               ScalarEvolution &SE, DominatorTree &DT) {
  if (Regs.insert(Reg))
    RateRegister(Reg, Regs, L, SE, DT);
}

void Cost::RateFormula(const Formula &F,
                       SmallPtrSet<const SCEV *, 16> &Regs,
                       const DenseSet<const SCEV *> &VisitedRegs,
                       const Loop *L,
                       const SmallVectorImpl<int64_t> &Offsets,
                       ScalarEvolution &SE, DominatorTree &DT) {
  // Tally up the registers.
  if (const SCEV *ScaledReg = F.ScaledReg) {
    if (VisitedRegs.count(ScaledReg)) {
      Loose();
      return;
    }
    RatePrimaryRegister(ScaledReg, Regs, L, SE, DT);
  }
  for (SmallVectorImpl<const SCEV *>::const_iterator I = F.BaseRegs.begin(),
       E = F.BaseRegs.end(); I != E; ++I) {
    const SCEV *BaseReg = *I;
    if (VisitedRegs.count(BaseReg)) {
      Loose();
      return;
    }
    RatePrimaryRegister(BaseReg, Regs, L, SE, DT);

    NumIVMuls += isa<SCEVMulExpr>(BaseReg) &&
                 BaseReg->hasComputableLoopEvolution(L);
  }

  if (F.BaseRegs.size() > 1)
    NumBaseAdds += F.BaseRegs.size() - 1;

  // Tally up the non-zero immediates.
  for (SmallVectorImpl<int64_t>::const_iterator I = Offsets.begin(),
       E = Offsets.end(); I != E; ++I) {
    int64_t Offset = (uint64_t)*I + F.AM.BaseOffs;
    if (F.AM.BaseGV)
      ImmCost += 64; // Handle symbolic values conservatively.
                     // TODO: This should probably be the pointer size.
    else if (Offset != 0)
      ImmCost += APInt(64, Offset, true).getMinSignedBits();
  }
}

/// Loose - Set this cost to a loosing value.
void Cost::Loose() {
  NumRegs = ~0u;
  AddRecCost = ~0u;
  NumIVMuls = ~0u;
  NumBaseAdds = ~0u;
  ImmCost = ~0u;
  SetupCost = ~0u;
}

/// operator< - Choose the lower cost.
bool Cost::operator<(const Cost &Other) const {
  if (NumRegs != Other.NumRegs)
    return NumRegs < Other.NumRegs;
  if (AddRecCost != Other.AddRecCost)
    return AddRecCost < Other.AddRecCost;
  if (NumIVMuls != Other.NumIVMuls)
    return NumIVMuls < Other.NumIVMuls;
  if (NumBaseAdds != Other.NumBaseAdds)
    return NumBaseAdds < Other.NumBaseAdds;
  if (ImmCost != Other.ImmCost)
    return ImmCost < Other.ImmCost;
  if (SetupCost != Other.SetupCost)
    return SetupCost < Other.SetupCost;
  return false;
}

void Cost::print(raw_ostream &OS) const {
  OS << NumRegs << " reg" << (NumRegs == 1 ? "" : "s");
  if (AddRecCost != 0)
    OS << ", with addrec cost " << AddRecCost;
  if (NumIVMuls != 0)
    OS << ", plus " << NumIVMuls << " IV mul" << (NumIVMuls == 1 ? "" : "s");
  if (NumBaseAdds != 0)
    OS << ", plus " << NumBaseAdds << " base add"
       << (NumBaseAdds == 1 ? "" : "s");
  if (ImmCost != 0)
    OS << ", plus " << ImmCost << " imm cost";
  if (SetupCost != 0)
    OS << ", plus " << SetupCost << " setup cost";
}

void Cost::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

/// LSRFixup - An operand value in an instruction which is to be replaced
/// with some equivalent, possibly strength-reduced, replacement.
struct LSRFixup {
  /// UserInst - The instruction which will be updated.
  Instruction *UserInst;

  /// OperandValToReplace - The operand of the instruction which will
  /// be replaced. The operand may be used more than once; every instance
  /// will be replaced.
  Value *OperandValToReplace;

  /// PostIncLoop - If this user is to use the post-incremented value of an
  /// induction variable, this variable is non-null and holds the loop
  /// associated with the induction variable.
  const Loop *PostIncLoop;

  /// LUIdx - The index of the LSRUse describing the expression which
  /// this fixup needs, minus an offset (below).
  size_t LUIdx;

  /// Offset - A constant offset to be added to the LSRUse expression.
  /// This allows multiple fixups to share the same LSRUse with different
  /// offsets, for example in an unrolled loop.
  int64_t Offset;

  LSRFixup();

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

LSRFixup::LSRFixup()
  : UserInst(0), OperandValToReplace(0), PostIncLoop(0),
    LUIdx(~size_t(0)), Offset(0) {}

void LSRFixup::print(raw_ostream &OS) const {
  OS << "UserInst=";
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

  if (LUIdx != ~size_t(0))
    OS << ", LUIdx=" << LUIdx;

  if (Offset != 0)
    OS << ", Offset=" << Offset;
}

void LSRFixup::dump() const {
  print(errs()); errs() << '\n';
}

namespace {

/// UniquifierDenseMapInfo - A DenseMapInfo implementation for holding
/// DenseMaps and DenseSets of sorted SmallVectors of const SCEV*.
struct UniquifierDenseMapInfo {
  static SmallVector<const SCEV *, 2> getEmptyKey() {
    SmallVector<const SCEV *, 2> V;
    V.push_back(reinterpret_cast<const SCEV *>(-1));
    return V;
  }

  static SmallVector<const SCEV *, 2> getTombstoneKey() {
    SmallVector<const SCEV *, 2> V;
    V.push_back(reinterpret_cast<const SCEV *>(-2));
    return V;
  }

  static unsigned getHashValue(const SmallVector<const SCEV *, 2> &V) {
    unsigned Result = 0;
    for (SmallVectorImpl<const SCEV *>::const_iterator I = V.begin(),
         E = V.end(); I != E; ++I)
      Result ^= DenseMapInfo<const SCEV *>::getHashValue(*I);
    return Result;
  }

  static bool isEqual(const SmallVector<const SCEV *, 2> &LHS,
                      const SmallVector<const SCEV *, 2> &RHS) {
    return LHS == RHS;
  }
};

/// LSRUse - This class holds the state that LSR keeps for each use in
/// IVUsers, as well as uses invented by LSR itself. It includes information
/// about what kinds of things can be folded into the user, information about
/// the user itself, and information about how the use may be satisfied.
/// TODO: Represent multiple users of the same expression in common?
class LSRUse {
  DenseSet<SmallVector<const SCEV *, 2>, UniquifierDenseMapInfo> Uniquifier;

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

  SmallVector<int64_t, 8> Offsets;
  int64_t MinOffset;
  int64_t MaxOffset;

  /// AllFixupsOutsideLoop - This records whether all of the fixups using this
  /// LSRUse are outside of the loop, in which case some special-case heuristics
  /// may be used.
  bool AllFixupsOutsideLoop;

  /// Formulae - A list of ways to build a value that can satisfy this user.
  /// After the list is populated, one of these is selected heuristically and
  /// used to formulate a replacement for OperandValToReplace in UserInst.
  SmallVector<Formula, 12> Formulae;

  /// Regs - The set of register candidates used by all formulae in this LSRUse.
  SmallPtrSet<const SCEV *, 4> Regs;

  LSRUse(KindType K, const Type *T) : Kind(K), AccessTy(T),
                                      MinOffset(INT64_MAX),
                                      MaxOffset(INT64_MIN),
                                      AllFixupsOutsideLoop(true) {}

  bool InsertFormula(size_t LUIdx, const Formula &F);

  void check() const;

  void print(raw_ostream &OS) const;
  void dump() const;
};

/// InsertFormula - If the given formula has not yet been inserted, add it to
/// the list, and return true. Return false otherwise.
bool LSRUse::InsertFormula(size_t LUIdx, const Formula &F) {
  SmallVector<const SCEV *, 2> Key = F.BaseRegs;
  if (F.ScaledReg) Key.push_back(F.ScaledReg);
  // Unstable sort by host order ok, because this is only used for uniquifying.
  std::sort(Key.begin(), Key.end());

  if (!Uniquifier.insert(Key).second)
    return false;

  // Using a register to hold the value of 0 is not profitable.
  assert((!F.ScaledReg || !F.ScaledReg->isZero()) &&
         "Zero allocated in a scaled register!");
#ifndef NDEBUG
  for (SmallVectorImpl<const SCEV *>::const_iterator I =
       F.BaseRegs.begin(), E = F.BaseRegs.end(); I != E; ++I)
    assert(!(*I)->isZero() && "Zero allocated in a base register!");
#endif

  // Add the formula to the list.
  Formulae.push_back(F);

  // Record registers now being used by this use.
  if (F.ScaledReg) Regs.insert(F.ScaledReg);
  Regs.insert(F.BaseRegs.begin(), F.BaseRegs.end());

  return true;
}

void LSRUse::print(raw_ostream &OS) const {
  OS << "LSR Use: Kind=";
  switch (Kind) {
  case Basic:    OS << "Basic"; break;
  case Special:  OS << "Special"; break;
  case ICmpZero: OS << "ICmpZero"; break;
  case Address:
    OS << "Address of ";
    if (AccessTy->isPointerTy())
      OS << "pointer"; // the full pointer type could be really verbose
    else
      OS << *AccessTy;
  }

  OS << ", Offsets={";
  for (SmallVectorImpl<int64_t>::const_iterator I = Offsets.begin(),
       E = Offsets.end(); I != E; ++I) {
    OS << *I;
    if (next(I) != E)
      OS << ',';
  }
  OS << '}';

  if (AllFixupsOutsideLoop)
    OS << ", all-fixups-outside-loop";
}

void LSRUse::dump() const {
  print(errs()); errs() << '\n';
}

/// isLegalUse - Test whether the use described by AM is "legal", meaning it can
/// be completely folded into the user instruction at isel time. This includes
/// address-mode folding and special icmp tricks.
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
    // There's not even a target hook for querying whether it would be legal to
    // fold a GV into an ICmp.
    if (AM.BaseGV)
      return false;

    // ICmp only has two operands; don't allow more than two non-trivial parts.
    if (AM.Scale != 0 && AM.HasBaseReg && AM.BaseOffs != 0)
      return false;

    // ICmp only supports no scale or a -1 scale, as we can "fold" a -1 scale by
    // putting the scaled register in the other operand of the icmp.
    if (AM.Scale != 0 && AM.Scale != -1)
      return false;

    // If we have low-level target information, ask the target if it can fold an
    // integer immediate on an icmp.
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

static bool isLegalUse(TargetLowering::AddrMode AM,
                       int64_t MinOffset, int64_t MaxOffset,
                       LSRUse::KindType Kind, const Type *AccessTy,
                       const TargetLowering *TLI) {
  // Check for overflow.
  if (((int64_t)((uint64_t)AM.BaseOffs + MinOffset) > AM.BaseOffs) !=
      (MinOffset > 0))
    return false;
  AM.BaseOffs = (uint64_t)AM.BaseOffs + MinOffset;
  if (isLegalUse(AM, Kind, AccessTy, TLI)) {
    AM.BaseOffs = (uint64_t)AM.BaseOffs - MinOffset;
    // Check for overflow.
    if (((int64_t)((uint64_t)AM.BaseOffs + MaxOffset) > AM.BaseOffs) !=
        (MaxOffset > 0))
      return false;
    AM.BaseOffs = (uint64_t)AM.BaseOffs + MaxOffset;
    return isLegalUse(AM, Kind, AccessTy, TLI);
  }
  return false;
}

static bool isAlwaysFoldable(int64_t BaseOffs,
                             GlobalValue *BaseGV,
                             bool HasBaseReg,
                             LSRUse::KindType Kind, const Type *AccessTy,
                             const TargetLowering *TLI,
                             ScalarEvolution &SE) {
  // Fast-path: zero is always foldable.
  if (BaseOffs == 0 && !BaseGV) return true;

  // Conservatively, create an address with an immediate and a
  // base and a scale.
  TargetLowering::AddrMode AM;
  AM.BaseOffs = BaseOffs;
  AM.BaseGV = BaseGV;
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Kind == LSRUse::ICmpZero ? -1 : 1;

  return isLegalUse(AM, Kind, AccessTy, TLI);
}

static bool isAlwaysFoldable(const SCEV *S,
                             int64_t MinOffset, int64_t MaxOffset,
                             bool HasBaseReg,
                             LSRUse::KindType Kind, const Type *AccessTy,
                             const TargetLowering *TLI,
                             ScalarEvolution &SE) {
  // Fast-path: zero is always foldable.
  if (S->isZero()) return true;

  // Conservatively, create an address with an immediate and a
  // base and a scale.
  int64_t BaseOffs = ExtractImmediate(S, SE);
  GlobalValue *BaseGV = ExtractSymbol(S, SE);

  // If there's anything else involved, it's not foldable.
  if (!S->isZero()) return false;

  // Fast-path: zero is always foldable.
  if (BaseOffs == 0 && !BaseGV) return true;

  // Conservatively, create an address with an immediate and a
  // base and a scale.
  TargetLowering::AddrMode AM;
  AM.BaseOffs = BaseOffs;
  AM.BaseGV = BaseGV;
  AM.HasBaseReg = HasBaseReg;
  AM.Scale = Kind == LSRUse::ICmpZero ? -1 : 1;

  return isLegalUse(AM, MinOffset, MaxOffset, Kind, AccessTy, TLI);
}

/// FormulaSorter - This class implements an ordering for formulae which sorts
/// the by their standalone cost.
class FormulaSorter {
  /// These two sets are kept empty, so that we compute standalone costs.
  DenseSet<const SCEV *> VisitedRegs;
  SmallPtrSet<const SCEV *, 16> Regs;
  Loop *L;
  LSRUse *LU;
  ScalarEvolution &SE;
  DominatorTree &DT;

public:
  FormulaSorter(Loop *l, LSRUse &lu, ScalarEvolution &se, DominatorTree &dt)
    : L(l), LU(&lu), SE(se), DT(dt) {}

  bool operator()(const Formula &A, const Formula &B) {
    Cost CostA;
    CostA.RateFormula(A, Regs, VisitedRegs, L, LU->Offsets, SE, DT);
    Regs.clear();
    Cost CostB;
    CostB.RateFormula(B, Regs, VisitedRegs, L, LU->Offsets, SE, DT);
    Regs.clear();
    return CostA < CostB;
  }
};

/// LSRInstance - This class holds state for the main loop strength reduction
/// logic.
class LSRInstance {
  IVUsers &IU;
  ScalarEvolution &SE;
  DominatorTree &DT;
  const TargetLowering *const TLI;
  Loop *const L;
  bool Changed;

  /// IVIncInsertPos - This is the insert position that the current loop's
  /// induction variable increment should be placed. In simple loops, this is
  /// the latch block's terminator. But in more complicated cases, this is a
  /// position which will dominate all the in-loop post-increment users.
  Instruction *IVIncInsertPos;

  /// Factors - Interesting factors between use strides.
  SmallSetVector<int64_t, 8> Factors;

  /// Types - Interesting use types, to facilitate truncation reuse.
  SmallSetVector<const Type *, 4> Types;

  /// Fixups - The list of operands which are to be replaced.
  SmallVector<LSRFixup, 16> Fixups;

  /// Uses - The list of interesting uses.
  SmallVector<LSRUse, 16> Uses;

  /// RegUses - Track which uses use which register candidates.
  RegUseTracker RegUses;

  void OptimizeShadowIV();
  bool FindIVUserForCond(ICmpInst *Cond, IVStrideUse *&CondUse);
  ICmpInst *OptimizeMax(ICmpInst *Cond, IVStrideUse* &CondUse);
  bool OptimizeLoopTermCond();

  void CollectInterestingTypesAndFactors();
  void CollectFixupsAndInitialFormulae();

  LSRFixup &getNewFixup() {
    Fixups.push_back(LSRFixup());
    return Fixups.back();
  }

  // Support for sharing of LSRUses between LSRFixups.
  typedef DenseMap<const SCEV *, size_t> UseMapTy;
  UseMapTy UseMap;

  bool reconcileNewOffset(LSRUse &LU, int64_t NewOffset,
                          LSRUse::KindType Kind, const Type *AccessTy);

  std::pair<size_t, int64_t> getUse(const SCEV *&Expr,
                                    LSRUse::KindType Kind,
                                    const Type *AccessTy);

public:
  void InsertInitialFormula(const SCEV *S, Loop *L, LSRUse &LU, size_t LUIdx);
  void InsertSupplementalFormula(const SCEV *S, LSRUse &LU, size_t LUIdx);
  void CountRegisters(const Formula &F, size_t LUIdx);
  bool InsertFormula(LSRUse &LU, unsigned LUIdx, const Formula &F);

  void CollectLoopInvariantFixupsAndFormulae();

  void GenerateReassociations(LSRUse &LU, unsigned LUIdx, Formula Base,
                              unsigned Depth = 0);
  void GenerateCombinations(LSRUse &LU, unsigned LUIdx, Formula Base);
  void GenerateSymbolicOffsets(LSRUse &LU, unsigned LUIdx, Formula Base);
  void GenerateConstantOffsets(LSRUse &LU, unsigned LUIdx, Formula Base);
  void GenerateICmpZeroScales(LSRUse &LU, unsigned LUIdx, Formula Base);
  void GenerateScales(LSRUse &LU, unsigned LUIdx, Formula Base);
  void GenerateTruncates(LSRUse &LU, unsigned LUIdx, Formula Base);
  void GenerateCrossUseConstantOffsets();
  void GenerateAllReuseFormulae();

  void FilterOutUndesirableDedicatedRegisters();
  void NarrowSearchSpaceUsingHeuristics();

  void SolveRecurse(SmallVectorImpl<const Formula *> &Solution,
                    Cost &SolutionCost,
                    SmallVectorImpl<const Formula *> &Workspace,
                    const Cost &CurCost,
                    const SmallPtrSet<const SCEV *, 16> &CurRegs,
                    DenseSet<const SCEV *> &VisitedRegs) const;
  void Solve(SmallVectorImpl<const Formula *> &Solution) const;

  Value *Expand(const LSRFixup &LF,
                const Formula &F,
                BasicBlock::iterator IP, Loop *L, Instruction *IVIncInsertPos,
                SCEVExpander &Rewriter,
                SmallVectorImpl<WeakVH> &DeadInsts,
                ScalarEvolution &SE, DominatorTree &DT) const;
  void RewriteForPHI(PHINode *PN, const LSRFixup &LF,
                     const Formula &F,
                     Loop *L, Instruction *IVIncInsertPos,
                     SCEVExpander &Rewriter,
                     SmallVectorImpl<WeakVH> &DeadInsts,
                     ScalarEvolution &SE, DominatorTree &DT,
                     Pass *P) const;
  void Rewrite(const LSRFixup &LF,
               const Formula &F,
               Loop *L, Instruction *IVIncInsertPos,
               SCEVExpander &Rewriter,
               SmallVectorImpl<WeakVH> &DeadInsts,
               ScalarEvolution &SE, DominatorTree &DT,
               Pass *P) const;
  void ImplementSolution(const SmallVectorImpl<const Formula *> &Solution,
                         Pass *P);

  LSRInstance(const TargetLowering *tli, Loop *l, Pass *P);

  bool getChanged() const { return Changed; }

  void print_factors_and_types(raw_ostream &OS) const;
  void print_fixups(raw_ostream &OS) const;
  void print_uses(raw_ostream &OS) const;
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

  for (IVUsers::const_iterator UI = IU.begin(), E = IU.end();
       UI != E; /* empty */) {
    IVUsers::const_iterator CandidateUI = UI;
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

/// FindIVUserForCond - If Cond has an operand that is an expression of an IV,
/// set the IV user and stride information and return true, otherwise return
/// false.
bool LSRInstance::FindIVUserForCond(ICmpInst *Cond,
                                    IVStrideUse *&CondUse) {
  for (IVUsers::iterator UI = IU.begin(), E = IU.end(); UI != E; ++UI)
    if (UI->getUser() == Cond) {
      // NOTE: we could handle setcc instructions with multiple uses here, but
      // InstCombine does it as well for simple uses, it's not clear that it
      // occurs enough in real life to handle.
      CondUse = UI;
      return true;
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

/// OptimizeLoopTermCond - Change loop terminating condition to use the
/// postinc iv when possible.
bool
LSRInstance::OptimizeLoopTermCond() {
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
    ICmpInst *Cond = cast<ICmpInst>(TermBr->getCondition());
    if (!FindIVUserForCond(Cond, CondUse))
      continue;

    // If the trip count is computed in terms of a max (due to ScalarEvolution
    // being unable to find a sufficient guard, for example), change the loop
    // comparison to use SLT or ULT instead of NE.
    // One consequence of doing this now is that it disrupts the count-down
    // optimization. That's not always a bad thing though, because in such
    // cases it may still be worthwhile to avoid a max.
    Cond = OptimizeMax(Cond, CondUse);

    // If this exiting block dominates the latch block, it may also use
    // the post-inc value if it won't be shared with other uses.
    // Check for dominance.
    if (!DT.dominates(ExitingBlock, LatchBlock))
      continue;

    // Conservatively avoid trying to use the post-inc value in non-latch
    // exits if there may be pre-inc users in intervening blocks.
    if (LatchBlock != ExitingBlock)
      for (IVUsers::const_iterator UI = IU.begin(), E = IU.end(); UI != E; ++UI)
        // Test if the use is reachable from the exiting block. This dominator
        // query is a conservative approximation of reachability.
        if (&*UI != CondUse &&
            !DT.properlyDominates(UI->getUser()->getParent(), ExitingBlock)) {
          // Conservatively assume there may be reuse if the quotient of their
          // strides could be a legal scale.
          const SCEV *A = CondUse->getStride();
          const SCEV *B = UI->getStride();
          if (SE.getTypeSizeInBits(A->getType()) !=
              SE.getTypeSizeInBits(B->getType())) {
            if (SE.getTypeSizeInBits(A->getType()) >
                SE.getTypeSizeInBits(B->getType()))
              B = SE.getSignExtendExpr(B, A->getType());
            else
              A = SE.getSignExtendExpr(A, B->getType());
          }
          if (const SCEVConstant *D =
                dyn_cast_or_null<SCEVConstant>(getExactSDiv(B, A, SE))) {
            // Stride of one or negative one can have reuse with non-addresses.
            if (D->getValue()->isOne() ||
                D->getValue()->isAllOnesValue())
              goto decline_post_inc;
            // Avoid weird situations.
            if (D->getValue()->getValue().getMinSignedBits() >= 64 ||
                D->getValue()->getValue().isMinSignedValue())
              goto decline_post_inc;
            // Without TLI, assume that any stride might be valid, and so any
            // use might be shared.
            if (!TLI)
              goto decline_post_inc;
            // Check for possible scaled-address reuse.
            const Type *AccessTy = getAccessType(UI->getUser());
            TargetLowering::AddrMode AM;
            AM.Scale = D->getValue()->getSExtValue();
            if (TLI->isLegalAddressingMode(AM, AccessTy))
              goto decline_post_inc;
            AM.Scale = -AM.Scale;
            if (TLI->isLegalAddressingMode(AM, AccessTy))
              goto decline_post_inc;
          }
        }

    DEBUG(dbgs() << "  Change loop exiting icmp to use postinc iv: "
                 << *Cond << '\n');

    // It's possible for the setcc instruction to be anywhere in the loop, and
    // possible for it to have multiple users.  If it is not immediately before
    // the exiting block branch, move it.
    if (&*++BasicBlock::iterator(Cond) != TermBr) {
      if (Cond->hasOneUse()) {
        Cond->moveBefore(TermBr);
      } else {
        // Clone the terminating condition and insert into the loopend.
        ICmpInst *OldCond = Cond;
        Cond = cast<ICmpInst>(Cond->clone());
        Cond->setName(L->getHeader()->getName() + ".termcond");
        ExitingBlock->getInstList().insert(TermBr, Cond);

        // Clone the IVUse, as the old use still exists!
        CondUse = &IU.AddUser(CondUse->getStride(), CondUse->getOffset(),
                              Cond, CondUse->getOperandValToReplace());
        TermBr->replaceUsesOfWith(OldCond, Cond);
      }
    }

    // If we get to here, we know that we can transform the setcc instruction to
    // use the post-incremented version of the IV, allowing us to coalesce the
    // live ranges for the IV correctly.
    CondUse->setOffset(SE.getMinusSCEV(CondUse->getOffset(),
                                       CondUse->getStride()));
    CondUse->setIsUseOfPostIncrementedValue(true);
    Changed = true;

    PostIncs.insert(Cond);
  decline_post_inc:;
  }

  // Determine an insertion point for the loop induction variable increment. It
  // must dominate all the post-inc comparisons we just set up, and it must
  // dominate the loop latch edge.
  IVIncInsertPos = L->getLoopLatch()->getTerminator();
  for (SmallPtrSet<Instruction *, 4>::const_iterator I = PostIncs.begin(),
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

bool
LSRInstance::reconcileNewOffset(LSRUse &LU, int64_t NewOffset,
                                LSRUse::KindType Kind, const Type *AccessTy) {
  int64_t NewMinOffset = LU.MinOffset;
  int64_t NewMaxOffset = LU.MaxOffset;
  const Type *NewAccessTy = AccessTy;

  // Check for a mismatched kind. It's tempting to collapse mismatched kinds to
  // something conservative, however this can pessimize in the case that one of
  // the uses will have all its uses outside the loop, for example.
  if (LU.Kind != Kind)
    return false;
  // Conservatively assume HasBaseReg is true for now.
  if (NewOffset < LU.MinOffset) {
    if (!isAlwaysFoldable(LU.MaxOffset - NewOffset, 0, /*HasBaseReg=*/true,
                          Kind, AccessTy, TLI, SE))
      return false;
    NewMinOffset = NewOffset;
  } else if (NewOffset > LU.MaxOffset) {
    if (!isAlwaysFoldable(NewOffset - LU.MinOffset, 0, /*HasBaseReg=*/true,
                          Kind, AccessTy, TLI, SE))
      return false;
    NewMaxOffset = NewOffset;
  }
  // Check for a mismatched access type, and fall back conservatively as needed.
  if (Kind == LSRUse::Address && AccessTy != LU.AccessTy)
    NewAccessTy = Type::getVoidTy(AccessTy->getContext());

  // Update the use.
  LU.MinOffset = NewMinOffset;
  LU.MaxOffset = NewMaxOffset;
  LU.AccessTy = NewAccessTy;
  if (NewOffset != LU.Offsets.back())
    LU.Offsets.push_back(NewOffset);
  return true;
}

/// getUse - Return an LSRUse index and an offset value for a fixup which
/// needs the given expression, with the given kind and optional access type.
/// Either reuse an exisitng use or create a new one, as needed.
std::pair<size_t, int64_t>
LSRInstance::getUse(const SCEV *&Expr,
                    LSRUse::KindType Kind, const Type *AccessTy) {
  const SCEV *Copy = Expr;
  int64_t Offset = ExtractImmediate(Expr, SE);

  // Basic uses can't accept any offset, for example.
  if (!isAlwaysFoldable(Offset, 0, /*HasBaseReg=*/true,
                        Kind, AccessTy, TLI, SE)) {
    Expr = Copy;
    Offset = 0;
  }

  std::pair<UseMapTy::iterator, bool> P =
    UseMap.insert(std::make_pair(Expr, 0));
  if (!P.second) {
    // A use already existed with this base.
    size_t LUIdx = P.first->second;
    LSRUse &LU = Uses[LUIdx];
    if (reconcileNewOffset(LU, Offset, Kind, AccessTy))
      // Reuse this use.
      return std::make_pair(LUIdx, Offset);
  }

  // Create a new use.
  size_t LUIdx = Uses.size();
  P.first->second = LUIdx;
  Uses.push_back(LSRUse(Kind, AccessTy));
  LSRUse &LU = Uses[LUIdx];

  // We don't need to track redundant offsets, but we don't need to go out
  // of our way here to avoid them.
  if (LU.Offsets.empty() || Offset != LU.Offsets.back())
    LU.Offsets.push_back(Offset);

  LU.MinOffset = Offset;
  LU.MaxOffset = Offset;
  return std::make_pair(LUIdx, Offset);
}

void LSRInstance::CollectInterestingTypesAndFactors() {
  SmallSetVector<const SCEV *, 4> Strides;

  // Collect interesting types and strides.
  for (IVUsers::const_iterator UI = IU.begin(), E = IU.end(); UI != E; ++UI) {
    const SCEV *Stride = UI->getStride();

    // Collect interesting types.
    Types.insert(SE.getEffectiveSCEVType(Stride->getType()));

    // Add the stride for this loop.
    Strides.insert(Stride);

    // Add strides for other mentioned loops.
    for (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(UI->getOffset());
         AR; AR = dyn_cast<SCEVAddRecExpr>(AR->getStart()))
      Strides.insert(AR->getStepRecurrence(SE));
  }

  // Compute interesting factors from the set of interesting strides.
  for (SmallSetVector<const SCEV *, 4>::const_iterator
       I = Strides.begin(), E = Strides.end(); I != E; ++I)
    for (SmallSetVector<const SCEV *, 4>::const_iterator NewStrideIter =
         next(I); NewStrideIter != E; ++NewStrideIter) {
      const SCEV *OldStride = *I;
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
            dyn_cast_or_null<SCEVConstant>(getExactSDiv(NewStride, OldStride,
                                                        SE, true))) {
        if (Factor->getValue()->getValue().getMinSignedBits() <= 64)
          Factors.insert(Factor->getValue()->getValue().getSExtValue());
      } else if (const SCEVConstant *Factor =
                   dyn_cast_or_null<SCEVConstant>(getExactSDiv(OldStride, NewStride,
                                                               SE, true))) {
        if (Factor->getValue()->getValue().getMinSignedBits() <= 64)
          Factors.insert(Factor->getValue()->getValue().getSExtValue());
      }
    }

  // If all uses use the same type, don't bother looking for truncation-based
  // reuse.
  if (Types.size() == 1)
    Types.clear();

  DEBUG(print_factors_and_types(dbgs()));
}

void LSRInstance::CollectFixupsAndInitialFormulae() {
  for (IVUsers::const_iterator UI = IU.begin(), E = IU.end(); UI != E; ++UI) {
    // Record the uses.
    LSRFixup &LF = getNewFixup();
    LF.UserInst = UI->getUser();
    LF.OperandValToReplace = UI->getOperandValToReplace();
    if (UI->isUseOfPostIncrementedValue())
      LF.PostIncLoop = L;

    LSRUse::KindType Kind = LSRUse::Basic;
    const Type *AccessTy = 0;
    if (isAddressUse(LF.UserInst, LF.OperandValToReplace)) {
      Kind = LSRUse::Address;
      AccessTy = getAccessType(LF.UserInst);
    }

    const SCEV *S = IU.getCanonicalExpr(*UI);

    // Equality (== and !=) ICmps are special. We can rewrite (i == N) as
    // (N - i == 0), and this allows (N - i) to be the expression that we work
    // with rather than just N or i, so we can consider the register
    // requirements for both N and i at the same time. Limiting this code to
    // equality icmps is not a problem because all interesting loops use
    // equality icmps, thanks to IndVarSimplify.
    if (ICmpInst *CI = dyn_cast<ICmpInst>(LF.UserInst))
      if (CI->isEquality()) {
        // Swap the operands if needed to put the OperandValToReplace on the
        // left, for consistency.
        Value *NV = CI->getOperand(1);
        if (NV == LF.OperandValToReplace) {
          CI->setOperand(1, CI->getOperand(0));
          CI->setOperand(0, NV);
        }

        // x == y  -->  x - y == 0
        const SCEV *N = SE.getSCEV(NV);
        if (N->isLoopInvariant(L)) {
          Kind = LSRUse::ICmpZero;
          S = SE.getMinusSCEV(N, S);
        }

        // -1 and the negations of all interesting strides (except the negation
        // of -1) are now also interesting.
        for (size_t i = 0, e = Factors.size(); i != e; ++i)
          if (Factors[i] != -1)
            Factors.insert(-(uint64_t)Factors[i]);
        Factors.insert(-1);
      }

    // Set up the initial formula for this use.
    std::pair<size_t, int64_t> P = getUse(S, Kind, AccessTy);
    LF.LUIdx = P.first;
    LF.Offset = P.second;
    LSRUse &LU = Uses[LF.LUIdx];
    LU.AllFixupsOutsideLoop &= !L->contains(LF.UserInst);

    // If this is the first use of this LSRUse, give it a formula.
    if (LU.Formulae.empty()) {
      InsertInitialFormula(S, L, LU, LF.LUIdx);
      CountRegisters(LU.Formulae.back(), LF.LUIdx);
    }
  }

  DEBUG(print_fixups(dbgs()));
}

void
LSRInstance::InsertInitialFormula(const SCEV *S, Loop *L,
                                  LSRUse &LU, size_t LUIdx) {
  Formula F;
  F.InitialMatch(S, L, SE, DT);
  bool Inserted = InsertFormula(LU, LUIdx, F);
  assert(Inserted && "Initial formula already exists!"); (void)Inserted;
}

void
LSRInstance::InsertSupplementalFormula(const SCEV *S,
                                       LSRUse &LU, size_t LUIdx) {
  Formula F;
  F.BaseRegs.push_back(S);
  F.AM.HasBaseReg = true;
  bool Inserted = InsertFormula(LU, LUIdx, F);
  assert(Inserted && "Supplemental formula already exists!"); (void)Inserted;
}

/// CountRegisters - Note which registers are used by the given formula,
/// updating RegUses.
void LSRInstance::CountRegisters(const Formula &F, size_t LUIdx) {
  if (F.ScaledReg)
    RegUses.CountRegister(F.ScaledReg, LUIdx);
  for (SmallVectorImpl<const SCEV *>::const_iterator I = F.BaseRegs.begin(),
       E = F.BaseRegs.end(); I != E; ++I)
    RegUses.CountRegister(*I, LUIdx);
}

/// InsertFormula - If the given formula has not yet been inserted, add it to
/// the list, and return true. Return false otherwise.
bool LSRInstance::InsertFormula(LSRUse &LU, unsigned LUIdx, const Formula &F) {
  if (!LU.InsertFormula(LUIdx, F))
    return false;

  CountRegisters(F, LUIdx);
  return true;
}

/// CollectLoopInvariantFixupsAndFormulae - Check for other uses of
/// loop-invariant values which we're tracking. These other uses will pin these
/// values in registers, making them less profitable for elimination.
/// TODO: This currently misses non-constant addrec step registers.
/// TODO: Should this give more weight to users inside the loop?
void
LSRInstance::CollectLoopInvariantFixupsAndFormulae() {
  SmallVector<const SCEV *, 8> Worklist(RegUses.begin(), RegUses.end());
  SmallPtrSet<const SCEV *, 8> Inserted;

  while (!Worklist.empty()) {
    const SCEV *S = Worklist.pop_back_val();

    if (const SCEVNAryExpr *N = dyn_cast<SCEVNAryExpr>(S))
      Worklist.insert(Worklist.end(), N->op_begin(), N->op_end());
    else if (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(S))
      Worklist.push_back(C->getOperand());
    else if (const SCEVUDivExpr *D = dyn_cast<SCEVUDivExpr>(S)) {
      Worklist.push_back(D->getLHS());
      Worklist.push_back(D->getRHS());
    } else if (const SCEVUnknown *U = dyn_cast<SCEVUnknown>(S)) {
      if (!Inserted.insert(U)) continue;
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

        LSRFixup &LF = getNewFixup();
        LF.UserInst = const_cast<Instruction *>(UserInst);
        LF.OperandValToReplace = UI.getUse();
        std::pair<size_t, int64_t> P = getUse(S, LSRUse::Basic, 0);
        LF.LUIdx = P.first;
        LF.Offset = P.second;
        LSRUse &LU = Uses[LF.LUIdx];
        LU.AllFixupsOutsideLoop &= L->contains(LF.UserInst);
        InsertSupplementalFormula(U, LU, LF.LUIdx);
        CountRegisters(LU.Formulae.back(), Uses.size() - 1);
        break;
      }
    }
  }
}

/// CollectSubexprs - Split S into subexpressions which can be pulled out into
/// separate registers. If C is non-null, multiply each subexpression by C.
static void CollectSubexprs(const SCEV *S, const SCEVConstant *C,
                            SmallVectorImpl<const SCEV *> &Ops,
                            ScalarEvolution &SE) {
  if (const SCEVAddExpr *Add = dyn_cast<SCEVAddExpr>(S)) {
    // Break out add operands.
    for (SCEVAddExpr::op_iterator I = Add->op_begin(), E = Add->op_end();
         I != E; ++I)
      CollectSubexprs(*I, C, Ops, SE);
    return;
  } else if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // Split a non-zero base out of an addrec.
    if (!AR->getStart()->isZero()) {
      CollectSubexprs(SE.getAddRecExpr(SE.getIntegerSCEV(0, AR->getType()),
                                       AR->getStepRecurrence(SE),
                                       AR->getLoop()), C, Ops, SE);
      CollectSubexprs(AR->getStart(), C, Ops, SE);
      return;
    }
  } else if (const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(S)) {
    // Break (C * (a + b + c)) into C*a + C*b + C*c.
    if (Mul->getNumOperands() == 2)
      if (const SCEVConstant *Op0 =
            dyn_cast<SCEVConstant>(Mul->getOperand(0))) {
        CollectSubexprs(Mul->getOperand(1),
                        C ? cast<SCEVConstant>(SE.getMulExpr(C, Op0)) : Op0,
                        Ops, SE);
        return;
      }
  }

  // Otherwise use the value itself.
  Ops.push_back(C ? SE.getMulExpr(C, S) : S);
}

/// GenerateReassociations - Split out subexpressions from adds and the bases of
/// addrecs.
void LSRInstance::GenerateReassociations(LSRUse &LU, unsigned LUIdx,
                                         Formula Base,
                                         unsigned Depth) {
  // Arbitrarily cap recursion to protect compile time.
  if (Depth >= 3) return;

  for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i) {
    const SCEV *BaseReg = Base.BaseRegs[i];

    SmallVector<const SCEV *, 8> AddOps;
    CollectSubexprs(BaseReg, 0, AddOps, SE);
    if (AddOps.size() == 1) continue;

    for (SmallVectorImpl<const SCEV *>::const_iterator J = AddOps.begin(),
         JE = AddOps.end(); J != JE; ++J) {
      // Don't pull a constant into a register if the constant could be folded
      // into an immediate field.
      if (isAlwaysFoldable(*J, LU.MinOffset, LU.MaxOffset,
                           Base.getNumRegs() > 1,
                           LU.Kind, LU.AccessTy, TLI, SE))
        continue;

      // Collect all operands except *J.
      SmallVector<const SCEV *, 8> InnerAddOps;
      for (SmallVectorImpl<const SCEV *>::const_iterator K = AddOps.begin(),
           KE = AddOps.end(); K != KE; ++K)
        if (K != J)
          InnerAddOps.push_back(*K);

      // Don't leave just a constant behind in a register if the constant could
      // be folded into an immediate field.
      if (InnerAddOps.size() == 1 &&
          isAlwaysFoldable(InnerAddOps[0], LU.MinOffset, LU.MaxOffset,
                           Base.getNumRegs() > 1,
                           LU.Kind, LU.AccessTy, TLI, SE))
        continue;

      Formula F = Base;
      F.BaseRegs[i] = SE.getAddExpr(InnerAddOps);
      F.BaseRegs.push_back(*J);
      if (InsertFormula(LU, LUIdx, F))
        // If that formula hadn't been seen before, recurse to find more like
        // it.
        GenerateReassociations(LU, LUIdx, LU.Formulae.back(), Depth+1);
    }
  }
}

/// GenerateCombinations - Generate a formula consisting of all of the
/// loop-dominating registers added into a single register.
void LSRInstance::GenerateCombinations(LSRUse &LU, unsigned LUIdx,
                                       Formula Base) {
  // This method is only intersting on a plurality of registers.
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
    const SCEV *Sum = SE.getAddExpr(Ops);
    // TODO: If Sum is zero, it probably means ScalarEvolution missed an
    // opportunity to fold something. For now, just ignore such cases
    // rather than procede with zero in a register.
    if (!Sum->isZero()) {
      F.BaseRegs.push_back(Sum);
      (void)InsertFormula(LU, LUIdx, F);
    }
  }
}

/// GenerateSymbolicOffsets - Generate reuse formulae using symbolic offsets.
void LSRInstance::GenerateSymbolicOffsets(LSRUse &LU, unsigned LUIdx,
                                          Formula Base) {
  // We can't add a symbolic offset if the address already contains one.
  if (Base.AM.BaseGV) return;

  for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i) {
    const SCEV *G = Base.BaseRegs[i];
    GlobalValue *GV = ExtractSymbol(G, SE);
    if (G->isZero() || !GV)
      continue;
    Formula F = Base;
    F.AM.BaseGV = GV;
    if (!isLegalUse(F.AM, LU.MinOffset, LU.MaxOffset,
                    LU.Kind, LU.AccessTy, TLI))
      continue;
    F.BaseRegs[i] = G;
    (void)InsertFormula(LU, LUIdx, F);
  }
}

/// GenerateConstantOffsets - Generate reuse formulae using symbolic offsets.
void LSRInstance::GenerateConstantOffsets(LSRUse &LU, unsigned LUIdx,
                                          Formula Base) {
  // TODO: For now, just add the min and max offset, because it usually isn't
  // worthwhile looking at everything inbetween.
  SmallVector<int64_t, 4> Worklist;
  Worklist.push_back(LU.MinOffset);
  if (LU.MaxOffset != LU.MinOffset)
    Worklist.push_back(LU.MaxOffset);

  for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i) {
    const SCEV *G = Base.BaseRegs[i];

    for (SmallVectorImpl<int64_t>::const_iterator I = Worklist.begin(),
         E = Worklist.end(); I != E; ++I) {
      Formula F = Base;
      F.AM.BaseOffs = (uint64_t)Base.AM.BaseOffs - *I;
      if (isLegalUse(F.AM, LU.MinOffset - *I, LU.MaxOffset - *I,
                     LU.Kind, LU.AccessTy, TLI)) {
        F.BaseRegs[i] = SE.getAddExpr(G, SE.getIntegerSCEV(*I, G->getType()));

        (void)InsertFormula(LU, LUIdx, F);
      }
    }

    int64_t Imm = ExtractImmediate(G, SE);
    if (G->isZero() || Imm == 0)
      continue;
    Formula F = Base;
    F.AM.BaseOffs = (uint64_t)F.AM.BaseOffs + Imm;
    if (!isLegalUse(F.AM, LU.MinOffset, LU.MaxOffset,
                    LU.Kind, LU.AccessTy, TLI))
      continue;
    F.BaseRegs[i] = G;
    (void)InsertFormula(LU, LUIdx, F);
  }
}

/// GenerateICmpZeroScales - For ICmpZero, check to see if we can scale up
/// the comparison. For example, x == y -> x*c == y*c.
void LSRInstance::GenerateICmpZeroScales(LSRUse &LU, unsigned LUIdx,
                                         Formula Base) {
  if (LU.Kind != LSRUse::ICmpZero) return;

  // Determine the integer type for the base formula.
  const Type *IntTy = Base.getType();
  if (!IntTy) return;
  if (SE.getTypeSizeInBits(IntTy) > 64) return;

  // Don't do this if there is more than one offset.
  if (LU.MinOffset != LU.MaxOffset) return;

  assert(!Base.AM.BaseGV && "ICmpZero use is not legal!");

  // Check each interesting stride.
  for (SmallSetVector<int64_t, 8>::const_iterator
       I = Factors.begin(), E = Factors.end(); I != E; ++I) {
    int64_t Factor = *I;
    Formula F = Base;

    // Check that the multiplication doesn't overflow.
    if (F.AM.BaseOffs == INT64_MIN && Factor == -1)
      continue;
    F.AM.BaseOffs = (uint64_t)Base.AM.BaseOffs * Factor;
    if (F.AM.BaseOffs / Factor != Base.AM.BaseOffs)
      continue;

    // Check that multiplying with the use offset doesn't overflow.
    int64_t Offset = LU.MinOffset;
    if (Offset == INT64_MIN && Factor == -1)
      continue;
    Offset = (uint64_t)Offset * Factor;
    if (Offset / Factor != LU.MinOffset)
      continue;

    // Check that this scale is legal.
    if (!isLegalUse(F.AM, Offset, Offset, LU.Kind, LU.AccessTy, TLI))
      continue;

    // Compensate for the use having MinOffset built into it.
    F.AM.BaseOffs = (uint64_t)F.AM.BaseOffs + Offset - LU.MinOffset;

    const SCEV *FactorS = SE.getIntegerSCEV(Factor, IntTy);

    // Check that multiplying with each base register doesn't overflow.
    for (size_t i = 0, e = F.BaseRegs.size(); i != e; ++i) {
      F.BaseRegs[i] = SE.getMulExpr(F.BaseRegs[i], FactorS);
      if (getExactSDiv(F.BaseRegs[i], FactorS, SE) != Base.BaseRegs[i])
        goto next;
    }

    // Check that multiplying with the scaled register doesn't overflow.
    if (F.ScaledReg) {
      F.ScaledReg = SE.getMulExpr(F.ScaledReg, FactorS);
      if (getExactSDiv(F.ScaledReg, FactorS, SE) != Base.ScaledReg)
        continue;
    }

    // If we make it here and it's legal, add it.
    (void)InsertFormula(LU, LUIdx, F);
  next:;
  }
}

/// GenerateScales - Generate stride factor reuse formulae by making use of
/// scaled-offset address modes, for example.
void LSRInstance::GenerateScales(LSRUse &LU, unsigned LUIdx,
                                 Formula Base) {
  // Determine the integer type for the base formula.
  const Type *IntTy = Base.getType();
  if (!IntTy) return;

  // If this Formula already has a scaled register, we can't add another one.
  if (Base.AM.Scale != 0) return;

  // Check each interesting stride.
  for (SmallSetVector<int64_t, 8>::const_iterator
       I = Factors.begin(), E = Factors.end(); I != E; ++I) {
    int64_t Factor = *I;

    Base.AM.Scale = Factor;
    Base.AM.HasBaseReg = Base.BaseRegs.size() > 1;
    // Check whether this scale is going to be legal.
    if (!isLegalUse(Base.AM, LU.MinOffset, LU.MaxOffset,
                    LU.Kind, LU.AccessTy, TLI)) {
      // As a special-case, handle special out-of-loop Basic users specially.
      // TODO: Reconsider this special case.
      if (LU.Kind == LSRUse::Basic &&
          isLegalUse(Base.AM, LU.MinOffset, LU.MaxOffset,
                     LSRUse::Special, LU.AccessTy, TLI) &&
          LU.AllFixupsOutsideLoop)
        LU.Kind = LSRUse::Special;
      else
        continue;
    }
    // For an ICmpZero, negating a solitary base register won't lead to
    // new solutions.
    if (LU.Kind == LSRUse::ICmpZero &&
        !Base.AM.HasBaseReg && Base.AM.BaseOffs == 0 && !Base.AM.BaseGV)
      continue;
    // For each addrec base reg, apply the scale, if possible.
    for (size_t i = 0, e = Base.BaseRegs.size(); i != e; ++i)
      if (const SCEVAddRecExpr *AR =
            dyn_cast<SCEVAddRecExpr>(Base.BaseRegs[i])) {
        const SCEV *FactorS = SE.getIntegerSCEV(Factor, IntTy);
        if (FactorS->isZero())
          continue;
        // Divide out the factor, ignoring high bits, since we'll be
        // scaling the value back up in the end.
        if (const SCEV *Quotient = getExactSDiv(AR, FactorS, SE, true)) {
          // TODO: This could be optimized to avoid all the copying.
          Formula F = Base;
          F.ScaledReg = Quotient;
          std::swap(F.BaseRegs[i], F.BaseRegs.back());
          F.BaseRegs.pop_back();
          (void)InsertFormula(LU, LUIdx, F);
        }
      }
  }
}

/// GenerateTruncates - Generate reuse formulae from different IV types.
void LSRInstance::GenerateTruncates(LSRUse &LU, unsigned LUIdx,
                                    Formula Base) {
  // This requires TargetLowering to tell us which truncates are free.
  if (!TLI) return;

  // Don't bother truncating symbolic values.
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

      // TODO: This assumes we've done basic processing on all uses and
      // have an idea what the register usage is.
      if (!F.hasRegsUsedByUsesOtherThan(LUIdx, RegUses))
        continue;

      (void)InsertFormula(LU, LUIdx, F);
    }
  }
}

namespace {

/// WorkItem - Helper class for GenerateCrossUseConstantOffsets. It's used to
/// defer modifications so that the search phase doesn't have to worry about
/// the data structures moving underneath it.
struct WorkItem {
  size_t LUIdx;
  int64_t Imm;
  const SCEV *OrigReg;

  WorkItem(size_t LI, int64_t I, const SCEV *R)
    : LUIdx(LI), Imm(I), OrigReg(R) {}

  void print(raw_ostream &OS) const;
  void dump() const;
};

}

void WorkItem::print(raw_ostream &OS) const {
  OS << "in formulae referencing " << *OrigReg << " in use " << LUIdx
     << " , add offset " << Imm;
}

void WorkItem::dump() const {
  print(errs()); errs() << '\n';
}

/// GenerateCrossUseConstantOffsets - Look for registers which are a constant
/// distance apart and try to form reuse opportunities between them.
void LSRInstance::GenerateCrossUseConstantOffsets() {
  // Group the registers by their value without any added constant offset.
  typedef std::map<int64_t, const SCEV *> ImmMapTy;
  typedef DenseMap<const SCEV *, ImmMapTy> RegMapTy;
  RegMapTy Map;
  DenseMap<const SCEV *, SmallBitVector> UsedByIndicesMap;
  SmallVector<const SCEV *, 8> Sequence;
  for (RegUseTracker::const_iterator I = RegUses.begin(), E = RegUses.end();
       I != E; ++I) {
    const SCEV *Reg = *I;
    int64_t Imm = ExtractImmediate(Reg, SE);
    std::pair<RegMapTy::iterator, bool> Pair =
      Map.insert(std::make_pair(Reg, ImmMapTy()));
    if (Pair.second)
      Sequence.push_back(Reg);
    Pair.first->second.insert(std::make_pair(Imm, *I));
    UsedByIndicesMap[Reg] |= RegUses.getUsedByIndices(*I);
  }

  // Now examine each set of registers with the same base value. Build up
  // a list of work to do and do the work in a separate step so that we're
  // not adding formulae and register counts while we're searching.
  SmallVector<WorkItem, 32> WorkItems;
  SmallSet<std::pair<size_t, int64_t>, 32> UniqueItems;
  for (SmallVectorImpl<const SCEV *>::const_iterator I = Sequence.begin(),
       E = Sequence.end(); I != E; ++I) {
    const SCEV *Reg = *I;
    const ImmMapTy &Imms = Map.find(Reg)->second;

    // It's not worthwhile looking for reuse if there's only one offset.
    if (Imms.size() == 1)
      continue;

    DEBUG(dbgs() << "Generating cross-use offsets for " << *Reg << ':';
          for (ImmMapTy::const_iterator J = Imms.begin(), JE = Imms.end();
               J != JE; ++J)
            dbgs() << ' ' << J->first;
          dbgs() << '\n');

    // Examine each offset.
    for (ImmMapTy::const_iterator J = Imms.begin(), JE = Imms.end();
         J != JE; ++J) {
      const SCEV *OrigReg = J->second;

      int64_t JImm = J->first;
      const SmallBitVector &UsedByIndices = RegUses.getUsedByIndices(OrigReg);

      if (!isa<SCEVConstant>(OrigReg) &&
          UsedByIndicesMap[Reg].count() == 1) {
        DEBUG(dbgs() << "Skipping cross-use reuse for " << *OrigReg << '\n');
        continue;
      }

      // Conservatively examine offsets between this orig reg a few selected
      // other orig regs.
      ImmMapTy::const_iterator OtherImms[] = {
        Imms.begin(), prior(Imms.end()),
        Imms.upper_bound((Imms.begin()->first + prior(Imms.end())->first) / 2)
      };
      for (size_t i = 0, e = array_lengthof(OtherImms); i != e; ++i) {
        ImmMapTy::const_iterator M = OtherImms[i];
        if (M == J || M == JE) continue;

        // Compute the difference between the two.
        int64_t Imm = (uint64_t)JImm - M->first;
        for (int LUIdx = UsedByIndices.find_first(); LUIdx != -1;
             LUIdx = UsedByIndices.find_next(LUIdx))
          // Make a memo of this use, offset, and register tuple.
          if (UniqueItems.insert(std::make_pair(LUIdx, Imm)))
            WorkItems.push_back(WorkItem(LUIdx, Imm, OrigReg));
      }
    }
  }

  Map.clear();
  Sequence.clear();
  UsedByIndicesMap.clear();
  UniqueItems.clear();

  // Now iterate through the worklist and add new formulae.
  for (SmallVectorImpl<WorkItem>::const_iterator I = WorkItems.begin(),
       E = WorkItems.end(); I != E; ++I) {
    const WorkItem &WI = *I;
    size_t LUIdx = WI.LUIdx;
    LSRUse &LU = Uses[LUIdx];
    int64_t Imm = WI.Imm;
    const SCEV *OrigReg = WI.OrigReg;

    const Type *IntTy = SE.getEffectiveSCEVType(OrigReg->getType());
    const SCEV *NegImmS = SE.getSCEV(ConstantInt::get(IntTy, -(uint64_t)Imm));
    unsigned BitWidth = SE.getTypeSizeInBits(IntTy);

    // TODO: Use a more targetted data structure.
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
        if (!isLegalUse(NewF.AM, LU.MinOffset, LU.MaxOffset,
                        LU.Kind, LU.AccessTy, TLI))
          continue;
        NewF.ScaledReg = SE.getAddExpr(NegImmS, NewF.ScaledReg);

        // If the new scale is a constant in a register, and adding the constant
        // value to the immediate would produce a value closer to zero than the
        // immediate itself, then the formula isn't worthwhile.
        if (const SCEVConstant *C = dyn_cast<SCEVConstant>(NewF.ScaledReg))
          if (C->getValue()->getValue().isNegative() !=
                (NewF.AM.BaseOffs < 0) &&
              (C->getValue()->getValue().abs() * APInt(BitWidth, F.AM.Scale))
                .ule(APInt(BitWidth, NewF.AM.BaseOffs).abs()))
            continue;

        // OK, looks good.
        (void)InsertFormula(LU, LUIdx, NewF);
      } else {
        // Use the immediate in a base register.
        for (size_t N = 0, NE = F.BaseRegs.size(); N != NE; ++N) {
          const SCEV *BaseReg = F.BaseRegs[N];
          if (BaseReg != OrigReg)
            continue;
          Formula NewF = F;
          NewF.AM.BaseOffs = (uint64_t)NewF.AM.BaseOffs + Imm;
          if (!isLegalUse(NewF.AM, LU.MinOffset, LU.MaxOffset,
                          LU.Kind, LU.AccessTy, TLI))
            continue;
          NewF.BaseRegs[N] = SE.getAddExpr(NegImmS, BaseReg);

          // If the new formula has a constant in a register, and adding the
          // constant value to the immediate would produce a value closer to
          // zero than the immediate itself, then the formula isn't worthwhile.
          for (SmallVectorImpl<const SCEV *>::const_iterator
               J = NewF.BaseRegs.begin(), JE = NewF.BaseRegs.end();
               J != JE; ++J)
            if (const SCEVConstant *C = dyn_cast<SCEVConstant>(*J))
              if (C->getValue()->getValue().isNegative() !=
                    (NewF.AM.BaseOffs < 0) &&
                  C->getValue()->getValue().abs()
                    .ule(APInt(BitWidth, NewF.AM.BaseOffs).abs()))
                goto skip_formula;

          // Ok, looks good.
          (void)InsertFormula(LU, LUIdx, NewF);
          break;
        skip_formula:;
        }
      }
    }
  }
}

/// GenerateAllReuseFormulae - Generate formulae for each use.
void
LSRInstance::GenerateAllReuseFormulae() {
  // This is split into multiple loops so that hasRegsUsedByUsesOtherThan
  // queries are more precise.
  for (size_t LUIdx = 0, NumUses = Uses.size(); LUIdx != NumUses; ++LUIdx) {
    LSRUse &LU = Uses[LUIdx];
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateReassociations(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateCombinations(LU, LUIdx, LU.Formulae[i]);
  }
  for (size_t LUIdx = 0, NumUses = Uses.size(); LUIdx != NumUses; ++LUIdx) {
    LSRUse &LU = Uses[LUIdx];
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateSymbolicOffsets(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateConstantOffsets(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateICmpZeroScales(LU, LUIdx, LU.Formulae[i]);
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateScales(LU, LUIdx, LU.Formulae[i]);
  }
  for (size_t LUIdx = 0, NumUses = Uses.size(); LUIdx != NumUses; ++LUIdx) {
    LSRUse &LU = Uses[LUIdx];
    for (size_t i = 0, f = LU.Formulae.size(); i != f; ++i)
      GenerateTruncates(LU, LUIdx, LU.Formulae[i]);
  }

  GenerateCrossUseConstantOffsets();
}

/// If their are multiple formulae with the same set of registers used
/// by other uses, pick the best one and delete the others.
void LSRInstance::FilterOutUndesirableDedicatedRegisters() {
#ifndef NDEBUG
  bool Changed = false;
#endif

  // Collect the best formula for each unique set of shared registers. This
  // is reset for each use.
  typedef DenseMap<SmallVector<const SCEV *, 2>, size_t, UniquifierDenseMapInfo>
    BestFormulaeTy;
  BestFormulaeTy BestFormulae;

  for (size_t LUIdx = 0, NumUses = Uses.size(); LUIdx != NumUses; ++LUIdx) {
    LSRUse &LU = Uses[LUIdx];
    FormulaSorter Sorter(L, LU, SE, DT);

    // Clear out the set of used regs; it will be recomputed.
    LU.Regs.clear();

    for (size_t FIdx = 0, NumForms = LU.Formulae.size();
         FIdx != NumForms; ++FIdx) {
      Formula &F = LU.Formulae[FIdx];

      SmallVector<const SCEV *, 2> Key;
      for (SmallVectorImpl<const SCEV *>::const_iterator J = F.BaseRegs.begin(),
           JE = F.BaseRegs.end(); J != JE; ++J) {
        const SCEV *Reg = *J;
        if (RegUses.isRegUsedByUsesOtherThan(Reg, LUIdx))
          Key.push_back(Reg);
      }
      if (F.ScaledReg &&
          RegUses.isRegUsedByUsesOtherThan(F.ScaledReg, LUIdx))
        Key.push_back(F.ScaledReg);
      // Unstable sort by host order ok, because this is only used for
      // uniquifying.
      std::sort(Key.begin(), Key.end());

      std::pair<BestFormulaeTy::const_iterator, bool> P =
        BestFormulae.insert(std::make_pair(Key, FIdx));
      if (!P.second) {
        Formula &Best = LU.Formulae[P.first->second];
        if (Sorter.operator()(F, Best))
          std::swap(F, Best);
        DEBUG(dbgs() << "Filtering out "; F.print(dbgs());
              dbgs() << "\n"
                        "  in favor of "; Best.print(dbgs());
              dbgs() << '\n');
#ifndef NDEBUG
        Changed = true;
#endif
        std::swap(F, LU.Formulae.back());
        LU.Formulae.pop_back();
        --FIdx;
        --NumForms;
        continue;
      }
      if (F.ScaledReg) LU.Regs.insert(F.ScaledReg);
      LU.Regs.insert(F.BaseRegs.begin(), F.BaseRegs.end());
    }
    BestFormulae.clear();
  }

  DEBUG(if (Changed) {
          dbgs() << "\n"
                    "After filtering out undesirable candidates:\n";
          print_uses(dbgs());
        });
}

/// NarrowSearchSpaceUsingHeuristics - If there are an extrordinary number of
/// formulae to choose from, use some rough heuristics to prune down the number
/// of formulae. This keeps the main solver from taking an extrordinary amount
/// of time in some worst-case scenarios.
void LSRInstance::NarrowSearchSpaceUsingHeuristics() {
  // This is a rough guess that seems to work fairly well.
  const size_t Limit = UINT16_MAX;

  SmallPtrSet<const SCEV *, 4> Taken;
  for (;;) {
    // Estimate the worst-case number of solutions we might consider. We almost
    // never consider this many solutions because we prune the search space,
    // but the pruning isn't always sufficient.
    uint32_t Power = 1;
    for (SmallVectorImpl<LSRUse>::const_iterator I = Uses.begin(),
         E = Uses.end(); I != E; ++I) {
      size_t FSize = I->Formulae.size();
      if (FSize >= Limit) {
        Power = Limit;
        break;
      }
      Power *= FSize;
      if (Power >= Limit)
        break;
    }
    if (Power < Limit)
      break;

    // Ok, we have too many of formulae on our hands to conveniently handle.
    // Use a rough heuristic to thin out the list.

    // Pick the register which is used by the most LSRUses, which is likely
    // to be a good reuse register candidate.
    const SCEV *Best = 0;
    unsigned BestNum = 0;
    for (RegUseTracker::const_iterator I = RegUses.begin(), E = RegUses.end();
         I != E; ++I) {
      const SCEV *Reg = *I;
      if (Taken.count(Reg))
        continue;
      if (!Best)
        Best = Reg;
      else {
        unsigned Count = RegUses.getUsedByIndices(Reg).count();
        if (Count > BestNum) {
          Best = Reg;
          BestNum = Count;
        }
      }
    }

    DEBUG(dbgs() << "Narrowing the search space by assuming " << *Best
                 << " will yeild profitable reuse.\n");
    Taken.insert(Best);

    // In any use with formulae which references this register, delete formulae
    // which don't reference it.
    for (SmallVectorImpl<LSRUse>::iterator I = Uses.begin(),
         E = Uses.end(); I != E; ++I) {
      LSRUse &LU = *I;
      if (!LU.Regs.count(Best)) continue;

      // Clear out the set of used regs; it will be recomputed.
      LU.Regs.clear();

      for (size_t i = 0, e = LU.Formulae.size(); i != e; ++i) {
        Formula &F = LU.Formulae[i];
        if (!F.referencesReg(Best)) {
          DEBUG(dbgs() << "  Deleting "; F.print(dbgs()); dbgs() << '\n');
          std::swap(LU.Formulae.back(), F);
          LU.Formulae.pop_back();
          --e;
          --i;
          continue;
        }

        if (F.ScaledReg) LU.Regs.insert(F.ScaledReg);
        LU.Regs.insert(F.BaseRegs.begin(), F.BaseRegs.end());
      }
    }

    DEBUG(dbgs() << "After pre-selection:\n";
          print_uses(dbgs()));
  }
}

/// SolveRecurse - This is the recursive solver.
void LSRInstance::SolveRecurse(SmallVectorImpl<const Formula *> &Solution,
                               Cost &SolutionCost,
                               SmallVectorImpl<const Formula *> &Workspace,
                               const Cost &CurCost,
                               const SmallPtrSet<const SCEV *, 16> &CurRegs,
                               DenseSet<const SCEV *> &VisitedRegs) const {
  // Some ideas:
  //  - prune more:
  //    - use more aggressive filtering
  //    - sort the formula so that the most profitable solutions are found first
  //    - sort the uses too
  //  - search faster:
  //    - dont compute a cost, and then compare. compare while computing a cost
  //      and bail early.
  //    - track register sets with SmallBitVector

  const LSRUse &LU = Uses[Workspace.size()];

  // If this use references any register that's already a part of the
  // in-progress solution, consider it a requirement that a formula must
  // reference that register in order to be considered. This prunes out
  // unprofitable searching.
  SmallSetVector<const SCEV *, 4> ReqRegs;
  for (SmallPtrSet<const SCEV *, 16>::const_iterator I = CurRegs.begin(),
       E = CurRegs.end(); I != E; ++I)
    if (LU.Regs.count(*I))
      ReqRegs.insert(*I);

  bool AnySatisfiedReqRegs = false;
  SmallPtrSet<const SCEV *, 16> NewRegs;
  Cost NewCost;
retry:
  for (SmallVectorImpl<Formula>::const_iterator I = LU.Formulae.begin(),
       E = LU.Formulae.end(); I != E; ++I) {
    const Formula &F = *I;

    // Ignore formulae which do not use any of the required registers.
    for (SmallSetVector<const SCEV *, 4>::const_iterator J = ReqRegs.begin(),
         JE = ReqRegs.end(); J != JE; ++J) {
      const SCEV *Reg = *J;
      if ((!F.ScaledReg || F.ScaledReg != Reg) &&
          std::find(F.BaseRegs.begin(), F.BaseRegs.end(), Reg) ==
          F.BaseRegs.end())
        goto skip;
    }
    AnySatisfiedReqRegs = true;

    // Evaluate the cost of the current formula. If it's already worse than
    // the current best, prune the search at that point.
    NewCost = CurCost;
    NewRegs = CurRegs;
    NewCost.RateFormula(F, NewRegs, VisitedRegs, L, LU.Offsets, SE, DT);
    if (NewCost < SolutionCost) {
      Workspace.push_back(&F);
      if (Workspace.size() != Uses.size()) {
        SolveRecurse(Solution, SolutionCost, Workspace, NewCost,
                     NewRegs, VisitedRegs);
        if (F.getNumRegs() == 1 && Workspace.size() == 1)
          VisitedRegs.insert(F.ScaledReg ? F.ScaledReg : F.BaseRegs[0]);
      } else {
        DEBUG(dbgs() << "New best at "; NewCost.print(dbgs());
              dbgs() << ". Regs:";
              for (SmallPtrSet<const SCEV *, 16>::const_iterator
                   I = NewRegs.begin(), E = NewRegs.end(); I != E; ++I)
                dbgs() << ' ' << **I;
              dbgs() << '\n');

        SolutionCost = NewCost;
        Solution = Workspace;
      }
      Workspace.pop_back();
    }
  skip:;
  }

  // If none of the formulae had all of the required registers, relax the
  // constraint so that we don't exclude all formulae.
  if (!AnySatisfiedReqRegs) {
    ReqRegs.clear();
    goto retry;
  }
}

void LSRInstance::Solve(SmallVectorImpl<const Formula *> &Solution) const {
  SmallVector<const Formula *, 8> Workspace;
  Cost SolutionCost;
  SolutionCost.Loose();
  Cost CurCost;
  SmallPtrSet<const SCEV *, 16> CurRegs;
  DenseSet<const SCEV *> VisitedRegs;
  Workspace.reserve(Uses.size());

  SolveRecurse(Solution, SolutionCost, Workspace, CurCost,
               CurRegs, VisitedRegs);

  // Ok, we've now made all our decisions.
  DEBUG(dbgs() << "\n"
                  "The chosen solution requires "; SolutionCost.print(dbgs());
        dbgs() << ":\n";
        for (size_t i = 0, e = Uses.size(); i != e; ++i) {
          dbgs() << "  ";
          Uses[i].print(dbgs());
          dbgs() << "\n"
                    "    ";
          Solution[i]->print(dbgs());
          dbgs() << '\n';
        });
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

Value *LSRInstance::Expand(const LSRFixup &LF,
                           const Formula &F,
                           BasicBlock::iterator IP,
                           Loop *L, Instruction *IVIncInsertPos,
                           SCEVExpander &Rewriter,
                           SmallVectorImpl<WeakVH> &DeadInsts,
                           ScalarEvolution &SE, DominatorTree &DT) const {
  const LSRUse &LU = Uses[LF.LUIdx];

  // Then, collect some instructions which we will remain dominated by when
  // expanding the replacement. These must be dominated by any operands that
  // will be required in the expansion.
  SmallVector<Instruction *, 4> Inputs;
  if (Instruction *I = dyn_cast<Instruction>(LF.OperandValToReplace))
    Inputs.push_back(I);
  if (LU.Kind == LSRUse::ICmpZero)
    if (Instruction *I =
          dyn_cast<Instruction>(cast<ICmpInst>(LF.UserInst)->getOperand(1)))
      Inputs.push_back(I);
  if (LF.PostIncLoop && !L->contains(LF.UserInst))
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

  // Inform the Rewriter if we have a post-increment use, so that it can
  // perform an advantageous expansion.
  Rewriter.setPostInc(LF.PostIncLoop);

  // This is the type that the user actually needs.
  const Type *OpTy = LF.OperandValToReplace->getType();
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
    const SCEV *Start = Reg;
    while (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(Start)) {
      if (AR->getLoop() == LF.PostIncLoop) {
        Reg = SE.getAddExpr(Reg, AR->getStepRecurrence(SE));
        // If the user is inside the loop, insert the code after the increment
        // so that it is dominated by its operand. If the original insert point
        // was already dominated by the increment, keep it, because there may
        // be loop-variant operands that need to be respected also.
        if (L->contains(LF.UserInst) && !DT.dominates(IVIncInsertPos, IP))
          IP = IVIncInsertPos;
        break;
      }
      Start = AR->getStart();
    }

    Ops.push_back(SE.getUnknown(Rewriter.expandCodeFor(Reg, 0, IP)));
  }

  // Expand the ScaledReg portion.
  Value *ICmpScaledV = 0;
  if (F.AM.Scale != 0) {
    const SCEV *ScaledS = F.ScaledReg;

    // If we're expanding for a post-inc user for the add-rec's loop, make the
    // post-inc adjustment.
    if (const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(ScaledS))
      if (AR->getLoop() == LF.PostIncLoop)
        ScaledS = SE.getAddExpr(ScaledS, AR->getStepRecurrence(SE));

    if (LU.Kind == LSRUse::ICmpZero) {
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
      ScaledS = SE.getMulExpr(ScaledS,
                              SE.getIntegerSCEV(F.AM.Scale,
                                                ScaledS->getType()));
      Ops.push_back(ScaledS);
    }
  }

  // Expand the immediate portions.
  if (F.AM.BaseGV)
    Ops.push_back(SE.getSCEV(F.AM.BaseGV));
  int64_t Offset = (uint64_t)F.AM.BaseOffs + LF.Offset;
  if (Offset != 0) {
    if (LU.Kind == LSRUse::ICmpZero) {
      // The other interesting way of "folding" with an ICmpZero is to use a
      // negated immediate.
      if (!ICmpScaledV)
        ICmpScaledV = ConstantInt::get(IntTy, -Offset);
      else {
        Ops.push_back(SE.getUnknown(ICmpScaledV));
        ICmpScaledV = ConstantInt::get(IntTy, Offset);
      }
    } else {
      // Just add the immediate values. These again are expected to be matched
      // as part of the address.
      Ops.push_back(SE.getIntegerSCEV(Offset, IntTy));
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
  if (LU.Kind == LSRUse::ICmpZero) {
    ICmpInst *CI = cast<ICmpInst>(LF.UserInst);
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
                                           -(uint64_t)Offset);
      if (C->getType() != OpTy)
        C = ConstantExpr::getCast(CastInst::getCastOpcode(C, false,
                                                          OpTy, false),
                                  C, OpTy);

      CI->setOperand(1, C);
    }
  }

  return FullV;
}

/// RewriteForPHI - Helper for Rewrite. PHI nodes are special because the use
/// of their operands effectively happens in their predecessor blocks, so the
/// expression may need to be expanded in multiple places.
void LSRInstance::RewriteForPHI(PHINode *PN,
                                const LSRFixup &LF,
                                const Formula &F,
                                Loop *L, Instruction *IVIncInsertPos,
                                SCEVExpander &Rewriter,
                                SmallVectorImpl<WeakVH> &DeadInsts,
                                ScalarEvolution &SE, DominatorTree &DT,
                                Pass *P) const {
  DenseMap<BasicBlock *, Value *> Inserted;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
    if (PN->getIncomingValue(i) == LF.OperandValToReplace) {
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
        Value *FullV = Expand(LF, F, BB->getTerminator(), L, IVIncInsertPos,
                              Rewriter, DeadInsts, SE, DT);

        // If this is reuse-by-noop-cast, insert the noop cast.
        const Type *OpTy = LF.OperandValToReplace->getType();
        if (FullV->getType() != OpTy)
          FullV =
            CastInst::Create(CastInst::getCastOpcode(FullV, false,
                                                     OpTy, false),
                             FullV, LF.OperandValToReplace->getType(),
                             "tmp", BB->getTerminator());

        PN->setIncomingValue(i, FullV);
        Pair.first->second = FullV;
      }
    }
}

/// Rewrite - Emit instructions for the leading candidate expression for this
/// LSRUse (this is called "expanding"), and update the UserInst to reference
/// the newly expanded value.
void LSRInstance::Rewrite(const LSRFixup &LF,
                          const Formula &F,
                          Loop *L, Instruction *IVIncInsertPos,
                          SCEVExpander &Rewriter,
                          SmallVectorImpl<WeakVH> &DeadInsts,
                          ScalarEvolution &SE, DominatorTree &DT,
                          Pass *P) const {
  // First, find an insertion point that dominates UserInst. For PHI nodes,
  // find the nearest block which dominates all the relevant uses.
  if (PHINode *PN = dyn_cast<PHINode>(LF.UserInst)) {
    RewriteForPHI(PN, LF, F, L, IVIncInsertPos, Rewriter, DeadInsts, SE, DT, P);
  } else {
    Value *FullV = Expand(LF, F, LF.UserInst, L, IVIncInsertPos,
                          Rewriter, DeadInsts, SE, DT);

    // If this is reuse-by-noop-cast, insert the noop cast.
    const Type *OpTy = LF.OperandValToReplace->getType();
    if (FullV->getType() != OpTy) {
      Instruction *Cast =
        CastInst::Create(CastInst::getCastOpcode(FullV, false, OpTy, false),
                         FullV, OpTy, "tmp", LF.UserInst);
      FullV = Cast;
    }

    // Update the user. ICmpZero is handled specially here (for now) because
    // Expand may have updated one of the operands of the icmp already, and
    // its new value may happen to be equal to LF.OperandValToReplace, in
    // which case doing replaceUsesOfWith leads to replacing both operands
    // with the same value. TODO: Reorganize this.
    if (Uses[LF.LUIdx].Kind == LSRUse::ICmpZero)
      LF.UserInst->setOperand(0, FullV);
    else
      LF.UserInst->replaceUsesOfWith(LF.OperandValToReplace, FullV);
  }

  DeadInsts.push_back(LF.OperandValToReplace);
}

void
LSRInstance::ImplementSolution(const SmallVectorImpl<const Formula *> &Solution,
                               Pass *P) {
  // Keep track of instructions we may have made dead, so that
  // we can remove them after we are done working.
  SmallVector<WeakVH, 16> DeadInsts;

  SCEVExpander Rewriter(SE);
  Rewriter.disableCanonicalMode();
  Rewriter.setIVIncInsertPos(L, IVIncInsertPos);

  // Expand the new value definitions and update the users.
  for (size_t i = 0, e = Fixups.size(); i != e; ++i) {
    size_t LUIdx = Fixups[i].LUIdx;

    Rewrite(Fixups[i], *Solution[LUIdx], L, IVIncInsertPos, Rewriter,
            DeadInsts, SE, DT, P);

    Changed = true;
  }

  // Clean up after ourselves. This must be done before deleting any
  // instructions.
  Rewriter.clear();

  Changed |= DeleteTriviallyDeadInstructions(DeadInsts);
}

LSRInstance::LSRInstance(const TargetLowering *tli, Loop *l, Pass *P)
  : IU(P->getAnalysis<IVUsers>()),
    SE(P->getAnalysis<ScalarEvolution>()),
    DT(P->getAnalysis<DominatorTree>()),
    TLI(tli), L(l), Changed(false), IVIncInsertPos(0) {

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm()) return;

  // If there's no interesting work to be done, bail early.
  if (IU.empty()) return;

  DEBUG(dbgs() << "\nLSR on loop ";
        WriteAsOperand(dbgs(), L->getHeader(), /*PrintType=*/false);
        dbgs() << ":\n");

  /// OptimizeShadowIV - If IV is used in a int-to-float cast
  /// inside the loop then try to eliminate the cast opeation.
  OptimizeShadowIV();

  // Change loop terminating condition to use the postinc iv when possible.
  Changed |= OptimizeLoopTermCond();

  CollectInterestingTypesAndFactors();
  CollectFixupsAndInitialFormulae();
  CollectLoopInvariantFixupsAndFormulae();

  DEBUG(dbgs() << "LSR found " << Uses.size() << " uses:\n";
        print_uses(dbgs()));

  // Now use the reuse data to generate a bunch of interesting ways
  // to formulate the values needed for the uses.
  GenerateAllReuseFormulae();

  DEBUG(dbgs() << "\n"
                  "After generating reuse formulae:\n";
        print_uses(dbgs()));

  FilterOutUndesirableDedicatedRegisters();
  NarrowSearchSpaceUsingHeuristics();

  SmallVector<const Formula *, 8> Solution;
  Solve(Solution);
  assert(Solution.size() == Uses.size() && "Malformed solution!");

  // Release memory that is no longer needed.
  Factors.clear();
  Types.clear();
  RegUses.clear();

#ifndef NDEBUG
  // Formulae should be legal.
  for (SmallVectorImpl<LSRUse>::const_iterator I = Uses.begin(),
       E = Uses.end(); I != E; ++I) {
     const LSRUse &LU = *I;
     for (SmallVectorImpl<Formula>::const_iterator J = LU.Formulae.begin(),
          JE = LU.Formulae.end(); J != JE; ++J)
        assert(isLegalUse(J->AM, LU.MinOffset, LU.MaxOffset,
                          LU.Kind, LU.AccessTy, TLI) &&
               "Illegal formula generated!");
  };
#endif

  // Now that we've decided what we want, make it so.
  ImplementSolution(Solution, P);
}

void LSRInstance::print_factors_and_types(raw_ostream &OS) const {
  if (Factors.empty() && Types.empty()) return;

  OS << "LSR has identified the following interesting factors and types: ";
  bool First = true;

  for (SmallSetVector<int64_t, 8>::const_iterator
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
}

void LSRInstance::print_fixups(raw_ostream &OS) const {
  OS << "LSR is examining the following fixup sites:\n";
  for (SmallVectorImpl<LSRFixup>::const_iterator I = Fixups.begin(),
       E = Fixups.end(); I != E; ++I) {
    const LSRFixup &LF = *I;
    dbgs() << "  ";
    LF.print(OS);
    OS << '\n';
  }
}

void LSRInstance::print_uses(raw_ostream &OS) const {
  OS << "LSR is examining the following uses:\n";
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
      OS << '\n';
    }
  }
}

void LSRInstance::print(raw_ostream &OS) const {
  print_factors_and_types(OS);
  print_fixups(OS);
  print_uses(OS);
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
  explicit LoopStrengthReduce(const TargetLowering *tli = 0);

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
