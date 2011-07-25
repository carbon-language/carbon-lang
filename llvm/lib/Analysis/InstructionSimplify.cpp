//===- InstructionSimplify.cpp - Fold instruction operands ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements routines for folding instructions into simpler forms
// that do not require creating new instructions.  This does constant folding
// ("add i32 1, 1" -> "2") but can also handle non-constant operands, either
// returning a constant ("and i32 %x, 0" -> "0") or an already existing value
// ("and i32 %x, %x" -> "%x").  All operands are assumed to have already been
// simplified: This is usually true and assuming it simplifies the logic (if
// they have not been simplified then results are correct but maybe suboptimal).
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "instsimplify"
#include "llvm/Operator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/ConstantRange.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;
using namespace llvm::PatternMatch;

enum { RecursionLimit = 3 };

STATISTIC(NumExpand,  "Number of expansions");
STATISTIC(NumFactor , "Number of factorizations");
STATISTIC(NumReassoc, "Number of reassociations");

static Value *SimplifyAndInst(Value *, Value *, const TargetData *,
                              const DominatorTree *, unsigned);
static Value *SimplifyBinOp(unsigned, Value *, Value *, const TargetData *,
                            const DominatorTree *, unsigned);
static Value *SimplifyCmpInst(unsigned, Value *, Value *, const TargetData *,
                              const DominatorTree *, unsigned);
static Value *SimplifyOrInst(Value *, Value *, const TargetData *,
                             const DominatorTree *, unsigned);
static Value *SimplifyXorInst(Value *, Value *, const TargetData *,
                              const DominatorTree *, unsigned);

/// ValueDominatesPHI - Does the given value dominate the specified phi node?
static bool ValueDominatesPHI(Value *V, PHINode *P, const DominatorTree *DT) {
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I)
    // Arguments and constants dominate all instructions.
    return true;

  // If we have a DominatorTree then do a precise test.
  if (DT)
    return DT->dominates(I, P);

  // Otherwise, if the instruction is in the entry block, and is not an invoke,
  // then it obviously dominates all phi nodes.
  if (I->getParent() == &I->getParent()->getParent()->getEntryBlock() &&
      !isa<InvokeInst>(I))
    return true;

  return false;
}

/// ExpandBinOp - Simplify "A op (B op' C)" by distributing op over op', turning
/// it into "(A op B) op' (A op C)".  Here "op" is given by Opcode and "op'" is
/// given by OpcodeToExpand, while "A" corresponds to LHS and "B op' C" to RHS.
/// Also performs the transform "(A op' B) op C" -> "(A op C) op' (B op C)".
/// Returns the simplified value, or null if no simplification was performed.
static Value *ExpandBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                          unsigned OpcToExpand, const TargetData *TD,
                          const DominatorTree *DT, unsigned MaxRecurse) {
  Instruction::BinaryOps OpcodeToExpand = (Instruction::BinaryOps)OpcToExpand;
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  // Check whether the expression has the form "(A op' B) op C".
  if (BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS))
    if (Op0->getOpcode() == OpcodeToExpand) {
      // It does!  Try turning it into "(A op C) op' (B op C)".
      Value *A = Op0->getOperand(0), *B = Op0->getOperand(1), *C = RHS;
      // Do "A op C" and "B op C" both simplify?
      if (Value *L = SimplifyBinOp(Opcode, A, C, TD, DT, MaxRecurse))
        if (Value *R = SimplifyBinOp(Opcode, B, C, TD, DT, MaxRecurse)) {
          // They do! Return "L op' R" if it simplifies or is already available.
          // If "L op' R" equals "A op' B" then "L op' R" is just the LHS.
          if ((L == A && R == B) || (Instruction::isCommutative(OpcodeToExpand)
                                     && L == B && R == A)) {
            ++NumExpand;
            return LHS;
          }
          // Otherwise return "L op' R" if it simplifies.
          if (Value *V = SimplifyBinOp(OpcodeToExpand, L, R, TD, DT,
                                       MaxRecurse)) {
            ++NumExpand;
            return V;
          }
        }
    }

  // Check whether the expression has the form "A op (B op' C)".
  if (BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS))
    if (Op1->getOpcode() == OpcodeToExpand) {
      // It does!  Try turning it into "(A op B) op' (A op C)".
      Value *A = LHS, *B = Op1->getOperand(0), *C = Op1->getOperand(1);
      // Do "A op B" and "A op C" both simplify?
      if (Value *L = SimplifyBinOp(Opcode, A, B, TD, DT, MaxRecurse))
        if (Value *R = SimplifyBinOp(Opcode, A, C, TD, DT, MaxRecurse)) {
          // They do! Return "L op' R" if it simplifies or is already available.
          // If "L op' R" equals "B op' C" then "L op' R" is just the RHS.
          if ((L == B && R == C) || (Instruction::isCommutative(OpcodeToExpand)
                                     && L == C && R == B)) {
            ++NumExpand;
            return RHS;
          }
          // Otherwise return "L op' R" if it simplifies.
          if (Value *V = SimplifyBinOp(OpcodeToExpand, L, R, TD, DT,
                                       MaxRecurse)) {
            ++NumExpand;
            return V;
          }
        }
    }

  return 0;
}

/// FactorizeBinOp - Simplify "LHS Opcode RHS" by factorizing out a common term
/// using the operation OpCodeToExtract.  For example, when Opcode is Add and
/// OpCodeToExtract is Mul then this tries to turn "(A*B)+(A*C)" into "A*(B+C)".
/// Returns the simplified value, or null if no simplification was performed.
static Value *FactorizeBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                             unsigned OpcToExtract, const TargetData *TD,
                             const DominatorTree *DT, unsigned MaxRecurse) {
  Instruction::BinaryOps OpcodeToExtract = (Instruction::BinaryOps)OpcToExtract;
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);

  if (!Op0 || Op0->getOpcode() != OpcodeToExtract ||
      !Op1 || Op1->getOpcode() != OpcodeToExtract)
    return 0;

  // The expression has the form "(A op' B) op (C op' D)".
  Value *A = Op0->getOperand(0), *B = Op0->getOperand(1);
  Value *C = Op1->getOperand(0), *D = Op1->getOperand(1);

  // Use left distributivity, i.e. "X op' (Y op Z) = (X op' Y) op (X op' Z)".
  // Does the instruction have the form "(A op' B) op (A op' D)" or, in the
  // commutative case, "(A op' B) op (C op' A)"?
  if (A == C || (Instruction::isCommutative(OpcodeToExtract) && A == D)) {
    Value *DD = A == C ? D : C;
    // Form "A op' (B op DD)" if it simplifies completely.
    // Does "B op DD" simplify?
    if (Value *V = SimplifyBinOp(Opcode, B, DD, TD, DT, MaxRecurse)) {
      // It does!  Return "A op' V" if it simplifies or is already available.
      // If V equals B then "A op' V" is just the LHS.  If V equals DD then
      // "A op' V" is just the RHS.
      if (V == B || V == DD) {
        ++NumFactor;
        return V == B ? LHS : RHS;
      }
      // Otherwise return "A op' V" if it simplifies.
      if (Value *W = SimplifyBinOp(OpcodeToExtract, A, V, TD, DT, MaxRecurse)) {
        ++NumFactor;
        return W;
      }
    }
  }

  // Use right distributivity, i.e. "(X op Y) op' Z = (X op' Z) op (Y op' Z)".
  // Does the instruction have the form "(A op' B) op (C op' B)" or, in the
  // commutative case, "(A op' B) op (B op' D)"?
  if (B == D || (Instruction::isCommutative(OpcodeToExtract) && B == C)) {
    Value *CC = B == D ? C : D;
    // Form "(A op CC) op' B" if it simplifies completely..
    // Does "A op CC" simplify?
    if (Value *V = SimplifyBinOp(Opcode, A, CC, TD, DT, MaxRecurse)) {
      // It does!  Return "V op' B" if it simplifies or is already available.
      // If V equals A then "V op' B" is just the LHS.  If V equals CC then
      // "V op' B" is just the RHS.
      if (V == A || V == CC) {
        ++NumFactor;
        return V == A ? LHS : RHS;
      }
      // Otherwise return "V op' B" if it simplifies.
      if (Value *W = SimplifyBinOp(OpcodeToExtract, V, B, TD, DT, MaxRecurse)) {
        ++NumFactor;
        return W;
      }
    }
  }

  return 0;
}

/// SimplifyAssociativeBinOp - Generic simplifications for associative binary
/// operations.  Returns the simpler value, or null if none was found.
static Value *SimplifyAssociativeBinOp(unsigned Opc, Value *LHS, Value *RHS,
                                       const TargetData *TD,
                                       const DominatorTree *DT,
                                       unsigned MaxRecurse) {
  Instruction::BinaryOps Opcode = (Instruction::BinaryOps)Opc;
  assert(Instruction::isAssociative(Opcode) && "Not an associative operation!");

  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  BinaryOperator *Op0 = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *Op1 = dyn_cast<BinaryOperator>(RHS);

  // Transform: "(A op B) op C" ==> "A op (B op C)" if it simplifies completely.
  if (Op0 && Op0->getOpcode() == Opcode) {
    Value *A = Op0->getOperand(0);
    Value *B = Op0->getOperand(1);
    Value *C = RHS;

    // Does "B op C" simplify?
    if (Value *V = SimplifyBinOp(Opcode, B, C, TD, DT, MaxRecurse)) {
      // It does!  Return "A op V" if it simplifies or is already available.
      // If V equals B then "A op V" is just the LHS.
      if (V == B) return LHS;
      // Otherwise return "A op V" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, A, V, TD, DT, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // Transform: "A op (B op C)" ==> "(A op B) op C" if it simplifies completely.
  if (Op1 && Op1->getOpcode() == Opcode) {
    Value *A = LHS;
    Value *B = Op1->getOperand(0);
    Value *C = Op1->getOperand(1);

    // Does "A op B" simplify?
    if (Value *V = SimplifyBinOp(Opcode, A, B, TD, DT, MaxRecurse)) {
      // It does!  Return "V op C" if it simplifies or is already available.
      // If V equals B then "V op C" is just the RHS.
      if (V == B) return RHS;
      // Otherwise return "V op C" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, V, C, TD, DT, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // The remaining transforms require commutativity as well as associativity.
  if (!Instruction::isCommutative(Opcode))
    return 0;

  // Transform: "(A op B) op C" ==> "(C op A) op B" if it simplifies completely.
  if (Op0 && Op0->getOpcode() == Opcode) {
    Value *A = Op0->getOperand(0);
    Value *B = Op0->getOperand(1);
    Value *C = RHS;

    // Does "C op A" simplify?
    if (Value *V = SimplifyBinOp(Opcode, C, A, TD, DT, MaxRecurse)) {
      // It does!  Return "V op B" if it simplifies or is already available.
      // If V equals A then "V op B" is just the LHS.
      if (V == A) return LHS;
      // Otherwise return "V op B" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, V, B, TD, DT, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  // Transform: "A op (B op C)" ==> "B op (C op A)" if it simplifies completely.
  if (Op1 && Op1->getOpcode() == Opcode) {
    Value *A = LHS;
    Value *B = Op1->getOperand(0);
    Value *C = Op1->getOperand(1);

    // Does "C op A" simplify?
    if (Value *V = SimplifyBinOp(Opcode, C, A, TD, DT, MaxRecurse)) {
      // It does!  Return "B op V" if it simplifies or is already available.
      // If V equals C then "B op V" is just the RHS.
      if (V == C) return RHS;
      // Otherwise return "B op V" if it simplifies.
      if (Value *W = SimplifyBinOp(Opcode, B, V, TD, DT, MaxRecurse)) {
        ++NumReassoc;
        return W;
      }
    }
  }

  return 0;
}

/// ThreadBinOpOverSelect - In the case of a binary operation with a select
/// instruction as an operand, try to simplify the binop by seeing whether
/// evaluating it on both branches of the select results in the same value.
/// Returns the common value if so, otherwise returns null.
static Value *ThreadBinOpOverSelect(unsigned Opcode, Value *LHS, Value *RHS,
                                    const TargetData *TD,
                                    const DominatorTree *DT,
                                    unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  SelectInst *SI;
  if (isa<SelectInst>(LHS)) {
    SI = cast<SelectInst>(LHS);
  } else {
    assert(isa<SelectInst>(RHS) && "No select instruction operand!");
    SI = cast<SelectInst>(RHS);
  }

  // Evaluate the BinOp on the true and false branches of the select.
  Value *TV;
  Value *FV;
  if (SI == LHS) {
    TV = SimplifyBinOp(Opcode, SI->getTrueValue(), RHS, TD, DT, MaxRecurse);
    FV = SimplifyBinOp(Opcode, SI->getFalseValue(), RHS, TD, DT, MaxRecurse);
  } else {
    TV = SimplifyBinOp(Opcode, LHS, SI->getTrueValue(), TD, DT, MaxRecurse);
    FV = SimplifyBinOp(Opcode, LHS, SI->getFalseValue(), TD, DT, MaxRecurse);
  }

  // If they simplified to the same value, then return the common value.
  // If they both failed to simplify then return null.
  if (TV == FV)
    return TV;

  // If one branch simplified to undef, return the other one.
  if (TV && isa<UndefValue>(TV))
    return FV;
  if (FV && isa<UndefValue>(FV))
    return TV;

  // If applying the operation did not change the true and false select values,
  // then the result of the binop is the select itself.
  if (TV == SI->getTrueValue() && FV == SI->getFalseValue())
    return SI;

  // If one branch simplified and the other did not, and the simplified
  // value is equal to the unsimplified one, return the simplified value.
  // For example, select (cond, X, X & Z) & Z -> X & Z.
  if ((FV && !TV) || (TV && !FV)) {
    // Check that the simplified value has the form "X op Y" where "op" is the
    // same as the original operation.
    Instruction *Simplified = dyn_cast<Instruction>(FV ? FV : TV);
    if (Simplified && Simplified->getOpcode() == Opcode) {
      // The value that didn't simplify is "UnsimplifiedLHS op UnsimplifiedRHS".
      // We already know that "op" is the same as for the simplified value.  See
      // if the operands match too.  If so, return the simplified value.
      Value *UnsimplifiedBranch = FV ? SI->getTrueValue() : SI->getFalseValue();
      Value *UnsimplifiedLHS = SI == LHS ? UnsimplifiedBranch : LHS;
      Value *UnsimplifiedRHS = SI == LHS ? RHS : UnsimplifiedBranch;
      if (Simplified->getOperand(0) == UnsimplifiedLHS &&
          Simplified->getOperand(1) == UnsimplifiedRHS)
        return Simplified;
      if (Simplified->isCommutative() &&
          Simplified->getOperand(1) == UnsimplifiedLHS &&
          Simplified->getOperand(0) == UnsimplifiedRHS)
        return Simplified;
    }
  }

  return 0;
}

/// ThreadCmpOverSelect - In the case of a comparison with a select instruction,
/// try to simplify the comparison by seeing whether both branches of the select
/// result in the same value.  Returns the common value if so, otherwise returns
/// null.
static Value *ThreadCmpOverSelect(CmpInst::Predicate Pred, Value *LHS,
                                  Value *RHS, const TargetData *TD,
                                  const DominatorTree *DT,
                                  unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  // Make sure the select is on the LHS.
  if (!isa<SelectInst>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(isa<SelectInst>(LHS) && "Not comparing with a select instruction!");
  SelectInst *SI = cast<SelectInst>(LHS);

  // Now that we have "cmp select(Cond, TV, FV), RHS", analyse it.
  // Does "cmp TV, RHS" simplify?
  if (Value *TCmp = SimplifyCmpInst(Pred, SI->getTrueValue(), RHS, TD, DT,
                                    MaxRecurse)) {
    // It does!  Does "cmp FV, RHS" simplify?
    if (Value *FCmp = SimplifyCmpInst(Pred, SI->getFalseValue(), RHS, TD, DT,
                                      MaxRecurse)) {
      // It does!  If they simplified to the same value, then use it as the
      // result of the original comparison.
      if (TCmp == FCmp)
        return TCmp;
      Value *Cond = SI->getCondition();
      // If the false value simplified to false, then the result of the compare
      // is equal to "Cond && TCmp".  This also catches the case when the false
      // value simplified to false and the true value to true, returning "Cond".
      if (match(FCmp, m_Zero()))
        if (Value *V = SimplifyAndInst(Cond, TCmp, TD, DT, MaxRecurse))
          return V;
      // If the true value simplified to true, then the result of the compare
      // is equal to "Cond || FCmp".
      if (match(TCmp, m_One()))
        if (Value *V = SimplifyOrInst(Cond, FCmp, TD, DT, MaxRecurse))
          return V;
      // Finally, if the false value simplified to true and the true value to
      // false, then the result of the compare is equal to "!Cond".
      if (match(FCmp, m_One()) && match(TCmp, m_Zero()))
        if (Value *V =
            SimplifyXorInst(Cond, Constant::getAllOnesValue(Cond->getType()),
                            TD, DT, MaxRecurse))
          return V;
    }
  }

  return 0;
}

/// ThreadBinOpOverPHI - In the case of a binary operation with an operand that
/// is a PHI instruction, try to simplify the binop by seeing whether evaluating
/// it on the incoming phi values yields the same result for every value.  If so
/// returns the common value, otherwise returns null.
static Value *ThreadBinOpOverPHI(unsigned Opcode, Value *LHS, Value *RHS,
                                 const TargetData *TD, const DominatorTree *DT,
                                 unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  PHINode *PI;
  if (isa<PHINode>(LHS)) {
    PI = cast<PHINode>(LHS);
    // Bail out if RHS and the phi may be mutually interdependent due to a loop.
    if (!ValueDominatesPHI(RHS, PI, DT))
      return 0;
  } else {
    assert(isa<PHINode>(RHS) && "No PHI instruction operand!");
    PI = cast<PHINode>(RHS);
    // Bail out if LHS and the phi may be mutually interdependent due to a loop.
    if (!ValueDominatesPHI(LHS, PI, DT))
      return 0;
  }

  // Evaluate the BinOp on the incoming phi values.
  Value *CommonValue = 0;
  for (unsigned i = 0, e = PI->getNumIncomingValues(); i != e; ++i) {
    Value *Incoming = PI->getIncomingValue(i);
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PI) continue;
    Value *V = PI == LHS ?
      SimplifyBinOp(Opcode, Incoming, RHS, TD, DT, MaxRecurse) :
      SimplifyBinOp(Opcode, LHS, Incoming, TD, DT, MaxRecurse);
    // If the operation failed to simplify, or simplified to a different value
    // to previously, then give up.
    if (!V || (CommonValue && V != CommonValue))
      return 0;
    CommonValue = V;
  }

  return CommonValue;
}

/// ThreadCmpOverPHI - In the case of a comparison with a PHI instruction, try
/// try to simplify the comparison by seeing whether comparing with all of the
/// incoming phi values yields the same result every time.  If so returns the
/// common result, otherwise returns null.
static Value *ThreadCmpOverPHI(CmpInst::Predicate Pred, Value *LHS, Value *RHS,
                               const TargetData *TD, const DominatorTree *DT,
                               unsigned MaxRecurse) {
  // Recursion is always used, so bail out at once if we already hit the limit.
  if (!MaxRecurse--)
    return 0;

  // Make sure the phi is on the LHS.
  if (!isa<PHINode>(LHS)) {
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }
  assert(isa<PHINode>(LHS) && "Not comparing with a phi instruction!");
  PHINode *PI = cast<PHINode>(LHS);

  // Bail out if RHS and the phi may be mutually interdependent due to a loop.
  if (!ValueDominatesPHI(RHS, PI, DT))
    return 0;

  // Evaluate the BinOp on the incoming phi values.
  Value *CommonValue = 0;
  for (unsigned i = 0, e = PI->getNumIncomingValues(); i != e; ++i) {
    Value *Incoming = PI->getIncomingValue(i);
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PI) continue;
    Value *V = SimplifyCmpInst(Pred, Incoming, RHS, TD, DT, MaxRecurse);
    // If the operation failed to simplify, or simplified to a different value
    // to previously, then give up.
    if (!V || (CommonValue && V != CommonValue))
      return 0;
    CommonValue = V;
  }

  return CommonValue;
}

/// SimplifyAddInst - Given operands for an Add, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAddInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const TargetData *TD, const DominatorTree *DT,
                              unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Add, CLHS->getType(),
                                      Ops, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X + undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // X + 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X + (Y - X) -> Y
  // (Y - X) + X -> Y
  // Eg: X + -X -> 0
  Value *Y = 0;
  if (match(Op1, m_Sub(m_Value(Y), m_Specific(Op0))) ||
      match(Op0, m_Sub(m_Value(Y), m_Specific(Op1))))
    return Y;

  // X + ~X -> -1   since   ~X = -X-1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  /// i1 add -> xor.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, TD, DT, MaxRecurse-1))
      return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Add, Op0, Op1, TD, DT,
                                          MaxRecurse))
    return V;

  // Mul distributes over Add.  Try some generic simplifications based on this.
  if (Value *V = FactorizeBinOp(Instruction::Add, Op0, Op1, Instruction::Mul,
                                TD, DT, MaxRecurse))
    return V;

  // Threading Add over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A + select(cond, B, C)" means evaluating
  // "A+B" and "A+C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A+B" and "A+C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return 0;
}

Value *llvm::SimplifyAddInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyAddInst(Op0, Op1, isNSW, isNUW, TD, DT, RecursionLimit);
}

/// SimplifySubInst - Given operands for a Sub, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifySubInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const TargetData *TD, const DominatorTree *DT,
                              unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0))
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Sub, CLHS->getType(),
                                      Ops, TD);
    }

  // X - undef -> undef
  // undef - X -> undef
  if (match(Op0, m_Undef()) || match(Op1, m_Undef()))
    return UndefValue::get(Op0->getType());

  // X - 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X - X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // (X*2) - X -> X
  // (X<<1) - X -> X
  Value *X = 0;
  if (match(Op0, m_Mul(m_Specific(Op1), m_ConstantInt<2>())) ||
      match(Op0, m_Shl(m_Specific(Op1), m_One())))
    return Op1;

  // (X + Y) - Z -> X + (Y - Z) or Y + (X - Z) if everything simplifies.
  // For example, (X + Y) - Y -> X; (Y + X) - Y -> X
  Value *Y = 0, *Z = Op1;
  if (MaxRecurse && match(Op0, m_Add(m_Value(X), m_Value(Y)))) { // (X + Y) - Z
    // See if "V === Y - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, Y, Z, TD, DT, MaxRecurse-1))
      // It does!  Now see if "X + V" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, X, V, TD, DT,
                                   MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
    // See if "V === X - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Z, TD, DT, MaxRecurse-1))
      // It does!  Now see if "Y + V" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, Y, V, TD, DT,
                                   MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
  }

  // X - (Y + Z) -> (X - Y) - Z or (X - Z) - Y if everything simplifies.
  // For example, X - (X + 1) -> -1
  X = Op0;
  if (MaxRecurse && match(Op1, m_Add(m_Value(Y), m_Value(Z)))) { // X - (Y + Z)
    // See if "V === X - Y" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Y, TD, DT, MaxRecurse-1))
      // It does!  Now see if "V - Z" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Sub, V, Z, TD, DT,
                                   MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
    // See if "V === X - Z" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, X, Z, TD, DT, MaxRecurse-1))
      // It does!  Now see if "V - Y" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Sub, V, Y, TD, DT,
                                   MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }
  }

  // Z - (X - Y) -> (Z - X) + Y if everything simplifies.
  // For example, X - (X - Y) -> Y.
  Z = Op0;
  if (MaxRecurse && match(Op1, m_Sub(m_Value(X), m_Value(Y)))) // Z - (X - Y)
    // See if "V === Z - X" simplifies.
    if (Value *V = SimplifyBinOp(Instruction::Sub, Z, X, TD, DT, MaxRecurse-1))
      // It does!  Now see if "V + Y" simplifies.
      if (Value *W = SimplifyBinOp(Instruction::Add, V, Y, TD, DT,
                                   MaxRecurse-1)) {
        // It does, we successfully reassociated!
        ++NumReassoc;
        return W;
      }

  // Mul distributes over Sub.  Try some generic simplifications based on this.
  if (Value *V = FactorizeBinOp(Instruction::Sub, Op0, Op1, Instruction::Mul,
                                TD, DT, MaxRecurse))
    return V;

  // i1 sub -> xor.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, TD, DT, MaxRecurse-1))
      return V;

  // Threading Sub over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A - select(cond, B, C)" means evaluating
  // "A-B" and "A-C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A-B" and "A-C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return 0;
}

Value *llvm::SimplifySubInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifySubInst(Op0, Op1, isNSW, isNUW, TD, DT, RecursionLimit);
}

/// SimplifyMulInst - Given operands for a Mul, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyMulInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Mul, CLHS->getType(),
                                      Ops, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X * undef -> 0
  if (match(Op1, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // X * 0 -> 0
  if (match(Op1, m_Zero()))
    return Op1;

  // X * 1 -> X
  if (match(Op1, m_One()))
    return Op0;

  // (X / Y) * Y -> X if the division is exact.
  Value *X = 0, *Y = 0;
  if ((match(Op0, m_IDiv(m_Value(X), m_Value(Y))) && Y == Op1) || // (X / Y) * Y
      (match(Op1, m_IDiv(m_Value(X), m_Value(Y))) && Y == Op0)) { // Y * (X / Y)
    BinaryOperator *Div = cast<BinaryOperator>(Y == Op1 ? Op0 : Op1);
    if (Div->isExact())
      return X;
  }

  // i1 mul -> and.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyAndInst(Op0, Op1, TD, DT, MaxRecurse-1))
      return V;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Mul, Op0, Op1, TD, DT,
                                          MaxRecurse))
    return V;

  // Mul distributes over Add.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::Mul, Op0, Op1, Instruction::Add,
                             TD, DT, MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::Mul, Op0, Op1, TD, DT,
                                         MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::Mul, Op0, Op1, TD, DT,
                                      MaxRecurse))
      return V;

  return 0;
}

Value *llvm::SimplifyMulInst(Value *Op0, Value *Op1, const TargetData *TD,
                             const DominatorTree *DT) {
  return ::SimplifyMulInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyDiv - Given operands for an SDiv or UDiv, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyDiv(Instruction::BinaryOps Opcode, Value *Op0, Value *Op1,
                          const TargetData *TD, const DominatorTree *DT,
                          unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, TD);
    }
  }

  bool isSigned = Opcode == Instruction::SDiv;

  // X / undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // undef / X -> 0
  if (match(Op0, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // 0 / X -> 0, we don't need to preserve faults!
  if (match(Op0, m_Zero()))
    return Op0;

  // X / 1 -> X
  if (match(Op1, m_One()))
    return Op0;

  if (Op0->getType()->isIntegerTy(1))
    // It can't be division by zero, hence it must be division by one.
    return Op0;

  // X / X -> 1
  if (Op0 == Op1)
    return ConstantInt::get(Op0->getType(), 1);

  // (X * Y) / Y -> X if the multiplication does not overflow.
  Value *X = 0, *Y = 0;
  if (match(Op0, m_Mul(m_Value(X), m_Value(Y))) && (X == Op1 || Y == Op1)) {
    if (Y != Op1) std::swap(X, Y); // Ensure expression is (X * Y) / Y, Y = Op1
    BinaryOperator *Mul = cast<BinaryOperator>(Op0);
    // If the Mul knows it does not overflow, then we are good to go.
    if ((isSigned && Mul->hasNoSignedWrap()) ||
        (!isSigned && Mul->hasNoUnsignedWrap()))
      return X;
    // If X has the form X = A / Y then X * Y cannot overflow.
    if (BinaryOperator *Div = dyn_cast<BinaryOperator>(X))
      if (Div->getOpcode() == Opcode && Div->getOperand(1) == Y)
        return X;
  }

  // (X rem Y) / Y -> 0
  if ((isSigned && match(Op0, m_SRem(m_Value(), m_Specific(Op1)))) ||
      (!isSigned && match(Op0, m_URem(m_Value(), m_Specific(Op1)))))
    return Constant::getNullValue(Op0->getType());

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, TD, DT, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, TD, DT, MaxRecurse))
      return V;

  return 0;
}

/// SimplifySDivInst - Given operands for an SDiv, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifySDivInst(Value *Op0, Value *Op1, const TargetData *TD,
                               const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyDiv(Instruction::SDiv, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  return 0;
}

Value *llvm::SimplifySDivInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifySDivInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyUDivInst - Given operands for a UDiv, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyUDivInst(Value *Op0, Value *Op1, const TargetData *TD,
                               const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyDiv(Instruction::UDiv, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  return 0;
}

Value *llvm::SimplifyUDivInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifyUDivInst(Op0, Op1, TD, DT, RecursionLimit);
}

static Value *SimplifyFDivInst(Value *Op0, Value *Op1, const TargetData *,
                               const DominatorTree *, unsigned) {
  // undef / X -> undef    (the undef could be a snan).
  if (match(Op0, m_Undef()))
    return Op0;

  // X / undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  return 0;
}

Value *llvm::SimplifyFDivInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifyFDivInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyRem - Given operands for an SRem or URem, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyRem(Instruction::BinaryOps Opcode, Value *Op0, Value *Op1,
                          const TargetData *TD, const DominatorTree *DT,
                          unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, TD);
    }
  }

  // X % undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // undef % X -> 0
  if (match(Op0, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // 0 % X -> 0, we don't need to preserve faults!
  if (match(Op0, m_Zero()))
    return Op0;

  // X % 0 -> undef, we don't need to preserve faults!
  if (match(Op1, m_Zero()))
    return UndefValue::get(Op0->getType());

  // X % 1 -> 0
  if (match(Op1, m_One()))
    return Constant::getNullValue(Op0->getType());

  if (Op0->getType()->isIntegerTy(1))
    // It can't be remainder by zero, hence it must be remainder by one.
    return Constant::getNullValue(Op0->getType());

  // X % X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, TD, DT, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, TD, DT, MaxRecurse))
      return V;

  return 0;
}

/// SimplifySRemInst - Given operands for an SRem, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifySRemInst(Value *Op0, Value *Op1, const TargetData *TD,
                               const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyRem(Instruction::SRem, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  return 0;
}

Value *llvm::SimplifySRemInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifySRemInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyURemInst - Given operands for a URem, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyURemInst(Value *Op0, Value *Op1, const TargetData *TD,
                               const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyRem(Instruction::URem, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  return 0;
}

Value *llvm::SimplifyURemInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifyURemInst(Op0, Op1, TD, DT, RecursionLimit);
}

static Value *SimplifyFRemInst(Value *Op0, Value *Op1, const TargetData *,
                               const DominatorTree *, unsigned) {
  // undef % X -> undef    (the undef could be a snan).
  if (match(Op0, m_Undef()))
    return Op0;

  // X % undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  return 0;
}

Value *llvm::SimplifyFRemInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifyFRemInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyShift - Given operands for an Shl, LShr or AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyShift(unsigned Opcode, Value *Op0, Value *Op1,
                            const TargetData *TD, const DominatorTree *DT,
                            unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, TD);
    }
  }

  // 0 shift by X -> 0
  if (match(Op0, m_Zero()))
    return Op0;

  // X shift by 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X shift by undef -> undef because it may shift by the bitwidth.
  if (match(Op1, m_Undef()))
    return Op1;

  // Shifting by the bitwidth or more is undefined.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1))
    if (CI->getValue().getLimitedValue() >=
        Op0->getType()->getScalarSizeInBits())
      return UndefValue::get(Op0->getType());

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Opcode, Op0, Op1, TD, DT, MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Opcode, Op0, Op1, TD, DT, MaxRecurse))
      return V;

  return 0;
}

/// SimplifyShlInst - Given operands for an Shl, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                              const TargetData *TD, const DominatorTree *DT,
                              unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::Shl, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  // undef << X -> 0
  if (match(Op0, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // (X >> A) << A -> X
  Value *X;
  if (match(Op0, m_Shr(m_Value(X), m_Specific(Op1))) &&
      cast<PossiblyExactOperator>(Op0)->isExact())
    return X;
  return 0;
}

Value *llvm::SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                             const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyShlInst(Op0, Op1, isNSW, isNUW, TD, DT, RecursionLimit);
}

/// SimplifyLShrInst - Given operands for an LShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                               const TargetData *TD, const DominatorTree *DT,
                               unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::LShr, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  // undef >>l X -> 0
  if (match(Op0, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // (X << A) >> A -> X
  Value *X;
  if (match(Op0, m_Shl(m_Value(X), m_Specific(Op1))) &&
      cast<OverflowingBinaryOperator>(Op0)->hasNoUnsignedWrap())
    return X;

  return 0;
}

Value *llvm::SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                              const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyLShrInst(Op0, Op1, isExact, TD, DT, RecursionLimit);
}

/// SimplifyAShrInst - Given operands for an AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                               const TargetData *TD, const DominatorTree *DT,
                               unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::AShr, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  // all ones >>a X -> all ones
  if (match(Op0, m_AllOnes()))
    return Op0;

  // undef >>a X -> all ones
  if (match(Op0, m_Undef()))
    return Constant::getAllOnesValue(Op0->getType());

  // (X << A) >> A -> X
  Value *X;
  if (match(Op0, m_Shl(m_Value(X), m_Specific(Op1))) &&
      cast<OverflowingBinaryOperator>(Op0)->hasNoSignedWrap())
    return X;

  return 0;
}

Value *llvm::SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                              const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyAShrInst(Op0, Op1, isExact, TD, DT, RecursionLimit);
}

/// SimplifyAndInst - Given operands for an And, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAndInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::And, CLHS->getType(),
                                      Ops, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X & undef -> 0
  if (match(Op1, m_Undef()))
    return Constant::getNullValue(Op0->getType());

  // X & X = X
  if (Op0 == Op1)
    return Op0;

  // X & 0 = 0
  if (match(Op1, m_Zero()))
    return Op1;

  // X & -1 = X
  if (match(Op1, m_AllOnes()))
    return Op0;

  // A & ~A  =  ~A & A  =  0
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getNullValue(Op0->getType());

  // (A | ?) & A = A
  Value *A = 0, *B = 0;
  if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
      (A == Op1 || B == Op1))
    return Op1;

  // A & (A | ?) = A
  if (match(Op1, m_Or(m_Value(A), m_Value(B))) &&
      (A == Op0 || B == Op0))
    return Op0;

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::And, Op0, Op1, TD, DT,
                                          MaxRecurse))
    return V;

  // And distributes over Or.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::And, Op0, Op1, Instruction::Or,
                             TD, DT, MaxRecurse))
    return V;

  // And distributes over Xor.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::And, Op0, Op1, Instruction::Xor,
                             TD, DT, MaxRecurse))
    return V;

  // Or distributes over And.  Try some generic simplifications based on this.
  if (Value *V = FactorizeBinOp(Instruction::And, Op0, Op1, Instruction::Or,
                                TD, DT, MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::And, Op0, Op1, TD, DT,
                                         MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::And, Op0, Op1, TD, DT,
                                      MaxRecurse))
      return V;

  return 0;
}

Value *llvm::SimplifyAndInst(Value *Op0, Value *Op1, const TargetData *TD,
                             const DominatorTree *DT) {
  return ::SimplifyAndInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyOrInst - Given operands for an Or, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyOrInst(Value *Op0, Value *Op1, const TargetData *TD,
                             const DominatorTree *DT, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Or, CLHS->getType(),
                                      Ops, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X | undef -> -1
  if (match(Op1, m_Undef()))
    return Constant::getAllOnesValue(Op0->getType());

  // X | X = X
  if (Op0 == Op1)
    return Op0;

  // X | 0 = X
  if (match(Op1, m_Zero()))
    return Op0;

  // X | -1 = -1
  if (match(Op1, m_AllOnes()))
    return Op1;

  // A | ~A  =  ~A | A  =  -1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  // (A & ?) | A = A
  Value *A = 0, *B = 0;
  if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
      (A == Op1 || B == Op1))
    return Op1;

  // A | (A & ?) = A
  if (match(Op1, m_And(m_Value(A), m_Value(B))) &&
      (A == Op0 || B == Op0))
    return Op0;

  // ~(A & ?) | A = -1
  if (match(Op0, m_Not(m_And(m_Value(A), m_Value(B)))) &&
      (A == Op1 || B == Op1))
    return Constant::getAllOnesValue(Op1->getType());

  // A | ~(A & ?) = -1
  if (match(Op1, m_Not(m_And(m_Value(A), m_Value(B)))) &&
      (A == Op0 || B == Op0))
    return Constant::getAllOnesValue(Op0->getType());

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Or, Op0, Op1, TD, DT,
                                          MaxRecurse))
    return V;

  // Or distributes over And.  Try some generic simplifications based on this.
  if (Value *V = ExpandBinOp(Instruction::Or, Op0, Op1, Instruction::And,
                             TD, DT, MaxRecurse))
    return V;

  // And distributes over Or.  Try some generic simplifications based on this.
  if (Value *V = FactorizeBinOp(Instruction::Or, Op0, Op1, Instruction::And,
                                TD, DT, MaxRecurse))
    return V;

  // If the operation is with the result of a select instruction, check whether
  // operating on either branch of the select always yields the same value.
  if (isa<SelectInst>(Op0) || isa<SelectInst>(Op1))
    if (Value *V = ThreadBinOpOverSelect(Instruction::Or, Op0, Op1, TD, DT,
                                         MaxRecurse))
      return V;

  // If the operation is with the result of a phi instruction, check whether
  // operating on all incoming values of the phi always yields the same value.
  if (isa<PHINode>(Op0) || isa<PHINode>(Op1))
    if (Value *V = ThreadBinOpOverPHI(Instruction::Or, Op0, Op1, TD, DT,
                                      MaxRecurse))
      return V;

  return 0;
}

Value *llvm::SimplifyOrInst(Value *Op0, Value *Op1, const TargetData *TD,
                            const DominatorTree *DT) {
  return ::SimplifyOrInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyXorInst - Given operands for a Xor, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyXorInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::Xor, CLHS->getType(),
                                      Ops, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // A ^ undef -> undef
  if (match(Op1, m_Undef()))
    return Op1;

  // A ^ 0 = A
  if (match(Op1, m_Zero()))
    return Op0;

  // A ^ A = 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // A ^ ~A  =  ~A ^ A  =  -1
  if (match(Op0, m_Not(m_Specific(Op1))) ||
      match(Op1, m_Not(m_Specific(Op0))))
    return Constant::getAllOnesValue(Op0->getType());

  // Try some generic simplifications for associative operations.
  if (Value *V = SimplifyAssociativeBinOp(Instruction::Xor, Op0, Op1, TD, DT,
                                          MaxRecurse))
    return V;

  // And distributes over Xor.  Try some generic simplifications based on this.
  if (Value *V = FactorizeBinOp(Instruction::Xor, Op0, Op1, Instruction::And,
                                TD, DT, MaxRecurse))
    return V;

  // Threading Xor over selects and phi nodes is pointless, so don't bother.
  // Threading over the select in "A ^ select(cond, B, C)" means evaluating
  // "A^B" and "A^C" and seeing if they are equal; but they are equal if and
  // only if B and C are equal.  If B and C are equal then (since we assume
  // that operands have already been simplified) "select(cond, B, C)" should
  // have been simplified to the common value of B and C already.  Analysing
  // "A^B" and "A^C" thus gains nothing, but costs compile time.  Similarly
  // for threading over phi nodes.

  return 0;
}

Value *llvm::SimplifyXorInst(Value *Op0, Value *Op1, const TargetData *TD,
                             const DominatorTree *DT) {
  return ::SimplifyXorInst(Op0, Op1, TD, DT, RecursionLimit);
}

static Type *GetCompareTy(Value *Op) {
  return CmpInst::makeCmpResultType(Op->getType());
}

/// ExtractEquivalentCondition - Rummage around inside V looking for something
/// equivalent to the comparison "LHS Pred RHS".  Return such a value if found,
/// otherwise return null.  Helper function for analyzing max/min idioms.
static Value *ExtractEquivalentCondition(Value *V, CmpInst::Predicate Pred,
                                         Value *LHS, Value *RHS) {
  SelectInst *SI = dyn_cast<SelectInst>(V);
  if (!SI)
    return 0;
  CmpInst *Cmp = dyn_cast<CmpInst>(SI->getCondition());
  if (!Cmp)
    return 0;
  Value *CmpLHS = Cmp->getOperand(0), *CmpRHS = Cmp->getOperand(1);
  if (Pred == Cmp->getPredicate() && LHS == CmpLHS && RHS == CmpRHS)
    return Cmp;
  if (Pred == CmpInst::getSwappedPredicate(Cmp->getPredicate()) &&
      LHS == CmpRHS && RHS == CmpLHS)
    return Cmp;
  return 0;
}

/// SimplifyICmpInst - Given operands for an ICmpInst, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               const TargetData *TD, const DominatorTree *DT,
                               unsigned MaxRecurse) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isIntPredicate(Pred) && "Not an integer compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS)) {
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, TD);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }

  Type *ITy = GetCompareTy(LHS); // The return type.
  Type *OpTy = LHS->getType();   // The operand type.

  // icmp X, X -> true/false
  // X icmp undef -> true/false.  For example, icmp ugt %X, undef -> false
  // because X could be 0.
  if (LHS == RHS || isa<UndefValue>(RHS))
    return ConstantInt::get(ITy, CmpInst::isTrueWhenEqual(Pred));

  // Special case logic when the operands have i1 type.
  if (OpTy->isIntegerTy(1) || (OpTy->isVectorTy() &&
       cast<VectorType>(OpTy)->getElementType()->isIntegerTy(1))) {
    switch (Pred) {
    default: break;
    case ICmpInst::ICMP_EQ:
      // X == 1 -> X
      if (match(RHS, m_One()))
        return LHS;
      break;
    case ICmpInst::ICMP_NE:
      // X != 0 -> X
      if (match(RHS, m_Zero()))
        return LHS;
      break;
    case ICmpInst::ICMP_UGT:
      // X >u 0 -> X
      if (match(RHS, m_Zero()))
        return LHS;
      break;
    case ICmpInst::ICMP_UGE:
      // X >=u 1 -> X
      if (match(RHS, m_One()))
        return LHS;
      break;
    case ICmpInst::ICMP_SLT:
      // X <s 0 -> X
      if (match(RHS, m_Zero()))
        return LHS;
      break;
    case ICmpInst::ICMP_SLE:
      // X <=s -1 -> X
      if (match(RHS, m_One()))
        return LHS;
      break;
    }
  }

  // icmp <alloca*>, <global/alloca*/null> - Different stack variables have
  // different addresses, and what's more the address of a stack variable is
  // never null or equal to the address of a global.  Note that generalizing
  // to the case where LHS is a global variable address or null is pointless,
  // since if both LHS and RHS are constants then we already constant folded
  // the compare, and if only one of them is then we moved it to RHS already.
  if (isa<AllocaInst>(LHS) && (isa<GlobalValue>(RHS) || isa<AllocaInst>(RHS) ||
                               isa<ConstantPointerNull>(RHS)))
    // We already know that LHS != RHS.
    return ConstantInt::get(ITy, CmpInst::isFalseWhenEqual(Pred));

  // If we are comparing with zero then try hard since this is a common case.
  if (match(RHS, m_Zero())) {
    bool LHSKnownNonNegative, LHSKnownNegative;
    switch (Pred) {
    default:
      assert(false && "Unknown ICmp predicate!");
    case ICmpInst::ICMP_ULT:
      // getNullValue also works for vectors, unlike getFalse.
      return Constant::getNullValue(ITy);
    case ICmpInst::ICMP_UGE:
      // getAllOnesValue also works for vectors, unlike getTrue.
      return ConstantInt::getAllOnesValue(ITy);
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_ULE:
      if (isKnownNonZero(LHS, TD))
        return Constant::getNullValue(ITy);
      break;
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_UGT:
      if (isKnownNonZero(LHS, TD))
        return ConstantInt::getAllOnesValue(ITy);
      break;
    case ICmpInst::ICMP_SLT:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, TD);
      if (LHSKnownNegative)
        return ConstantInt::getAllOnesValue(ITy);
      if (LHSKnownNonNegative)
        return Constant::getNullValue(ITy);
      break;
    case ICmpInst::ICMP_SLE:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, TD);
      if (LHSKnownNegative)
        return ConstantInt::getAllOnesValue(ITy);
      if (LHSKnownNonNegative && isKnownNonZero(LHS, TD))
        return Constant::getNullValue(ITy);
      break;
    case ICmpInst::ICMP_SGE:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, TD);
      if (LHSKnownNegative)
        return Constant::getNullValue(ITy);
      if (LHSKnownNonNegative)
        return ConstantInt::getAllOnesValue(ITy);
      break;
    case ICmpInst::ICMP_SGT:
      ComputeSignBit(LHS, LHSKnownNonNegative, LHSKnownNegative, TD);
      if (LHSKnownNegative)
        return Constant::getNullValue(ITy);
      if (LHSKnownNonNegative && isKnownNonZero(LHS, TD))
        return ConstantInt::getAllOnesValue(ITy);
      break;
    }
  }

  // See if we are doing a comparison with a constant integer.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
    // Rule out tautological comparisons (eg., ult 0 or uge 0).
    ConstantRange RHS_CR = ICmpInst::makeConstantRange(Pred, CI->getValue());
    if (RHS_CR.isEmptySet())
      return ConstantInt::getFalse(CI->getContext());
    if (RHS_CR.isFullSet())
      return ConstantInt::getTrue(CI->getContext());

    // Many binary operators with constant RHS have easy to compute constant
    // range.  Use them to check whether the comparison is a tautology.
    uint32_t Width = CI->getBitWidth();
    APInt Lower = APInt(Width, 0);
    APInt Upper = APInt(Width, 0);
    ConstantInt *CI2;
    if (match(LHS, m_URem(m_Value(), m_ConstantInt(CI2)))) {
      // 'urem x, CI2' produces [0, CI2).
      Upper = CI2->getValue();
    } else if (match(LHS, m_SRem(m_Value(), m_ConstantInt(CI2)))) {
      // 'srem x, CI2' produces (-|CI2|, |CI2|).
      Upper = CI2->getValue().abs();
      Lower = (-Upper) + 1;
    } else if (match(LHS, m_UDiv(m_Value(), m_ConstantInt(CI2)))) {
      // 'udiv x, CI2' produces [0, UINT_MAX / CI2].
      APInt NegOne = APInt::getAllOnesValue(Width);
      if (!CI2->isZero())
        Upper = NegOne.udiv(CI2->getValue()) + 1;
    } else if (match(LHS, m_SDiv(m_Value(), m_ConstantInt(CI2)))) {
      // 'sdiv x, CI2' produces [INT_MIN / CI2, INT_MAX / CI2].
      APInt IntMin = APInt::getSignedMinValue(Width);
      APInt IntMax = APInt::getSignedMaxValue(Width);
      APInt Val = CI2->getValue().abs();
      if (!Val.isMinValue()) {
        Lower = IntMin.sdiv(Val);
        Upper = IntMax.sdiv(Val) + 1;
      }
    } else if (match(LHS, m_LShr(m_Value(), m_ConstantInt(CI2)))) {
      // 'lshr x, CI2' produces [0, UINT_MAX >> CI2].
      APInt NegOne = APInt::getAllOnesValue(Width);
      if (CI2->getValue().ult(Width))
        Upper = NegOne.lshr(CI2->getValue()) + 1;
    } else if (match(LHS, m_AShr(m_Value(), m_ConstantInt(CI2)))) {
      // 'ashr x, CI2' produces [INT_MIN >> CI2, INT_MAX >> CI2].
      APInt IntMin = APInt::getSignedMinValue(Width);
      APInt IntMax = APInt::getSignedMaxValue(Width);
      if (CI2->getValue().ult(Width)) {
        Lower = IntMin.ashr(CI2->getValue());
        Upper = IntMax.ashr(CI2->getValue()) + 1;
      }
    } else if (match(LHS, m_Or(m_Value(), m_ConstantInt(CI2)))) {
      // 'or x, CI2' produces [CI2, UINT_MAX].
      Lower = CI2->getValue();
    } else if (match(LHS, m_And(m_Value(), m_ConstantInt(CI2)))) {
      // 'and x, CI2' produces [0, CI2].
      Upper = CI2->getValue() + 1;
    }
    if (Lower != Upper) {
      ConstantRange LHS_CR = ConstantRange(Lower, Upper);
      if (RHS_CR.contains(LHS_CR))
        return ConstantInt::getTrue(RHS->getContext());
      if (RHS_CR.inverse().contains(LHS_CR))
        return ConstantInt::getFalse(RHS->getContext());
    }
  }

  // Compare of cast, for example (zext X) != 0 -> X != 0
  if (isa<CastInst>(LHS) && (isa<Constant>(RHS) || isa<CastInst>(RHS))) {
    Instruction *LI = cast<CastInst>(LHS);
    Value *SrcOp = LI->getOperand(0);
    Type *SrcTy = SrcOp->getType();
    Type *DstTy = LI->getType();

    // Turn icmp (ptrtoint x), (ptrtoint/constant) into a compare of the input
    // if the integer type is the same size as the pointer type.
    if (MaxRecurse && TD && isa<PtrToIntInst>(LI) &&
        TD->getPointerSizeInBits() == DstTy->getPrimitiveSizeInBits()) {
      if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
        // Transfer the cast to the constant.
        if (Value *V = SimplifyICmpInst(Pred, SrcOp,
                                        ConstantExpr::getIntToPtr(RHSC, SrcTy),
                                        TD, DT, MaxRecurse-1))
          return V;
      } else if (PtrToIntInst *RI = dyn_cast<PtrToIntInst>(RHS)) {
        if (RI->getOperand(0)->getType() == SrcTy)
          // Compare without the cast.
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, RI->getOperand(0),
                                          TD, DT, MaxRecurse-1))
            return V;
      }
    }

    if (isa<ZExtInst>(LHS)) {
      // Turn icmp (zext X), (zext Y) into a compare of X and Y if they have the
      // same type.
      if (ZExtInst *RI = dyn_cast<ZExtInst>(RHS)) {
        if (MaxRecurse && SrcTy == RI->getOperand(0)->getType())
          // Compare X and Y.  Note that signed predicates become unsigned.
          if (Value *V = SimplifyICmpInst(ICmpInst::getUnsignedPredicate(Pred),
                                          SrcOp, RI->getOperand(0), TD, DT,
                                          MaxRecurse-1))
            return V;
      }
      // Turn icmp (zext X), Cst into a compare of X and Cst if Cst is extended
      // too.  If not, then try to deduce the result of the comparison.
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
        // Compute the constant that would happen if we truncated to SrcTy then
        // reextended to DstTy.
        Constant *Trunc = ConstantExpr::getTrunc(CI, SrcTy);
        Constant *RExt = ConstantExpr::getCast(CastInst::ZExt, Trunc, DstTy);

        // If the re-extended constant didn't change then this is effectively
        // also a case of comparing two zero-extended values.
        if (RExt == CI && MaxRecurse)
          if (Value *V = SimplifyICmpInst(ICmpInst::getUnsignedPredicate(Pred),
                                          SrcOp, Trunc, TD, DT, MaxRecurse-1))
            return V;

        // Otherwise the upper bits of LHS are zero while RHS has a non-zero bit
        // there.  Use this to work out the result of the comparison.
        if (RExt != CI) {
          switch (Pred) {
          default:
            assert(false && "Unknown ICmp predicate!");
          // LHS <u RHS.
          case ICmpInst::ICMP_EQ:
          case ICmpInst::ICMP_UGT:
          case ICmpInst::ICMP_UGE:
            return ConstantInt::getFalse(CI->getContext());

          case ICmpInst::ICMP_NE:
          case ICmpInst::ICMP_ULT:
          case ICmpInst::ICMP_ULE:
            return ConstantInt::getTrue(CI->getContext());

          // LHS is non-negative.  If RHS is negative then LHS >s LHS.  If RHS
          // is non-negative then LHS <s RHS.
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_SGE:
            return CI->getValue().isNegative() ?
              ConstantInt::getTrue(CI->getContext()) :
              ConstantInt::getFalse(CI->getContext());

          case ICmpInst::ICMP_SLT:
          case ICmpInst::ICMP_SLE:
            return CI->getValue().isNegative() ?
              ConstantInt::getFalse(CI->getContext()) :
              ConstantInt::getTrue(CI->getContext());
          }
        }
      }
    }

    if (isa<SExtInst>(LHS)) {
      // Turn icmp (sext X), (sext Y) into a compare of X and Y if they have the
      // same type.
      if (SExtInst *RI = dyn_cast<SExtInst>(RHS)) {
        if (MaxRecurse && SrcTy == RI->getOperand(0)->getType())
          // Compare X and Y.  Note that the predicate does not change.
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, RI->getOperand(0),
                                          TD, DT, MaxRecurse-1))
            return V;
      }
      // Turn icmp (sext X), Cst into a compare of X and Cst if Cst is extended
      // too.  If not, then try to deduce the result of the comparison.
      else if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
        // Compute the constant that would happen if we truncated to SrcTy then
        // reextended to DstTy.
        Constant *Trunc = ConstantExpr::getTrunc(CI, SrcTy);
        Constant *RExt = ConstantExpr::getCast(CastInst::SExt, Trunc, DstTy);

        // If the re-extended constant didn't change then this is effectively
        // also a case of comparing two sign-extended values.
        if (RExt == CI && MaxRecurse)
          if (Value *V = SimplifyICmpInst(Pred, SrcOp, Trunc, TD, DT,
                                          MaxRecurse-1))
            return V;

        // Otherwise the upper bits of LHS are all equal, while RHS has varying
        // bits there.  Use this to work out the result of the comparison.
        if (RExt != CI) {
          switch (Pred) {
          default:
            assert(false && "Unknown ICmp predicate!");
          case ICmpInst::ICMP_EQ:
            return ConstantInt::getFalse(CI->getContext());
          case ICmpInst::ICMP_NE:
            return ConstantInt::getTrue(CI->getContext());

          // If RHS is non-negative then LHS <s RHS.  If RHS is negative then
          // LHS >s RHS.
          case ICmpInst::ICMP_SGT:
          case ICmpInst::ICMP_SGE:
            return CI->getValue().isNegative() ?
              ConstantInt::getTrue(CI->getContext()) :
              ConstantInt::getFalse(CI->getContext());
          case ICmpInst::ICMP_SLT:
          case ICmpInst::ICMP_SLE:
            return CI->getValue().isNegative() ?
              ConstantInt::getFalse(CI->getContext()) :
              ConstantInt::getTrue(CI->getContext());

          // If LHS is non-negative then LHS <u RHS.  If LHS is negative then
          // LHS >u RHS.
          case ICmpInst::ICMP_UGT:
          case ICmpInst::ICMP_UGE:
            // Comparison is true iff the LHS <s 0.
            if (MaxRecurse)
              if (Value *V = SimplifyICmpInst(ICmpInst::ICMP_SLT, SrcOp,
                                              Constant::getNullValue(SrcTy),
                                              TD, DT, MaxRecurse-1))
                return V;
            break;
          case ICmpInst::ICMP_ULT:
          case ICmpInst::ICMP_ULE:
            // Comparison is true iff the LHS >=s 0.
            if (MaxRecurse)
              if (Value *V = SimplifyICmpInst(ICmpInst::ICMP_SGE, SrcOp,
                                              Constant::getNullValue(SrcTy),
                                              TD, DT, MaxRecurse-1))
                return V;
            break;
          }
        }
      }
    }
  }

  // Special logic for binary operators.
  BinaryOperator *LBO = dyn_cast<BinaryOperator>(LHS);
  BinaryOperator *RBO = dyn_cast<BinaryOperator>(RHS);
  if (MaxRecurse && (LBO || RBO)) {
    // Analyze the case when either LHS or RHS is an add instruction.
    Value *A = 0, *B = 0, *C = 0, *D = 0;
    // LHS = A + B (or A and B are null); RHS = C + D (or C and D are null).
    bool NoLHSWrapProblem = false, NoRHSWrapProblem = false;
    if (LBO && LBO->getOpcode() == Instruction::Add) {
      A = LBO->getOperand(0); B = LBO->getOperand(1);
      NoLHSWrapProblem = ICmpInst::isEquality(Pred) ||
        (CmpInst::isUnsigned(Pred) && LBO->hasNoUnsignedWrap()) ||
        (CmpInst::isSigned(Pred) && LBO->hasNoSignedWrap());
    }
    if (RBO && RBO->getOpcode() == Instruction::Add) {
      C = RBO->getOperand(0); D = RBO->getOperand(1);
      NoRHSWrapProblem = ICmpInst::isEquality(Pred) ||
        (CmpInst::isUnsigned(Pred) && RBO->hasNoUnsignedWrap()) ||
        (CmpInst::isSigned(Pred) && RBO->hasNoSignedWrap());
    }

    // icmp (X+Y), X -> icmp Y, 0 for equalities or if there is no overflow.
    if ((A == RHS || B == RHS) && NoLHSWrapProblem)
      if (Value *V = SimplifyICmpInst(Pred, A == RHS ? B : A,
                                      Constant::getNullValue(RHS->getType()),
                                      TD, DT, MaxRecurse-1))
        return V;

    // icmp X, (X+Y) -> icmp 0, Y for equalities or if there is no overflow.
    if ((C == LHS || D == LHS) && NoRHSWrapProblem)
      if (Value *V = SimplifyICmpInst(Pred,
                                      Constant::getNullValue(LHS->getType()),
                                      C == LHS ? D : C, TD, DT, MaxRecurse-1))
        return V;

    // icmp (X+Y), (X+Z) -> icmp Y,Z for equalities or if there is no overflow.
    if (A && C && (A == C || A == D || B == C || B == D) &&
        NoLHSWrapProblem && NoRHSWrapProblem) {
      // Determine Y and Z in the form icmp (X+Y), (X+Z).
      Value *Y = (A == C || A == D) ? B : A;
      Value *Z = (C == A || C == B) ? D : C;
      if (Value *V = SimplifyICmpInst(Pred, Y, Z, TD, DT, MaxRecurse-1))
        return V;
    }
  }

  if (LBO && match(LBO, m_URem(m_Value(), m_Specific(RHS)))) {
    bool KnownNonNegative, KnownNegative;
    switch (Pred) {
    default:
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE:
      ComputeSignBit(LHS, KnownNonNegative, KnownNegative, TD);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE:
      // getNullValue also works for vectors, unlike getFalse.
      return Constant::getNullValue(ITy);
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE:
      ComputeSignBit(LHS, KnownNonNegative, KnownNegative, TD);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE:
      // getAllOnesValue also works for vectors, unlike getTrue.
      return Constant::getAllOnesValue(ITy);
    }
  }
  if (RBO && match(RBO, m_URem(m_Value(), m_Specific(LHS)))) {
    bool KnownNonNegative, KnownNegative;
    switch (Pred) {
    default:
      break;
    case ICmpInst::ICMP_SGT:
    case ICmpInst::ICMP_SGE:
      ComputeSignBit(RHS, KnownNonNegative, KnownNegative, TD);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_NE:
    case ICmpInst::ICMP_UGT:
    case ICmpInst::ICMP_UGE:
      // getAllOnesValue also works for vectors, unlike getTrue.
      return Constant::getAllOnesValue(ITy);
    case ICmpInst::ICMP_SLT:
    case ICmpInst::ICMP_SLE:
      ComputeSignBit(RHS, KnownNonNegative, KnownNegative, TD);
      if (!KnownNonNegative)
        break;
      // fall-through
    case ICmpInst::ICMP_EQ:
    case ICmpInst::ICMP_ULT:
    case ICmpInst::ICMP_ULE:
      // getNullValue also works for vectors, unlike getFalse.
      return Constant::getNullValue(ITy);
    }
  }

  if (MaxRecurse && LBO && RBO && LBO->getOpcode() == RBO->getOpcode() &&
      LBO->getOperand(1) == RBO->getOperand(1)) {
    switch (LBO->getOpcode()) {
    default: break;
    case Instruction::UDiv:
    case Instruction::LShr:
      if (ICmpInst::isSigned(Pred))
        break;
      // fall-through
    case Instruction::SDiv:
    case Instruction::AShr:
      if (!LBO->isExact() || !RBO->isExact())
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), TD, DT, MaxRecurse-1))
        return V;
      break;
    case Instruction::Shl: {
      bool NUW = LBO->hasNoUnsignedWrap() && LBO->hasNoUnsignedWrap();
      bool NSW = LBO->hasNoSignedWrap() && RBO->hasNoSignedWrap();
      if (!NUW && !NSW)
        break;
      if (!NSW && ICmpInst::isSigned(Pred))
        break;
      if (Value *V = SimplifyICmpInst(Pred, LBO->getOperand(0),
                                      RBO->getOperand(0), TD, DT, MaxRecurse-1))
        return V;
      break;
    }
    }
  }

  // Simplify comparisons involving max/min.
  Value *A, *B;
  CmpInst::Predicate P = CmpInst::BAD_ICMP_PREDICATE;
  CmpInst::Predicate EqP; // Chosen so that "A == max/min(A,B)" iff "A EqP B".

  // Signed variants on "max(a,b)>=a -> true".
  if (match(LHS, m_SMax(m_Value(A), m_Value(B))) && (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // smax(A, B) pred A.
    EqP = CmpInst::ICMP_SGE; // "A == smax(A, B)" iff "A sge B".
    // We analyze this as smax(A, B) pred A.
    P = Pred;
  } else if (match(RHS, m_SMax(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred smax(A, B).
    EqP = CmpInst::ICMP_SGE; // "A == smax(A, B)" iff "A sge B".
    // We analyze this as smax(A, B) swapped-pred A.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(LHS, m_SMin(m_Value(A), m_Value(B))) &&
             (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // smin(A, B) pred A.
    EqP = CmpInst::ICMP_SLE; // "A == smin(A, B)" iff "A sle B".
    // We analyze this as smax(-A, -B) swapped-pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(RHS, m_SMin(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred smin(A, B).
    EqP = CmpInst::ICMP_SLE; // "A == smin(A, B)" iff "A sle B".
    // We analyze this as smax(-A, -B) pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = Pred;
  }
  if (P != CmpInst::BAD_ICMP_PREDICATE) {
    // Cases correspond to "max(A, B) p A".
    switch (P) {
    default:
      break;
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_SLE:
      // Equivalent to "A EqP B".  This may be the same as the condition tested
      // in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, EqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, EqP, A, B))
        return V;
      // Otherwise, see if "A EqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(EqP, A, B, TD, DT, MaxRecurse-1))
          return V;
      break;
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_SGT: {
      CmpInst::Predicate InvEqP = CmpInst::getInversePredicate(EqP);
      // Equivalent to "A InvEqP B".  This may be the same as the condition
      // tested in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, InvEqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, InvEqP, A, B))
        return V;
      // Otherwise, see if "A InvEqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(InvEqP, A, B, TD, DT, MaxRecurse-1))
          return V;
      break;
    }
    case CmpInst::ICMP_SGE:
      // Always true.
      return Constant::getAllOnesValue(ITy);
    case CmpInst::ICMP_SLT:
      // Always false.
      return Constant::getNullValue(ITy);
    }
  }

  // Unsigned variants on "max(a,b)>=a -> true".
  P = CmpInst::BAD_ICMP_PREDICATE;
  if (match(LHS, m_UMax(m_Value(A), m_Value(B))) && (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // umax(A, B) pred A.
    EqP = CmpInst::ICMP_UGE; // "A == umax(A, B)" iff "A uge B".
    // We analyze this as umax(A, B) pred A.
    P = Pred;
  } else if (match(RHS, m_UMax(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred umax(A, B).
    EqP = CmpInst::ICMP_UGE; // "A == umax(A, B)" iff "A uge B".
    // We analyze this as umax(A, B) swapped-pred A.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(LHS, m_UMin(m_Value(A), m_Value(B))) &&
             (A == RHS || B == RHS)) {
    if (A != RHS) std::swap(A, B); // umin(A, B) pred A.
    EqP = CmpInst::ICMP_ULE; // "A == umin(A, B)" iff "A ule B".
    // We analyze this as umax(-A, -B) swapped-pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = CmpInst::getSwappedPredicate(Pred);
  } else if (match(RHS, m_UMin(m_Value(A), m_Value(B))) &&
             (A == LHS || B == LHS)) {
    if (A != LHS) std::swap(A, B); // A pred umin(A, B).
    EqP = CmpInst::ICMP_ULE; // "A == umin(A, B)" iff "A ule B".
    // We analyze this as umax(-A, -B) pred -A.
    // Note that we do not need to actually form -A or -B thanks to EqP.
    P = Pred;
  }
  if (P != CmpInst::BAD_ICMP_PREDICATE) {
    // Cases correspond to "max(A, B) p A".
    switch (P) {
    default:
      break;
    case CmpInst::ICMP_EQ:
    case CmpInst::ICMP_ULE:
      // Equivalent to "A EqP B".  This may be the same as the condition tested
      // in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, EqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, EqP, A, B))
        return V;
      // Otherwise, see if "A EqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(EqP, A, B, TD, DT, MaxRecurse-1))
          return V;
      break;
    case CmpInst::ICMP_NE:
    case CmpInst::ICMP_UGT: {
      CmpInst::Predicate InvEqP = CmpInst::getInversePredicate(EqP);
      // Equivalent to "A InvEqP B".  This may be the same as the condition
      // tested in the max/min; if so, we can just return that.
      if (Value *V = ExtractEquivalentCondition(LHS, InvEqP, A, B))
        return V;
      if (Value *V = ExtractEquivalentCondition(RHS, InvEqP, A, B))
        return V;
      // Otherwise, see if "A InvEqP B" simplifies.
      if (MaxRecurse)
        if (Value *V = SimplifyICmpInst(InvEqP, A, B, TD, DT, MaxRecurse-1))
          return V;
      break;
    }
    case CmpInst::ICMP_UGE:
      // Always true.
      return Constant::getAllOnesValue(ITy);
    case CmpInst::ICMP_ULT:
      // Always false.
      return Constant::getNullValue(ITy);
    }
  }

  // Variants on "max(x,y) >= min(x,z)".
  Value *C, *D;
  if (match(LHS, m_SMax(m_Value(A), m_Value(B))) &&
      match(RHS, m_SMin(m_Value(C), m_Value(D))) &&
      (A == C || A == D || B == C || B == D)) {
    // max(x, ?) pred min(x, ?).
    if (Pred == CmpInst::ICMP_SGE)
      // Always true.
      return Constant::getAllOnesValue(ITy);
    if (Pred == CmpInst::ICMP_SLT)
      // Always false.
      return Constant::getNullValue(ITy);
  } else if (match(LHS, m_SMin(m_Value(A), m_Value(B))) &&
             match(RHS, m_SMax(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // min(x, ?) pred max(x, ?).
    if (Pred == CmpInst::ICMP_SLE)
      // Always true.
      return Constant::getAllOnesValue(ITy);
    if (Pred == CmpInst::ICMP_SGT)
      // Always false.
      return Constant::getNullValue(ITy);
  } else if (match(LHS, m_UMax(m_Value(A), m_Value(B))) &&
             match(RHS, m_UMin(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // max(x, ?) pred min(x, ?).
    if (Pred == CmpInst::ICMP_UGE)
      // Always true.
      return Constant::getAllOnesValue(ITy);
    if (Pred == CmpInst::ICMP_ULT)
      // Always false.
      return Constant::getNullValue(ITy);
  } else if (match(LHS, m_UMin(m_Value(A), m_Value(B))) &&
             match(RHS, m_UMax(m_Value(C), m_Value(D))) &&
             (A == C || A == D || B == C || B == D)) {
    // min(x, ?) pred max(x, ?).
    if (Pred == CmpInst::ICMP_ULE)
      // Always true.
      return Constant::getAllOnesValue(ITy);
    if (Pred == CmpInst::ICMP_UGT)
      // Always false.
      return Constant::getNullValue(ITy);
  }

  // If the comparison is with the result of a select instruction, check whether
  // comparing with either branch of the select always yields the same value.
  if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
    if (Value *V = ThreadCmpOverSelect(Pred, LHS, RHS, TD, DT, MaxRecurse))
      return V;

  // If the comparison is with the result of a phi instruction, check whether
  // doing the compare with each incoming phi value yields a common result.
  if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
    if (Value *V = ThreadCmpOverPHI(Pred, LHS, RHS, TD, DT, MaxRecurse))
      return V;

  return 0;
}

Value *llvm::SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyICmpInst(Predicate, LHS, RHS, TD, DT, RecursionLimit);
}

/// SimplifyFCmpInst - Given operands for an FCmpInst, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                               const TargetData *TD, const DominatorTree *DT,
                               unsigned MaxRecurse) {
  CmpInst::Predicate Pred = (CmpInst::Predicate)Predicate;
  assert(CmpInst::isFPPredicate(Pred) && "Not an FP compare!");

  if (Constant *CLHS = dyn_cast<Constant>(LHS)) {
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantFoldCompareInstOperands(Pred, CLHS, CRHS, TD);

    // If we have a constant, make sure it is on the RHS.
    std::swap(LHS, RHS);
    Pred = CmpInst::getSwappedPredicate(Pred);
  }

  // Fold trivial predicates.
  if (Pred == FCmpInst::FCMP_FALSE)
    return ConstantInt::get(GetCompareTy(LHS), 0);
  if (Pred == FCmpInst::FCMP_TRUE)
    return ConstantInt::get(GetCompareTy(LHS), 1);

  if (isa<UndefValue>(RHS))                  // fcmp pred X, undef -> undef
    return UndefValue::get(GetCompareTy(LHS));

  // fcmp x,x -> true/false.  Not all compares are foldable.
  if (LHS == RHS) {
    if (CmpInst::isTrueWhenEqual(Pred))
      return ConstantInt::get(GetCompareTy(LHS), 1);
    if (CmpInst::isFalseWhenEqual(Pred))
      return ConstantInt::get(GetCompareTy(LHS), 0);
  }

  // Handle fcmp with constant RHS
  if (Constant *RHSC = dyn_cast<Constant>(RHS)) {
    // If the constant is a nan, see if we can fold the comparison based on it.
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(RHSC)) {
      if (CFP->getValueAPF().isNaN()) {
        if (FCmpInst::isOrdered(Pred))   // True "if ordered and foo"
          return ConstantInt::getFalse(CFP->getContext());
        assert(FCmpInst::isUnordered(Pred) &&
               "Comparison must be either ordered or unordered!");
        // True if unordered.
        return ConstantInt::getTrue(CFP->getContext());
      }
      // Check whether the constant is an infinity.
      if (CFP->getValueAPF().isInfinity()) {
        if (CFP->getValueAPF().isNegative()) {
          switch (Pred) {
          case FCmpInst::FCMP_OLT:
            // No value is ordered and less than negative infinity.
            return ConstantInt::getFalse(CFP->getContext());
          case FCmpInst::FCMP_UGE:
            // All values are unordered with or at least negative infinity.
            return ConstantInt::getTrue(CFP->getContext());
          default:
            break;
          }
        } else {
          switch (Pred) {
          case FCmpInst::FCMP_OGT:
            // No value is ordered and greater than infinity.
            return ConstantInt::getFalse(CFP->getContext());
          case FCmpInst::FCMP_ULE:
            // All values are unordered with and at most infinity.
            return ConstantInt::getTrue(CFP->getContext());
          default:
            break;
          }
        }
      }
    }
  }

  // If the comparison is with the result of a select instruction, check whether
  // comparing with either branch of the select always yields the same value.
  if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
    if (Value *V = ThreadCmpOverSelect(Pred, LHS, RHS, TD, DT, MaxRecurse))
      return V;

  // If the comparison is with the result of a phi instruction, check whether
  // doing the compare with each incoming phi value yields a common result.
  if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
    if (Value *V = ThreadCmpOverPHI(Pred, LHS, RHS, TD, DT, MaxRecurse))
      return V;

  return 0;
}

Value *llvm::SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyFCmpInst(Predicate, LHS, RHS, TD, DT, RecursionLimit);
}

/// SimplifySelectInst - Given operands for a SelectInst, see if we can fold
/// the result.  If not, this returns null.
Value *llvm::SimplifySelectInst(Value *CondVal, Value *TrueVal, Value *FalseVal,
                                const TargetData *TD, const DominatorTree *) {
  // select true, X, Y  -> X
  // select false, X, Y -> Y
  if (ConstantInt *CB = dyn_cast<ConstantInt>(CondVal))
    return CB->getZExtValue() ? TrueVal : FalseVal;

  // select C, X, X -> X
  if (TrueVal == FalseVal)
    return TrueVal;

  if (isa<UndefValue>(CondVal)) {  // select undef, X, Y -> X or Y
    if (isa<Constant>(TrueVal))
      return TrueVal;
    return FalseVal;
  }
  if (isa<UndefValue>(TrueVal))   // select C, undef, X -> X
    return FalseVal;
  if (isa<UndefValue>(FalseVal))   // select C, X, undef -> X
    return TrueVal;

  return 0;
}

/// SimplifyGEPInst - Given operands for an GetElementPtrInst, see if we can
/// fold the result.  If not, this returns null.
Value *llvm::SimplifyGEPInst(ArrayRef<Value *> Ops,
                             const TargetData *TD, const DominatorTree *) {
  // The type of the GEP pointer operand.
  PointerType *PtrTy = cast<PointerType>(Ops[0]->getType());

  // getelementptr P -> P.
  if (Ops.size() == 1)
    return Ops[0];

  if (isa<UndefValue>(Ops[0])) {
    // Compute the (pointer) type returned by the GEP instruction.
    Type *LastType = GetElementPtrInst::getIndexedType(PtrTy, Ops.slice(1));
    Type *GEPTy = PointerType::get(LastType, PtrTy->getAddressSpace());
    return UndefValue::get(GEPTy);
  }

  if (Ops.size() == 2) {
    // getelementptr P, 0 -> P.
    if (ConstantInt *C = dyn_cast<ConstantInt>(Ops[1]))
      if (C->isZero())
        return Ops[0];
    // getelementptr P, N -> P if P points to a type of zero size.
    if (TD) {
      Type *Ty = PtrTy->getElementType();
      if (Ty->isSized() && TD->getTypeAllocSize(Ty) == 0)
        return Ops[0];
    }
  }

  // Check to see if this is constant foldable.
  for (unsigned i = 0, e = Ops.size(); i != e; ++i)
    if (!isa<Constant>(Ops[i]))
      return 0;

  return ConstantExpr::getGetElementPtr(cast<Constant>(Ops[0]), Ops.slice(1));
}

/// SimplifyPHINode - See if we can fold the given phi.  If not, returns null.
static Value *SimplifyPHINode(PHINode *PN, const DominatorTree *DT) {
  // If all of the PHI's incoming values are the same then replace the PHI node
  // with the common value.
  Value *CommonValue = 0;
  bool HasUndefInput = false;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    Value *Incoming = PN->getIncomingValue(i);
    // If the incoming value is the phi node itself, it can safely be skipped.
    if (Incoming == PN) continue;
    if (isa<UndefValue>(Incoming)) {
      // Remember that we saw an undef value, but otherwise ignore them.
      HasUndefInput = true;
      continue;
    }
    if (CommonValue && Incoming != CommonValue)
      return 0;  // Not the same, bail out.
    CommonValue = Incoming;
  }

  // If CommonValue is null then all of the incoming values were either undef or
  // equal to the phi node itself.
  if (!CommonValue)
    return UndefValue::get(PN->getType());

  // If we have a PHI node like phi(X, undef, X), where X is defined by some
  // instruction, we cannot return X as the result of the PHI node unless it
  // dominates the PHI block.
  if (HasUndefInput)
    return ValueDominatesPHI(CommonValue, PN, DT) ? CommonValue : 0;

  return CommonValue;
}


//=== Helper functions for higher up the class hierarchy.

/// SimplifyBinOp - Given operands for a BinaryOperator, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                            const TargetData *TD, const DominatorTree *DT,
                            unsigned MaxRecurse) {
  switch (Opcode) {
  case Instruction::Add:
    return SimplifyAddInst(LHS, RHS, /*isNSW*/false, /*isNUW*/false,
                           TD, DT, MaxRecurse);
  case Instruction::Sub:
    return SimplifySubInst(LHS, RHS, /*isNSW*/false, /*isNUW*/false,
                           TD, DT, MaxRecurse);
  case Instruction::Mul:  return SimplifyMulInst (LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::SDiv: return SimplifySDivInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::UDiv: return SimplifyUDivInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::FDiv: return SimplifyFDivInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::SRem: return SimplifySRemInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::URem: return SimplifyURemInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::FRem: return SimplifyFRemInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::Shl:
    return SimplifyShlInst(LHS, RHS, /*isNSW*/false, /*isNUW*/false,
                           TD, DT, MaxRecurse);
  case Instruction::LShr:
    return SimplifyLShrInst(LHS, RHS, /*isExact*/false, TD, DT, MaxRecurse);
  case Instruction::AShr:
    return SimplifyAShrInst(LHS, RHS, /*isExact*/false, TD, DT, MaxRecurse);
  case Instruction::And: return SimplifyAndInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::Or:  return SimplifyOrInst (LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::Xor: return SimplifyXorInst(LHS, RHS, TD, DT, MaxRecurse);
  default:
    if (Constant *CLHS = dyn_cast<Constant>(LHS))
      if (Constant *CRHS = dyn_cast<Constant>(RHS)) {
        Constant *COps[] = {CLHS, CRHS};
        return ConstantFoldInstOperands(Opcode, LHS->getType(), COps, TD);
      }

    // If the operation is associative, try some generic simplifications.
    if (Instruction::isAssociative(Opcode))
      if (Value *V = SimplifyAssociativeBinOp(Opcode, LHS, RHS, TD, DT,
                                              MaxRecurse))
        return V;

    // If the operation is with the result of a select instruction, check whether
    // operating on either branch of the select always yields the same value.
    if (isa<SelectInst>(LHS) || isa<SelectInst>(RHS))
      if (Value *V = ThreadBinOpOverSelect(Opcode, LHS, RHS, TD, DT,
                                           MaxRecurse))
        return V;

    // If the operation is with the result of a phi instruction, check whether
    // operating on all incoming values of the phi always yields the same value.
    if (isa<PHINode>(LHS) || isa<PHINode>(RHS))
      if (Value *V = ThreadBinOpOverPHI(Opcode, LHS, RHS, TD, DT, MaxRecurse))
        return V;

    return 0;
  }
}

Value *llvm::SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                           const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyBinOp(Opcode, LHS, RHS, TD, DT, RecursionLimit);
}

/// SimplifyCmpInst - Given operands for a CmpInst, see if we can
/// fold the result.
static Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                              const TargetData *TD, const DominatorTree *DT,
                              unsigned MaxRecurse) {
  if (CmpInst::isIntPredicate((CmpInst::Predicate)Predicate))
    return SimplifyICmpInst(Predicate, LHS, RHS, TD, DT, MaxRecurse);
  return SimplifyFCmpInst(Predicate, LHS, RHS, TD, DT, MaxRecurse);
}

Value *llvm::SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                             const TargetData *TD, const DominatorTree *DT) {
  return ::SimplifyCmpInst(Predicate, LHS, RHS, TD, DT, RecursionLimit);
}

/// SimplifyInstruction - See if we can compute a simplified version of this
/// instruction.  If not, this returns null.
Value *llvm::SimplifyInstruction(Instruction *I, const TargetData *TD,
                                 const DominatorTree *DT) {
  Value *Result;

  switch (I->getOpcode()) {
  default:
    Result = ConstantFoldInstruction(I, TD);
    break;
  case Instruction::Add:
    Result = SimplifyAddInst(I->getOperand(0), I->getOperand(1),
                             cast<BinaryOperator>(I)->hasNoSignedWrap(),
                             cast<BinaryOperator>(I)->hasNoUnsignedWrap(),
                             TD, DT);
    break;
  case Instruction::Sub:
    Result = SimplifySubInst(I->getOperand(0), I->getOperand(1),
                             cast<BinaryOperator>(I)->hasNoSignedWrap(),
                             cast<BinaryOperator>(I)->hasNoUnsignedWrap(),
                             TD, DT);
    break;
  case Instruction::Mul:
    Result = SimplifyMulInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::SDiv:
    Result = SimplifySDivInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::UDiv:
    Result = SimplifyUDivInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::FDiv:
    Result = SimplifyFDivInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::SRem:
    Result = SimplifySRemInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::URem:
    Result = SimplifyURemInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::FRem:
    Result = SimplifyFRemInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::Shl:
    Result = SimplifyShlInst(I->getOperand(0), I->getOperand(1),
                             cast<BinaryOperator>(I)->hasNoSignedWrap(),
                             cast<BinaryOperator>(I)->hasNoUnsignedWrap(),
                             TD, DT);
    break;
  case Instruction::LShr:
    Result = SimplifyLShrInst(I->getOperand(0), I->getOperand(1),
                              cast<BinaryOperator>(I)->isExact(),
                              TD, DT);
    break;
  case Instruction::AShr:
    Result = SimplifyAShrInst(I->getOperand(0), I->getOperand(1),
                              cast<BinaryOperator>(I)->isExact(),
                              TD, DT);
    break;
  case Instruction::And:
    Result = SimplifyAndInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::Or:
    Result = SimplifyOrInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::Xor:
    Result = SimplifyXorInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::ICmp:
    Result = SimplifyICmpInst(cast<ICmpInst>(I)->getPredicate(),
                              I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::FCmp:
    Result = SimplifyFCmpInst(cast<FCmpInst>(I)->getPredicate(),
                              I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::Select:
    Result = SimplifySelectInst(I->getOperand(0), I->getOperand(1),
                                I->getOperand(2), TD, DT);
    break;
  case Instruction::GetElementPtr: {
    SmallVector<Value*, 8> Ops(I->op_begin(), I->op_end());
    Result = SimplifyGEPInst(Ops, TD, DT);
    break;
  }
  case Instruction::PHI:
    Result = SimplifyPHINode(cast<PHINode>(I), DT);
    break;
  }

  /// If called on unreachable code, the above logic may report that the
  /// instruction simplified to itself.  Make life easier for users by
  /// detecting that case here, returning a safe value instead.
  return Result == I ? UndefValue::get(I->getType()) : Result;
}

/// ReplaceAndSimplifyAllUses - Perform From->replaceAllUsesWith(To) and then
/// delete the From instruction.  In addition to a basic RAUW, this does a
/// recursive simplification of the newly formed instructions.  This catches
/// things where one simplification exposes other opportunities.  This only
/// simplifies and deletes scalar operations, it does not change the CFG.
///
void llvm::ReplaceAndSimplifyAllUses(Instruction *From, Value *To,
                                     const TargetData *TD,
                                     const DominatorTree *DT) {
  assert(From != To && "ReplaceAndSimplifyAllUses(X,X) is not valid!");

  // FromHandle/ToHandle - This keeps a WeakVH on the from/to values so that
  // we can know if it gets deleted out from under us or replaced in a
  // recursive simplification.
  WeakVH FromHandle(From);
  WeakVH ToHandle(To);

  while (!From->use_empty()) {
    // Update the instruction to use the new value.
    Use &TheUse = From->use_begin().getUse();
    Instruction *User = cast<Instruction>(TheUse.getUser());
    TheUse = To;

    // Check to see if the instruction can be folded due to the operand
    // replacement.  For example changing (or X, Y) into (or X, -1) can replace
    // the 'or' with -1.
    Value *SimplifiedVal;
    {
      // Sanity check to make sure 'User' doesn't dangle across
      // SimplifyInstruction.
      AssertingVH<> UserHandle(User);

      SimplifiedVal = SimplifyInstruction(User, TD, DT);
      if (SimplifiedVal == 0) continue;
    }

    // Recursively simplify this user to the new value.
    ReplaceAndSimplifyAllUses(User, SimplifiedVal, TD, DT);
    From = dyn_cast_or_null<Instruction>((Value*)FromHandle);
    To = ToHandle;

    assert(ToHandle && "To value deleted by recursive simplification?");

    // If the recursive simplification ended up revisiting and deleting
    // 'From' then we're done.
    if (From == 0)
      return;
  }

  // If 'From' has value handles referring to it, do a real RAUW to update them.
  From->replaceAllUsesWith(To);

  From->eraseFromParent();
}
