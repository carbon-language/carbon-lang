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
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Dominators.h"
#include "llvm/Support/PatternMatch.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Target/TargetData.h"
using namespace llvm;
using namespace llvm::PatternMatch;

#define RecursionLimit 3

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

  // Now that we have "cmp select(cond, TV, FV), RHS", analyse it.
  // Does "cmp TV, RHS" simplify?
  if (Value *TCmp = SimplifyCmpInst(Pred, SI->getTrueValue(), RHS, TD, DT,
                                    MaxRecurse))
    // It does!  Does "cmp FV, RHS" simplify?
    if (Value *FCmp = SimplifyCmpInst(Pred, SI->getFalseValue(), RHS, TD, DT,
                                      MaxRecurse))
      // It does!  If they simplified to the same value, then use it as the
      // result of the original comparison.
      if (TCmp == FCmp)
        return TCmp;
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
                                      Ops, 2, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X + undef -> undef
  if (isa<UndefValue>(Op1))
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
                                      Ops, 2, TD);
    }

  // X - undef -> undef
  // undef - X -> undef
  if (isa<UndefValue>(Op0) || isa<UndefValue>(Op1))
    return UndefValue::get(Op0->getType());

  // X - 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X - X -> 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // (X + Y) - Y -> X
  // (Y + X) - Y -> X
  Value *X = 0;
  if (match(Op0, m_Add(m_Value(X), m_Specific(Op1))) ||
      match(Op0, m_Add(m_Specific(Op1), m_Value(X))))
    return X;

  /// i1 sub -> xor.
  if (MaxRecurse && Op0->getType()->isIntegerTy(1))
    if (Value *V = SimplifyXorInst(Op0, Op1, TD, DT, MaxRecurse-1))
      return V;

  // Mul distributes over Sub.  Try some generic simplifications based on this.
  if (Value *V = FactorizeBinOp(Instruction::Sub, Op0, Op1, Instruction::Mul,
                                TD, DT, MaxRecurse))
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
                                      Ops, 2, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X * undef -> 0
  if (isa<UndefValue>(Op1))
    return Constant::getNullValue(Op0->getType());

  // X * 0 -> 0
  if (match(Op1, m_Zero()))
    return Op1;

  // X * 1 -> X
  if (match(Op1, m_One()))
    return Op0;

  /// i1 mul -> and.
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

/// SimplifyShift - Given operands for an Shl, LShr or AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyShift(unsigned Opcode, Value *Op0, Value *Op1,
                            const TargetData *TD, const DominatorTree *DT,
                            unsigned MaxRecurse) {
  if (Constant *C0 = dyn_cast<Constant>(Op0)) {
    if (Constant *C1 = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { C0, C1 };
      return ConstantFoldInstOperands(Opcode, C0->getType(), Ops, 2, TD);
    }
  }

  // 0 shift by X -> 0
  if (match(Op0, m_Zero()))
    return Op0;

  // X shift by 0 -> X
  if (match(Op1, m_Zero()))
    return Op0;

  // X shift by undef -> undef because it may shift by the bitwidth.
  if (isa<UndefValue>(Op1))
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
static Value *SimplifyShlInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::Shl, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  // undef << X -> 0
  if (isa<UndefValue>(Op0))
    return Constant::getNullValue(Op0->getType());

  return 0;
}

Value *llvm::SimplifyShlInst(Value *Op0, Value *Op1, const TargetData *TD,
                             const DominatorTree *DT) {
  return ::SimplifyShlInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyLShrInst - Given operands for an LShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyLShrInst(Value *Op0, Value *Op1, const TargetData *TD,
                               const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::LShr, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  // undef >>l X -> 0
  if (isa<UndefValue>(Op0))
    return Constant::getNullValue(Op0->getType());

  return 0;
}

Value *llvm::SimplifyLShrInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifyLShrInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyAShrInst - Given operands for an AShr, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAShrInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT, unsigned MaxRecurse) {
  if (Value *V = SimplifyShift(Instruction::AShr, Op0, Op1, TD, DT, MaxRecurse))
    return V;

  // all ones >>a X -> all ones
  if (match(Op0, m_AllOnes()))
    return Op0;

  // undef >>a X -> all ones
  if (isa<UndefValue>(Op0))
    return Constant::getAllOnesValue(Op0->getType());

  return 0;
}

Value *llvm::SimplifyAShrInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT) {
  return ::SimplifyAShrInst(Op0, Op1, TD, DT, RecursionLimit);
}

/// SimplifyAndInst - Given operands for an And, see if we can
/// fold the result.  If not, this returns null.
static Value *SimplifyAndInst(Value *Op0, Value *Op1, const TargetData *TD,
                              const DominatorTree *DT, unsigned MaxRecurse) {
  if (Constant *CLHS = dyn_cast<Constant>(Op0)) {
    if (Constant *CRHS = dyn_cast<Constant>(Op1)) {
      Constant *Ops[] = { CLHS, CRHS };
      return ConstantFoldInstOperands(Instruction::And, CLHS->getType(),
                                      Ops, 2, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X & undef -> 0
  if (isa<UndefValue>(Op1))
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
  Value *A = 0, *B = 0;
  if ((match(Op0, m_Not(m_Value(A))) && A == Op1) ||
      (match(Op1, m_Not(m_Value(A))) && A == Op0))
    return Constant::getNullValue(Op0->getType());

  // (A | ?) & A = A
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
                                      Ops, 2, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // X | undef -> -1
  if (isa<UndefValue>(Op1))
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
  Value *A = 0, *B = 0;
  if ((match(Op0, m_Not(m_Value(A))) && A == Op1) ||
      (match(Op1, m_Not(m_Value(A))) && A == Op0))
    return Constant::getAllOnesValue(Op0->getType());

  // (A & ?) | A = A
  if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
      (A == Op1 || B == Op1))
    return Op1;

  // A | (A & ?) = A
  if (match(Op1, m_And(m_Value(A), m_Value(B))) &&
      (A == Op0 || B == Op0))
    return Op0;

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
                                      Ops, 2, TD);
    }

    // Canonicalize the constant to the RHS.
    std::swap(Op0, Op1);
  }

  // A ^ undef -> undef
  if (isa<UndefValue>(Op1))
    return Op1;

  // A ^ 0 = A
  if (match(Op1, m_Zero()))
    return Op0;

  // A ^ A = 0
  if (Op0 == Op1)
    return Constant::getNullValue(Op0->getType());

  // A ^ ~A  =  ~A ^ A  =  -1
  Value *A = 0;
  if ((match(Op0, m_Not(m_Value(A))) && A == Op1) ||
      (match(Op1, m_Not(m_Value(A))) && A == Op0))
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

static const Type *GetCompareTy(Value *Op) {
  return CmpInst::makeCmpResultType(Op->getType());
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

  const Type *ITy = GetCompareTy(LHS); // The return type.
  const Type *OpTy = LHS->getType();   // The operand type.

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

  // See if we are doing a comparison with a constant.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(RHS)) {
    switch (Pred) {
    default: break;
    case ICmpInst::ICMP_UGT:
      if (CI->isMaxValue(false))                 // A >u MAX -> FALSE
        return ConstantInt::getFalse(CI->getContext());
      break;
    case ICmpInst::ICMP_UGE:
      if (CI->isMinValue(false))                 // A >=u MIN -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    case ICmpInst::ICMP_ULT:
      if (CI->isMinValue(false))                 // A <u MIN -> FALSE
        return ConstantInt::getFalse(CI->getContext());
      break;
    case ICmpInst::ICMP_ULE:
      if (CI->isMaxValue(false))                 // A <=u MAX -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    case ICmpInst::ICMP_SGT:
      if (CI->isMaxValue(true))                  // A >s MAX -> FALSE
        return ConstantInt::getFalse(CI->getContext());
      break;
    case ICmpInst::ICMP_SGE:
      if (CI->isMinValue(true))                  // A >=s MIN -> TRUE
        return ConstantInt::getTrue(CI->getContext());
      break;
    case ICmpInst::ICMP_SLT:
      if (CI->isMinValue(true))                  // A <s MIN -> FALSE
        return ConstantInt::getFalse(CI->getContext());
      break;
    case ICmpInst::ICMP_SLE:
      if (CI->isMaxValue(true))                  // A <=s MAX -> TRUE
        return ConstantInt::getTrue(CI->getContext());
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
    // We already know that LHS != LHS.
    return ConstantInt::get(ITy, CmpInst::isFalseWhenEqual(Pred));

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

  if (isa<UndefValue>(TrueVal))   // select C, undef, X -> X
    return FalseVal;
  if (isa<UndefValue>(FalseVal))   // select C, X, undef -> X
    return TrueVal;
  if (isa<UndefValue>(CondVal)) {  // select undef, X, Y -> X or Y
    if (isa<Constant>(TrueVal))
      return TrueVal;
    return FalseVal;
  }

  return 0;
}

/// SimplifyGEPInst - Given operands for an GetElementPtrInst, see if we can
/// fold the result.  If not, this returns null.
Value *llvm::SimplifyGEPInst(Value *const *Ops, unsigned NumOps,
                             const TargetData *TD, const DominatorTree *) {
  // The type of the GEP pointer operand.
  const PointerType *PtrTy = cast<PointerType>(Ops[0]->getType());

  // getelementptr P -> P.
  if (NumOps == 1)
    return Ops[0];

  if (isa<UndefValue>(Ops[0])) {
    // Compute the (pointer) type returned by the GEP instruction.
    const Type *LastType = GetElementPtrInst::getIndexedType(PtrTy, &Ops[1],
                                                             NumOps-1);
    const Type *GEPTy = PointerType::get(LastType, PtrTy->getAddressSpace());
    return UndefValue::get(GEPTy);
  }

  if (NumOps == 2) {
    // getelementptr P, 0 -> P.
    if (ConstantInt *C = dyn_cast<ConstantInt>(Ops[1]))
      if (C->isZero())
        return Ops[0];
    // getelementptr P, N -> P if P points to a type of zero size.
    if (TD) {
      const Type *Ty = PtrTy->getElementType();
      if (Ty->isSized() && TD->getTypeAllocSize(Ty) == 0)
        return Ops[0];
    }
  }

  // Check to see if this is constant foldable.
  for (unsigned i = 0; i != NumOps; ++i)
    if (!isa<Constant>(Ops[i]))
      return 0;

  return ConstantExpr::getGetElementPtr(cast<Constant>(Ops[0]),
                                        (Constant *const*)Ops+1, NumOps-1);
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
  case Instruction::Add: return SimplifyAddInst(LHS, RHS, /* isNSW */ false,
                                                /* isNUW */ false, TD, DT,
                                                MaxRecurse);
  case Instruction::Sub: return SimplifySubInst(LHS, RHS, /* isNSW */ false,
                                                /* isNUW */ false, TD, DT,
                                                MaxRecurse);
  case Instruction::Mul: return SimplifyMulInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::Shl: return SimplifyShlInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::LShr: return SimplifyLShrInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::AShr: return SimplifyAShrInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::And: return SimplifyAndInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::Or:  return SimplifyOrInst(LHS, RHS, TD, DT, MaxRecurse);
  case Instruction::Xor: return SimplifyXorInst(LHS, RHS, TD, DT, MaxRecurse);
  default:
    if (Constant *CLHS = dyn_cast<Constant>(LHS))
      if (Constant *CRHS = dyn_cast<Constant>(RHS)) {
        Constant *COps[] = {CLHS, CRHS};
        return ConstantFoldInstOperands(Opcode, LHS->getType(), COps, 2, TD);
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
  case Instruction::Shl:
    Result = SimplifyShlInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::LShr:
    Result = SimplifyLShrInst(I->getOperand(0), I->getOperand(1), TD, DT);
    break;
  case Instruction::AShr:
    Result = SimplifyAShrInst(I->getOperand(0), I->getOperand(1), TD, DT);
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
    Result = SimplifyGEPInst(&Ops[0], Ops.size(), TD, DT);
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
