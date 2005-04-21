//===-- llvm/Support/PatternMatch.h - Match on the LLVM IR ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on the LLVM IR.  The power of these routines is
// that it allows you to write concise patterns that are expressive and easy to
// understand.  The other major advantage of this is that is allows to you
// trivially capture/bind elements in the pattern to variables.  For example,
// you can do something like this:
//
//  Value *Exp = ...
//  Value *X, *Y;  ConstantInt *C1, *C2;      // (X & C1) | (Y & C2)
//  if (match(Exp, m_Or(m_And(m_Value(X), m_ConstantInt(C1)),
//                      m_And(m_Value(Y), m_ConstantInt(C2))))) {
//    ... Pattern is matched and variables are bound ...
//  }
//
// This is primarily useful to things like the instruction combiner, but can
// also be useful for static analysis tools or code generators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PATTERNMATCH_H
#define LLVM_SUPPORT_PATTERNMATCH_H

#include "llvm/Constants.h"
#include "llvm/Instructions.h"

namespace llvm {
namespace PatternMatch {

template<typename Val, typename Pattern>
bool match(Val *V, const Pattern &P) {
  return const_cast<Pattern&>(P).match(V);
}

template<typename Class>
struct leaf_ty {
  template<typename ITy>
  bool match(ITy *V) { return isa<Class>(V); }
};

inline leaf_ty<Value> m_Value() { return leaf_ty<Value>(); }
inline leaf_ty<ConstantInt> m_ConstantInt() { return leaf_ty<ConstantInt>(); }

template<typename Class>
struct bind_ty {
  Class *&VR;
  bind_ty(Class *&V) : VR(V) {}

  template<typename ITy>
  bool match(ITy *V) {
    if (Class *CV = dyn_cast<Class>(V)) {
      VR = CV;
      return true;
    }
    return false;
  }
};

inline bind_ty<Value> m_Value(Value *&V) { return V; }
inline bind_ty<ConstantInt> m_ConstantInt(ConstantInt *&CI) { return CI; }

//===----------------------------------------------------------------------===//
// Matchers for specific binary operators
//

template<typename LHS_t, typename RHS_t, unsigned Opcode>
struct BinaryOp_match {
  LHS_t L;
  RHS_t R;

  BinaryOp_match(const LHS_t &LHS, const RHS_t &RHS) : L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      return I->getOpcode() == Opcode && L.match(I->getOperand(0)) &&
             R.match(I->getOperand(1));
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
    return false;
  }
};

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Add> m_Add(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Add>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Sub> m_Sub(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Sub>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Mul> m_Mul(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Mul>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Div> m_Div(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Div>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Rem> m_Rem(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Rem>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::And> m_And(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::And>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Rem> m_Or(const LHS &L,
                                                       const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Or>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Xor> m_Xor(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Xor>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Shl> m_Shl(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Shl>(L, R);
}

template<typename LHS, typename RHS>
inline BinaryOp_match<LHS, RHS, Instruction::Shr> m_Shr(const LHS &L,
                                                        const RHS &R) {
  return BinaryOp_match<LHS, RHS, Instruction::Shr>(L, R);
}

//===----------------------------------------------------------------------===//
// Matchers for binary classes
//

template<typename LHS_t, typename RHS_t, typename Class>
struct BinaryOpClass_match {
  Instruction::BinaryOps &Opcode;
  LHS_t L;
  RHS_t R;

  BinaryOpClass_match(Instruction::BinaryOps &Op, const LHS_t &LHS,
                      const RHS_t &RHS)
    : Opcode(Op), L(LHS), R(RHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Class *I = dyn_cast<Class>(V))
      if (L.match(I->getOperand(0)) && R.match(I->getOperand(1))) {
        Opcode = I->getOpcode();
        return true;
      }
#if 0  // Doesn't handle constantexprs yet!
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode() == Opcode && L.match(CE->getOperand(0)) &&
             R.match(CE->getOperand(1));
#endif
    return false;
  }
};

template<typename LHS, typename RHS>
inline BinaryOpClass_match<LHS, RHS, SetCondInst>
m_SetCond(Instruction::BinaryOps &Op, const LHS &L, const RHS &R) {
  return BinaryOpClass_match<LHS, RHS, SetCondInst>(Op, L, R);
}


//===----------------------------------------------------------------------===//
// Matchers for unary operators
//

template<typename LHS_t>
struct neg_match {
  LHS_t L;

  neg_match(const LHS_t &LHS) : L(LHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Sub)
        return matchIfNeg(I->getOperand(0), I->getOperand(1));
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::Sub)
        return matchIfNeg(CE->getOperand(0), CE->getOperand(1));
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return L.match(ConstantExpr::getNeg(CI));
    return false;
  }
private:
  bool matchIfNeg(Value *LHS, Value *RHS) {
    if (!LHS->getType()->isFloatingPoint())
      return LHS == Constant::getNullValue(LHS->getType()) && L.match(RHS);
    else
      return LHS == ConstantFP::get(LHS->getType(), -0.0) && L.match(RHS);
  }
};

template<typename LHS>
inline neg_match<LHS> m_Neg(const LHS &L) { return L; }


template<typename LHS_t>
struct not_match {
  LHS_t L;

  not_match(const LHS_t &LHS) : L(LHS) {}

  template<typename OpTy>
  bool match(OpTy *V) {
    if (Instruction *I = dyn_cast<Instruction>(V))
      if (I->getOpcode() == Instruction::Xor)
        return matchIfNot(I->getOperand(0), I->getOperand(1));
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      if (CE->getOpcode() == Instruction::Xor)
        return matchIfNot(CE->getOperand(0), CE->getOperand(1));
    if (ConstantInt *CI = dyn_cast<ConstantInt>(V))
      return L.match(ConstantExpr::getNot(CI));
    return false;
  }
private:
  bool matchIfNot(Value *LHS, Value *RHS) {
    if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(RHS))
      return CI->isAllOnesValue() && L.match(LHS);
    else if (ConstantIntegral *CI = dyn_cast<ConstantIntegral>(LHS))
      return CI->isAllOnesValue() && L.match(RHS);
    return false;
  }
};

template<typename LHS>
inline not_match<LHS> m_Not(const LHS &L) { return L; }

//===----------------------------------------------------------------------===//
// Matchers for control flow
//

template<typename Cond_t>
struct brc_match {
  Cond_t Cond;
  BasicBlock *&T, *&F;
  brc_match(const Cond_t &C, BasicBlock *&t, BasicBlock *&f)
    : Cond(C), T(t), F(f) {
  }

  template<typename OpTy>
  bool match(OpTy *V) {
    if (BranchInst *BI = dyn_cast<BranchInst>(V))
      if (BI->isConditional()) {
        if (Cond.match(BI->getCondition())) {
          T = BI->getSuccessor(0);
          F = BI->getSuccessor(1);
          return true;
        }
      }
    return false;
  }
};

template<typename Cond_t>
inline brc_match<Cond_t> m_Br(const Cond_t &C, BasicBlock *&T, BasicBlock *&F){
  return brc_match<Cond_t>(C, T, F);
}


}} // end llvm::match


#endif

