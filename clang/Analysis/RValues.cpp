//= RValues.cpp - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines RValue, LValue, and NonLValue, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#include "RValues.h"

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// SymbolManager.
//===----------------------------------------------------------------------===//

SymbolID SymbolManager::getSymbol(ParmVarDecl* D) {
  SymbolID& X = DataToSymbol[D];
  
  if (!X.isInitialized()) {
    X = SymbolToData.size();
    SymbolToData.push_back(D);
  }
  
  return X;
}

SymbolManager::SymbolManager() {}
SymbolManager::~SymbolManager() {}

//===----------------------------------------------------------------------===//
// ValueManager.
//===----------------------------------------------------------------------===//

ValueManager::~ValueManager() {
  // Note that the dstor for the contents of APSIntSet will never be called,
  // so we iterate over the set and invoke the dstor for each APSInt.  This
  // frees an aux. memory allocated to represent very large constants.
  for (APSIntSetTy::iterator I=APSIntSet.begin(), E=APSIntSet.end(); I!=E; ++I)
    I->getValue().~APSInt();
}

APSInt& ValueManager::getValue(const APSInt& X) {
  llvm::FoldingSetNodeID ID;
  void* InsertPos;
  typedef llvm::FoldingSetNodeWrapper<APSInt> FoldNodeTy;
  
  X.Profile(ID);
  FoldNodeTy* P = APSIntSet.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!P) {  
    P = (FoldNodeTy*) BPAlloc.Allocate<FoldNodeTy>();
    new (P) FoldNodeTy(X);
    APSIntSet.InsertNode(P, InsertPos);
  }
  
  return *P;
}

APSInt& ValueManager::getValue(uint64_t X, unsigned BitWidth, bool isUnsigned) {
  APSInt V(BitWidth, isUnsigned);
  V = X;  
  return getValue(V);
}

APSInt& ValueManager::getValue(uint64_t X, QualType T, SourceLocation Loc) {
  unsigned bits = Ctx.getTypeSize(T, Loc);
  APSInt V(bits, T->isUnsignedIntegerType());
  V = X;
  return getValue(V);
}

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

RValue RValue::Cast(ValueManager& ValMgr, Expr* CastExpr) const {
  switch (getBaseKind()) {
    default: assert(false && "Invalid RValue."); break;
    case LValueKind: return cast<LValue>(this)->Cast(ValMgr, CastExpr);
    case NonLValueKind: return cast<NonLValue>(this)->Cast(ValMgr, CastExpr);      
    case UninitializedKind: case InvalidKind: break;
  }
  
  return *this;
}
 
RValue LValue::Cast(ValueManager& ValMgr, Expr* CastExpr) const {
  if (CastExpr->getType()->isPointerType())
    return *this;
  
  assert (CastExpr->getType()->isIntegerType());

  if (!isa<ConcreteIntLValue>(*this))
    return InvalidValue();
  
  APSInt V = cast<ConcreteIntLValue>(this)->getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  return ConcreteInt(ValMgr.getValue(V));
}

RValue NonLValue::Cast(ValueManager& ValMgr, Expr* CastExpr) const {
  if (!isa<ConcreteInt>(this))
    return InvalidValue();
    
  APSInt V = cast<ConcreteInt>(this)->getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  
  if (CastExpr->getType()->isPointerType())
    return ConcreteIntLValue(ValMgr.getValue(V));
  else
    return ConcreteInt(ValMgr.getValue(V));
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-LValues.
//===----------------------------------------------------------------------===//

NonLValue NonLValue::UnaryMinus(ValueManager& ValMgr, UnaryOperator* U) const {
  switch (getSubKind()) {
    case ConcreteIntKind:
      return cast<ConcreteInt>(this)->UnaryMinus(ValMgr, U);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}

NonLValue NonLValue::BitwiseComplement(ValueManager& ValMgr) const {
  switch (getSubKind()) {
    case ConcreteIntKind:
      return cast<ConcreteInt>(this)->BitwiseComplement(ValMgr);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}


#define NONLVALUE_DISPATCH_CASE(k1,k2,Op)\
case (k1##Kind*NumNonLValueKind+k2##Kind):\
return cast<k1>(*this).Op(ValMgr,cast<k2>(RHS));

#define NONLVALUE_DISPATCH(Op)\
switch (getSubKind()*NumNonLValueKind+RHS.getSubKind()){\
NONLVALUE_DISPATCH_CASE(ConcreteInt,ConcreteInt,Op)\
default:\
if (getBaseKind() == UninitializedKind ||\
RHS.getBaseKind() == UninitializedKind)\
return cast<NonLValue>(UninitializedValue());\
assert (!isValid() || !RHS.isValid() && "Missing case.");\
break;\
}\
return cast<NonLValue>(InvalidValue());

NonLValue NonLValue::Add(ValueManager& ValMgr, const NonLValue& RHS) const {
  NONLVALUE_DISPATCH(Add)
}

NonLValue NonLValue::Sub(ValueManager& ValMgr, const NonLValue& RHS) const {
  NONLVALUE_DISPATCH(Sub)
}

NonLValue NonLValue::Mul(ValueManager& ValMgr, const NonLValue& RHS) const {
  NONLVALUE_DISPATCH(Mul)
}

NonLValue NonLValue::Div(ValueManager& ValMgr, const NonLValue& RHS) const {
  NONLVALUE_DISPATCH(Div)
}

NonLValue NonLValue::Rem(ValueManager& ValMgr, const NonLValue& RHS) const {
  NONLVALUE_DISPATCH(Rem)
}

NonLValue NonLValue::EQ(ValueManager& ValMgr, const NonLValue& RHS) const {  
  NONLVALUE_DISPATCH(EQ)
}

NonLValue NonLValue::NE(ValueManager& ValMgr, const NonLValue& RHS) const {
  NONLVALUE_DISPATCH(NE)
}

#undef NONLVALUE_DISPATCH_CASE
#undef NONLVALUE_DISPATCH

//===----------------------------------------------------------------------===//
// Transfer function dispatch for LValues.
//===----------------------------------------------------------------------===//


NonLValue LValue::EQ(ValueManager& ValMgr, const LValue& RHS) const {
  if (getSubKind() != RHS.getSubKind())
    return NonLValue::GetIntTruthValue(ValMgr, false);
  
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this LValue.");
      return cast<NonLValue>(InvalidValue());
    
    case ConcreteIntLValueKind: {
      bool b = cast<ConcreteIntLValue>(this)->getValue() ==
               cast<ConcreteIntLValue>(RHS).getValue();
            
      return NonLValue::GetIntTruthValue(ValMgr, b);
    }
      
    case LValueDeclKind: {
      bool b = cast<LValueDecl>(*this) == cast<LValueDecl>(RHS);
      return NonLValue::GetIntTruthValue(ValMgr, b);
    }
  }
}

NonLValue LValue::NE(ValueManager& ValMgr, const LValue& RHS) const {
  if (getSubKind() != RHS.getSubKind())
    return NonLValue::GetIntTruthValue(ValMgr, true);
  
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this LValue.");
      return cast<NonLValue>(InvalidValue());
      
    case ConcreteIntLValueKind: {
      bool b = cast<ConcreteIntLValue>(this)->getValue() !=
               cast<ConcreteIntLValue>(RHS).getValue();
      
      return NonLValue::GetIntTruthValue(ValMgr, b);
    }  
      
    case LValueDeclKind: {
      bool b = cast<LValueDecl>(*this) != cast<LValueDecl>(RHS);
      return NonLValue::GetIntTruthValue(ValMgr, b);
    }
  }
}


//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-LValues.
//===----------------------------------------------------------------------===//

NonLValue NonLValue::GetValue(ValueManager& ValMgr, uint64_t X, QualType T,
                              SourceLocation Loc) {
  
  return ConcreteInt(ValMgr.getValue(X, T, Loc));
}

NonLValue NonLValue::GetValue(ValueManager& ValMgr, IntegerLiteral* I) {
  return ConcreteInt(ValMgr.getValue(APSInt(I->getValue(),
                                            I->getType()->isUnsignedIntegerType())));
}

RValue RValue::GetSymbolValue(SymbolManager& SymMgr, ParmVarDecl* D) {
  QualType T = D->getType();
  
  if (T->isPointerType() || T->isReferenceType())
    return SymbolicLValue(SymMgr.getSymbol(D));
  else
    return SymbolicNonLValue(SymMgr.getSymbol(D));
}

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void RValue::print(std::ostream& Out) const {
  switch (getBaseKind()) {
    case InvalidKind:
      Out << "Invalid";
      break;
      
    case NonLValueKind:
      cast<NonLValue>(this)->print(Out);
      break;
      
    case LValueKind:
      cast<LValue>(this)->print(Out);
      break;
      
    case UninitializedKind:
      Out << "Uninitialized";
      break;
      
    default:
      assert (false && "Invalid RValue.");
  }
}

void NonLValue::print(std::ostream& Out) const {
  switch (getSubKind()) {  
    case ConcreteIntKind:
      Out << cast<ConcreteInt>(this)->getValue().toString();
      break;
      
    case SymbolicNonLValueKind:
      Out << '$' << cast<SymbolicNonLValue>(this)->getSymbolID();
      break;
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLValue.");
      break;
  }
}

void LValue::print(std::ostream& Out) const {
  switch (getSubKind()) {        
    case ConcreteIntLValueKind:
      Out << cast<ConcreteIntLValue>(this)->getValue().toString() 
          << " (LValue)";
      break;
      
    case SymbolicLValueKind:
      Out << '$' << cast<SymbolicLValue>(this)->getSymbolID();
      break;
      
    case LValueDeclKind:
      Out << '&' 
      << cast<LValueDecl>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    default:
      assert (false && "Pretty-printed not implemented for this LValue.");
      break;
  }
}
