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
// Values and ValueManager.
//===----------------------------------------------------------------------===//

ValueManager::~ValueManager() {
  // Note that the dstor for the contents of APSIntSet will never be called,
  // so we iterate over the set and invoke the dstor for each APSInt.  This
  // frees an aux. memory allocated to represent very large constants.
  for (APSIntSetTy::iterator I=APSIntSet.begin(), E=APSIntSet.end(); I!=E; ++I)
    I->getValue().~APSInt();
}

const APSInt& ValueManager::getValue(const APSInt& X) {
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

const APSInt& ValueManager::getValue(uint64_t X, unsigned BitWidth,
                                     bool isUnsigned) {
  APSInt V(BitWidth, isUnsigned);
  V = X;  
  return getValue(V);
}

const APSInt& ValueManager::getValue(uint64_t X, QualType T,
                                     SourceLocation Loc) {
  
  unsigned bits = Ctx.getTypeSize(T, Loc);
  APSInt V(bits, T->isUnsignedIntegerType());
  V = X;
  return getValue(V);
}

const SymIntConstraint&
ValueManager::getConstraint(SymbolID sym, BinaryOperator::Opcode Op,
                            const llvm::APSInt& V) {
  
  llvm::FoldingSetNodeID ID;
  SymIntConstraint::Profile(ID, sym, Op, V);
  void* InsertPos;
  
  SymIntConstraint* C = SymIntCSet.FindNodeOrInsertPos(ID, InsertPos);
  
  if (!C) {
    C = (SymIntConstraint*) BPAlloc.Allocate<SymIntConstraint>();
    new (C) SymIntConstraint(sym, Op, V);
    SymIntCSet.InsertNode(C, InsertPos);
  }
  
  return *C;
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

  if (!isa<lval::ConcreteInt>(*this))
    return InvalidValue();
  
  APSInt V = cast<lval::ConcreteInt>(this)->getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  return nonlval::ConcreteInt(ValMgr.getValue(V));
}

RValue NonLValue::Cast(ValueManager& ValMgr, Expr* CastExpr) const {
  if (!isa<nonlval::ConcreteInt>(this))
    return InvalidValue();
    
  APSInt V = cast<nonlval::ConcreteInt>(this)->getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  
  if (CastExpr->getType()->isPointerType())
    return lval::ConcreteInt(ValMgr.getValue(V));
  else
    return nonlval::ConcreteInt(ValMgr.getValue(V));
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-LValues.
//===----------------------------------------------------------------------===//

NonLValue NonLValue::UnaryMinus(ValueManager& ValMgr, UnaryOperator* U) const {
  switch (getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(this)->UnaryMinus(ValMgr, U);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}

NonLValue NonLValue::BitwiseComplement(ValueManager& ValMgr) const {
  switch (getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(this)->BitwiseComplement(ValMgr);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}


#define NONLVALUE_DISPATCH_CASE(k1,k2,Op)\
case (k1##Kind*nonlval::NumKind+k2##Kind):\
return cast<k1>(*this).Op(ValMgr,cast<k2>(RHS));

#define NONLVALUE_DISPATCH(Op)\
switch (getSubKind()*nonlval::NumKind+RHS.getSubKind()){\
NONLVALUE_DISPATCH_CASE(nonlval::ConcreteInt,nonlval::ConcreteInt,Op)\
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
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this LValue.");
      return cast<NonLValue>(InvalidValue());
    
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(RHS)) {
        bool b = cast<lval::ConcreteInt>(this)->getValue() ==
                 cast<lval::ConcreteInt>(RHS).getValue();
              
        return NonLValue::GetIntTruthValue(ValMgr, b);
      }
      else if (isa<lval::SymbolVal>(RHS)) {

        const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(RHS).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(this)->getValue());
        
        return nonlval::SymIntConstraintVal(C);        
      }
      
      break;
      
    case lval::SymbolValKind: {
      if (isa<lval::ConcreteInt>(RHS)) {

        const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(this)->getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(RHS).getValue());
        
        return nonlval::SymIntConstraintVal(C);
      }

      assert (!isa<lval::SymbolVal>(RHS) && "FIXME: Implement unification.");
      
      break;
    }
      
    case lval::DeclValKind:
      if (isa<lval::DeclVal>(RHS)) {        
        bool b = cast<lval::DeclVal>(*this) == cast<lval::DeclVal>(RHS);
        return NonLValue::GetIntTruthValue(ValMgr, b);
      }
      
      break;
  }
  
  return NonLValue::GetIntTruthValue(ValMgr, false);
}

NonLValue LValue::NE(ValueManager& ValMgr, const LValue& RHS) const {
  switch (getSubKind()) {
    default:
      assert(false && "NE not implemented for this LValue.");
      return cast<NonLValue>(InvalidValue());
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(RHS)) {
        bool b = cast<lval::ConcreteInt>(this)->getValue() !=
                 cast<lval::ConcreteInt>(RHS).getValue();
        
        return NonLValue::GetIntTruthValue(ValMgr, b);
      }
      else if (isa<lval::SymbolVal>(RHS)) {
        
        const SymIntConstraint& C =
        ValMgr.getConstraint(cast<lval::SymbolVal>(RHS).getSymbol(),
                             BinaryOperator::NE,
                             cast<lval::ConcreteInt>(this)->getValue());
        
        return nonlval::SymIntConstraintVal(C);        
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(RHS)) {
          
          const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(this)->getSymbol(),
                               BinaryOperator::NE,
                               cast<lval::ConcreteInt>(RHS).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        
        assert (!isa<lval::SymbolVal>(RHS) && "FIXME: Implement sym !=.");
        
        break;
      }
      
      case lval::DeclValKind:
      if (isa<lval::DeclVal>(RHS)) {        
        bool b = cast<lval::DeclVal>(*this) == cast<lval::DeclVal>(RHS);
        return NonLValue::GetIntTruthValue(ValMgr, b);
      }
      
      break;
  }
  
  return NonLValue::GetIntTruthValue(ValMgr, true);
}


//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-LValues.
//===----------------------------------------------------------------------===//

NonLValue NonLValue::GetValue(ValueManager& ValMgr, uint64_t X, QualType T,
                              SourceLocation Loc) {
  
  return nonlval::ConcreteInt(ValMgr.getValue(X, T, Loc));
}

NonLValue NonLValue::GetValue(ValueManager& ValMgr, IntegerLiteral* I) {
  return nonlval::ConcreteInt(ValMgr.getValue(APSInt(I->getValue(),
                                   I->getType()->isUnsignedIntegerType())));
}

RValue RValue::GetSymbolValue(SymbolManager& SymMgr, ParmVarDecl* D) {
  QualType T = D->getType();
  
  if (T->isPointerType() || T->isReferenceType())
    return lval::SymbolVal(SymMgr.getSymbol(D));
  else
    return nonlval::SymbolVal(SymMgr.getSymbol(D));
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

static void printOpcode(std::ostream& Out, BinaryOperator::Opcode Op) {
  switch (Op) {
    case BinaryOperator::EQ: Out << "=="; break;
    case BinaryOperator::NE: Out << "!="; break;
    default: assert(false && "Not yet implemented.");
  }        
}

void NonLValue::print(std::ostream& Out) const {
  switch (getSubKind()) {  
    case nonlval::ConcreteIntKind:
      Out << cast<nonlval::ConcreteInt>(this)->getValue().toString();
      break;
      
    case nonlval::SymbolValKind:
      Out << '$' << cast<nonlval::SymbolVal>(this)->getSymbol();
      break;
     
    case nonlval::SymIntConstraintValKind: {
      const nonlval::SymIntConstraintVal& C = 
        *cast<nonlval::SymIntConstraintVal>(this);
      
      Out << '$' << C.getConstraint().getSymbol() << ' ';
      printOpcode(Out, C.getConstraint().getOpcode());
      Out << ' ' << C.getConstraint().getInt().toString();
      break;
    }  
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLValue.");
      break;
  }
}


void LValue::print(std::ostream& Out) const {
  switch (getSubKind()) {        
    case lval::ConcreteIntKind:
      Out << cast<lval::ConcreteInt>(this)->getValue().toString() 
          << " (LValue)";
      break;
      
    case lval::SymbolValKind:
      Out << '$' << cast<lval::SymbolVal>(this)->getSymbol();
      break;
      
    case lval::SymIntConstraintValKind: {
      const lval::SymIntConstraintVal& C = 
        *cast<lval::SymIntConstraintVal>(this);
      
      Out << '$' << C.getConstraint().getSymbol() << ' ';
      printOpcode(Out, C.getConstraint().getOpcode());
      Out << ' ' << C.getConstraint().getInt().toString();
      break;
    }
      
    case lval::DeclValKind:
      Out << '&' 
      << cast<lval::DeclVal>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    default:
      assert (false && "Pretty-printed not implemented for this LValue.");
      break;
  }
}
