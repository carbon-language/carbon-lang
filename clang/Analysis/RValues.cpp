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

QualType SymbolData::getType() const {
  switch (getKind()) {
    default:
      assert (false && "getType() not implemented for this symbol.");
    
    case ParmKind:
      return static_cast<ParmVarDecl*>(getPtr())->getType();
  }
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

RValue RValue::EvalCast(ValueManager& ValMgr, Expr* CastExpr) const {
  switch (getBaseKind()) {
    default: assert(false && "Invalid RValue."); break;
    case LValueKind: return cast<LValue>(this)->EvalCast(ValMgr, CastExpr);
    case NonLValueKind: return cast<NonLValue>(this)->EvalCast(ValMgr, CastExpr);      
    case UninitializedKind: case InvalidKind: break;
  }
  
  return *this;
}
 

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-LValues.
//===----------------------------------------------------------------------===//
  
  // Binary Operators (except assignments and comma).

NonLValue NonLValue::EvalBinaryOp(ValueManager& ValMgr,
                                  BinaryOperator::Opcode Op,
                                  const NonLValue& RHS) const {
  
  if (isa<InvalidValue>(this) || isa<InvalidValue>(RHS))
    return cast<NonLValue>(InvalidValue());
  
  if (isa<UninitializedValue>(this) || isa<UninitializedValue>(RHS))
    return cast<NonLValue>(UninitializedValue());
  
  switch (getSubKind()) {
    default:
      assert (false && "Binary Operators not implemented for this NonLValue");
      
    case nonlval::ConcreteIntKind:
      
      if (isa<nonlval::ConcreteInt>(RHS)) {
        nonlval::ConcreteInt& self = cast<nonlval::ConcreteInt>(*this);
        return self.EvalBinaryOp(ValMgr, Op,
                                       cast<nonlval::ConcreteInt>(RHS));
      }
      else if(isa<InvalidValue>(RHS))
        return cast<NonLValue>(InvalidValue());
      else
        return RHS.EvalBinaryOp(ValMgr, Op, *this);
      
    case nonlval::SymbolValKind: {
      const nonlval::SymbolVal& self = cast<nonlval::SymbolVal>(*this);
      
      switch (RHS.getSubKind()) {
        default: assert ("Not Implemented." && false);
        case nonlval::ConcreteIntKind: {
          const SymIntConstraint& C =
            ValMgr.getConstraint(self.getSymbol(), Op,
                                 cast<nonlval::ConcreteInt>(RHS).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
      }
    }
  }
}

static const
llvm::APSInt& EvaluateAPSInt(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                             const llvm::APSInt& V1, const llvm::APSInt& V2) {
  
  switch (Op) {
    default:
      assert (false && "Invalid Opcode.");
      
    case BinaryOperator::Mul:
      return ValMgr.getValue( V1 * V2 );
      
    case BinaryOperator::Div:
      return ValMgr.getValue( V1 / V2 );
      
    case BinaryOperator::Rem:
      return ValMgr.getValue( V1 % V2 );
      
    case BinaryOperator::Add:
      return ValMgr.getValue( V1 + V2 );

    case BinaryOperator::Sub:
      return ValMgr.getValue( V1 - V2 );

#if 0
    case BinaryOperator::Shl:
      return ValMgr.getValue( V1 << V2 );
      
    case BinaryOperator::Shr:
      return ValMgr.getValue( V1 >> V2 );
#endif     
    
    case BinaryOperator::LT:
      return ValMgr.getTruthValue( V1 < V2 );
      
    case BinaryOperator::GT:
      return ValMgr.getTruthValue( V1 > V2 );
      
    case BinaryOperator::LE:
      return ValMgr.getTruthValue( V1 <= V2 );
      
    case BinaryOperator::GE:
      return ValMgr.getTruthValue( V1 >= V2 );
      
    case BinaryOperator::EQ:
      return ValMgr.getTruthValue( V1 == V2 );
      
    case BinaryOperator::NE:
      return ValMgr.getTruthValue( V1 != V2 );
      
    // Note: LAnd, LOr, Comma are handled specially by higher-level logic.
      
    case BinaryOperator::And:
      return ValMgr.getValue( V1 & V2 );
      
    case BinaryOperator::Or:
      return ValMgr.getValue( V1 | V2 );
  }
}

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalBinaryOp(ValueManager& ValMgr,
                                   BinaryOperator::Opcode Op,
                                   const nonlval::ConcreteInt& RHS) const {

  return EvaluateAPSInt(ValMgr, Op, getValue(), RHS.getValue());
}


  // Bitwise-Complement.

NonLValue NonLValue::EvalComplement(ValueManager& ValMgr) const {
  switch (getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(this)->EvalComplement(ValMgr);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalComplement(ValueManager& ValMgr) const {
  return ValMgr.getValue(~getValue()); 
}

  // Casts.

RValue NonLValue::EvalCast(ValueManager& ValMgr, Expr* CastExpr) const {
  if (!isa<nonlval::ConcreteInt>(this))
    return InvalidValue();
  
  APSInt V = cast<nonlval::ConcreteInt>(this)->getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  
  if (CastExpr->getType()->isPointerType())
    return lval::ConcreteInt(ValMgr.getValue(V));
  else
    return nonlval::ConcreteInt(ValMgr.getValue(V));
}

  // Unary Minus.

NonLValue NonLValue::EvalMinus(ValueManager& ValMgr, UnaryOperator* U) const {
  switch (getSubKind()) {
    case nonlval::ConcreteIntKind:
      return cast<nonlval::ConcreteInt>(this)->EvalMinus(ValMgr, U);
    default:
      return cast<NonLValue>(InvalidValue());
  }
}

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalMinus(ValueManager& ValMgr, UnaryOperator* U) const {
  assert (U->getType() == U->getSubExpr()->getType());  
  assert (U->getType()->isIntegerType());  
  return ValMgr.getValue(-getValue()); 
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for LValues.
//===----------------------------------------------------------------------===//

  // Binary Operators (except assignments and comma).

RValue LValue::EvalBinaryOp(ValueManager& ValMgr, 
                                  BinaryOperator::Opcode Op,
                                  const LValue& RHS) const {
  
  switch (Op) {
    default:
      assert (false && "Not yet implemented.");
      
    case BinaryOperator::EQ:
      return EQ(ValMgr, RHS);
    
    case BinaryOperator::NE:
      return NE(ValMgr, RHS);
  }
}


lval::ConcreteInt
lval::ConcreteInt::EvalBinaryOp(ValueManager& ValMgr,
                                BinaryOperator::Opcode Op,
                                const lval::ConcreteInt& RHS) const {
  
  assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub ||
          (Op >= BinaryOperator::LT && Op <= BinaryOperator::NE));
  
  return EvaluateAPSInt(ValMgr, Op, getValue(), RHS.getValue());
}

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

  // Casts.

RValue LValue::EvalCast(ValueManager& ValMgr, Expr* CastExpr) const {
  if (CastExpr->getType()->isPointerType())
    return *this;
  
  assert (CastExpr->getType()->isIntegerType());
  
  if (!isa<lval::ConcreteInt>(*this))
    return InvalidValue();
  
  APSInt V = cast<lval::ConcreteInt>(this)->getValue();
  QualType T = CastExpr->getType();
  V.setIsUnsigned(T->isUnsignedIntegerType() || T->isPointerType());
  V.extOrTrunc(ValMgr.getContext().getTypeSize(T, CastExpr->getLocStart()));
  return nonlval::ConcreteInt(ValMgr.getValue(V));
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

NonLValue NonLValue::GetIntTruthValue(ValueManager& ValMgr, bool b) {
  return nonlval::ConcreteInt(ValMgr.getTruthValue(b));
}

RValue RValue::GetSymbolValue(SymbolManager& SymMgr, ParmVarDecl* D) {
  QualType T = D->getType();
  
  if (T->isPointerType() || T->isReferenceType())
    return lval::SymbolVal(SymMgr.getSymbol(D));
  else
    return nonlval::SymbolVal(SymMgr.getSymbol(D));
}

void RValue::print() const {
  print(*llvm::cerr.stream());
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

      if (cast<nonlval::ConcreteInt>(this)->getValue().isUnsigned())
        Out << 'U';
      
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
      
      if (C.getConstraint().getInt().isUnsigned())
        Out << 'U';
      
      break;
    }  
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLValue.");
      break;
  }
}

#if 0
void LValue::print(std::ostream& Out) const {
  switch (getSubKind()) {        
    case lval::ConcreteIntKind:
      Out << cast<lval::ConcreteInt>(this)->getValue().toString() 
          << " (LValue)";
      break;
      
    case lval::SymbolValKind:
      Out << '$' << cast<lval::SymbolVal>(this)->getSymbol();
      break;

    case lval::DeclValKind:
      Out << '&' 
      << cast<lval::DeclVal>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    default:
      assert (false && "Pretty-printed not implemented for this LValue.");
      break;
  }
}
#endif
