//= RValues.cpp - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines RVal, LVal, and NonLVal, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/RValues.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/Streams.h"

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Symbol Iteration.
//===----------------------------------------------------------------------===//

RVal::symbol_iterator RVal::symbol_begin() const {
  
  // FIXME: This is a rat's nest.  Cleanup.

  if (isa<lval::SymbolVal>(this))
    return (symbol_iterator) (&Data);
  else if (isa<nonlval::SymbolVal>(this))
    return (symbol_iterator) (&Data);
  else if (isa<nonlval::SymIntConstraintVal>(this)) {
    const SymIntConstraint& C =
      cast<nonlval::SymIntConstraintVal>(this)->getConstraint();
    
    return (symbol_iterator) &C.getSymbol();
  }
  else if (isa<nonlval::LValAsInteger>(this)) {
    const nonlval::LValAsInteger& V = cast<nonlval::LValAsInteger>(*this);
    return  V.getPersistentLVal().symbol_begin();
  }
  else if (isa<lval::FieldOffset>(this)) {
    const lval::FieldOffset& V = cast<lval::FieldOffset>(*this);
    return V.getPersistentBase().symbol_begin();
  }
  return NULL;
}

RVal::symbol_iterator RVal::symbol_end() const {
  symbol_iterator X = symbol_begin();
  return X ? X+1 : NULL;
}

//===----------------------------------------------------------------------===//
// Useful predicates.
//===----------------------------------------------------------------------===//

bool RVal::isZeroConstant() const {
  if (isa<lval::ConcreteInt>(*this))
    return cast<lval::ConcreteInt>(*this).getValue() == 0;
  else if (isa<nonlval::ConcreteInt>(*this))
    return cast<nonlval::ConcreteInt>(*this).getValue() == 0;
  else
    return false;
}


//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-LVals.
//===----------------------------------------------------------------------===//

RVal nonlval::ConcreteInt::EvalBinOp(BasicValueFactory& BasicVals,
                                     BinaryOperator::Opcode Op,
                                     const nonlval::ConcreteInt& R) const {
  
  const llvm::APSInt* X =
    BasicVals.EvaluateAPSInt(Op, getValue(), R.getValue());
  
  if (X)
    return nonlval::ConcreteInt(*X);
  else
    return UndefinedVal();
}

  // Bitwise-Complement.

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalComplement(BasicValueFactory& BasicVals) const {
  return BasicVals.getValue(~getValue()); 
}

  // Unary Minus.

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalMinus(BasicValueFactory& BasicVals, UnaryOperator* U) const {
  assert (U->getType() == U->getSubExpr()->getType());  
  assert (U->getType()->isIntegerType());  
  return BasicVals.getValue(-getValue()); 
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for LVals.
//===----------------------------------------------------------------------===//

RVal
lval::ConcreteInt::EvalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                             const lval::ConcreteInt& R) const {
  
  assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub ||
          (Op >= BinaryOperator::LT && Op <= BinaryOperator::NE));
  
  const llvm::APSInt* X = BasicVals.EvaluateAPSInt(Op, getValue(), R.getValue());
  
  if (X)
    return lval::ConcreteInt(*X);
  else
    return UndefinedVal();
}

NonLVal LVal::EQ(BasicValueFactory& BasicVals, const LVal& R) const {
  
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this LVal.");
      break;
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(this)->getValue() ==
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<lval::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(this)->getValue());
        
        return nonlval::SymIntConstraintVal(C);        
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(R)) {
          
          const SymIntConstraint& C =
            BasicVals.getConstraint(cast<lval::SymbolVal>(this)->getSymbol(),
                                 BinaryOperator::EQ,
                                 cast<lval::ConcreteInt>(R).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        
        assert (!isa<lval::SymbolVal>(R) && "FIXME: Implement unification.");
        
        break;
      }
      
      case lval::DeclValKind:
      if (isa<lval::DeclVal>(R)) {        
        bool b = cast<lval::DeclVal>(*this) == cast<lval::DeclVal>(R);
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      
      break;
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, false);
}

NonLVal LVal::NE(BasicValueFactory& BasicVals, const LVal& R) const {
  switch (getSubKind()) {
    default:
      assert(false && "NE not implemented for this LVal.");
      break;
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(this)->getValue() !=
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<lval::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
        BasicVals.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                             BinaryOperator::NE,
                             cast<lval::ConcreteInt>(this)->getValue());
        
        return nonlval::SymIntConstraintVal(C);        
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(R)) {
          
          const SymIntConstraint& C =
          BasicVals.getConstraint(cast<lval::SymbolVal>(this)->getSymbol(),
                               BinaryOperator::NE,
                               cast<lval::ConcreteInt>(R).getValue());
          
          return nonlval::SymIntConstraintVal(C);
        }
        
        assert (!isa<lval::SymbolVal>(R) && "FIXME: Implement sym !=.");
        
        break;
      }
      
      case lval::DeclValKind:
        if (isa<lval::DeclVal>(R)) {        
          bool b = cast<lval::DeclVal>(*this) == cast<lval::DeclVal>(R);
          return NonLVal::MakeIntTruthVal(BasicVals, b);
        }
      
        break;
  }
  
  return NonLVal::MakeIntTruthVal(BasicVals, true);
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-LVals.
//===----------------------------------------------------------------------===//

NonLVal NonLVal::MakeVal(BasicValueFactory& BasicVals, uint64_t X, QualType T) {  
  return nonlval::ConcreteInt(BasicVals.getValue(X, T));
}

NonLVal NonLVal::MakeVal(BasicValueFactory& BasicVals, IntegerLiteral* I) {

  return nonlval::ConcreteInt(BasicVals.getValue(APSInt(I->getValue(),
                              I->getType()->isUnsignedIntegerType())));
}

NonLVal NonLVal::MakeIntTruthVal(BasicValueFactory& BasicVals, bool b) {
  return nonlval::ConcreteInt(BasicVals.getTruthValue(b));
}

RVal RVal::GetSymbolValue(SymbolManager& SymMgr, VarDecl* D) {

  QualType T = D->getType();
  
  if (T->isPointerLikeType() || T->isObjCQualifiedIdType())
    return lval::SymbolVal(SymMgr.getSymbol(D));
  
  return nonlval::SymbolVal(SymMgr.getSymbol(D));
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing LVals.
//===----------------------------------------------------------------------===//

LVal LVal::MakeVal(AddrLabelExpr* E) { return lval::GotoLabel(E->getLabel()); }

LVal LVal::MakeVal(StringLiteral* S) {
  return lval::StringLiteralVal(S);
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing RVals (both NonLVals and LVals).
//===----------------------------------------------------------------------===//

RVal RVal::MakeVal(BasicValueFactory& BasicVals, DeclRefExpr* E) {
  
  ValueDecl* D = cast<DeclRefExpr>(E)->getDecl();
  
  if (VarDecl* VD = dyn_cast<VarDecl>(D)) {
    return lval::DeclVal(VD);
  }
  else if (EnumConstantDecl* ED = dyn_cast<EnumConstantDecl>(D)) {
    
    // FIXME: Do we need to cache a copy of this enum, since it
    // already has persistent storage?  We do this because we
    // are comparing states using pointer equality.  Perhaps there is
    // a better way, since APInts are fairly lightweight.
    
    return nonlval::ConcreteInt(BasicVals.getValue(ED->getInitVal()));          
  }
  else if (FunctionDecl* FD = dyn_cast<FunctionDecl>(D)) {
    return lval::FuncVal(FD);
  }
  
  assert (false &&
          "ValueDecl support for this ValueDecl not implemented.");
  
  return UnknownVal();
}

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void RVal::printStdErr() const { print(*llvm::cerr.stream()); }

void RVal::print(std::ostream& Out) const {

  switch (getBaseKind()) {
      
    case UnknownKind:
      Out << "Invalid"; break;
      
    case NonLValKind:
      cast<NonLVal>(this)->print(Out); break;
      
    case LValKind:
      cast<LVal>(this)->print(Out); break;
      
    case UndefinedKind:
      Out << "Undefined"; break;
      
    default:
      assert (false && "Invalid RVal.");
  }
}

static void printOpcode(std::ostream& Out, BinaryOperator::Opcode Op) {
  
  switch (Op) {      
    case BinaryOperator::Mul: Out << '*'  ; break;
    case BinaryOperator::Div: Out << '/'  ; break;
    case BinaryOperator::Rem: Out << '%'  ; break;
    case BinaryOperator::Add: Out << '+'  ; break;
    case BinaryOperator::Sub: Out << '-'  ; break;
    case BinaryOperator::Shl: Out << "<<" ; break;
    case BinaryOperator::Shr: Out << ">>" ; break;
    case BinaryOperator::LT:  Out << "<"  ; break;
    case BinaryOperator::GT:  Out << '>'  ; break;
    case BinaryOperator::LE:  Out << "<=" ; break;
    case BinaryOperator::GE:  Out << ">=" ; break;    
    case BinaryOperator::EQ:  Out << "==" ; break;
    case BinaryOperator::NE:  Out << "!=" ; break;
    case BinaryOperator::And: Out << '&'  ; break;
    case BinaryOperator::Xor: Out << '^'  ; break;
    case BinaryOperator::Or:  Out << '|'  ; break;
      
    default: assert(false && "Not yet implemented.");
  }        
}

void NonLVal::print(std::ostream& Out) const {

  switch (getSubKind()) {  

    case nonlval::ConcreteIntKind:
      Out << cast<nonlval::ConcreteInt>(this)->getValue();

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
      Out << ' ' << C.getConstraint().getInt();
      
      if (C.getConstraint().getInt().isUnsigned())
        Out << 'U';
      
      break;
    }
    
    case nonlval::LValAsIntegerKind: {
      const nonlval::LValAsInteger& C = *cast<nonlval::LValAsInteger>(this);
      C.getLVal().print(Out);
      Out << " [as " << C.getNumBits() << " bit integer]";
      break;
    }
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLVal.");
      break;
  }
}

void LVal::print(std::ostream& Out) const {
  
  switch (getSubKind()) {        

    case lval::ConcreteIntKind:
      Out << cast<lval::ConcreteInt>(this)->getValue() << " (LVal)";
      break;
      
    case lval::SymbolValKind:
      Out << '$' << cast<lval::SymbolVal>(this)->getSymbol();
      break;
      
    case lval::GotoLabelKind:
      Out << "&&"
          << cast<lval::GotoLabel>(this)->getLabel()->getID()->getName();
      break;

    case lval::DeclValKind:
      Out << '&' 
          << cast<lval::DeclVal>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    case lval::FuncValKind:
      Out << "function " 
          << cast<lval::FuncVal>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    case lval::StringLiteralValKind:
      Out << "literal \""
          << cast<lval::StringLiteralVal>(this)->getLiteral()->getStrData()
          << "\"";
      break;
      
    case lval::FieldOffsetKind: {
      const lval::FieldOffset& C = *cast<lval::FieldOffset>(this);
      C.getBase().print(Out);
      Out << "." << C.getFieldDecl()->getName() << " (field LVal)";
      break;
    }
      
    case lval::ArrayOffsetKind: {
      const lval::ArrayOffset& C = *cast<lval::ArrayOffset>(this);
      C.getBase().print(Out);
      Out << "[";
      C.getOffset().print(Out);
      Out << "] (lval array entry)";
      break;
    }
      
    default:
      assert (false && "Pretty-printing not implemented for this LVal.");
      break;
  }
}
