//= RValues.cpp - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This files defines RVal, LVal, and NonLVal, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/RValues.h"
#include "llvm/Support/Streams.h"

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Symbol Iteration.
//===----------------------------------------------------------------------===//

RVal::symbol_iterator RVal::symbol_begin() const {
  if (isa<lval::SymbolVal>(this))
    return (symbol_iterator) (&Data);
  else if (isa<nonlval::SymbolVal>(this))
    return (symbol_iterator) (&Data);
  else if (isa<nonlval::SymIntConstraintVal>(this)) {
    const SymIntConstraint& C =
      cast<nonlval::SymIntConstraintVal>(this)->getConstraint();
    
    return (symbol_iterator) &C.getSymbol();
  }
  
  return NULL;
}

RVal::symbol_iterator RVal::symbol_end() const {
  symbol_iterator X = symbol_begin();
  return X ? X+1 : NULL;
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-LVals.
//===----------------------------------------------------------------------===//

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                                const nonlval::ConcreteInt& R) const {
  
  return ValMgr.EvaluateAPSInt(Op, getValue(), R.getValue());
}


  // Bitwise-Complement.

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalComplement(ValueManager& ValMgr) const {
  return ValMgr.getValue(~getValue()); 
}

  // Unary Minus.

nonlval::ConcreteInt
nonlval::ConcreteInt::EvalMinus(ValueManager& ValMgr, UnaryOperator* U) const {
  assert (U->getType() == U->getSubExpr()->getType());  
  assert (U->getType()->isIntegerType());  
  return ValMgr.getValue(-getValue()); 
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for LVals.
//===----------------------------------------------------------------------===//

lval::ConcreteInt
lval::ConcreteInt::EvalBinOp(ValueManager& ValMgr, BinaryOperator::Opcode Op,
                             const lval::ConcreteInt& R) const {
  
  assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub ||
          (Op >= BinaryOperator::LT && Op <= BinaryOperator::NE));
  
  return ValMgr.EvaluateAPSInt(Op, getValue(), R.getValue());
}

NonLVal LVal::EQ(ValueManager& ValMgr, const LVal& R) const {
  
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this LVal.");
      break;
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(this)->getValue() ==
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(ValMgr, b);
      }
      else if (isa<lval::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<lval::ConcreteInt>(this)->getValue());
        
        return nonlval::SymIntConstraintVal(C);        
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(R)) {
          
          const SymIntConstraint& C =
            ValMgr.getConstraint(cast<lval::SymbolVal>(this)->getSymbol(),
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
        return NonLVal::MakeIntTruthVal(ValMgr, b);
      }
      
      break;
  }
  
  return NonLVal::MakeIntTruthVal(ValMgr, false);
}

NonLVal LVal::NE(ValueManager& ValMgr, const LVal& R) const {
  switch (getSubKind()) {
    default:
      assert(false && "NE not implemented for this LVal.");
      break;
      
    case lval::ConcreteIntKind:
      if (isa<lval::ConcreteInt>(R)) {
        bool b = cast<lval::ConcreteInt>(this)->getValue() !=
                 cast<lval::ConcreteInt>(R).getValue();
        
        return NonLVal::MakeIntTruthVal(ValMgr, b);
      }
      else if (isa<lval::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
        ValMgr.getConstraint(cast<lval::SymbolVal>(R).getSymbol(),
                             BinaryOperator::NE,
                             cast<lval::ConcreteInt>(this)->getValue());
        
        return nonlval::SymIntConstraintVal(C);        
      }
      
      break;
      
      case lval::SymbolValKind: {
        if (isa<lval::ConcreteInt>(R)) {
          
          const SymIntConstraint& C =
          ValMgr.getConstraint(cast<lval::SymbolVal>(this)->getSymbol(),
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
          return NonLVal::MakeIntTruthVal(ValMgr, b);
        }
      
        break;
  }
  
  return NonLVal::MakeIntTruthVal(ValMgr, true);
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-LVals.
//===----------------------------------------------------------------------===//

NonLVal NonLVal::MakeVal(ValueManager& ValMgr, uint64_t X, QualType T,
                             SourceLocation Loc) {  

  return nonlval::ConcreteInt(ValMgr.getValue(X, T, Loc));
}

NonLVal NonLVal::MakeVal(ValueManager& ValMgr, IntegerLiteral* I) {

  return nonlval::ConcreteInt(ValMgr.getValue(APSInt(I->getValue(),
                              I->getType()->isUnsignedIntegerType())));
}

NonLVal NonLVal::MakeIntTruthVal(ValueManager& ValMgr, bool b) {

  return nonlval::ConcreteInt(ValMgr.getTruthValue(b));
}

RVal RVal::GetSymbolValue(SymbolManager& SymMgr, ParmVarDecl* D) {

  QualType T = D->getType();
  
  if (T->isPointerType() || T->isReferenceType())
    return lval::SymbolVal(SymMgr.getSymbol(D));
  else
    return nonlval::SymbolVal(SymMgr.getSymbol(D));
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing LVals.
//===----------------------------------------------------------------------===//

LVal LVal::MakeVal(AddrLabelExpr* E) { return lval::GotoLabel(E->getLabel()); }

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
      
    case UninitializedKind:
      Out << "Uninitialized"; break;
      
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
      assert (false && "Pretty-printed not implemented for this NonLVal.");
      break;
  }
}

void LVal::print(std::ostream& Out) const {
  
  switch (getSubKind()) {        

    case lval::ConcreteIntKind:
      Out << cast<lval::ConcreteInt>(this)->getValue().toString() 
          << " (LVal)";
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
      
    default:
      assert (false && "Pretty-printing not implemented for this LVal.");
      break;
  }
}
