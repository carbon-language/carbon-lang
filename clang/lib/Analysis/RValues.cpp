//= RValues.cpp - Abstract RValues for Path-Sens. Value Tracking -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SVal, Loc, and NonLoc, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/Support/Streams.h"

using namespace clang;
using llvm::dyn_cast;
using llvm::cast;
using llvm::APSInt;

//===----------------------------------------------------------------------===//
// Symbol Iteration.
//===----------------------------------------------------------------------===//

SVal::symbol_iterator SVal::symbol_begin() const {
  
  // FIXME: This is a rat's nest.  Cleanup.

  if (isa<loc::SymbolVal>(this))
    return (symbol_iterator) (&Data);
  else if (isa<nonloc::SymbolVal>(this))
    return (symbol_iterator) (&Data);
  else if (isa<nonloc::SymIntConstraintVal>(this)) {
    const SymIntConstraint& C =
      cast<nonloc::SymIntConstraintVal>(this)->getConstraint();
    
    return (symbol_iterator) &C.getSymbol();
  }
  else if (isa<nonloc::LocAsInteger>(this)) {
    const nonloc::LocAsInteger& V = cast<nonloc::LocAsInteger>(*this);
    return V.getPersistentLoc().symbol_begin();
  }
  
  // FIXME: We need to iterate over the symbols of regions.

  return NULL;
}

SVal::symbol_iterator SVal::symbol_end() const {
  symbol_iterator X = symbol_begin();
  return X ? X+1 : NULL;
}

//===----------------------------------------------------------------------===//
// Useful predicates.
//===----------------------------------------------------------------------===//

bool SVal::isZeroConstant() const {
  if (isa<loc::ConcreteInt>(*this))
    return cast<loc::ConcreteInt>(*this).getValue() == 0;
  else if (isa<nonloc::ConcreteInt>(*this))
    return cast<nonloc::ConcreteInt>(*this).getValue() == 0;
  else
    return false;
}


//===----------------------------------------------------------------------===//
// Transfer function dispatch for Non-Locs.
//===----------------------------------------------------------------------===//

SVal nonloc::ConcreteInt::EvalBinOp(BasicValueFactory& BasicVals,
                                     BinaryOperator::Opcode Op,
                                     const nonloc::ConcreteInt& R) const {
  
  const llvm::APSInt* X =
    BasicVals.EvaluateAPSInt(Op, getValue(), R.getValue());
  
  if (X)
    return nonloc::ConcreteInt(*X);
  else
    return UndefinedVal();
}

  // Bitwise-Complement.

nonloc::ConcreteInt
nonloc::ConcreteInt::EvalComplement(BasicValueFactory& BasicVals) const {
  return BasicVals.getValue(~getValue()); 
}

  // Unary Minus.

nonloc::ConcreteInt
nonloc::ConcreteInt::EvalMinus(BasicValueFactory& BasicVals, UnaryOperator* U) const {
  assert (U->getType() == U->getSubExpr()->getType());  
  assert (U->getType()->isIntegerType());  
  return BasicVals.getValue(-getValue()); 
}

//===----------------------------------------------------------------------===//
// Transfer function dispatch for Locs.
//===----------------------------------------------------------------------===//

SVal
loc::ConcreteInt::EvalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                             const loc::ConcreteInt& R) const {
  
  assert (Op == BinaryOperator::Add || Op == BinaryOperator::Sub ||
          (Op >= BinaryOperator::LT && Op <= BinaryOperator::NE));
  
  const llvm::APSInt* X = BasicVals.EvaluateAPSInt(Op, getValue(), R.getValue());
  
  if (X)
    return loc::ConcreteInt(*X);
  else
    return UndefinedVal();
}

NonLoc Loc::EQ(BasicValueFactory& BasicVals, const Loc& R) const {
  
  switch (getSubKind()) {
    default:
      assert(false && "EQ not implemented for this Loc.");
      break;
      
    case loc::ConcreteIntKind:
      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(this)->getValue() ==
                 cast<loc::ConcreteInt>(R).getValue();
        
        return NonLoc::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<loc::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
          BasicVals.getConstraint(cast<loc::SymbolVal>(R).getSymbol(),
                               BinaryOperator::EQ,
                               cast<loc::ConcreteInt>(this)->getValue());
        
        return nonloc::SymIntConstraintVal(C);        
      }
      
      break;
      
      case loc::SymbolValKind: {
        if (isa<loc::ConcreteInt>(R)) {
          
          const SymIntConstraint& C =
            BasicVals.getConstraint(cast<loc::SymbolVal>(this)->getSymbol(),
                                 BinaryOperator::EQ,
                                 cast<loc::ConcreteInt>(R).getValue());
          
          return nonloc::SymIntConstraintVal(C);
        }
        
        assert (!isa<loc::SymbolVal>(R) && "FIXME: Implement unification.");
        
        break;
      }
      
      case loc::MemRegionKind:
      if (isa<loc::MemRegionVal>(R)) {        
        bool b = cast<loc::MemRegionVal>(*this) == cast<loc::MemRegionVal>(R);
        return NonLoc::MakeIntTruthVal(BasicVals, b);
      }
      
      break;
  }
  
  return NonLoc::MakeIntTruthVal(BasicVals, false);
}

NonLoc Loc::NE(BasicValueFactory& BasicVals, const Loc& R) const {
  switch (getSubKind()) {
    default:
      assert(false && "NE not implemented for this Loc.");
      break;
      
    case loc::ConcreteIntKind:
      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(this)->getValue() !=
                 cast<loc::ConcreteInt>(R).getValue();
        
        return NonLoc::MakeIntTruthVal(BasicVals, b);
      }
      else if (isa<loc::SymbolVal>(R)) {
        
        const SymIntConstraint& C =
        BasicVals.getConstraint(cast<loc::SymbolVal>(R).getSymbol(),
                             BinaryOperator::NE,
                             cast<loc::ConcreteInt>(this)->getValue());
        
        return nonloc::SymIntConstraintVal(C);        
      }
      
      break;
      
      case loc::SymbolValKind: {
        if (isa<loc::ConcreteInt>(R)) {
          
          const SymIntConstraint& C =
          BasicVals.getConstraint(cast<loc::SymbolVal>(this)->getSymbol(),
                               BinaryOperator::NE,
                               cast<loc::ConcreteInt>(R).getValue());
          
          return nonloc::SymIntConstraintVal(C);
        }
        
        assert (!isa<loc::SymbolVal>(R) && "FIXME: Implement sym !=.");
        
        break;
      }
      
      case loc::MemRegionKind:
        if (isa<loc::MemRegionVal>(R)) {        
          bool b = cast<loc::MemRegionVal>(*this)==cast<loc::MemRegionVal>(R);
          return NonLoc::MakeIntTruthVal(BasicVals, b);
        }
      
        break;
  }
  
  return NonLoc::MakeIntTruthVal(BasicVals, true);
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Non-Locs.
//===----------------------------------------------------------------------===//

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, uint64_t X, QualType T) {  
  return nonloc::ConcreteInt(BasicVals.getValue(X, T));
}

NonLoc NonLoc::MakeVal(BasicValueFactory& BasicVals, IntegerLiteral* I) {

  return nonloc::ConcreteInt(BasicVals.getValue(APSInt(I->getValue(),
                              I->getType()->isUnsignedIntegerType())));
}

NonLoc NonLoc::MakeIntTruthVal(BasicValueFactory& BasicVals, bool b) {
  return nonloc::ConcreteInt(BasicVals.getTruthValue(b));
}

SVal SVal::GetSymbolValue(SymbolManager& SymMgr, VarDecl* D) {

  QualType T = D->getType();
  
  if (Loc::IsLocType(T))
    return loc::SymbolVal(SymMgr.getSymbol(D));
  
  return nonloc::SymbolVal(SymMgr.getSymbol(D));
}

//===----------------------------------------------------------------------===//
// Utility methods for constructing Locs.
//===----------------------------------------------------------------------===//

Loc Loc::MakeVal(AddrLabelExpr* E) { return loc::GotoLabel(E->getLabel()); }

Loc Loc::MakeVal(StringLiteral* S) {
  return loc::StringLiteralVal(S);
}

//===----------------------------------------------------------------------===//
// Pretty-Printing.
//===----------------------------------------------------------------------===//

void SVal::printStdErr() const { print(*llvm::cerr.stream()); }

void SVal::print(std::ostream& Out) const {

  switch (getBaseKind()) {
      
    case UnknownKind:
      Out << "Invalid"; break;
      
    case NonLocKind:
      cast<NonLoc>(this)->print(Out); break;
      
    case LocKind:
      cast<Loc>(this)->print(Out); break;
      
    case UndefinedKind:
      Out << "Undefined"; break;
      
    default:
      assert (false && "Invalid SVal.");
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

void NonLoc::print(std::ostream& Out) const {

  switch (getSubKind()) {  

    case nonloc::ConcreteIntKind:
      Out << cast<nonloc::ConcreteInt>(this)->getValue().getZExtValue();

      if (cast<nonloc::ConcreteInt>(this)->getValue().isUnsigned())
        Out << 'U';
      
      break;
      
    case nonloc::SymbolValKind:
      Out << '$' << cast<nonloc::SymbolVal>(this)->getSymbol();
      break;
     
    case nonloc::SymIntConstraintValKind: {
      const nonloc::SymIntConstraintVal& C = 
        *cast<nonloc::SymIntConstraintVal>(this);
      
      Out << '$' << C.getConstraint().getSymbol() << ' ';
      printOpcode(Out, C.getConstraint().getOpcode());
      Out << ' ' << C.getConstraint().getInt().getZExtValue();
      
      if (C.getConstraint().getInt().isUnsigned())
        Out << 'U';
      
      break;
    }
    
    case nonloc::LocAsIntegerKind: {
      const nonloc::LocAsInteger& C = *cast<nonloc::LocAsInteger>(this);
      C.getLoc().print(Out);
      Out << " [as " << C.getNumBits() << " bit integer]";
      break;
    }
      
    default:
      assert (false && "Pretty-printed not implemented for this NonLoc.");
      break;
  }
}

void Loc::print(std::ostream& Out) const {
  
  switch (getSubKind()) {        

    case loc::ConcreteIntKind:
      Out << cast<loc::ConcreteInt>(this)->getValue().getZExtValue()
          << " (Loc)";
      break;
      
    case loc::SymbolValKind:
      Out << '$' << cast<loc::SymbolVal>(this)->getSymbol();
      break;
      
    case loc::GotoLabelKind:
      Out << "&&"
          << cast<loc::GotoLabel>(this)->getLabel()->getID()->getName();
      break;

    case loc::MemRegionKind:
      Out << '&' << cast<loc::MemRegionVal>(this)->getRegion()->getString();
      break;
      
    case loc::FuncValKind:
      Out << "function " 
          << cast<loc::FuncVal>(this)->getDecl()->getIdentifier()->getName();
      break;
      
    case loc::StringLiteralValKind:
      Out << "literal \""
          << cast<loc::StringLiteralVal>(this)->getLiteral()->getStrData()
          << "\"";
      break;
      
    default:
      assert (false && "Pretty-printing not implemented for this Loc.");
      break;
  }
}
