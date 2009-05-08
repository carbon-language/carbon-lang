// GRSimpleVals.cpp - Transfer functions for tracking simple values -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines GRSimpleVals, a sub-class of GRTransferFuncs that
//  provides transfer functions for performing simple value tracking with
//  limited support for symbolics.
//
//===----------------------------------------------------------------------===//

#include "GRSimpleVals.h"
#include "BasicObjCFoundationChecks.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathSensitive/BugReporter.h"
#include "clang/Analysis/LocalCheckers.h"
#include "clang/Analysis/PathSensitive/GRExprEngine.h"
#include "llvm/Support/Compiler.h"
#include <sstream>

using namespace clang;

//===----------------------------------------------------------------------===//
// Transfer Function creation for External clients.
//===----------------------------------------------------------------------===//

GRTransferFuncs* clang::MakeGRSimpleValsTF() { return new GRSimpleVals(); }  

//===----------------------------------------------------------------------===//
// Transfer function for Casts.
//===----------------------------------------------------------------------===//

SVal GRSimpleVals::EvalCast(GRExprEngine& Eng, NonLoc X, QualType T) {
  
  if (!isa<nonloc::ConcreteInt>(X))
    return UnknownVal();

  bool isLocType = Loc::IsLocType(T);
  
  // Only handle casts from integers to integers.
  if (!isLocType && !T->isIntegerType())
    return UnknownVal();
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  
  llvm::APSInt V = cast<nonloc::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || Loc::IsLocType(T));
  V.extOrTrunc(Eng.getContext().getTypeSize(T));
  
  if (isLocType)
    return loc::ConcreteInt(BasicVals.getValue(V));
  else
    return nonloc::ConcreteInt(BasicVals.getValue(V));
}

// Casts.

SVal GRSimpleVals::EvalCast(GRExprEngine& Eng, Loc X, QualType T) {
  
  // Casts from pointers -> pointers, just return the lval.
  //
  // Casts from pointers -> references, just return the lval.  These
  //   can be introduced by the frontend for corner cases, e.g
  //   casting from va_list* to __builtin_va_list&.
  //
  assert (!X.isUnknownOrUndef());
  
  if (Loc::IsLocType(T) || T->isReferenceType())
    return X;
  
  // FIXME: Handle transparent unions where a value can be "transparently"
  //  lifted into a union type.
  if (T->isUnionType())
    return UnknownVal();
  
  assert (T->isIntegerType());
  BasicValueFactory& BasicVals = Eng.getBasicVals();
  unsigned BitWidth = Eng.getContext().getTypeSize(T);

  if (!isa<loc::ConcreteInt>(X))
    return nonloc::LocAsInteger::Make(BasicVals, X, BitWidth);
  
  llvm::APSInt V = cast<loc::ConcreteInt>(X).getValue();
  V.setIsUnsigned(T->isUnsignedIntegerType() || Loc::IsLocType(T));
  V.extOrTrunc(BitWidth);
  return nonloc::ConcreteInt(BasicVals.getValue(V));
}

// Unary operators.

SVal GRSimpleVals::EvalMinus(GRExprEngine& Eng, UnaryOperator* U, NonLoc X){
  
  switch (X.getSubKind()) {
      
    case nonloc::ConcreteIntKind:
      return cast<nonloc::ConcreteInt>(X).EvalMinus(Eng.getBasicVals(), U);
      
    default:
      return UnknownVal();
  }
}

SVal GRSimpleVals::EvalComplement(GRExprEngine& Eng, NonLoc X) {

  switch (X.getSubKind()) {
      
    case nonloc::ConcreteIntKind:
      return cast<nonloc::ConcreteInt>(X).EvalComplement(Eng.getBasicVals());
      
    default:
      return UnknownVal();
  }
}

// Binary operators.

static unsigned char LNotOpMap[] = {
  (unsigned char) BinaryOperator::GE,  /* LT => GE */
  (unsigned char) BinaryOperator::LE,  /* GT => LE */
  (unsigned char) BinaryOperator::GT,  /* LE => GT */
  (unsigned char) BinaryOperator::LT,  /* GE => LT */
  (unsigned char) BinaryOperator::NE,  /* EQ => NE */
  (unsigned char) BinaryOperator::EQ   /* NE => EQ */
};

SVal GRSimpleVals::DetermEvalBinOpNN(GRExprEngine& Eng,
                                     BinaryOperator::Opcode Op,
                                     NonLoc L, NonLoc R,
                                     QualType T)  {

  BasicValueFactory& BasicVals = Eng.getBasicVals();
  unsigned subkind = L.getSubKind();
  
  while (1) {
    
    switch (subkind) {
      default:
        return UnknownVal();
        
      case nonloc::LocAsIntegerKind: {
        Loc LL = cast<nonloc::LocAsInteger>(L).getLoc();        
        
        switch (R.getSubKind()) {
          case nonloc::LocAsIntegerKind:
            return EvalBinOp(Eng, Op, LL,
                             cast<nonloc::LocAsInteger>(R).getLoc());
            
          case nonloc::ConcreteIntKind: {
            // Transform the integer into a location and compare.
            ASTContext& Ctx = Eng.getContext();
            llvm::APSInt V = cast<nonloc::ConcreteInt>(R).getValue();
            V.setIsUnsigned(true);
            V.extOrTrunc(Ctx.getTypeSize(Ctx.VoidPtrTy));
            return EvalBinOp(Eng, Op, LL,
                             loc::ConcreteInt(BasicVals.getValue(V)));
          }
          
          default: 
            switch (Op) {
              case BinaryOperator::EQ:
                return NonLoc::MakeIntTruthVal(BasicVals, false);
              case BinaryOperator::NE:
                return NonLoc::MakeIntTruthVal(BasicVals, true);
              default:
                // This case also handles pointer arithmetic.
                return UnknownVal();
            }
        }
      }
        
      case nonloc::SymExprValKind: {
        // Logical not?        
        if (!(Op == BinaryOperator::EQ && R.isZeroConstant()))
          return UnknownVal();

        const SymExpr &SE=*cast<nonloc::SymExprVal>(L).getSymbolicExpression();
        
        // Only handle ($sym op constant) for now.
        if (const SymIntExpr *E = dyn_cast<SymIntExpr>(&SE)) {
          BinaryOperator::Opcode Opc = E->getOpcode();
        
          if (Opc < BinaryOperator::LT || Opc > BinaryOperator::NE)
            return UnknownVal();

          // For comparison operators, translate the constraint by
          // changing the opcode.        
          int idx = (unsigned) Opc - (unsigned) BinaryOperator::LT;
        
          assert (idx >= 0 && 
                  (unsigned) idx < sizeof(LNotOpMap)/sizeof(unsigned char));
        
          Opc = (BinaryOperator::Opcode) LNotOpMap[idx];          
          assert(E->getType(Eng.getContext()) == T);
          E = Eng.getSymbolManager().getSymIntExpr(E->getLHS(), Opc,
                                                   E->getRHS(), T);
          return nonloc::SymExprVal(E);
        }
        
        return UnknownVal();
      }
        
      case nonloc::ConcreteIntKind:
        
        if (isa<nonloc::ConcreteInt>(R)) {          
          const nonloc::ConcreteInt& L_CI = cast<nonloc::ConcreteInt>(L);
          const nonloc::ConcreteInt& R_CI = cast<nonloc::ConcreteInt>(R);
          return L_CI.EvalBinOp(BasicVals, Op, R_CI);          
        }
        else {
          subkind = R.getSubKind();
          NonLoc tmp = R;
          R = L;
          L = tmp;
          
          // Swap the operators.
          switch (Op) {
            case BinaryOperator::LT: Op = BinaryOperator::GT; break;
            case BinaryOperator::GT: Op = BinaryOperator::LT; break;
            case BinaryOperator::LE: Op = BinaryOperator::GE; break;
            case BinaryOperator::GE: Op = BinaryOperator::LE; break;
            default: break;
          }
          
          continue;
        }
        
      case nonloc::SymbolValKind:
        if (isa<nonloc::ConcreteInt>(R)) {
          ValueManager &ValMgr = Eng.getValueManager();
          return ValMgr.makeNonLoc(cast<nonloc::SymbolVal>(L).getSymbol(), Op,
                                   cast<nonloc::ConcreteInt>(R).getValue(), T);
        }
        else
          return UnknownVal();
    }
  }
}


// Binary Operators (except assignments and comma).

SVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             Loc L, Loc R) {
  
  switch (Op) {
    default:
      return UnknownVal();      
    case BinaryOperator::EQ:
    case BinaryOperator::NE:
      return EvalEquality(Eng, L, R, Op == BinaryOperator::EQ);
  }
}

SVal GRSimpleVals::EvalBinOp(GRExprEngine& Eng, BinaryOperator::Opcode Op,
                             Loc L, NonLoc R) {
  
  // Special case: 'R' is an integer that has the same width as a pointer and
  // we are using the integer location in a comparison.  Normally this cannot be
  // triggered, but transfer functions like those for OSCommpareAndSwapBarrier32
  // can generate comparisons that trigger this code.
  // FIXME: Are all locations guaranteed to have pointer width?
  if (BinaryOperator::isEqualityOp(Op)) {
    if (nonloc::ConcreteInt *RInt = dyn_cast<nonloc::ConcreteInt>(&R)) {
      const llvm::APSInt *X = &RInt->getValue();
      ASTContext &C = Eng.getContext();
      if (C.getTypeSize(C.VoidPtrTy) == X->getBitWidth()) {
        // Convert the signedness of the integer (if necessary).
        if (X->isSigned())
          X = &Eng.getBasicVals().getValue(*X, true);          
        
        return EvalBinOp(Eng, Op, L, loc::ConcreteInt(*X));
      }
    }
  }
  
  // Delegate pointer arithmetic to store manager.
  return Eng.getStoreManager().EvalBinOp(Op, L, R);
}

// Equality operators for Locs.  
// FIXME: All this logic will be revamped when we have MemRegion::getLocation()
// implemented.

SVal GRSimpleVals::EvalEquality(GRExprEngine& Eng, Loc L, Loc R, bool isEqual) {
  
  BasicValueFactory& BasicVals = Eng.getBasicVals();

  switch (L.getSubKind()) {

    default:
      assert(false && "EQ/NE not implemented for this Loc.");
      return UnknownVal();
      
    case loc::ConcreteIntKind:

      if (isa<loc::ConcreteInt>(R)) {
        bool b = cast<loc::ConcreteInt>(L).getValue() ==
                 cast<loc::ConcreteInt>(R).getValue();
        
        // Are we computing '!='?  Flip the result.
        if (!isEqual)
          b = !b;
        
        return NonLoc::MakeIntTruthVal(BasicVals, b);
      }
      else if (SymbolRef Sym = R.getAsSymbol()) {
        const SymIntExpr * SE =
        Eng.getSymbolManager().getSymIntExpr(Sym,
                                             isEqual ? BinaryOperator::EQ
                                                     : BinaryOperator::NE,
                                             cast<loc::ConcreteInt>(L).getValue(),
                                             Eng.getContext().IntTy);
        return nonloc::SymExprVal(SE);
      }
      
      break;
      
    case loc::MemRegionKind: {
      if (SymbolRef LSym = L.getAsLocSymbol()) {
        if (isa<loc::ConcreteInt>(R)) {
          const SymIntExpr *SE =
            Eng.getSymbolManager().getSymIntExpr(LSym,
                                           isEqual ? BinaryOperator::EQ
                                                   : BinaryOperator::NE,
                                           cast<loc::ConcreteInt>(R).getValue(),
                                           Eng.getContext().IntTy);
        
          return nonloc::SymExprVal(SE);
        }
      }
    }    
    
    // Fall-through.
      
    case loc::GotoLabelKind:
      return NonLoc::MakeIntTruthVal(BasicVals, isEqual ? L == R : L != R);
  }
  
  return NonLoc::MakeIntTruthVal(BasicVals, isEqual ? false : true);
}

//===----------------------------------------------------------------------===//
// Transfer function for function calls.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalCall(ExplodedNodeSet<GRState>& Dst,
                            GRExprEngine& Eng,
                            GRStmtNodeBuilder<GRState>& Builder,
                            CallExpr* CE, SVal L,
                            ExplodedNode<GRState>* Pred) {
  
  GRStateManager& StateMgr = Eng.getStateManager();
  const GRState* St = Builder.GetState(Pred);
  
  // Invalidate all arguments passed in by reference (Locs).

  for (CallExpr::arg_iterator I = CE->arg_begin(), E = CE->arg_end();
        I != E; ++I) {

    SVal V = StateMgr.GetSVal(St, *I);
    
    if (isa<loc::MemRegionVal>(V))
      St = StateMgr.BindLoc(St, cast<Loc>(V), UnknownVal());
    else if (isa<nonloc::LocAsInteger>(V))
      St = StateMgr.BindLoc(St, cast<nonloc::LocAsInteger>(V).getLoc(),
                            UnknownVal());
    
  }
  
  // Make up a symbol for the return value of this function.  
  // FIXME: We eventually should handle structs and other compound types
  // that are returned by value.
  QualType T = CE->getType();  
  if (Loc::IsLocType(T) || (T->isIntegerType() && T->isScalarType())) {    
    unsigned Count = Builder.getCurrentBlockCount();
    SVal X = Eng.getValueManager().getConjuredSymbolVal(CE, Count);
    St = StateMgr.BindExpr(St, CE, X, Eng.getCFG().isBlkExpr(CE), false);
  }  
    
  Builder.MakeNode(Dst, CE, Pred, St);
}

//===----------------------------------------------------------------------===//
// Transfer function for Objective-C message expressions.
//===----------------------------------------------------------------------===//

void GRSimpleVals::EvalObjCMessageExpr(ExplodedNodeSet<GRState>& Dst,
                                       GRExprEngine& Eng,
                                       GRStmtNodeBuilder<GRState>& Builder,
                                       ObjCMessageExpr* ME,
                                       ExplodedNode<GRState>* Pred) {
  
  
  // The basic transfer function logic for message expressions does nothing.
  // We just invalidate all arguments passed in by references.
  
  GRStateManager& StateMgr = Eng.getStateManager();
  const GRState* St = Builder.GetState(Pred);
  
  for (ObjCMessageExpr::arg_iterator I = ME->arg_begin(), E = ME->arg_end();
       I != E; ++I) {
    
    SVal V = StateMgr.GetSVal(St, *I);
    
    if (isa<Loc>(V))
      St = StateMgr.BindLoc(St, cast<Loc>(V), UnknownVal());
  }
  
  Builder.MakeNode(Dst, ME, Pred, St);
}
