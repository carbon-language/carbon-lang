//=-- ExprEngineC.cpp - ExprEngine support for C expressions ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines ExprEngine's support for C expressions.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/Analysis/Support/SaveAndRestore.h"

using namespace clang;
using namespace ento;
using llvm::APSInt;

void ExprEngine::VisitBinaryOperator(const BinaryOperator* B,
                                     ExplodedNode *Pred,
                                     ExplodedNodeSet &Dst) {

  Expr *LHS = B->getLHS()->IgnoreParens();
  Expr *RHS = B->getRHS()->IgnoreParens();
  
  // FIXME: Prechecks eventually go in ::Visit().
  ExplodedNodeSet CheckedSet;
  ExplodedNodeSet Tmp2;
  getCheckerManager().runCheckersForPreStmt(CheckedSet, Pred, B, *this);
    
  // With both the LHS and RHS evaluated, process the operation itself.    
  for (ExplodedNodeSet::iterator it=CheckedSet.begin(), ei=CheckedSet.end();
         it != ei; ++it) {
      
    const ProgramState *state = (*it)->getState();
    SVal LeftV = state->getSVal(LHS);
    SVal RightV = state->getSVal(RHS);
      
    BinaryOperator::Opcode Op = B->getOpcode();
      
    if (Op == BO_Assign) {
      // EXPERIMENTAL: "Conjured" symbols.
      // FIXME: Handle structs.
      if (RightV.isUnknown() ||
          !getConstraintManager().canReasonAbout(RightV)) {
        unsigned Count = Builder->getCurrentBlockCount();
        RightV = svalBuilder.getConjuredSymbolVal(NULL, B->getRHS(), Count);
      }
      // Simulate the effects of a "store":  bind the value of the RHS
      // to the L-Value represented by the LHS.
      SVal ExprVal = B->isLValue() ? LeftV : RightV;
      evalStore(Tmp2, B, LHS, *it, state->BindExpr(B, ExprVal), LeftV, RightV);
      continue;
    }
      
    if (!B->isAssignmentOp()) {
      // Process non-assignments except commas or short-circuited
      // logical expressions (LAnd and LOr).
      SVal Result = evalBinOp(state, Op, LeftV, RightV, B->getType());      
      if (Result.isUnknown()) {
        MakeNode(Tmp2, B, *it, state);
        continue;
      }        

      state = state->BindExpr(B, Result);      
      MakeNode(Tmp2, B, *it, state);
      continue;
    }
      
    assert (B->isCompoundAssignmentOp());
    
    switch (Op) {
      default:
        assert(0 && "Invalid opcode for compound assignment.");
      case BO_MulAssign: Op = BO_Mul; break;
      case BO_DivAssign: Op = BO_Div; break;
      case BO_RemAssign: Op = BO_Rem; break;
      case BO_AddAssign: Op = BO_Add; break;
      case BO_SubAssign: Op = BO_Sub; break;
      case BO_ShlAssign: Op = BO_Shl; break;
      case BO_ShrAssign: Op = BO_Shr; break;
      case BO_AndAssign: Op = BO_And; break;
      case BO_XorAssign: Op = BO_Xor; break;
      case BO_OrAssign:  Op = BO_Or;  break;
    }
      
    // Perform a load (the LHS).  This performs the checks for
    // null dereferences, and so on.
    ExplodedNodeSet Tmp;
    SVal location = LeftV;
    evalLoad(Tmp, LHS, *it, state, location);
    
    for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I != E;
         ++I) {

      state = (*I)->getState();
      SVal V = state->getSVal(LHS);
      
      // Get the computation type.
      QualType CTy =
        cast<CompoundAssignOperator>(B)->getComputationResultType();
      CTy = getContext().getCanonicalType(CTy);
      
      QualType CLHSTy =
        cast<CompoundAssignOperator>(B)->getComputationLHSType();
      CLHSTy = getContext().getCanonicalType(CLHSTy);
      
      QualType LTy = getContext().getCanonicalType(LHS->getType());
      
      // Promote LHS.
      V = svalBuilder.evalCast(V, CLHSTy, LTy);
      
      // Compute the result of the operation.
      SVal Result = svalBuilder.evalCast(evalBinOp(state, Op, V, RightV, CTy),
                                         B->getType(), CTy);
      
      // EXPERIMENTAL: "Conjured" symbols.
      // FIXME: Handle structs.
      
      SVal LHSVal;
      
      if (Result.isUnknown() ||
          !getConstraintManager().canReasonAbout(Result)) {
        
        unsigned Count = Builder->getCurrentBlockCount();
        
        // The symbolic value is actually for the type of the left-hand side
        // expression, not the computation type, as this is the value the
        // LValue on the LHS will bind to.
        LHSVal = svalBuilder.getConjuredSymbolVal(NULL, B->getRHS(), LTy,
                                                  Count);
        
        // However, we need to convert the symbol to the computation type.
        Result = svalBuilder.evalCast(LHSVal, CTy, LTy);
      }
      else {
        // The left-hand side may bind to a different value then the
        // computation type.
        LHSVal = svalBuilder.evalCast(Result, LTy, CTy);
      }
      
      // In C++, assignment and compound assignment operators return an 
      // lvalue.
      if (B->isLValue())
        state = state->BindExpr(B, location);
      else
        state = state->BindExpr(B, Result);
      
      evalStore(Tmp2, B, LHS, *I, state, location, LHSVal);
    }
  }
  
  // FIXME: postvisits eventually go in ::Visit()
  getCheckerManager().runCheckersForPostStmt(Dst, Tmp2, B, *this);
}

void ExprEngine::VisitBlockExpr(const BlockExpr *BE, ExplodedNode *Pred,
                                ExplodedNodeSet &Dst) {
  
  CanQualType T = getContext().getCanonicalType(BE->getType());
  SVal V = svalBuilder.getBlockPointer(BE->getBlockDecl(), T,
                                       Pred->getLocationContext());
  
  ExplodedNodeSet Tmp;
  MakeNode(Tmp, BE, Pred, Pred->getState()->BindExpr(BE, V),
           ProgramPoint::PostLValueKind);
  
  // FIXME: Move all post/pre visits to ::Visit().
  getCheckerManager().runCheckersForPostStmt(Dst, Tmp, BE, *this);
}

void ExprEngine::VisitCast(const CastExpr *CastE, const Expr *Ex, 
                           ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  
  ExplodedNodeSet dstPreStmt;
  getCheckerManager().runCheckersForPreStmt(dstPreStmt, Pred, CastE, *this);
  
  if (CastE->getCastKind() == CK_LValueToRValue ||
      CastE->getCastKind() == CK_GetObjCProperty) {
    for (ExplodedNodeSet::iterator I = dstPreStmt.begin(), E = dstPreStmt.end();
         I!=E; ++I) {
      ExplodedNode *subExprNode = *I;
      const ProgramState *state = subExprNode->getState();
      evalLoad(Dst, CastE, subExprNode, state, state->getSVal(Ex));
    }
    return;
  }
  
  // All other casts.  
  QualType T = CastE->getType();
  QualType ExTy = Ex->getType();
  
  if (const ExplicitCastExpr *ExCast=dyn_cast_or_null<ExplicitCastExpr>(CastE))
    T = ExCast->getTypeAsWritten();
  
  for (ExplodedNodeSet::iterator I = dstPreStmt.begin(), E = dstPreStmt.end();
       I != E; ++I) {
    
    Pred = *I;
    
    switch (CastE->getCastKind()) {
      case CK_LValueToRValue:
        assert(false && "LValueToRValue casts handled earlier.");
      case CK_GetObjCProperty:
        assert(false && "GetObjCProperty casts handled earlier.");
      case CK_ToVoid:
        Dst.Add(Pred);
        continue;
        // The analyzer doesn't do anything special with these casts,
        // since it understands retain/release semantics already.
      case CK_ObjCProduceObject:
      case CK_ObjCConsumeObject:
      case CK_ObjCReclaimReturnedObject: // Fall-through.
        // True no-ops.
      case CK_NoOp:
      case CK_FunctionToPointerDecay: {
        // Copy the SVal of Ex to CastE.
        const ProgramState *state = Pred->getState();
        SVal V = state->getSVal(Ex);
        state = state->BindExpr(CastE, V);
        MakeNode(Dst, CastE, Pred, state);
        continue;
      }
      case CK_Dependent:
      case CK_ArrayToPointerDecay:
      case CK_BitCast:
      case CK_LValueBitCast:
      case CK_IntegralCast:
      case CK_NullToPointer:
      case CK_IntegralToPointer:
      case CK_PointerToIntegral:
      case CK_PointerToBoolean:
      case CK_IntegralToBoolean:
      case CK_IntegralToFloating:
      case CK_FloatingToIntegral:
      case CK_FloatingToBoolean:
      case CK_FloatingCast:
      case CK_FloatingRealToComplex:
      case CK_FloatingComplexToReal:
      case CK_FloatingComplexToBoolean:
      case CK_FloatingComplexCast:
      case CK_FloatingComplexToIntegralComplex:
      case CK_IntegralRealToComplex:
      case CK_IntegralComplexToReal:
      case CK_IntegralComplexToBoolean:
      case CK_IntegralComplexCast:
      case CK_IntegralComplexToFloatingComplex:
      case CK_AnyPointerToObjCPointerCast:
      case CK_AnyPointerToBlockPointerCast:  
      case CK_ObjCObjectLValueCast: {
        // Delegate to SValBuilder to process.
        const ProgramState *state = Pred->getState();
        SVal V = state->getSVal(Ex);
        V = svalBuilder.evalCast(V, T, ExTy);
        state = state->BindExpr(CastE, V);
        MakeNode(Dst, CastE, Pred, state);
        continue;
      }
      case CK_DerivedToBase:
      case CK_UncheckedDerivedToBase: {
        // For DerivedToBase cast, delegate to the store manager.
        const ProgramState *state = Pred->getState();
        SVal val = state->getSVal(Ex);
        val = getStoreManager().evalDerivedToBase(val, T);
        state = state->BindExpr(CastE, val);
        MakeNode(Dst, CastE, Pred, state);
        continue;
      }
        // Various C++ casts that are not handled yet.
      case CK_Dynamic:
      case CK_ToUnion:
      case CK_BaseToDerived:
      case CK_NullToMemberPointer:
      case CK_BaseToDerivedMemberPointer:
      case CK_DerivedToBaseMemberPointer:
      case CK_UserDefinedConversion:
      case CK_ConstructorConversion:
      case CK_VectorSplat:
      case CK_MemberPointerToBoolean: {
        // Recover some path-sensitivty by conjuring a new value.
        QualType resultType = CastE->getType();
        if (CastE->isLValue())
          resultType = getContext().getPointerType(resultType);
        
        SVal result =
        svalBuilder.getConjuredSymbolVal(NULL, CastE, resultType,
                                         Builder->getCurrentBlockCount());
        
        const ProgramState *state = Pred->getState()->BindExpr(CastE, result);
        MakeNode(Dst, CastE, Pred, state);
        continue;
      }
    }
  }
}

void ExprEngine::VisitCompoundLiteralExpr(const CompoundLiteralExpr *CL,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  const InitListExpr *ILE 
    = cast<InitListExpr>(CL->getInitializer()->IgnoreParens());
  
  const ProgramState *state = Pred->getState();
  SVal ILV = state->getSVal(ILE);
  const LocationContext *LC = Pred->getLocationContext();
  state = state->bindCompoundLiteral(CL, LC, ILV);
  
  if (CL->isLValue())
    MakeNode(Dst, CL, Pred, state->BindExpr(CL, state->getLValue(CL, LC)));
  else
    MakeNode(Dst, CL, Pred, state->BindExpr(CL, ILV));
}

void ExprEngine::VisitDeclStmt(const DeclStmt *DS, ExplodedNode *Pred,
                               ExplodedNodeSet &Dst) {
  
  // FIXME: static variables may have an initializer, but the second
  //  time a function is called those values may not be current.
  //  This may need to be reflected in the CFG.
  
  // Assumption: The CFG has one DeclStmt per Decl.
  const Decl *D = *DS->decl_begin();
  
  if (!D || !isa<VarDecl>(D))
    return;
  
  // FIXME: all pre/post visits should eventually be handled by ::Visit().
  ExplodedNodeSet dstPreVisit;
  getCheckerManager().runCheckersForPreStmt(dstPreVisit, Pred, DS, *this);
  
  const VarDecl *VD = dyn_cast<VarDecl>(D);
  
  for (ExplodedNodeSet::iterator I = dstPreVisit.begin(), E = dstPreVisit.end();
       I!=E; ++I) {
    ExplodedNode *N = *I;
    const ProgramState *state = N->getState();
    
    // Decls without InitExpr are not initialized explicitly.
    const LocationContext *LC = N->getLocationContext();
    
    if (const Expr *InitEx = VD->getInit()) {
      SVal InitVal = state->getSVal(InitEx);
      
      // We bound the temp obj region to the CXXConstructExpr. Now recover
      // the lazy compound value when the variable is not a reference.
      if (AMgr.getLangOptions().CPlusPlus && VD->getType()->isRecordType() && 
          !VD->getType()->isReferenceType() && isa<loc::MemRegionVal>(InitVal)){
        InitVal = state->getSVal(cast<loc::MemRegionVal>(InitVal).getRegion());
        assert(isa<nonloc::LazyCompoundVal>(InitVal));
      }
      
      // Recover some path-sensitivity if a scalar value evaluated to
      // UnknownVal.
      if ((InitVal.isUnknown() ||
           !getConstraintManager().canReasonAbout(InitVal)) &&
          !VD->getType()->isReferenceType()) {
        InitVal = svalBuilder.getConjuredSymbolVal(NULL, InitEx,
                                                   Builder->getCurrentBlockCount());
      }
      
      evalBind(Dst, DS, N, state->getLValue(VD, LC), InitVal, true);
    }
    else {
      MakeNode(Dst, DS, N, state->bindDeclWithNoInit(state->getRegion(VD, LC)));
    }
  }
}

void ExprEngine::VisitLogicalExpr(const BinaryOperator* B, ExplodedNode *Pred,
                                  ExplodedNodeSet &Dst) {
  
  assert(B->getOpcode() == BO_LAnd ||
         B->getOpcode() == BO_LOr);
  
  const ProgramState *state = Pred->getState();
  SVal X = state->getSVal(B);
  assert(X.isUndef());
  
  const Expr *Ex = (const Expr*) cast<UndefinedVal>(X).getData();
  assert(Ex);
  
  if (Ex == B->getRHS()) {
    X = state->getSVal(Ex);
    
    // Handle undefined values.
    if (X.isUndef()) {
      MakeNode(Dst, B, Pred, state->BindExpr(B, X));
      return;
    }
    
    DefinedOrUnknownSVal XD = cast<DefinedOrUnknownSVal>(X);
    
    // We took the RHS.  Because the value of the '&&' or '||' expression must
    // evaluate to 0 or 1, we must assume the value of the RHS evaluates to 0
    // or 1.  Alternatively, we could take a lazy approach, and calculate this
    // value later when necessary.  We don't have the machinery in place for
    // this right now, and since most logical expressions are used for branches,
    // the payoff is not likely to be large.  Instead, we do eager evaluation.
    if (const ProgramState *newState = state->assume(XD, true))
      MakeNode(Dst, B, Pred,
               newState->BindExpr(B, svalBuilder.makeIntVal(1U, B->getType())));
    
    if (const ProgramState *newState = state->assume(XD, false))
      MakeNode(Dst, B, Pred,
               newState->BindExpr(B, svalBuilder.makeIntVal(0U, B->getType())));
  }
  else {
    // We took the LHS expression.  Depending on whether we are '&&' or
    // '||' we know what the value of the expression is via properties of
    // the short-circuiting.
    X = svalBuilder.makeIntVal(B->getOpcode() == BO_LAnd ? 0U : 1U,
                               B->getType());
    MakeNode(Dst, B, Pred, state->BindExpr(B, X));
  }
}

void ExprEngine::VisitInitListExpr(const InitListExpr *IE,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {

  const ProgramState *state = Pred->getState();
  QualType T = getContext().getCanonicalType(IE->getType());
  unsigned NumInitElements = IE->getNumInits();
  
  if (T->isArrayType() || T->isRecordType() || T->isVectorType()) {
    llvm::ImmutableList<SVal> vals = getBasicVals().getEmptySValList();
    
    // Handle base case where the initializer has no elements.
    // e.g: static int* myArray[] = {};
    if (NumInitElements == 0) {
      SVal V = svalBuilder.makeCompoundVal(T, vals);
      MakeNode(Dst, IE, Pred, state->BindExpr(IE, V));
      return;
    }
    
    for (InitListExpr::const_reverse_iterator it = IE->rbegin(),
         ei = IE->rend(); it != ei; ++it) {
      vals = getBasicVals().consVals(state->getSVal(cast<Expr>(*it)), vals);
    }
    
    MakeNode(Dst, IE, Pred,
             state->BindExpr(IE, svalBuilder.makeCompoundVal(T, vals)));
    return;
  }
  
  if (Loc::isLocType(T) || T->isIntegerType()) {
    assert(IE->getNumInits() == 1);
    const Expr *initEx = IE->getInit(0);
    MakeNode(Dst, IE, Pred, state->BindExpr(IE, state->getSVal(initEx)));
    return;
  }
  
  llvm_unreachable("unprocessed InitListExpr type");
}

void ExprEngine::VisitGuardedExpr(const Expr *Ex,
                                  const Expr *L, 
                                  const Expr *R,
                                  ExplodedNode *Pred,
                                  ExplodedNodeSet &Dst) {
  
  const ProgramState *state = Pred->getState();
  SVal X = state->getSVal(Ex);  
  assert (X.isUndef());  
  const Expr *SE = (Expr*) cast<UndefinedVal>(X).getData();
  assert(SE);
  X = state->getSVal(SE);
  
  // Make sure that we invalidate the previous binding.
  MakeNode(Dst, Ex, Pred, state->BindExpr(Ex, X, true));
}

void ExprEngine::
VisitOffsetOfExpr(const OffsetOfExpr *OOE, 
                  ExplodedNode *Pred, ExplodedNodeSet &Dst) {
  Expr::EvalResult Res;
  if (OOE->Evaluate(Res, getContext()) && Res.Val.isInt()) {
    const APSInt &IV = Res.Val.getInt();
    assert(IV.getBitWidth() == getContext().getTypeSize(OOE->getType()));
    assert(OOE->getType()->isIntegerType());
    assert(IV.isSigned() == OOE->getType()->isSignedIntegerOrEnumerationType());
    SVal X = svalBuilder.makeIntVal(IV);
    MakeNode(Dst, OOE, Pred, Pred->getState()->BindExpr(OOE, X));
    return;
  }
  // FIXME: Handle the case where __builtin_offsetof is not a constant.
  Dst.Add(Pred);
}


void ExprEngine::
VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *Ex,
                              ExplodedNode *Pred,
                              ExplodedNodeSet &Dst) {

  QualType T = Ex->getTypeOfArgument();
  
  if (Ex->getKind() == UETT_SizeOf) {
    if (!T->isIncompleteType() && !T->isConstantSizeType()) {
      assert(T->isVariableArrayType() && "Unknown non-constant-sized type.");
      
      // FIXME: Add support for VLA type arguments and VLA expressions.
      // When that happens, we should probably refactor VLASizeChecker's code.
      Dst.Add(Pred);
      return;
    }
    else if (T->getAs<ObjCObjectType>()) {
      // Some code tries to take the sizeof an ObjCObjectType, relying that
      // the compiler has laid out its representation.  Just report Unknown
      // for these.
      Dst.Add(Pred);
      return;
    }
  }
  
  Expr::EvalResult Result;
  Ex->Evaluate(Result, getContext());
  CharUnits amt = CharUnits::fromQuantity(Result.Val.getInt().getZExtValue());
  
  const ProgramState *state = Pred->getState();
  state = state->BindExpr(Ex, svalBuilder.makeIntVal(amt.getQuantity(),
                                                     Ex->getType()));
  MakeNode(Dst, Ex, Pred, state);
}

void ExprEngine::VisitUnaryOperator(const UnaryOperator* U, 
                                    ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {  
  switch (U->getOpcode()) {
    default:
      break;
    case UO_Real: {
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        
        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }
        
        // For all other types, UO_Real is an identity operation.
        assert (U->getType() == Ex->getType());
        const ProgramState *state = (*I)->getState();
        MakeNode(Dst, U, *I, state->BindExpr(U, state->getSVal(Ex)));
      }
      
      return;
    }
      
    case UO_Imag: {
      
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        // FIXME: We don't have complex SValues yet.
        if (Ex->getType()->isAnyComplexType()) {
          // Just report "Unknown."
          Dst.Add(*I);
          continue;
        }
        
        // For all other types, UO_Imag returns 0.
        const ProgramState *state = (*I)->getState();
        SVal X = svalBuilder.makeZeroVal(Ex->getType());
        MakeNode(Dst, U, *I, state->BindExpr(U, X));
      }
      
      return;
    }
      
    case UO_Plus:
      assert(!U->isLValue());
      // FALL-THROUGH.
    case UO_Deref:
    case UO_AddrOf:
    case UO_Extension: {
      
      // Unary "+" is a no-op, similar to a parentheses.  We still have places
      // where it may be a block-level expression, so we need to
      // generate an extra node that just propagates the value of the
      // subexpression.
      
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const ProgramState *state = (*I)->getState();
        MakeNode(Dst, U, *I, state->BindExpr(U, state->getSVal(Ex)));
      }
      
      return;
    }
      
    case UO_LNot:
    case UO_Minus:
    case UO_Not: {
      assert (!U->isLValue());
      const Expr *Ex = U->getSubExpr()->IgnoreParens();
      ExplodedNodeSet Tmp;
      Visit(Ex, Pred, Tmp);
      
      for (ExplodedNodeSet::iterator I=Tmp.begin(), E=Tmp.end(); I!=E; ++I) {
        const ProgramState *state = (*I)->getState();
        
        // Get the value of the subexpression.
        SVal V = state->getSVal(Ex);
        
        if (V.isUnknownOrUndef()) {
          MakeNode(Dst, U, *I, state->BindExpr(U, V));
          continue;
        }
        
        switch (U->getOpcode()) {
          default:
            assert(false && "Invalid Opcode.");
            break;
            
          case UO_Not:
            // FIXME: Do we need to handle promotions?
            state = state->BindExpr(U, evalComplement(cast<NonLoc>(V)));
            break;
            
          case UO_Minus:
            // FIXME: Do we need to handle promotions?
            state = state->BindExpr(U, evalMinus(cast<NonLoc>(V)));
            break;
            
          case UO_LNot:
            
            // C99 6.5.3.3: "The expression !E is equivalent to (0==E)."
            //
            //  Note: technically we do "E == 0", but this is the same in the
            //    transfer functions as "0 == E".
            SVal Result;
            
            if (isa<Loc>(V)) {
              Loc X = svalBuilder.makeNull();
              Result = evalBinOp(state, BO_EQ, cast<Loc>(V), X,
                                 U->getType());
            }
            else {
              nonloc::ConcreteInt X(getBasicVals().getValue(0, Ex->getType()));
              Result = evalBinOp(state, BO_EQ, cast<NonLoc>(V), X,
                                 U->getType());
            }
            
            state = state->BindExpr(U, Result);
            
            break;
        }
        
        MakeNode(Dst, U, *I, state);
      }
      
      return;
    }
  }
  
  // Handle ++ and -- (both pre- and post-increment).
  assert (U->isIncrementDecrementOp());
  ExplodedNodeSet Tmp;
  const Expr *Ex = U->getSubExpr()->IgnoreParens();
  Visit(Ex, Pred, Tmp);
  
  for (ExplodedNodeSet::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I) {
    
    const ProgramState *state = (*I)->getState();
    SVal loc = state->getSVal(Ex);
    
    // Perform a load.
    ExplodedNodeSet Tmp2;
    evalLoad(Tmp2, Ex, *I, state, loc);
    
    for (ExplodedNodeSet::iterator I2=Tmp2.begin(), E2=Tmp2.end();I2!=E2;++I2) {
      
      state = (*I2)->getState();
      SVal V2_untested = state->getSVal(Ex);
      
      // Propagate unknown and undefined values.
      if (V2_untested.isUnknownOrUndef()) {
        MakeNode(Dst, U, *I2, state->BindExpr(U, V2_untested));
        continue;
      }
      DefinedSVal V2 = cast<DefinedSVal>(V2_untested);
      
      // Handle all other values.
      BinaryOperator::Opcode Op = U->isIncrementOp() ? BO_Add
      : BO_Sub;
      
      // If the UnaryOperator has non-location type, use its type to create the
      // constant value. If the UnaryOperator has location type, create the
      // constant with int type and pointer width.
      SVal RHS;
      
      if (U->getType()->isAnyPointerType())
        RHS = svalBuilder.makeArrayIndex(1);
      else
        RHS = svalBuilder.makeIntVal(1, U->getType());
      
      SVal Result = evalBinOp(state, Op, V2, RHS, U->getType());
      
      // Conjure a new symbol if necessary to recover precision.
      if (Result.isUnknown() || !getConstraintManager().canReasonAbout(Result)){
        DefinedOrUnknownSVal SymVal =
        svalBuilder.getConjuredSymbolVal(NULL, Ex,
                                         Builder->getCurrentBlockCount());
        Result = SymVal;
        
        // If the value is a location, ++/-- should always preserve
        // non-nullness.  Check if the original value was non-null, and if so
        // propagate that constraint.
        if (Loc::isLocType(U->getType())) {
          DefinedOrUnknownSVal Constraint =
          svalBuilder.evalEQ(state, V2,svalBuilder.makeZeroVal(U->getType()));
          
          if (!state->assume(Constraint, true)) {
            // It isn't feasible for the original value to be null.
            // Propagate this constraint.
            Constraint = svalBuilder.evalEQ(state, SymVal,
                                            svalBuilder.makeZeroVal(U->getType()));
            
            
            state = state->assume(Constraint, false);
            assert(state);
          }
        }
      }
      
      // Since the lvalue-to-rvalue conversion is explicit in the AST,
      // we bind an l-value if the operator is prefix and an lvalue (in C++).
      if (U->isLValue())
        state = state->BindExpr(U, loc);
      else
        state = state->BindExpr(U, U->isPostfix() ? V2 : Result);
      
      // Perform the store.
      evalStore(Dst, NULL, U, *I2, state, loc, Result);
    }
  }
}
