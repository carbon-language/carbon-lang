//===--- CGExprAgg.cpp - Emit LLVM Code from Aggregate Expressions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Aggregate Expr nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "CGObjCRuntime.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;

//===----------------------------------------------------------------------===//
//                        Aggregate Expression Emitter
//===----------------------------------------------------------------------===//

namespace  {
class AggExprEmitter : public StmtVisitor<AggExprEmitter> {
  CodeGenFunction &CGF;
  CGBuilderTy &Builder;
  AggValueSlot Dest;
  bool IgnoreResult;

  ReturnValueSlot getReturnValueSlot() const {
    // If the destination slot requires garbage collection, we can't
    // use the real return value slot, because we have to use the GC
    // API.
    if (Dest.requiresGCollection()) return ReturnValueSlot();

    return ReturnValueSlot(Dest.getAddr(), Dest.isVolatile());
  }

  AggValueSlot EnsureSlot(QualType T) {
    if (!Dest.isIgnored()) return Dest;
    return CGF.CreateAggTemp(T, "agg.tmp.ensured");
  }

public:
  AggExprEmitter(CodeGenFunction &cgf, AggValueSlot Dest,
                 bool ignore)
    : CGF(cgf), Builder(CGF.Builder), Dest(Dest),
      IgnoreResult(ignore) {
  }

  //===--------------------------------------------------------------------===//
  //                               Utilities
  //===--------------------------------------------------------------------===//

  /// EmitAggLoadOfLValue - Given an expression with aggregate type that
  /// represents a value lvalue, this method emits the address of the lvalue,
  /// then loads the result into DestPtr.
  void EmitAggLoadOfLValue(const Expr *E);

  /// EmitFinalDestCopy - Perform the final copy to DestPtr, if desired.
  void EmitFinalDestCopy(const Expr *E, LValue Src, bool Ignore = false);
  void EmitFinalDestCopy(const Expr *E, RValue Src, bool Ignore = false);

  void EmitGCMove(const Expr *E, RValue Src);

  bool TypeRequiresGCollection(QualType T);

  //===--------------------------------------------------------------------===//
  //                            Visitor Methods
  //===--------------------------------------------------------------------===//

  void VisitStmt(Stmt *S) {
    CGF.ErrorUnsupported(S, "aggregate expression");
  }
  void VisitParenExpr(ParenExpr *PE) { Visit(PE->getSubExpr()); }
  void VisitGenericSelectionExpr(GenericSelectionExpr *GE) {
    Visit(GE->getResultExpr());
  }
  void VisitUnaryExtension(UnaryOperator *E) { Visit(E->getSubExpr()); }

  // l-values.
  void VisitDeclRefExpr(DeclRefExpr *DRE) { EmitAggLoadOfLValue(DRE); }
  void VisitMemberExpr(MemberExpr *ME) { EmitAggLoadOfLValue(ME); }
  void VisitUnaryDeref(UnaryOperator *E) { EmitAggLoadOfLValue(E); }
  void VisitStringLiteral(StringLiteral *E) { EmitAggLoadOfLValue(E); }
  void VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    EmitAggLoadOfLValue(E);
  }
  void VisitBlockDeclRefExpr(const BlockDeclRefExpr *E) {
    EmitAggLoadOfLValue(E);
  }
  void VisitPredefinedExpr(const PredefinedExpr *E) {
    EmitAggLoadOfLValue(E);
  }

  // Operators.
  void VisitCastExpr(CastExpr *E);
  void VisitCallExpr(const CallExpr *E);
  void VisitStmtExpr(const StmtExpr *E);
  void VisitBinaryOperator(const BinaryOperator *BO);
  void VisitPointerToDataMemberBinaryOperator(const BinaryOperator *BO);
  void VisitBinAssign(const BinaryOperator *E);
  void VisitBinComma(const BinaryOperator *E);

  void VisitObjCMessageExpr(ObjCMessageExpr *E);
  void VisitObjCIvarRefExpr(ObjCIvarRefExpr *E) {
    EmitAggLoadOfLValue(E);
  }
  void VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E);

  void VisitAbstractConditionalOperator(const AbstractConditionalOperator *CO);
  void VisitChooseExpr(const ChooseExpr *CE);
  void VisitInitListExpr(InitListExpr *E);
  void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
  void VisitCXXDefaultArgExpr(CXXDefaultArgExpr *DAE) {
    Visit(DAE->getExpr());
  }
  void VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E);
  void VisitCXXConstructExpr(const CXXConstructExpr *E);
  void VisitExprWithCleanups(ExprWithCleanups *E);
  void VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E);
  void VisitCXXTypeidExpr(CXXTypeidExpr *E) { EmitAggLoadOfLValue(E); }

  void VisitOpaqueValueExpr(OpaqueValueExpr *E);

  void VisitVAArgExpr(VAArgExpr *E);

  void EmitInitializationToLValue(Expr *E, LValue Address);
  void EmitNullInitializationToLValue(LValue Address);
  //  case Expr::ChooseExprClass:
  void VisitCXXThrowExpr(const CXXThrowExpr *E) { CGF.EmitCXXThrowExpr(E); }
};
}  // end anonymous namespace.

//===----------------------------------------------------------------------===//
//                                Utilities
//===----------------------------------------------------------------------===//

/// EmitAggLoadOfLValue - Given an expression with aggregate type that
/// represents a value lvalue, this method emits the address of the lvalue,
/// then loads the result into DestPtr.
void AggExprEmitter::EmitAggLoadOfLValue(const Expr *E) {
  LValue LV = CGF.EmitLValue(E);
  EmitFinalDestCopy(E, LV);
}

/// \brief True if the given aggregate type requires special GC API calls.
bool AggExprEmitter::TypeRequiresGCollection(QualType T) {
  // Only record types have members that might require garbage collection.
  const RecordType *RecordTy = T->getAs<RecordType>();
  if (!RecordTy) return false;

  // Don't mess with non-trivial C++ types.
  RecordDecl *Record = RecordTy->getDecl();
  if (isa<CXXRecordDecl>(Record) &&
      (!cast<CXXRecordDecl>(Record)->hasTrivialCopyConstructor() ||
       !cast<CXXRecordDecl>(Record)->hasTrivialDestructor()))
    return false;

  // Check whether the type has an object member.
  return Record->hasObjectMember();
}

/// \brief Perform the final move to DestPtr if RequiresGCollection is set.
///
/// The idea is that you do something like this:
///   RValue Result = EmitSomething(..., getReturnValueSlot());
///   EmitGCMove(E, Result);
/// If GC doesn't interfere, this will cause the result to be emitted
/// directly into the return value slot.  If GC does interfere, a final
/// move will be performed.
void AggExprEmitter::EmitGCMove(const Expr *E, RValue Src) {
  if (Dest.requiresGCollection()) {
    CharUnits size = CGF.getContext().getTypeSizeInChars(E->getType());
    const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
    llvm::Value *SizeVal = llvm::ConstantInt::get(SizeTy, size.getQuantity());
    CGF.CGM.getObjCRuntime().EmitGCMemmoveCollectable(CGF, Dest.getAddr(),
                                                    Src.getAggregateAddr(),
                                                    SizeVal);
  }
}

/// EmitFinalDestCopy - Perform the final copy to DestPtr, if desired.
void AggExprEmitter::EmitFinalDestCopy(const Expr *E, RValue Src, bool Ignore) {
  assert(Src.isAggregate() && "value must be aggregate value!");

  // If Dest is ignored, then we're evaluating an aggregate expression
  // in a context (like an expression statement) that doesn't care
  // about the result.  C says that an lvalue-to-rvalue conversion is
  // performed in these cases; C++ says that it is not.  In either
  // case, we don't actually need to do anything unless the value is
  // volatile.
  if (Dest.isIgnored()) {
    if (!Src.isVolatileQualified() ||
        CGF.CGM.getLangOptions().CPlusPlus ||
        (IgnoreResult && Ignore))
      return;

    // If the source is volatile, we must read from it; to do that, we need
    // some place to put it.
    Dest = CGF.CreateAggTemp(E->getType(), "agg.tmp");
  }

  if (Dest.requiresGCollection()) {
    CharUnits size = CGF.getContext().getTypeSizeInChars(E->getType());
    const llvm::Type *SizeTy = CGF.ConvertType(CGF.getContext().getSizeType());
    llvm::Value *SizeVal = llvm::ConstantInt::get(SizeTy, size.getQuantity());
    CGF.CGM.getObjCRuntime().EmitGCMemmoveCollectable(CGF,
                                                      Dest.getAddr(),
                                                      Src.getAggregateAddr(),
                                                      SizeVal);
    return;
  }
  // If the result of the assignment is used, copy the LHS there also.
  // FIXME: Pass VolatileDest as well.  I think we also need to merge volatile
  // from the source as well, as we can't eliminate it if either operand
  // is volatile, unless copy has volatile for both source and destination..
  CGF.EmitAggregateCopy(Dest.getAddr(), Src.getAggregateAddr(), E->getType(),
                        Dest.isVolatile()|Src.isVolatileQualified());
}

/// EmitFinalDestCopy - Perform the final copy to DestPtr, if desired.
void AggExprEmitter::EmitFinalDestCopy(const Expr *E, LValue Src, bool Ignore) {
  assert(Src.isSimple() && "Can't have aggregate bitfield, vector, etc");

  EmitFinalDestCopy(E, RValue::getAggregate(Src.getAddress(),
                                            Src.isVolatileQualified()),
                    Ignore);
}

//===----------------------------------------------------------------------===//
//                            Visitor Methods
//===----------------------------------------------------------------------===//

void AggExprEmitter::VisitOpaqueValueExpr(OpaqueValueExpr *e) {
  EmitFinalDestCopy(e, CGF.getOpaqueLValueMapping(e));
}

void
AggExprEmitter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  if (E->getType().isPODType(CGF.getContext())) {
    // For a POD type, just emit a load of the lvalue + a copy, because our
    // compound literal might alias the destination.
    // FIXME: This is a band-aid; the real problem appears to be in our handling
    // of assignments, where we store directly into the LHS without checking
    // whether anything in the RHS aliases.
    EmitAggLoadOfLValue(E);
    return;
  }
  
  AggValueSlot Slot = EnsureSlot(E->getType());
  CGF.EmitAggExpr(E->getInitializer(), Slot);
}


void AggExprEmitter::VisitCastExpr(CastExpr *E) {
  switch (E->getCastKind()) {
  case CK_Dynamic: {
    assert(isa<CXXDynamicCastExpr>(E) && "CK_Dynamic without a dynamic_cast?");
    LValue LV = CGF.EmitCheckedLValue(E->getSubExpr());
    // FIXME: Do we also need to handle property references here?
    if (LV.isSimple())
      CGF.EmitDynamicCast(LV.getAddress(), cast<CXXDynamicCastExpr>(E));
    else
      CGF.CGM.ErrorUnsupported(E, "non-simple lvalue dynamic_cast");
    
    if (!Dest.isIgnored())
      CGF.CGM.ErrorUnsupported(E, "lvalue dynamic_cast with a destination");
    break;
  }
      
  case CK_ToUnion: {
    if (Dest.isIgnored()) break;

    // GCC union extension
    QualType Ty = E->getSubExpr()->getType();
    QualType PtrTy = CGF.getContext().getPointerType(Ty);
    llvm::Value *CastPtr = Builder.CreateBitCast(Dest.getAddr(),
                                                 CGF.ConvertType(PtrTy));
    EmitInitializationToLValue(E->getSubExpr(),
                               CGF.MakeAddrLValue(CastPtr, Ty));
    break;
  }

  case CK_DerivedToBase:
  case CK_BaseToDerived:
  case CK_UncheckedDerivedToBase: {
    assert(0 && "cannot perform hierarchy conversion in EmitAggExpr: "
                "should have been unpacked before we got here");
    break;
  }

  case CK_GetObjCProperty: {
    LValue LV = CGF.EmitLValue(E->getSubExpr());
    assert(LV.isPropertyRef());
    RValue RV = CGF.EmitLoadOfPropertyRefLValue(LV, getReturnValueSlot());
    EmitGCMove(E, RV);
    break;
  }

  case CK_LValueToRValue: // hope for downstream optimization
  case CK_NoOp:
  case CK_UserDefinedConversion:
  case CK_ConstructorConversion:
    assert(CGF.getContext().hasSameUnqualifiedType(E->getSubExpr()->getType(),
                                                   E->getType()) &&
           "Implicit cast types must be compatible");
    Visit(E->getSubExpr());
    break;
      
  case CK_LValueBitCast:
    llvm_unreachable("should not be emitting lvalue bitcast as rvalue");
    break;

  case CK_Dependent:
  case CK_BitCast:
  case CK_ArrayToPointerDecay:
  case CK_FunctionToPointerDecay:
  case CK_NullToPointer:
  case CK_NullToMemberPointer:
  case CK_BaseToDerivedMemberPointer:
  case CK_DerivedToBaseMemberPointer:
  case CK_MemberPointerToBoolean:
  case CK_IntegralToPointer:
  case CK_PointerToIntegral:
  case CK_PointerToBoolean:
  case CK_ToVoid:
  case CK_VectorSplat:
  case CK_IntegralCast:
  case CK_IntegralToBoolean:
  case CK_IntegralToFloating:
  case CK_FloatingToIntegral:
  case CK_FloatingToBoolean:
  case CK_FloatingCast:
  case CK_AnyPointerToObjCPointerCast:
  case CK_AnyPointerToBlockPointerCast:
  case CK_ObjCObjectLValueCast:
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
  case CK_ObjCProduceObject:
  case CK_ObjCConsumeObject:
    llvm_unreachable("cast kind invalid for aggregate types");
  }
}

void AggExprEmitter::VisitCallExpr(const CallExpr *E) {
  if (E->getCallReturnType()->isReferenceType()) {
    EmitAggLoadOfLValue(E);
    return;
  }

  RValue RV = CGF.EmitCallExpr(E, getReturnValueSlot());
  EmitGCMove(E, RV);
}

void AggExprEmitter::VisitObjCMessageExpr(ObjCMessageExpr *E) {
  RValue RV = CGF.EmitObjCMessageExpr(E, getReturnValueSlot());
  EmitGCMove(E, RV);
}

void AggExprEmitter::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *E) {
  llvm_unreachable("direct property access not surrounded by "
                   "lvalue-to-rvalue cast");
}

void AggExprEmitter::VisitBinComma(const BinaryOperator *E) {
  CGF.EmitIgnoredExpr(E->getLHS());
  Visit(E->getRHS());
}

void AggExprEmitter::VisitStmtExpr(const StmtExpr *E) {
  CodeGenFunction::StmtExprEvaluation eval(CGF);
  CGF.EmitCompoundStmt(*E->getSubStmt(), true, Dest);
}

void AggExprEmitter::VisitBinaryOperator(const BinaryOperator *E) {
  if (E->getOpcode() == BO_PtrMemD || E->getOpcode() == BO_PtrMemI)
    VisitPointerToDataMemberBinaryOperator(E);
  else
    CGF.ErrorUnsupported(E, "aggregate binary expression");
}

void AggExprEmitter::VisitPointerToDataMemberBinaryOperator(
                                                    const BinaryOperator *E) {
  LValue LV = CGF.EmitPointerToDataMemberBinaryExpr(E);
  EmitFinalDestCopy(E, LV);
}

void AggExprEmitter::VisitBinAssign(const BinaryOperator *E) {
  // For an assignment to work, the value on the right has
  // to be compatible with the value on the left.
  assert(CGF.getContext().hasSameUnqualifiedType(E->getLHS()->getType(),
                                                 E->getRHS()->getType())
         && "Invalid assignment");

  if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E->getLHS()))
    if (const VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl()))
      if (VD->hasAttr<BlocksAttr>() &&
          E->getRHS()->HasSideEffects(CGF.getContext())) {
        // When __block variable on LHS, the RHS must be evaluated first 
        // as it may change the 'forwarding' field via call to Block_copy.
        LValue RHS = CGF.EmitLValue(E->getRHS());
        LValue LHS = CGF.EmitLValue(E->getLHS());
        bool GCollection = false;
        if (CGF.getContext().getLangOptions().getGCMode())
          GCollection = TypeRequiresGCollection(E->getLHS()->getType());
        Dest = AggValueSlot::forLValue(LHS, true, GCollection);
        EmitFinalDestCopy(E, RHS, true);
        return;
      }
  
  LValue LHS = CGF.EmitLValue(E->getLHS());

  // We have to special case property setters, otherwise we must have
  // a simple lvalue (no aggregates inside vectors, bitfields).
  if (LHS.isPropertyRef()) {
    const ObjCPropertyRefExpr *RE = LHS.getPropertyRefExpr();
    QualType ArgType = RE->getSetterArgType();
    RValue Src;
    if (ArgType->isReferenceType())
      Src = CGF.EmitReferenceBindingToExpr(E->getRHS(), 0);
    else {
      AggValueSlot Slot = EnsureSlot(E->getRHS()->getType());
      CGF.EmitAggExpr(E->getRHS(), Slot);
      Src = Slot.asRValue();
    }
    CGF.EmitStoreThroughPropertyRefLValue(Src, LHS);
  } else {
    bool GCollection = false;
    if (CGF.getContext().getLangOptions().getGCMode())
      GCollection = TypeRequiresGCollection(E->getLHS()->getType());

    // Codegen the RHS so that it stores directly into the LHS.
    AggValueSlot LHSSlot = AggValueSlot::forLValue(LHS, true, 
                                                   GCollection);
    CGF.EmitAggExpr(E->getRHS(), LHSSlot, false);
    EmitFinalDestCopy(E, LHS, true);
  }
}

void AggExprEmitter::
VisitAbstractConditionalOperator(const AbstractConditionalOperator *E) {
  llvm::BasicBlock *LHSBlock = CGF.createBasicBlock("cond.true");
  llvm::BasicBlock *RHSBlock = CGF.createBasicBlock("cond.false");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("cond.end");

  // Bind the common expression if necessary.
  CodeGenFunction::OpaqueValueMapping binding(CGF, E);

  CodeGenFunction::ConditionalEvaluation eval(CGF);
  CGF.EmitBranchOnBoolExpr(E->getCond(), LHSBlock, RHSBlock);

  // Save whether the destination's lifetime is externally managed.
  bool DestLifetimeManaged = Dest.isLifetimeExternallyManaged();

  eval.begin(CGF);
  CGF.EmitBlock(LHSBlock);
  Visit(E->getTrueExpr());
  eval.end(CGF);

  assert(CGF.HaveInsertPoint() && "expression evaluation ended with no IP!");
  CGF.Builder.CreateBr(ContBlock);

  // If the result of an agg expression is unused, then the emission
  // of the LHS might need to create a destination slot.  That's fine
  // with us, and we can safely emit the RHS into the same slot, but
  // we shouldn't claim that its lifetime is externally managed.
  Dest.setLifetimeExternallyManaged(DestLifetimeManaged);

  eval.begin(CGF);
  CGF.EmitBlock(RHSBlock);
  Visit(E->getFalseExpr());
  eval.end(CGF);

  CGF.EmitBlock(ContBlock);
}

void AggExprEmitter::VisitChooseExpr(const ChooseExpr *CE) {
  Visit(CE->getChosenSubExpr(CGF.getContext()));
}

void AggExprEmitter::VisitVAArgExpr(VAArgExpr *VE) {
  llvm::Value *ArgValue = CGF.EmitVAListRef(VE->getSubExpr());
  llvm::Value *ArgPtr = CGF.EmitVAArg(ArgValue, VE->getType());

  if (!ArgPtr) {
    CGF.ErrorUnsupported(VE, "aggregate va_arg expression");
    return;
  }

  EmitFinalDestCopy(VE, CGF.MakeAddrLValue(ArgPtr, VE->getType()));
}

void AggExprEmitter::VisitCXXBindTemporaryExpr(CXXBindTemporaryExpr *E) {
  // Ensure that we have a slot, but if we already do, remember
  // whether its lifetime was externally managed.
  bool WasManaged = Dest.isLifetimeExternallyManaged();
  Dest = EnsureSlot(E->getType());
  Dest.setLifetimeExternallyManaged();

  Visit(E->getSubExpr());

  // Set up the temporary's destructor if its lifetime wasn't already
  // being managed.
  if (!WasManaged)
    CGF.EmitCXXTemporary(E->getTemporary(), Dest.getAddr());
}

void
AggExprEmitter::VisitCXXConstructExpr(const CXXConstructExpr *E) {
  AggValueSlot Slot = EnsureSlot(E->getType());
  CGF.EmitCXXConstructExpr(E, Slot);
}

void AggExprEmitter::VisitExprWithCleanups(ExprWithCleanups *E) {
  CGF.EmitExprWithCleanups(E, Dest);
}

void AggExprEmitter::VisitCXXScalarValueInitExpr(CXXScalarValueInitExpr *E) {
  QualType T = E->getType();
  AggValueSlot Slot = EnsureSlot(T);
  EmitNullInitializationToLValue(CGF.MakeAddrLValue(Slot.getAddr(), T));
}

void AggExprEmitter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  QualType T = E->getType();
  AggValueSlot Slot = EnsureSlot(T);
  EmitNullInitializationToLValue(CGF.MakeAddrLValue(Slot.getAddr(), T));
}

/// isSimpleZero - If emitting this value will obviously just cause a store of
/// zero to memory, return true.  This can return false if uncertain, so it just
/// handles simple cases.
static bool isSimpleZero(const Expr *E, CodeGenFunction &CGF) {
  E = E->IgnoreParens();

  // 0
  if (const IntegerLiteral *IL = dyn_cast<IntegerLiteral>(E))
    return IL->getValue() == 0;
  // +0.0
  if (const FloatingLiteral *FL = dyn_cast<FloatingLiteral>(E))
    return FL->getValue().isPosZero();
  // int()
  if ((isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E)) &&
      CGF.getTypes().isZeroInitializable(E->getType()))
    return true;
  // (int*)0 - Null pointer expressions.
  if (const CastExpr *ICE = dyn_cast<CastExpr>(E))
    return ICE->getCastKind() == CK_NullToPointer;
  // '\0'
  if (const CharacterLiteral *CL = dyn_cast<CharacterLiteral>(E))
    return CL->getValue() == 0;
  
  // Otherwise, hard case: conservatively return false.
  return false;
}


void 
AggExprEmitter::EmitInitializationToLValue(Expr* E, LValue LV) {
  QualType type = LV.getType();
  // FIXME: Ignore result?
  // FIXME: Are initializers affected by volatile?
  if (Dest.isZeroed() && isSimpleZero(E, CGF)) {
    // Storing "i32 0" to a zero'd memory location is a noop.
  } else if (isa<ImplicitValueInitExpr>(E)) {
    EmitNullInitializationToLValue(LV);
  } else if (type->isReferenceType()) {
    RValue RV = CGF.EmitReferenceBindingToExpr(E, /*InitializedDecl=*/0);
    CGF.EmitStoreThroughLValue(RV, LV, type);
  } else if (type->isAnyComplexType()) {
    CGF.EmitComplexExprIntoAddr(E, LV.getAddress(), false);
  } else if (CGF.hasAggregateLLVMType(type)) {
    CGF.EmitAggExpr(E, AggValueSlot::forLValue(LV, true, false,
                                               Dest.isZeroed()));
  } else if (LV.isSimple()) {
    CGF.EmitScalarInit(E, /*D=*/0, LV, /*Captured=*/false);
  } else {
    CGF.EmitStoreThroughLValue(RValue::get(CGF.EmitScalarExpr(E)), LV, type);
  }
}

void AggExprEmitter::EmitNullInitializationToLValue(LValue lv) {
  QualType type = lv.getType();

  // If the destination slot is already zeroed out before the aggregate is
  // copied into it, we don't have to emit any zeros here.
  if (Dest.isZeroed() && CGF.getTypes().isZeroInitializable(type))
    return;
  
  if (!CGF.hasAggregateLLVMType(type)) {
    // For non-aggregates, we can store zero
    llvm::Value *null = llvm::Constant::getNullValue(CGF.ConvertType(type));
    CGF.EmitStoreThroughLValue(RValue::get(null), lv, type);
  } else {
    // There's a potential optimization opportunity in combining
    // memsets; that would be easy for arrays, but relatively
    // difficult for structures with the current code.
    CGF.EmitNullInitialization(lv.getAddress(), lv.getType());
  }
}

void AggExprEmitter::VisitInitListExpr(InitListExpr *E) {
#if 0
  // FIXME: Assess perf here?  Figure out what cases are worth optimizing here
  // (Length of globals? Chunks of zeroed-out space?).
  //
  // If we can, prefer a copy from a global; this is a lot less code for long
  // globals, and it's easier for the current optimizers to analyze.
  if (llvm::Constant* C = CGF.CGM.EmitConstantExpr(E, E->getType(), &CGF)) {
    llvm::GlobalVariable* GV =
    new llvm::GlobalVariable(CGF.CGM.getModule(), C->getType(), true,
                             llvm::GlobalValue::InternalLinkage, C, "");
    EmitFinalDestCopy(E, CGF.MakeAddrLValue(GV, E->getType()));
    return;
  }
#endif
  if (E->hadArrayRangeDesignator())
    CGF.ErrorUnsupported(E, "GNU array range designator extension");

  llvm::Value *DestPtr = Dest.getAddr();

  // Handle initialization of an array.
  if (E->getType()->isArrayType()) {
    const llvm::PointerType *APType =
      cast<llvm::PointerType>(DestPtr->getType());
    const llvm::ArrayType *AType =
      cast<llvm::ArrayType>(APType->getElementType());

    uint64_t NumInitElements = E->getNumInits();

    if (E->getNumInits() > 0) {
      QualType T1 = E->getType();
      QualType T2 = E->getInit(0)->getType();
      if (CGF.getContext().hasSameUnqualifiedType(T1, T2)) {
        EmitAggLoadOfLValue(E->getInit(0));
        return;
      }
    }

    uint64_t NumArrayElements = AType->getNumElements();
    QualType ElementType = CGF.getContext().getCanonicalType(E->getType());
    ElementType = CGF.getContext().getAsArrayType(ElementType)->getElementType();
    ElementType = CGF.getContext().getQualifiedType(ElementType, 
                                                    Dest.getQualifiers());

    bool hasNonTrivialCXXConstructor = false;
    if (CGF.getContext().getLangOptions().CPlusPlus)
      if (const RecordType *RT = CGF.getContext()
                        .getBaseElementType(ElementType)->getAs<RecordType>()) {
        const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
        hasNonTrivialCXXConstructor = !RD->hasTrivialDefaultConstructor();
      }

    for (uint64_t i = 0; i != NumArrayElements; ++i) {
      // If we're done emitting initializers and the destination is known-zeroed
      // then we're done.
      if (i == NumInitElements &&
          Dest.isZeroed() &&
          CGF.getTypes().isZeroInitializable(ElementType) &&
          !hasNonTrivialCXXConstructor)
        break;

      llvm::Value *NextVal = Builder.CreateStructGEP(DestPtr, i, ".array");
      LValue LV = CGF.MakeAddrLValue(NextVal, ElementType);
      
      if (i < NumInitElements)
        EmitInitializationToLValue(E->getInit(i), LV);
      else if (Expr *filler = E->getArrayFiller())
        EmitInitializationToLValue(filler, LV);
      else
        EmitNullInitializationToLValue(LV);
      
      // If the GEP didn't get used because of a dead zero init or something
      // else, clean it up for -O0 builds and general tidiness.
      if (llvm::GetElementPtrInst *GEP =
            dyn_cast<llvm::GetElementPtrInst>(NextVal))
        if (GEP->use_empty())
          GEP->eraseFromParent();
    }
    return;
  }

  assert(E->getType()->isRecordType() && "Only support structs/unions here!");

  // Do struct initialization; this code just sets each individual member
  // to the approprate value.  This makes bitfield support automatic;
  // the disadvantage is that the generated code is more difficult for
  // the optimizer, especially with bitfields.
  unsigned NumInitElements = E->getNumInits();
  RecordDecl *SD = E->getType()->getAs<RecordType>()->getDecl();
  
  if (E->getType()->isUnionType()) {
    // Only initialize one field of a union. The field itself is
    // specified by the initializer list.
    if (!E->getInitializedFieldInUnion()) {
      // Empty union; we have nothing to do.

#ifndef NDEBUG
      // Make sure that it's really an empty and not a failure of
      // semantic analysis.
      for (RecordDecl::field_iterator Field = SD->field_begin(),
                                   FieldEnd = SD->field_end();
           Field != FieldEnd; ++Field)
        assert(Field->isUnnamedBitfield() && "Only unnamed bitfields allowed");
#endif
      return;
    }

    // FIXME: volatility
    FieldDecl *Field = E->getInitializedFieldInUnion();

    LValue FieldLoc = CGF.EmitLValueForFieldInitialization(DestPtr, Field, 0);
    if (NumInitElements) {
      // Store the initializer into the field
      EmitInitializationToLValue(E->getInit(0), FieldLoc);
    } else {
      // Default-initialize to null.
      EmitNullInitializationToLValue(FieldLoc);
    }

    return;
  }

  // Here we iterate over the fields; this makes it simpler to both
  // default-initialize fields and skip over unnamed fields.
  unsigned CurInitVal = 0;
  for (RecordDecl::field_iterator Field = SD->field_begin(),
                               FieldEnd = SD->field_end();
       Field != FieldEnd; ++Field) {
    // We're done once we hit the flexible array member
    if (Field->getType()->isIncompleteArrayType())
      break;

    if (Field->isUnnamedBitfield())
      continue;

    // Don't emit GEP before a noop store of zero.
    if (CurInitVal == NumInitElements && Dest.isZeroed() &&
        CGF.getTypes().isZeroInitializable(E->getType()))
      break;
    
    // FIXME: volatility
    LValue FieldLoc = CGF.EmitLValueForFieldInitialization(DestPtr, *Field, 0);
    // We never generate write-barries for initialized fields.
    FieldLoc.setNonGC(true);
    
    if (CurInitVal < NumInitElements) {
      // Store the initializer into the field.
      EmitInitializationToLValue(E->getInit(CurInitVal++), FieldLoc);
    } else {
      // We're out of initalizers; default-initialize to null
      EmitNullInitializationToLValue(FieldLoc);
    }
    
    // If the GEP didn't get used because of a dead zero init or something
    // else, clean it up for -O0 builds and general tidiness.
    if (FieldLoc.isSimple())
      if (llvm::GetElementPtrInst *GEP =
            dyn_cast<llvm::GetElementPtrInst>(FieldLoc.getAddress()))
        if (GEP->use_empty())
          GEP->eraseFromParent();
  }
}

//===----------------------------------------------------------------------===//
//                        Entry Points into this File
//===----------------------------------------------------------------------===//

/// GetNumNonZeroBytesInInit - Get an approximate count of the number of
/// non-zero bytes that will be stored when outputting the initializer for the
/// specified initializer expression.
static CharUnits GetNumNonZeroBytesInInit(const Expr *E, CodeGenFunction &CGF) {
  E = E->IgnoreParens();

  // 0 and 0.0 won't require any non-zero stores!
  if (isSimpleZero(E, CGF)) return CharUnits::Zero();

  // If this is an initlist expr, sum up the size of sizes of the (present)
  // elements.  If this is something weird, assume the whole thing is non-zero.
  const InitListExpr *ILE = dyn_cast<InitListExpr>(E);
  if (ILE == 0 || !CGF.getTypes().isZeroInitializable(ILE->getType()))
    return CGF.getContext().getTypeSizeInChars(E->getType());
  
  // InitListExprs for structs have to be handled carefully.  If there are
  // reference members, we need to consider the size of the reference, not the
  // referencee.  InitListExprs for unions and arrays can't have references.
  if (const RecordType *RT = E->getType()->getAs<RecordType>()) {
    if (!RT->isUnionType()) {
      RecordDecl *SD = E->getType()->getAs<RecordType>()->getDecl();
      CharUnits NumNonZeroBytes = CharUnits::Zero();
      
      unsigned ILEElement = 0;
      for (RecordDecl::field_iterator Field = SD->field_begin(),
           FieldEnd = SD->field_end(); Field != FieldEnd; ++Field) {
        // We're done once we hit the flexible array member or run out of
        // InitListExpr elements.
        if (Field->getType()->isIncompleteArrayType() ||
            ILEElement == ILE->getNumInits())
          break;
        if (Field->isUnnamedBitfield())
          continue;

        const Expr *E = ILE->getInit(ILEElement++);
        
        // Reference values are always non-null and have the width of a pointer.
        if (Field->getType()->isReferenceType())
          NumNonZeroBytes += CGF.getContext().toCharUnitsFromBits(
              CGF.getContext().Target.getPointerWidth(0));
        else
          NumNonZeroBytes += GetNumNonZeroBytesInInit(E, CGF);
      }
      
      return NumNonZeroBytes;
    }
  }
  
  
  CharUnits NumNonZeroBytes = CharUnits::Zero();
  for (unsigned i = 0, e = ILE->getNumInits(); i != e; ++i)
    NumNonZeroBytes += GetNumNonZeroBytesInInit(ILE->getInit(i), CGF);
  return NumNonZeroBytes;
}

/// CheckAggExprForMemSetUse - If the initializer is large and has a lot of
/// zeros in it, emit a memset and avoid storing the individual zeros.
///
static void CheckAggExprForMemSetUse(AggValueSlot &Slot, const Expr *E,
                                     CodeGenFunction &CGF) {
  // If the slot is already known to be zeroed, nothing to do.  Don't mess with
  // volatile stores.
  if (Slot.isZeroed() || Slot.isVolatile() || Slot.getAddr() == 0) return;

  // C++ objects with a user-declared constructor don't need zero'ing.
  if (CGF.getContext().getLangOptions().CPlusPlus)
    if (const RecordType *RT = CGF.getContext()
                       .getBaseElementType(E->getType())->getAs<RecordType>()) {
      const CXXRecordDecl *RD = cast<CXXRecordDecl>(RT->getDecl());
      if (RD->hasUserDeclaredConstructor())
        return;
    }

  // If the type is 16-bytes or smaller, prefer individual stores over memset.
  std::pair<CharUnits, CharUnits> TypeInfo =
    CGF.getContext().getTypeInfoInChars(E->getType());
  if (TypeInfo.first <= CharUnits::fromQuantity(16))
    return;

  // Check to see if over 3/4 of the initializer are known to be zero.  If so,
  // we prefer to emit memset + individual stores for the rest.
  CharUnits NumNonZeroBytes = GetNumNonZeroBytesInInit(E, CGF);
  if (NumNonZeroBytes*4 > TypeInfo.first)
    return;
  
  // Okay, it seems like a good idea to use an initial memset, emit the call.
  llvm::Constant *SizeVal = CGF.Builder.getInt64(TypeInfo.first.getQuantity());
  CharUnits Align = TypeInfo.second;

  llvm::Value *Loc = Slot.getAddr();
  const llvm::Type *BP = llvm::Type::getInt8PtrTy(CGF.getLLVMContext());
  
  Loc = CGF.Builder.CreateBitCast(Loc, BP);
  CGF.Builder.CreateMemSet(Loc, CGF.Builder.getInt8(0), SizeVal, 
                           Align.getQuantity(), false);
  
  // Tell the AggExprEmitter that the slot is known zero.
  Slot.setZeroed();
}




/// EmitAggExpr - Emit the computation of the specified expression of aggregate
/// type.  The result is computed into DestPtr.  Note that if DestPtr is null,
/// the value of the aggregate expression is not needed.  If VolatileDest is
/// true, DestPtr cannot be 0.
///
/// \param IsInitializer - true if this evaluation is initializing an
/// object whose lifetime is already being managed.
void CodeGenFunction::EmitAggExpr(const Expr *E, AggValueSlot Slot,
                                  bool IgnoreResult) {
  assert(E && hasAggregateLLVMType(E->getType()) &&
         "Invalid aggregate expression to emit");
  assert((Slot.getAddr() != 0 || Slot.isIgnored()) &&
         "slot has bits but no address");

  // Optimize the slot if possible.
  CheckAggExprForMemSetUse(Slot, E, *this);
 
  AggExprEmitter(*this, Slot, IgnoreResult).Visit(const_cast<Expr*>(E));
}

LValue CodeGenFunction::EmitAggExprToLValue(const Expr *E) {
  assert(hasAggregateLLVMType(E->getType()) && "Invalid argument!");
  llvm::Value *Temp = CreateMemTemp(E->getType());
  LValue LV = MakeAddrLValue(Temp, E->getType());
  EmitAggExpr(E, AggValueSlot::forLValue(LV, false));
  return LV;
}

void CodeGenFunction::EmitAggregateCopy(llvm::Value *DestPtr,
                                        llvm::Value *SrcPtr, QualType Ty,
                                        bool isVolatile) {
  assert(!Ty->isAnyComplexType() && "Shouldn't happen for complex");

  if (getContext().getLangOptions().CPlusPlus) {
    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      CXXRecordDecl *Record = cast<CXXRecordDecl>(RT->getDecl());
      assert((Record->hasTrivialCopyConstructor() || 
              Record->hasTrivialCopyAssignment()) &&
             "Trying to aggregate-copy a type without a trivial copy "
             "constructor or assignment operator");
      // Ignore empty classes in C++.
      if (Record->isEmpty())
        return;
    }
  }
  
  // Aggregate assignment turns into llvm.memcpy.  This is almost valid per
  // C99 6.5.16.1p3, which states "If the value being stored in an object is
  // read from another object that overlaps in anyway the storage of the first
  // object, then the overlap shall be exact and the two objects shall have
  // qualified or unqualified versions of a compatible type."
  //
  // memcpy is not defined if the source and destination pointers are exactly
  // equal, but other compilers do this optimization, and almost every memcpy
  // implementation handles this case safely.  If there is a libc that does not
  // safely handle this, we can add a target hook.

  // Get size and alignment info for this aggregate.
  std::pair<CharUnits, CharUnits> TypeInfo = 
    getContext().getTypeInfoInChars(Ty);

  // FIXME: Handle variable sized types.

  // FIXME: If we have a volatile struct, the optimizer can remove what might
  // appear to be `extra' memory ops:
  //
  // volatile struct { int i; } a, b;
  //
  // int main() {
  //   a = b;
  //   a = b;
  // }
  //
  // we need to use a different call here.  We use isVolatile to indicate when
  // either the source or the destination is volatile.

  const llvm::PointerType *DPT = cast<llvm::PointerType>(DestPtr->getType());
  const llvm::Type *DBP =
    llvm::Type::getInt8PtrTy(getLLVMContext(), DPT->getAddressSpace());
  DestPtr = Builder.CreateBitCast(DestPtr, DBP, "tmp");

  const llvm::PointerType *SPT = cast<llvm::PointerType>(SrcPtr->getType());
  const llvm::Type *SBP =
    llvm::Type::getInt8PtrTy(getLLVMContext(), SPT->getAddressSpace());
  SrcPtr = Builder.CreateBitCast(SrcPtr, SBP, "tmp");

  // Don't do any of the memmove_collectable tests if GC isn't set.
  if (CGM.getLangOptions().getGCMode() == LangOptions::NonGC) {
    // fall through
  } else if (const RecordType *RecordTy = Ty->getAs<RecordType>()) {
    RecordDecl *Record = RecordTy->getDecl();
    if (Record->hasObjectMember()) {
      CharUnits size = TypeInfo.first;
      const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
      llvm::Value *SizeVal = llvm::ConstantInt::get(SizeTy, size.getQuantity());
      CGM.getObjCRuntime().EmitGCMemmoveCollectable(*this, DestPtr, SrcPtr, 
                                                    SizeVal);
      return;
    }
  } else if (Ty->isArrayType()) {
    QualType BaseType = getContext().getBaseElementType(Ty);
    if (const RecordType *RecordTy = BaseType->getAs<RecordType>()) {
      if (RecordTy->getDecl()->hasObjectMember()) {
        CharUnits size = TypeInfo.first;
        const llvm::Type *SizeTy = ConvertType(getContext().getSizeType());
        llvm::Value *SizeVal = 
          llvm::ConstantInt::get(SizeTy, size.getQuantity());
        CGM.getObjCRuntime().EmitGCMemmoveCollectable(*this, DestPtr, SrcPtr, 
                                                      SizeVal);
        return;
      }
    }
  }
  
  Builder.CreateMemCpy(DestPtr, SrcPtr,
                       llvm::ConstantInt::get(IntPtrTy, 
                                              TypeInfo.first.getQuantity()),
                       TypeInfo.second.getQuantity(), isVolatile);
}
