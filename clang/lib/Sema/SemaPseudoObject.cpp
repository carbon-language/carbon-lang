//===--- SemaPseudoObject.cpp - Semantic Analysis for Pseudo-Objects ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for expressions involving
//  pseudo-object references.  Pseudo-objects are conceptual objects
//  whose storage is entirely abstract and all accesses to which are
//  translated through some sort of abstraction barrier.
//
//  For example, Objective-C objects can have "properties", either
//  declared or undeclared.  A property may be accessed by writing
//    expr.prop
//  where 'expr' is an r-value of Objective-C pointer type and 'prop'
//  is the name of the property.  If this expression is used in a context
//  needing an r-value, it is treated as if it were a message-send
//  of the associated 'getter' selector, typically:
//    [expr prop]
//  If it is used as the LHS of a simple assignment, it is treated
//  as a message-send of the associated 'setter' selector, typically:
//    [expr setProp: RHS]
//  If it is used as the LHS of a compound assignment, or the operand
//  of a unary increment or decrement, both are required;  for example,
//  'expr.prop *= 100' would be translated to:
//    [expr setProp: [expr prop] * 100]
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Initialization.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;
using namespace sema;

static ObjCMethodDecl *LookupMethodInReceiverType(Sema &S, Selector sel,
                                            const ObjCPropertyRefExpr *PRE) {
  if (PRE->isObjectReceiver()) {
    const ObjCObjectPointerType *PT =
      PRE->getBase()->getType()->castAs<ObjCObjectPointerType>();
    return S.LookupMethodInObjectType(sel, PT->getPointeeType(), true);
  }

  if (PRE->isSuperReceiver()) {
    if (const ObjCObjectPointerType *PT =
        PRE->getSuperReceiverType()->getAs<ObjCObjectPointerType>())
      return S.LookupMethodInObjectType(sel, PT->getPointeeType(), true);

    return S.LookupMethodInObjectType(sel, PRE->getSuperReceiverType(), false);
  }

  assert(PRE->isClassReceiver() && "Invalid expression");
  QualType IT = S.Context.getObjCInterfaceType(PRE->getClassReceiver());
  return S.LookupMethodInObjectType(sel, IT, false);
}

ExprResult Sema::checkPseudoObjectRValue(Expr *E) {
  assert(E->getValueKind() == VK_LValue &&
         E->getObjectKind() == OK_ObjCProperty);
  const ObjCPropertyRefExpr *PRE = E->getObjCProperty();

  QualType ReceiverType;
  if (PRE->isObjectReceiver())
    ReceiverType = PRE->getBase()->getType();
  else if (PRE->isSuperReceiver())
    ReceiverType = PRE->getSuperReceiverType();
  else
    ReceiverType = Context.getObjCInterfaceType(PRE->getClassReceiver());
    
  ExprValueKind VK = VK_RValue;
  QualType T;
  if (PRE->isImplicitProperty()) {
    if (ObjCMethodDecl *GetterMethod = 
          PRE->getImplicitPropertyGetter()) {
      T = getMessageSendResultType(ReceiverType, GetterMethod,
                                   PRE->isClassReceiver(), 
                                   PRE->isSuperReceiver());
      VK = Expr::getValueKindForType(GetterMethod->getResultType());
    } else {
      Diag(PRE->getLocation(), diag::err_getter_not_found)
            << PRE->getBase()->getType();
      return ExprError();
    }
  } else {
    ObjCPropertyDecl *prop = PRE->getExplicitProperty();

    ObjCMethodDecl *getter =
      LookupMethodInReceiverType(*this, prop->getGetterName(), PRE);
    if (getter && !getter->hasRelatedResultType())
      DiagnosePropertyAccessorMismatch(prop, getter, PRE->getLocation());
    if (!getter) getter = prop->getGetterMethodDecl();

    // Figure out the type of the expression.  Mostly this is the
    // result type of the getter, if possible.
    if (getter) {
      T = getMessageSendResultType(ReceiverType, getter, 
                                   PRE->isClassReceiver(), 
                                   PRE->isSuperReceiver());
      VK = Expr::getValueKindForType(getter->getResultType());

      // As a special case, if the method returns 'id', try to get a
      // better type from the property.
      if (VK == VK_RValue && T->isObjCIdType() &&
          prop->getType()->isObjCRetainableType())
        T = prop->getType();
    } else {
      T = prop->getType();
      VK = Expr::getValueKindForType(T);
      T = T.getNonLValueExprType(Context);
    }
  }

  E->setType(T);
  E = ImplicitCastExpr::Create(Context, T, CK_GetObjCProperty, E, 0, VK);
  
  ExprResult Result = MaybeBindToTemporary(E);
  if (!Result.isInvalid())
    E = Result.take();

  return Owned(E);
}

namespace {
  struct PseudoObjectInfo {
    const ObjCPropertyRefExpr *RefExpr;
    bool HasSetter;
    Selector SetterSelector;
    ParmVarDecl *SetterParam;
    QualType SetterParamType;

    void setSetter(ObjCMethodDecl *setter) {
      HasSetter = true;
      SetterParam = *setter->param_begin();
      SetterParamType = SetterParam->getType().getUnqualifiedType();
    }

    PseudoObjectInfo(Sema &S, Expr *E)
      : RefExpr(E->getObjCProperty()), HasSetter(false), SetterParam(0) {

      assert(E->getValueKind() == VK_LValue &&
             E->getObjectKind() == OK_ObjCProperty);

      // Try to find a setter.

      // For implicit properties, just trust the lookup we already did.
      if (RefExpr->isImplicitProperty()) {
        if (ObjCMethodDecl *setter = RefExpr->getImplicitPropertySetter()) {
          setSetter(setter);
          SetterSelector = setter->getSelector();
        } else {
          IdentifierInfo *getterName =
            RefExpr->getImplicitPropertyGetter()->getSelector()
              .getIdentifierInfoForSlot(0);
          SetterSelector = 
            SelectorTable::constructSetterName(S.PP.getIdentifierTable(),
                                               S.PP.getSelectorTable(),
                                               getterName);
        }
        return;
      }

      // For explicit properties, this is more involved.
      ObjCPropertyDecl *prop = RefExpr->getExplicitProperty();
      SetterSelector = prop->getSetterName();

      // Do a normal method lookup first.
      if (ObjCMethodDecl *setter =
            LookupMethodInReceiverType(S, SetterSelector, RefExpr)) {
        setSetter(setter);
        return;
      }

      // If that failed, trust the type on the @property declaration.
      if (!prop->isReadOnly()) {
        HasSetter = true;
        SetterParamType = prop->getType().getUnqualifiedType();
      }
    }
  };
}

/// Check an increment or decrement of a pseudo-object expression.
ExprResult Sema::checkPseudoObjectIncDec(Scope *S, SourceLocation opcLoc,
                                         UnaryOperatorKind opcode, Expr *op) {
  assert(UnaryOperator::isIncrementDecrementOp(opcode));
  PseudoObjectInfo info(*this, op);

  // If there's no setter, we have no choice but to try to assign to
  // the result of the getter.
  if (!info.HasSetter) {
    QualType resultType = info.RefExpr->getGetterResultType();
    assert(!resultType.isNull() && "property has no setter and no getter!");

    // Only do this if the getter returns an l-value reference type.
    if (const LValueReferenceType *refType
          = resultType->getAs<LValueReferenceType>()) {
      op = ImplicitCastExpr::Create(Context, refType->getPointeeType(),
                                    CK_GetObjCProperty, op, 0, VK_LValue);
      return BuildUnaryOp(S, opcLoc, opcode, op);
    }

    // Otherwise, it's an error.
    Diag(opcLoc, diag::err_nosetter_property_incdec)
      << unsigned(info.RefExpr->isImplicitProperty())
      << unsigned(UnaryOperator::isDecrementOp(opcode))
      << info.SetterSelector
      << op->getSourceRange();
    return ExprError();
  }

  // ++/-- behave like compound assignments, i.e. they need a getter.
  QualType getterResultType = info.RefExpr->getGetterResultType();
  if (getterResultType.isNull()) {
    assert(info.RefExpr->isImplicitProperty());
    Diag(opcLoc, diag::err_nogetter_property_incdec)
      << unsigned(UnaryOperator::isDecrementOp(opcode))
      << info.RefExpr->getImplicitPropertyGetter()->getSelector()
      << op->getSourceRange();
    return ExprError();
  }

  // HACK: change the type of the operand to prevent further placeholder
  // transformation.
  op->setType(getterResultType.getNonLValueExprType(Context));
  op->setObjectKind(OK_Ordinary);
  
  ExprResult result = CreateBuiltinUnaryOp(opcLoc, opcode, op);
  if (result.isInvalid()) return ExprError();

  // Change the object kind back.
  op->setObjectKind(OK_ObjCProperty);
  return result;
}

ExprResult Sema::checkPseudoObjectAssignment(Scope *S, SourceLocation opcLoc,
                                             BinaryOperatorKind opcode,
                                             Expr *LHS, Expr *RHS) {
  assert(BinaryOperator::isAssignmentOp(opcode));
  PseudoObjectInfo info(*this, LHS);

  // If there's no setter, we have no choice but to try to assign to
  // the result of the getter.
  if (!info.HasSetter) {
    QualType resultType = info.RefExpr->getGetterResultType();
    assert(!resultType.isNull() && "property has no setter and no getter!");

    // Only do this if the getter returns an l-value reference type.
    if (const LValueReferenceType *refType
          = resultType->getAs<LValueReferenceType>()) {
      LHS = ImplicitCastExpr::Create(Context, refType->getPointeeType(),
                                     CK_GetObjCProperty, LHS, 0, VK_LValue);
      return BuildBinOp(S, opcLoc, opcode, LHS, RHS);
    }

    // Otherwise, it's an error.
    Diag(opcLoc, diag::err_nosetter_property_assignment)
      << unsigned(info.RefExpr->isImplicitProperty())
      << info.SetterSelector
      << LHS->getSourceRange() << RHS->getSourceRange();
    return ExprError();
  }

  // If there is a setter, we definitely want to use it.

  // If this is a simple assignment, just initialize the parameter
  // with the RHS.
  if (opcode == BO_Assign) {
    LHS->setType(info.SetterParamType.getNonLValueExprType(Context));

    // Under certain circumstances, we need to type-check the RHS as a
    // straight-up parameter initialization.  This gives somewhat
    // inferior diagnostics, so we try to avoid it.

    if (RHS->isTypeDependent()) {
      // Just build the expression.

    } else if ((getLangOptions().CPlusPlus && LHS->getType()->isRecordType()) ||
               (getLangOptions().ObjCAutoRefCount &&
                info.SetterParam &&
                info.SetterParam->hasAttr<NSConsumedAttr>())) {
      InitializedEntity param = (info.SetterParam
        ? InitializedEntity::InitializeParameter(Context, info.SetterParam)
        : InitializedEntity::InitializeParameter(Context, info.SetterParamType,
                                                 /*consumed*/ false));
      ExprResult arg = PerformCopyInitialization(param, opcLoc, RHS);
      if (arg.isInvalid()) return ExprError();
      RHS = arg.take();

      // Warn about assignments of +1 objects to unsafe pointers in ARC.
      // CheckAssignmentOperands does this on the other path.
      if (getLangOptions().ObjCAutoRefCount)
        checkUnsafeExprAssigns(opcLoc, LHS, RHS);
    } else {
      ExprResult RHSResult = Owned(RHS);

      LHS->setObjectKind(OK_Ordinary);
      QualType resultType = CheckAssignmentOperands(LHS, RHSResult, opcLoc,
                                                    /*compound*/ QualType());
      LHS->setObjectKind(OK_ObjCProperty);

      if (!RHSResult.isInvalid()) RHS = RHSResult.take();
      if (resultType.isNull()) return ExprError();
    }

    // Warn about property sets in ARC that might cause retain cycles.
    if (getLangOptions().ObjCAutoRefCount && !info.RefExpr->isSuperReceiver())
      checkRetainCycles(const_cast<Expr*>(info.RefExpr->getBase()), RHS);

    return new (Context) BinaryOperator(LHS, RHS, opcode, RHS->getType(),
                                        RHS->getValueKind(),
                                        RHS->getObjectKind(),
                                        opcLoc);
  }

  // If this is a compound assignment, we need to use the getter, too.
  QualType getterResultType = info.RefExpr->getGetterResultType();
  if (getterResultType.isNull()) {
    Diag(opcLoc, diag::err_nogetter_property_compound_assignment)
      << LHS->getSourceRange() << RHS->getSourceRange();
    return ExprError();
  }

  // HACK: change the type of the LHS to prevent further placeholder
  // transformation.
  LHS->setType(getterResultType.getNonLValueExprType(Context));
  LHS->setObjectKind(OK_Ordinary);
  
  ExprResult result = CreateBuiltinBinOp(opcLoc, opcode, LHS, RHS);
  if (result.isInvalid()) return ExprError();

  // Change the object kind back.
  LHS->setObjectKind(OK_ObjCProperty);
  return result;
}
