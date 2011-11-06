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

namespace {
  // Basically just a very focused copy of TreeTransform.
  template <class T> struct Rebuilder {
    Sema &S;
    Rebuilder(Sema &S) : S(S) {}

    T &getDerived() { return static_cast<T&>(*this); }

    Expr *rebuild(Expr *e) {
      // Fast path: nothing to look through.
      if (typename T::specific_type *specific
            = dyn_cast<typename T::specific_type>(e))
        return getDerived().rebuildSpecific(specific);

      // Otherwise, we should look through and rebuild anything that
      // IgnoreParens would.

      if (ParenExpr *parens = dyn_cast<ParenExpr>(e)) {
        e = rebuild(parens->getSubExpr());
        return new (S.Context) ParenExpr(parens->getLParen(),
                                         parens->getRParen(),
                                         e);
      }

      if (UnaryOperator *uop = dyn_cast<UnaryOperator>(e)) {
        assert(uop->getOpcode() == UO_Extension);
        e = rebuild(uop->getSubExpr());
        return new (S.Context) UnaryOperator(e, uop->getOpcode(),
                                             uop->getType(),
                                             uop->getValueKind(),
                                             uop->getObjectKind(),
                                             uop->getOperatorLoc());
      }

      if (GenericSelectionExpr *gse = dyn_cast<GenericSelectionExpr>(e)) {
        assert(!gse->isResultDependent());
        unsigned resultIndex = gse->getResultIndex();
        unsigned numAssocs = gse->getNumAssocs();

        SmallVector<Expr*, 8> assocs(numAssocs);
        SmallVector<TypeSourceInfo*, 8> assocTypes(numAssocs);

        for (unsigned i = 0; i != numAssocs; ++i) {
          Expr *assoc = gse->getAssocExpr(i);
          if (i == resultIndex) assoc = rebuild(assoc);
          assocs[i] = assoc;
          assocTypes[i] = gse->getAssocTypeSourceInfo(i);
        }

        return new (S.Context) GenericSelectionExpr(S.Context,
                                                    gse->getGenericLoc(),
                                                    gse->getControllingExpr(),
                                                    assocTypes.data(),
                                                    assocs.data(),
                                                    numAssocs,
                                                    gse->getDefaultLoc(),
                                                    gse->getRParenLoc(),
                                      gse->containsUnexpandedParameterPack(),
                                                    resultIndex);
      }

      llvm_unreachable("bad expression to rebuild!");
    }
  };

  struct ObjCPropertyRefRebuilder : Rebuilder<ObjCPropertyRefRebuilder> {
    Expr *NewBase;
    ObjCPropertyRefRebuilder(Sema &S, Expr *newBase)
      : Rebuilder(S), NewBase(newBase) {}

    typedef ObjCPropertyRefExpr specific_type;
    Expr *rebuildSpecific(ObjCPropertyRefExpr *refExpr) {
      // Fortunately, the constraint that we're rebuilding something
      // with a base limits the number of cases here.
      assert(refExpr->getBase());

      if (refExpr->isExplicitProperty()) {
        return new (S.Context)
          ObjCPropertyRefExpr(refExpr->getExplicitProperty(),
                              refExpr->getType(), refExpr->getValueKind(),
                              refExpr->getObjectKind(), refExpr->getLocation(),
                              NewBase);
      }
      return new (S.Context)
        ObjCPropertyRefExpr(refExpr->getImplicitPropertyGetter(),
                            refExpr->getImplicitPropertySetter(),
                            refExpr->getType(), refExpr->getValueKind(),
                            refExpr->getObjectKind(),refExpr->getLocation(),
                            NewBase);
    }
  };

  class PseudoOpBuilder {
  public:
    Sema &S;
    unsigned ResultIndex;
    SourceLocation GenericLoc;
    SmallVector<Expr *, 4> Semantics;

    PseudoOpBuilder(Sema &S, SourceLocation genericLoc)
      : S(S), ResultIndex(PseudoObjectExpr::NoResult),
        GenericLoc(genericLoc) {}

    /// Add a normal semantic expression.
    void addSemanticExpr(Expr *semantic) {
      Semantics.push_back(semantic);
    }

    /// Add the 'result' semantic expression.
    void addResultSemanticExpr(Expr *resultExpr) {
      assert(ResultIndex == PseudoObjectExpr::NoResult);
      ResultIndex = Semantics.size();
      Semantics.push_back(resultExpr);
    }

    ExprResult buildRValueOperation(Expr *op);
    ExprResult buildAssignmentOperation(Scope *Sc,
                                        SourceLocation opLoc,
                                        BinaryOperatorKind opcode,
                                        Expr *LHS, Expr *RHS);
    ExprResult buildIncDecOperation(Scope *Sc, SourceLocation opLoc,
                                    UnaryOperatorKind opcode,
                                    Expr *op);

    ExprResult complete(Expr *syntacticForm);

    OpaqueValueExpr *capture(Expr *op);
    OpaqueValueExpr *captureValueAsResult(Expr *op);

    void setResultToLastSemantic() {
      assert(ResultIndex == PseudoObjectExpr::NoResult);
      ResultIndex = Semantics.size() - 1;
    }

    /// Return true if assignments have a non-void result.
    virtual bool assignmentsHaveResult() { return true; }

    virtual Expr *rebuildAndCaptureObject(Expr *) = 0;
    virtual ExprResult buildGet() = 0;
    virtual ExprResult buildSet(Expr *, SourceLocation,
                                bool captureSetValueAsResult) = 0;
  };

  /// A PseudoOpBuilder for Objective-C @properties.
  class ObjCPropertyOpBuilder : public PseudoOpBuilder {
    ObjCPropertyRefExpr *RefExpr;
    OpaqueValueExpr *InstanceReceiver;
    ObjCMethodDecl *Getter;

    ObjCMethodDecl *Setter;
    Selector SetterSelector;

  public:
    ObjCPropertyOpBuilder(Sema &S, ObjCPropertyRefExpr *refExpr) :
      PseudoOpBuilder(S, refExpr->getLocation()), RefExpr(refExpr),
      InstanceReceiver(0), Getter(0), Setter(0) {
    }

    ExprResult buildRValueOperation(Expr *op);
    ExprResult buildAssignmentOperation(Scope *Sc,
                                        SourceLocation opLoc,
                                        BinaryOperatorKind opcode,
                                        Expr *LHS, Expr *RHS);
    ExprResult buildIncDecOperation(Scope *Sc, SourceLocation opLoc,
                                    UnaryOperatorKind opcode,
                                    Expr *op);

    bool tryBuildGetOfReference(Expr *op, ExprResult &result);
    bool findSetter();
    bool findGetter();

    Expr *rebuildAndCaptureObject(Expr *syntacticBase);
    ExprResult buildGet();
    ExprResult buildSet(Expr *op, SourceLocation, bool);
  };
}

/// Capture the given expression in an OpaqueValueExpr.
OpaqueValueExpr *PseudoOpBuilder::capture(Expr *e) {
  // Make a new OVE whose source is the given expression.
  OpaqueValueExpr *captured = 
    new (S.Context) OpaqueValueExpr(GenericLoc, e->getType(),
                                    e->getValueKind());
  captured->setSourceExpr(e);
  
  // Make sure we bind that in the semantics.
  addSemanticExpr(captured);
  return captured;
}

/// Capture the given expression as the result of this pseudo-object
/// operation.  This routine is safe against expressions which may
/// already be captured.
///
/// \param Returns the captured expression, which will be the
///   same as the input if the input was already captured
OpaqueValueExpr *PseudoOpBuilder::captureValueAsResult(Expr *e) {
  assert(ResultIndex == PseudoObjectExpr::NoResult);

  // If the expression hasn't already been captured, just capture it
  // and set the new semantic 
  if (!isa<OpaqueValueExpr>(e)) {
    OpaqueValueExpr *cap = capture(e);
    setResultToLastSemantic();
    return cap;
  }

  // Otherwise, it must already be one of our semantic expressions;
  // set ResultIndex to its index.
  unsigned index = 0;
  for (;; ++index) {
    assert(index < Semantics.size() &&
           "captured expression not found in semantics!");
    if (e == Semantics[index]) break;
  }
  ResultIndex = index;
  return cast<OpaqueValueExpr>(e);
}

/// The routine which creates the final PseudoObjectExpr.
ExprResult PseudoOpBuilder::complete(Expr *syntactic) {
  return PseudoObjectExpr::Create(S.Context, syntactic,
                                  Semantics, ResultIndex);
}

/// The main skeleton for building an r-value operation.
ExprResult PseudoOpBuilder::buildRValueOperation(Expr *op) {
  Expr *syntacticBase = rebuildAndCaptureObject(op);

  ExprResult getExpr = buildGet();
  if (getExpr.isInvalid()) return ExprError();
  addResultSemanticExpr(getExpr.take());

  return complete(syntacticBase);
}

/// The basic skeleton for building a simple or compound
/// assignment operation.
ExprResult
PseudoOpBuilder::buildAssignmentOperation(Scope *Sc, SourceLocation opcLoc,
                                          BinaryOperatorKind opcode,
                                          Expr *LHS, Expr *RHS) {
  assert(BinaryOperator::isAssignmentOp(opcode));

  Expr *syntacticLHS = rebuildAndCaptureObject(LHS);
  OpaqueValueExpr *capturedRHS = capture(RHS);

  Expr *syntactic;

  ExprResult result;
  if (opcode == BO_Assign) {
    result = capturedRHS;
    syntactic = new (S.Context) BinaryOperator(syntacticLHS, capturedRHS,
                                               opcode, capturedRHS->getType(),
                                               capturedRHS->getValueKind(),
                                               OK_Ordinary, opcLoc);
  } else {
    ExprResult opLHS = buildGet();
    if (opLHS.isInvalid()) return ExprError();

    // Build an ordinary, non-compound operation.
    BinaryOperatorKind nonCompound =
      BinaryOperator::getOpForCompoundAssignment(opcode);
    result = S.BuildBinOp(Sc, opcLoc, nonCompound,
                          opLHS.take(), capturedRHS);
    if (result.isInvalid()) return ExprError();

    syntactic =
      new (S.Context) CompoundAssignOperator(syntacticLHS, capturedRHS, opcode,
                                             result.get()->getType(),
                                             result.get()->getValueKind(),
                                             OK_Ordinary,
                                             opLHS.get()->getType(),
                                             result.get()->getType(),
                                             opcLoc);
  }

  // The result of the assignment, if not void, is the value set into
  // the l-value.
  result = buildSet(result.take(), opcLoc, assignmentsHaveResult());
  if (result.isInvalid()) return ExprError();
  addSemanticExpr(result.take());

  return complete(syntactic);
}

/// The basic skeleton for building an increment or decrement
/// operation.
ExprResult
PseudoOpBuilder::buildIncDecOperation(Scope *Sc, SourceLocation opcLoc,
                                      UnaryOperatorKind opcode,
                                      Expr *op) {
  assert(UnaryOperator::isIncrementDecrementOp(opcode));

  Expr *syntacticOp = rebuildAndCaptureObject(op);

  // Load the value.
  ExprResult result = buildGet();
  if (result.isInvalid()) return ExprError();

  QualType resultType = result.get()->getType();

  // That's the postfix result.
  if (UnaryOperator::isPostfix(opcode) && assignmentsHaveResult()) {
    result = capture(result.take());
    setResultToLastSemantic();
  }

  // Add or subtract a literal 1.
  llvm::APInt oneV(S.Context.getTypeSize(S.Context.IntTy), 1);
  Expr *one = IntegerLiteral::Create(S.Context, oneV, S.Context.IntTy,
                                     GenericLoc);

  if (UnaryOperator::isIncrementOp(opcode)) {
    result = S.BuildBinOp(Sc, opcLoc, BO_Add, result.take(), one);
  } else {
    result = S.BuildBinOp(Sc, opcLoc, BO_Sub, result.take(), one);
  }
  if (result.isInvalid()) return ExprError();

  // Store that back into the result.  The value stored is the result
  // of a prefix operation.
  result = buildSet(result.take(), opcLoc,
             UnaryOperator::isPrefix(opcode) && assignmentsHaveResult());
  if (result.isInvalid()) return ExprError();
  addSemanticExpr(result.take());

  UnaryOperator *syntactic =
    new (S.Context) UnaryOperator(syntacticOp, opcode, resultType,
                                  VK_LValue, OK_Ordinary, opcLoc);
  return complete(syntactic);
}


//===----------------------------------------------------------------------===//
//  Objective-C @property and implicit property references
//===----------------------------------------------------------------------===//

/// Look up a method in the receiver type of an Objective-C property
/// reference.
static ObjCMethodDecl *LookupMethodInReceiverType(Sema &S, Selector sel,
                                            const ObjCPropertyRefExpr *PRE) {
  if (PRE->isObjectReceiver()) {
    const ObjCObjectPointerType *PT =
      PRE->getBase()->getType()->castAs<ObjCObjectPointerType>();

    // Special case for 'self' in class method implementations.
    if (PT->isObjCClassType() &&
        S.isSelfExpr(const_cast<Expr*>(PRE->getBase()))) {
      // This cast is safe because isSelfExpr is only true within
      // methods.
      ObjCMethodDecl *method =
        cast<ObjCMethodDecl>(S.CurContext->getNonClosureAncestor());
      return S.LookupMethodInObjectType(sel,
                 S.Context.getObjCInterfaceType(method->getClassInterface()),
                                        /*instance*/ false);
    }

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

bool ObjCPropertyOpBuilder::findGetter() {
  if (Getter) return true;

  Getter = LookupMethodInReceiverType(S, RefExpr->getGetterSelector(), RefExpr);
  return (Getter != 0);
}

/// Try to find the most accurate setter declaration for the property
/// reference.
///
/// \return true if a setter was found, in which case Setter 
bool ObjCPropertyOpBuilder::findSetter() {
  // For implicit properties, just trust the lookup we already did.
  if (RefExpr->isImplicitProperty()) {
    if (ObjCMethodDecl *setter = RefExpr->getImplicitPropertySetter()) {
      Setter = setter;
      SetterSelector = setter->getSelector();
      return true;
    } else {
      IdentifierInfo *getterName =
        RefExpr->getImplicitPropertyGetter()->getSelector()
          .getIdentifierInfoForSlot(0);
      SetterSelector =
        SelectorTable::constructSetterName(S.PP.getIdentifierTable(),
                                           S.PP.getSelectorTable(),
                                           getterName);
      return false;
    }
  }

  // For explicit properties, this is more involved.
  ObjCPropertyDecl *prop = RefExpr->getExplicitProperty();
  SetterSelector = prop->getSetterName();

  // Do a normal method lookup first.
  if (ObjCMethodDecl *setter =
        LookupMethodInReceiverType(S, SetterSelector, RefExpr)) {
    Setter = setter;
    return true;
  }

  // That can fail in the somewhat crazy situation that we're
  // type-checking a message send within the @interface declaration
  // that declared the @property.  But it's not clear that that's
  // valuable to support.

  return false;
}

/// Capture the base object of an Objective-C property expression.
Expr *ObjCPropertyOpBuilder::rebuildAndCaptureObject(Expr *syntacticBase) {
  assert(InstanceReceiver == 0);

  // If we have a base, capture it in an OVE and rebuild the syntactic
  // form to use the OVE as its base.
  if (RefExpr->isObjectReceiver()) {
    InstanceReceiver = capture(RefExpr->getBase());

    syntacticBase =
      ObjCPropertyRefRebuilder(S, InstanceReceiver).rebuild(syntacticBase);
  }

  return syntacticBase;
}

/// Load from an Objective-C property reference.
ExprResult ObjCPropertyOpBuilder::buildGet() {
  findGetter();
  assert(Getter);
  
  QualType receiverType;
  SourceLocation superLoc;
  if (RefExpr->isClassReceiver()) {
    receiverType = S.Context.getObjCInterfaceType(RefExpr->getClassReceiver());
  } else if (RefExpr->isSuperReceiver()) {
    superLoc = RefExpr->getReceiverLocation();
    receiverType = RefExpr->getSuperReceiverType();
  } else {
    assert(InstanceReceiver);
    receiverType = InstanceReceiver->getType();
  }

  // Build a message-send.
  ExprResult msg;
  if (Getter->isInstanceMethod() || RefExpr->isObjectReceiver()) {
    assert(InstanceReceiver || RefExpr->isSuperReceiver());
    msg = S.BuildInstanceMessage(InstanceReceiver, receiverType, superLoc,
                                 Getter->getSelector(), Getter,
                                 GenericLoc, GenericLoc, GenericLoc,
                                 MultiExprArg());
  } else {
    TypeSourceInfo *receiverTypeInfo = 0;
    if (!RefExpr->isSuperReceiver())
      receiverTypeInfo = S.Context.getTrivialTypeSourceInfo(receiverType);

    msg = S.BuildClassMessage(receiverTypeInfo, receiverType, superLoc,
                              Getter->getSelector(), Getter,
                              GenericLoc, GenericLoc, GenericLoc,
                              MultiExprArg());
  }
  return msg;
}

/// Store to an Objective-C property reference.
///
/// \param bindSetValueAsResult - If true, capture the actual
///   value being set as the value of the property operation.
ExprResult ObjCPropertyOpBuilder::buildSet(Expr *op, SourceLocation opcLoc,
                                           bool captureSetValueAsResult) {
  bool hasSetter = findSetter();
  assert(hasSetter); (void) hasSetter;

  QualType receiverType;
  SourceLocation superLoc;
  if (RefExpr->isClassReceiver()) {
    receiverType = S.Context.getObjCInterfaceType(RefExpr->getClassReceiver());
  } else if (RefExpr->isSuperReceiver()) {
    superLoc = RefExpr->getReceiverLocation();
    receiverType = RefExpr->getSuperReceiverType();
  } else {
    assert(InstanceReceiver);
    receiverType = InstanceReceiver->getType();
  }

  // Use assignment constraints when possible; they give us better
  // diagnostics.  "When possible" basically means anything except a
  // C++ class type.
  if (!S.getLangOptions().CPlusPlus || !op->getType()->isRecordType()) {
    QualType paramType = (*Setter->param_begin())->getType();
    if (!S.getLangOptions().CPlusPlus || !paramType->isRecordType()) {
      ExprResult opResult = op;
      Sema::AssignConvertType assignResult
        = S.CheckSingleAssignmentConstraints(paramType, opResult);
      if (S.DiagnoseAssignmentResult(assignResult, opcLoc, paramType,
                                     op->getType(), opResult.get(),
                                     Sema::AA_Assigning))
        return ExprError();

      op = opResult.take();
      assert(op && "successful assignment left argument invalid?");
    }
  }

  // Arguments.
  Expr *args[] = { op };

  // Build a message-send.
  ExprResult msg;
  if (Setter->isInstanceMethod() || RefExpr->isObjectReceiver()) {
    msg = S.BuildInstanceMessage(InstanceReceiver, receiverType, superLoc,
                                 SetterSelector, Setter,
                                 GenericLoc, GenericLoc, GenericLoc,
                                 MultiExprArg(args, 1));
  } else {
    TypeSourceInfo *receiverTypeInfo = 0;
    if (!RefExpr->isSuperReceiver())
      receiverTypeInfo = S.Context.getTrivialTypeSourceInfo(receiverType);

    msg = S.BuildClassMessage(receiverTypeInfo, receiverType, superLoc,
                              SetterSelector, Setter,
                              GenericLoc, GenericLoc, GenericLoc,
                              MultiExprArg(args, 1));
  }

  if (!msg.isInvalid() && captureSetValueAsResult) {
    ObjCMessageExpr *msgExpr =
      cast<ObjCMessageExpr>(msg.get()->IgnoreImplicit());
    Expr *arg = msgExpr->getArg(0);
    msgExpr->setArg(0, captureValueAsResult(arg));
  }

  return msg;
}

/// @property-specific behavior for doing lvalue-to-rvalue conversion.
ExprResult ObjCPropertyOpBuilder::buildRValueOperation(Expr *op) {
  // Explicit properties always have getters, but implicit ones don't.
  // Check that before proceeding.
  if (RefExpr->isImplicitProperty() &&
      !RefExpr->getImplicitPropertyGetter()) {
    S.Diag(RefExpr->getLocation(), diag::err_getter_not_found)
      << RefExpr->getBase()->getType();
    return ExprError();
  }

  ExprResult result = PseudoOpBuilder::buildRValueOperation(op);
  if (result.isInvalid()) return ExprError();

  if (RefExpr->isExplicitProperty() && !Getter->hasRelatedResultType())
    S.DiagnosePropertyAccessorMismatch(RefExpr->getExplicitProperty(),
                                       Getter, RefExpr->getLocation());

  // As a special case, if the method returns 'id', try to get
  // a better type from the property.
  if (RefExpr->isExplicitProperty() && result.get()->isRValue() &&
      result.get()->getType()->isObjCIdType()) {
    QualType propType = RefExpr->getExplicitProperty()->getType();
    if (const ObjCObjectPointerType *ptr
          = propType->getAs<ObjCObjectPointerType>()) {
      if (!ptr->isObjCIdType())
        result = S.ImpCastExprToType(result.get(), propType, CK_BitCast);
    }
  }

  return result;
}

/// Try to build this as a call to a getter that returns a reference.
///
/// \return true if it was possible, whether or not it actually
///   succeeded
bool ObjCPropertyOpBuilder::tryBuildGetOfReference(Expr *op,
                                                   ExprResult &result) {
  if (!S.getLangOptions().CPlusPlus) return false;

  findGetter();
  assert(Getter && "property has no setter and no getter!");

  // Only do this if the getter returns an l-value reference type.
  QualType resultType = Getter->getResultType();
  if (!resultType->isLValueReferenceType()) return false;

  result = buildRValueOperation(op);
  return true;
}

/// @property-specific behavior for doing assignments.
ExprResult
ObjCPropertyOpBuilder::buildAssignmentOperation(Scope *Sc,
                                                SourceLocation opcLoc,
                                                BinaryOperatorKind opcode,
                                                Expr *LHS, Expr *RHS) {
  assert(BinaryOperator::isAssignmentOp(opcode));

  // If there's no setter, we have no choice but to try to assign to
  // the result of the getter.
  if (!findSetter()) {
    ExprResult result;
    if (tryBuildGetOfReference(LHS, result)) {
      if (result.isInvalid()) return ExprError();
      return S.BuildBinOp(Sc, opcLoc, opcode, result.take(), RHS);
    }

    // Otherwise, it's an error.
    S.Diag(opcLoc, diag::err_nosetter_property_assignment)
      << unsigned(RefExpr->isImplicitProperty())
      << SetterSelector
      << LHS->getSourceRange() << RHS->getSourceRange();
    return ExprError();
  }

  // If there is a setter, we definitely want to use it.

  // Verify that we can do a compound assignment.
  if (opcode != BO_Assign && !findGetter()) {
    S.Diag(opcLoc, diag::err_nogetter_property_compound_assignment)
      << LHS->getSourceRange() << RHS->getSourceRange();
    return ExprError();
  }

  ExprResult result =
    PseudoOpBuilder::buildAssignmentOperation(Sc, opcLoc, opcode, LHS, RHS);
  if (result.isInvalid()) return ExprError();

  // Various warnings about property assignments in ARC.
  if (S.getLangOptions().ObjCAutoRefCount && InstanceReceiver) {
    S.checkRetainCycles(InstanceReceiver->getSourceExpr(), RHS);
    S.checkUnsafeExprAssigns(opcLoc, LHS, RHS);
  }

  return result;
}

/// @property-specific behavior for doing increments and decrements.
ExprResult
ObjCPropertyOpBuilder::buildIncDecOperation(Scope *Sc, SourceLocation opcLoc,
                                            UnaryOperatorKind opcode,
                                            Expr *op) {
  // If there's no setter, we have no choice but to try to assign to
  // the result of the getter.
  if (!findSetter()) {
    ExprResult result;
    if (tryBuildGetOfReference(op, result)) {
      if (result.isInvalid()) return ExprError();
      return S.BuildUnaryOp(Sc, opcLoc, opcode, result.take());
    }

    // Otherwise, it's an error.
    S.Diag(opcLoc, diag::err_nosetter_property_incdec)
      << unsigned(RefExpr->isImplicitProperty())
      << unsigned(UnaryOperator::isDecrementOp(opcode))
      << SetterSelector
      << op->getSourceRange();
    return ExprError();
  }

  // If there is a setter, we definitely want to use it.

  // We also need a getter.
  if (!findGetter()) {
    assert(RefExpr->isImplicitProperty());
    S.Diag(opcLoc, diag::err_nogetter_property_incdec)
      << unsigned(UnaryOperator::isDecrementOp(opcode))
      << RefExpr->getImplicitPropertyGetter()->getSelector() // FIXME!
      << op->getSourceRange();
    return ExprError();
  }

  return PseudoOpBuilder::buildIncDecOperation(Sc, opcLoc, opcode, op);
}

//===----------------------------------------------------------------------===//
//  General Sema routines.
//===----------------------------------------------------------------------===//

ExprResult Sema::checkPseudoObjectRValue(Expr *E) {
  Expr *opaqueRef = E->IgnoreParens();
  if (ObjCPropertyRefExpr *refExpr
        = dyn_cast<ObjCPropertyRefExpr>(opaqueRef)) {
    ObjCPropertyOpBuilder builder(*this, refExpr);
    return builder.buildRValueOperation(E);
  } else {
    llvm_unreachable("unknown pseudo-object kind!");
  }
}

/// Check an increment or decrement of a pseudo-object expression.
ExprResult Sema::checkPseudoObjectIncDec(Scope *Sc, SourceLocation opcLoc,
                                         UnaryOperatorKind opcode, Expr *op) {
  // Do nothing if the operand is dependent.
  if (op->isTypeDependent())
    return new (Context) UnaryOperator(op, opcode, Context.DependentTy,
                                       VK_RValue, OK_Ordinary, opcLoc);

  assert(UnaryOperator::isIncrementDecrementOp(opcode));
  Expr *opaqueRef = op->IgnoreParens();
  if (ObjCPropertyRefExpr *refExpr
        = dyn_cast<ObjCPropertyRefExpr>(opaqueRef)) {
    ObjCPropertyOpBuilder builder(*this, refExpr);
    return builder.buildIncDecOperation(Sc, opcLoc, opcode, op);
  } else {
    llvm_unreachable("unknown pseudo-object kind!");
  }
}

ExprResult Sema::checkPseudoObjectAssignment(Scope *S, SourceLocation opcLoc,
                                             BinaryOperatorKind opcode,
                                             Expr *LHS, Expr *RHS) {
  // Do nothing if either argument is dependent.
  if (LHS->isTypeDependent() || RHS->isTypeDependent())
    return new (Context) BinaryOperator(LHS, RHS, opcode, Context.DependentTy,
                                        VK_RValue, OK_Ordinary, opcLoc);

  // Filter out non-overload placeholder types in the RHS.
  if (const BuiltinType *PTy = RHS->getType()->getAsPlaceholderType()) {
    if (PTy->getKind() != BuiltinType::Overload) {
      ExprResult result = CheckPlaceholderExpr(RHS);
      if (result.isInvalid()) return ExprError();
      RHS = result.take();
    }
  }

  Expr *opaqueRef = LHS->IgnoreParens();
  if (ObjCPropertyRefExpr *refExpr
        = dyn_cast<ObjCPropertyRefExpr>(opaqueRef)) {
    ObjCPropertyOpBuilder builder(*this, refExpr);
    return builder.buildAssignmentOperation(S, opcLoc, opcode, LHS, RHS);
  } else {
    llvm_unreachable("unknown pseudo-object kind!");
  }
}
