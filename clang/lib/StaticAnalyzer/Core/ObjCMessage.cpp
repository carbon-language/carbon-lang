//===- ObjCMessage.cpp - Wrapper for ObjC messages and dot syntax -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines ObjCMessage which serves as a common wrapper for ObjC
// message expressions or implicit messages for loading/storing ObjC properties.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"

using namespace clang;
using namespace ento;

QualType ObjCMessage::getType(ASTContext &ctx) const {
  assert(isValid() && "This ObjCMessage is uninitialized!");
  if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
    return msgE->getType();
  const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
  if (isPropertySetter())
    return ctx.VoidTy;
  return propE->getType();
}

Selector ObjCMessage::getSelector() const {
  assert(isValid() && "This ObjCMessage is uninitialized!");
  if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
    return msgE->getSelector();
  const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
  if (isPropertySetter())
    return propE->getSetterSelector();
  return propE->getGetterSelector();
}

const ObjCMethodDecl *ObjCMessage::getMethodDecl() const {
  assert(isValid() && "This ObjCMessage is uninitialized!");
  if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
    return msgE->getMethodDecl();
  const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
  if (propE->isImplicitProperty())
    return isPropertySetter() ? propE->getImplicitPropertySetter()
                              : propE->getImplicitPropertyGetter();
  return 0;
}

const ObjCInterfaceDecl *ObjCMessage::getReceiverInterface() const {
  assert(isValid() && "This ObjCMessage is uninitialized!");
  if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
    return msgE->getReceiverInterface();
  const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
  if (propE->isClassReceiver())
    return propE->getClassReceiver();
  QualType recT;
  if (const Expr *recE = getInstanceReceiver())
    recT = recE->getType();
  else {
    assert(propE->isSuperReceiver());
    recT = propE->getSuperReceiverType();
  }
  if (const ObjCObjectPointerType *Ptr = recT->getAs<ObjCObjectPointerType>())
    return Ptr->getInterfaceDecl();
  return 0;
}

const Expr *ObjCMessage::getArgExpr(unsigned i) const {
  assert(isValid() && "This ObjCMessage is uninitialized!");
  assert(i < getNumArgs() && "Invalid index for argument");
  if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
    return msgE->getArg(i);
  assert(isPropertySetter());
  if (const BinaryOperator *bop = dyn_cast<BinaryOperator>(OriginE))
    if (bop->isAssignmentOp())
      return bop->getRHS();
  return 0;
}

QualType CallOrObjCMessage::getResultType(ASTContext &ctx) const {
  if (CallE) {
    const Expr *Callee = CallE->getCallee();
    if (const FunctionDecl *FD = State->getSVal(Callee).getAsFunctionDecl())
      return FD->getResultType();
    return CallE->getType();
  }
  return Msg.getResultType(ctx);
}

SVal CallOrObjCMessage::getArgSValAsScalarOrLoc(unsigned i) const {
  assert(i < getNumArgs());
  if (CallE) return State->getSValAsScalarOrLoc(CallE->getArg(i));
  QualType argT = Msg.getArgType(i);
  if (Loc::isLocType(argT) || argT->isIntegerType())
    return Msg.getArgSVal(i, State);
  return UnknownVal();
}
