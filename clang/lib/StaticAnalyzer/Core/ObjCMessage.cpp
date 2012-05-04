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
#include "clang/AST/DeclCXX.h"

using namespace clang;
using namespace ento;

QualType CallOrObjCMessage::getResultType(ASTContext &ctx) const {
  QualType resultTy;
  bool isLVal = false;

  if (isObjCMessage()) {
    resultTy = Msg.getResultType(ctx);
  } else if (const CXXConstructExpr *Ctor =
              CallE.dyn_cast<const CXXConstructExpr *>()) {
    resultTy = Ctor->getType();
  } else {
    const CallExpr *FunctionCall = CallE.get<const CallExpr *>();

    isLVal = FunctionCall->isLValue();
    const Expr *Callee = FunctionCall->getCallee();
    if (const FunctionDecl *FD = State->getSVal(Callee, LCtx).getAsFunctionDecl())
      resultTy = FD->getResultType();
    else
      resultTy = FunctionCall->getType();
  }

  if (isLVal)
    resultTy = ctx.getPointerType(resultTy);

  return resultTy;
}

SVal CallOrObjCMessage::getFunctionCallee() const {
  assert(isFunctionCall());
  assert(!isCXXCall());
  const Expr *Fun = CallE.get<const CallExpr *>()->getCallee()->IgnoreParens();
  return State->getSVal(Fun, LCtx);
}

SVal CallOrObjCMessage::getCXXCallee() const {
  assert(isCXXCall());
  const CallExpr *ActualCall = CallE.get<const CallExpr *>();
  const Expr *callee =
    cast<CXXMemberCallExpr>(ActualCall)->getImplicitObjectArgument();
  
  // FIXME: Will eventually need to cope with member pointers.  This is
  // a limitation in getImplicitObjectArgument().
  if (!callee)
    return UnknownVal();
  
  return State->getSVal(callee, LCtx);
}

SVal
CallOrObjCMessage::getInstanceMessageReceiver(const LocationContext *LC) const {
  assert(isObjCMessage());
  return Msg.getInstanceReceiverSVal(State, LC);
}

const Decl *CallOrObjCMessage::getDecl() const {
  if (isCXXCall()) {
    const CXXMemberCallExpr *CE =
        cast<CXXMemberCallExpr>(CallE.dyn_cast<const CallExpr *>());
    assert(CE);
    return CE->getMethodDecl();
  } else if (isObjCMessage()) {
    return Msg.getMethodDecl();
  } else if (isFunctionCall()) {
    // In case of a C style call, use the path sensitive information to find
    // the function declaration.
    SVal CalleeVal = getFunctionCallee();
    return CalleeVal.getAsFunctionDecl();
  }
  return 0;
}

bool CallOrObjCMessage::isCallbackArg(unsigned Idx, const Type *T) const {
  // If the parameter is 0, it's harmless.
  if (getArgSVal(Idx).isZeroConstant())
    return false;
    
  // If a parameter is a block or a callback, assume it can modify pointer.
  if (T->isBlockPointerType() || T->isFunctionPointerType())
    return true;

  // Check if a callback is passed inside a struct (for both, struct passed by
  // reference and by value). Dig just one level into the struct for now.
  if (const PointerType *PT = dyn_cast<PointerType>(T))
    T = PT->getPointeeType().getTypePtr();

  if (const RecordType *RT = T->getAsStructureType()) {
    const RecordDecl *RD = RT->getDecl();
    for (RecordDecl::field_iterator I = RD->field_begin(),
                                    E = RD->field_end(); I != E; ++I ) {
      const Type *FieldT = I->getType().getTypePtr();
      if (FieldT->isBlockPointerType() || FieldT->isFunctionPointerType())
        return true;
    }
  }
  return false;
}

bool CallOrObjCMessage::hasNonZeroCallbackArg() const {
  unsigned NumOfArgs = getNumArgs();

  // Process ObjC message first.
  if (!CallE) {
    const ObjCMethodDecl *D = Msg.getMethodDecl();
    unsigned Idx = 0;
    for (ObjCMethodDecl::param_const_iterator I = D->param_begin(),
                                     E = D->param_end(); I != E; ++I, ++Idx) {
      if (NumOfArgs <= Idx)
        break;

      if (isCallbackArg(Idx, (*I)->getType().getTypePtr()))
        return true;
    }
    return false;
  }

  // Else, assume we are dealing with a Function call.
  const FunctionDecl *FD = 0;
  if (const CXXConstructExpr *Ctor =
        CallE.dyn_cast<const CXXConstructExpr *>())
    FD = Ctor->getConstructor();

  const CallExpr * CE = CallE.get<const CallExpr *>();
  FD = dyn_cast_or_null<FunctionDecl>(CE->getCalleeDecl());

  // If calling using a function pointer, assume the function does not
  // have a callback. TODO: We could check the types of the arguments here.
  if (!FD)
    return false;

  unsigned Idx = 0;
  for (FunctionDecl::param_const_iterator I = FD->param_begin(),
                                      E = FD->param_end(); I != E; ++I, ++Idx) {
    if (NumOfArgs <= Idx)
      break;

    if (isCallbackArg(Idx, (*I)->getType().getTypePtr()))
      return true;
  }
  return false;
}

bool CallOrObjCMessage::isCFCGAllowingEscape(StringRef FName) {
  if (FName[0] == 'C' && (FName[1] == 'F' || FName[1] == 'G'))
         if (StrInStrNoCase(FName, "InsertValue") != StringRef::npos||
             StrInStrNoCase(FName, "AddValue") != StringRef::npos ||
             StrInStrNoCase(FName, "SetValue") != StringRef::npos ||
             StrInStrNoCase(FName, "WithData") != StringRef::npos ||
             StrInStrNoCase(FName, "AppendValue") != StringRef::npos||
             StrInStrNoCase(FName, "SetAttribute") != StringRef::npos) {
       return true;
     }
  return false;
}


