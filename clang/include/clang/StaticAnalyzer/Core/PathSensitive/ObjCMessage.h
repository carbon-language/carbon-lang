//===- ObjCMessage.h - Wrapper for ObjC messages and dot syntax ---*- C++ -*--//
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

#ifndef LLVM_CLANG_STATICANALYZER_PATHSENSITIVE_OBJCMESSAGE
#define LLVM_CLANG_STATICANALYZER_PATHSENSITIVE_OBJCMESSAGE

#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/AST/ExprObjC.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace ento {
using llvm::StrInStrNoCase;

/// \brief Represents both explicit ObjC message expressions and implicit
/// messages that are sent for handling properties in dot syntax.
class ObjCMessage {
  const ObjCMessageExpr *Msg;
  const ObjCPropertyRefExpr *PE;
  const bool IsPropSetter;
public:
  ObjCMessage() : Msg(0), PE(0), IsPropSetter(false) {}

  ObjCMessage(const ObjCMessageExpr *E, const ObjCPropertyRefExpr *pe = 0,
              bool isSetter = false)
    : Msg(E), PE(pe), IsPropSetter(isSetter) {
    assert(E && "should not be initialized with null expression");
  }

  bool isValid() const { return Msg; }
  
  bool isPureMessageExpr() const { return !PE; }

  bool isPropertyGetter() const { return PE && !IsPropSetter; }

  bool isPropertySetter() const {
    return IsPropSetter;
  }

  const ObjCMessageExpr *getMessageExpr() const {
    return Msg;
  }

  QualType getType(ASTContext &ctx) const {
    return Msg->getType();
  }

  QualType getResultType(ASTContext &ctx) const {
    if (const ObjCMethodDecl *MD = Msg->getMethodDecl())
      return MD->getResultType();
    return getType(ctx);
  }

  ObjCMethodFamily getMethodFamily() const {
    return Msg->getMethodFamily();
  }

  Selector getSelector() const {
    return Msg->getSelector();
  }

  const Expr *getInstanceReceiver() const {
    return Msg->getInstanceReceiver();
  }

  SVal getInstanceReceiverSVal(ProgramStateRef State,
                               const LocationContext *LC) const {
    if (!isInstanceMessage())
      return UndefinedVal();
    if (const Expr *Ex = getInstanceReceiver())
      return State->getSValAsScalarOrLoc(Ex, LC);

    // An instance message with no expression means we are sending to super.
    // In this case the object reference is the same as 'self'.
    const ImplicitParamDecl *SelfDecl = LC->getSelfDecl();
    assert(SelfDecl && "No message receiver Expr, but not in an ObjC method");
    return State->getSVal(State->getRegion(SelfDecl, LC));
  }

  bool isInstanceMessage() const {
    return Msg->isInstanceMessage();
  }

  const ObjCMethodDecl *getMethodDecl() const {
    return Msg->getMethodDecl();
  }

  const ObjCInterfaceDecl *getReceiverInterface() const {
    return Msg->getReceiverInterface();
  }

  SourceLocation getSuperLoc() const {
    if (PE)
      return PE->getReceiverLocation();
    return Msg->getSuperLoc();
  }  

  SourceRange getSourceRange() const LLVM_READONLY {
    if (PE)
      return PE->getSourceRange();
    return Msg->getSourceRange();
  }

  unsigned getNumArgs() const {
    return Msg->getNumArgs();
  }

  SVal getArgSVal(unsigned i,
                  const LocationContext *LCtx,
                  ProgramStateRef state) const {
    assert(i < getNumArgs() && "Invalid index for argument");
    return state->getSVal(Msg->getArg(i), LCtx);
  }

  QualType getArgType(unsigned i) const {
    assert(i < getNumArgs() && "Invalid index for argument");
    return Msg->getArg(i)->getType();
  }

  const Expr *getArgExpr(unsigned i) const {
    assert(i < getNumArgs() && "Invalid index for argument");
    return Msg->getArg(i);
  }

  SourceRange getArgSourceRange(unsigned i) const {
    const Expr *argE = getArgExpr(i);
    return argE->getSourceRange();
  }

  SourceRange getReceiverSourceRange() const {
    if (PE) {
      if (PE->isObjectReceiver())
        return PE->getBase()->getSourceRange();
    }
    else {
      return Msg->getReceiverRange();
    }

    // FIXME: This isn't a range.
    return PE->getReceiverLocation();
  }
};

}
}

#endif
