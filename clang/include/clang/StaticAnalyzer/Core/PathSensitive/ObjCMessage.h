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
#include "llvm/ADT/PointerUnion.h"

namespace clang {
namespace ento {

/// \brief Represents both explicit ObjC message expressions and implicit
/// messages that are sent for handling properties in dot syntax.
class ObjCMessage {
  const Expr *MsgOrPropE;
  const Expr *OriginE;
  bool IsPropSetter;
  SVal SetterArgV;

protected:
  ObjCMessage(const Expr *E, const Expr *origE, bool isSetter, SVal setArgV)
    : MsgOrPropE(E), OriginE(origE),
      IsPropSetter(isSetter), SetterArgV(setArgV) { }

public:
  ObjCMessage() : MsgOrPropE(0), OriginE(0) { }

  ObjCMessage(const ObjCMessageExpr *E)
    : MsgOrPropE(E), OriginE(E) {
    assert(E && "should not be initialized with null expression");
  }

  bool isValid() const { return MsgOrPropE != 0; }
  bool isInvalid() const { return !isValid(); }

  bool isMessageExpr() const {
    return isValid() && isa<ObjCMessageExpr>(MsgOrPropE);
  }

  bool isPropertyGetter() const {
    return isValid() &&
           isa<ObjCPropertyRefExpr>(MsgOrPropE) && !IsPropSetter;
  }

  bool isPropertySetter() const {
    return isValid() &&
           isa<ObjCPropertyRefExpr>(MsgOrPropE) && IsPropSetter;
  }
  
  const Expr *getOriginExpr() const { return OriginE; }

  QualType getType(ASTContext &ctx) const;

  QualType getResultType(ASTContext &ctx) const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      if (const ObjCMethodDecl *MD = msgE->getMethodDecl())
        return MD->getResultType();
    return getType(ctx);
  }

  ObjCMethodFamily getMethodFamily() const;

  Selector getSelector() const;

  const Expr *getInstanceReceiver() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return msgE->getInstanceReceiver();
    const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
    if (propE->isObjectReceiver())
      return propE->getBase();
    return 0;
  }

  SVal getInstanceReceiverSVal(const ProgramState *State,
                               const LocationContext *LC) const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (!isInstanceMessage())
      return UndefinedVal();
    if (const Expr *Ex = getInstanceReceiver())
      return State->getSValAsScalarOrLoc(Ex);

    // An instance message with no expression means we are sending to super.
    // In this case the object reference is the same as 'self'.
    const ImplicitParamDecl *SelfDecl = LC->getSelfDecl();
    assert(SelfDecl && "No message receiver Expr, but not in an ObjC method");
    return State->getSVal(State->getRegion(SelfDecl, LC));
  }

  bool isInstanceMessage() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return msgE->isInstanceMessage();
    const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
    // FIXME: 'super' may be super class.
    return propE->isObjectReceiver() || propE->isSuperReceiver();
  }

  const ObjCMethodDecl *getMethodDecl() const;

  const ObjCInterfaceDecl *getReceiverInterface() const;

  SourceLocation getSuperLoc() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return msgE->getSuperLoc();
    return cast<ObjCPropertyRefExpr>(MsgOrPropE)->getReceiverLocation();
  }

  const Expr *getMsgOrPropExpr() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    return MsgOrPropE;
  }

  SourceRange getSourceRange() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    return MsgOrPropE->getSourceRange();
  }

  unsigned getNumArgs() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return msgE->getNumArgs();
    return isPropertySetter() ? 1 : 0;
  }

  SVal getArgSVal(unsigned i, const ProgramState *state) const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    assert(i < getNumArgs() && "Invalid index for argument");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return state->getSVal(msgE->getArg(i));
    assert(isPropertySetter());
    return SetterArgV;
  }

  QualType getArgType(unsigned i) const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    assert(i < getNumArgs() && "Invalid index for argument");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return msgE->getArg(i)->getType();
    assert(isPropertySetter());
    return cast<ObjCPropertyRefExpr>(MsgOrPropE)->getType();
  }

  const Expr *getArgExpr(unsigned i) const;

  SourceRange getArgSourceRange(unsigned i) const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    assert(i < getNumArgs() && "Invalid index for argument");
    if (const Expr *argE = getArgExpr(i))
      return argE->getSourceRange();
    return OriginE->getSourceRange();
  }

  SourceRange getReceiverSourceRange() const {
    assert(isValid() && "This ObjCMessage is uninitialized!");
    if (const ObjCMessageExpr *msgE = dyn_cast<ObjCMessageExpr>(MsgOrPropE))
      return msgE->getReceiverRange();

    const ObjCPropertyRefExpr *propE = cast<ObjCPropertyRefExpr>(MsgOrPropE);
    if (propE->isObjectReceiver())
      return propE->getBase()->getSourceRange();

    // FIXME: This isn't a range.
    return propE->getReceiverLocation();
  }
};

class ObjCPropertyGetter : public ObjCMessage {
public:
  ObjCPropertyGetter(const ObjCPropertyRefExpr *propE, const Expr *originE)
    : ObjCMessage(propE, originE, false, SVal()) {
    assert(propE && originE &&
           "should not be initialized with null expressions");
  }
};

class ObjCPropertySetter : public ObjCMessage {
public:
  ObjCPropertySetter(const ObjCPropertyRefExpr *propE, const Expr *storeE,
                     SVal argV)
    : ObjCMessage(propE, storeE, true, argV) {
    assert(propE && storeE &&"should not be initialized with null expressions");
  }
};

/// \brief Common wrapper for a call expression, ObjC message, or C++ 
/// constructor, mainly to provide a common interface for their arguments.
class CallOrObjCMessage {
  llvm::PointerUnion<const CallExpr *, const CXXConstructExpr *> CallE;
  ObjCMessage Msg;
  const ProgramState *State;
public:
  CallOrObjCMessage(const CallExpr *callE, const ProgramState *state)
    : CallE(callE), State(state) {}
  CallOrObjCMessage(const CXXConstructExpr *consE, const ProgramState *state)
    : CallE(consE), State(state) {}
  CallOrObjCMessage(const ObjCMessage &msg, const ProgramState *state)
    : CallE((CallExpr *)0), Msg(msg), State(state) {}

  QualType getResultType(ASTContext &ctx) const;
  
  bool isFunctionCall() const {
    return CallE && CallE.is<const CallExpr *>();
  }

  bool isCXXConstructExpr() const {
    return CallE && CallE.is<const CXXConstructExpr *>();
  }

  bool isObjCMessage() const {
    return !CallE;
  }

  bool isCXXCall() const {
    const CallExpr *ActualCallE = CallE.dyn_cast<const CallExpr *>();
    return ActualCallE && isa<CXXMemberCallExpr>(ActualCallE);
  }

  const Expr *getOriginExpr() const {
    if (!CallE)
      return Msg.getOriginExpr();
    if (const CXXConstructExpr *Ctor =
          CallE.dyn_cast<const CXXConstructExpr *>())
      return Ctor;
    return CallE.get<const CallExpr *>();
  }
  
  SVal getFunctionCallee() const;
  SVal getCXXCallee() const;
  SVal getInstanceMessageReceiver(const LocationContext *LC) const;

  unsigned getNumArgs() const {
    if (!CallE)
      return Msg.getNumArgs();
    if (const CXXConstructExpr *Ctor =
          CallE.dyn_cast<const CXXConstructExpr *>())
      return Ctor->getNumArgs();
    return CallE.get<const CallExpr *>()->getNumArgs();
  }

  SVal getArgSVal(unsigned i) const {
    assert(i < getNumArgs());
    if (!CallE)
      return Msg.getArgSVal(i, State);
    return State->getSVal(getArg(i));
  }

  const Expr *getArg(unsigned i) const {
    assert(i < getNumArgs());
    if (!CallE)
      return Msg.getArgExpr(i);
    if (const CXXConstructExpr *Ctor =
          CallE.dyn_cast<const CXXConstructExpr *>())
      return Ctor->getArg(i);
    return CallE.get<const CallExpr *>()->getArg(i);
  }

  SourceRange getArgSourceRange(unsigned i) const {
    assert(i < getNumArgs());
    if (CallE)
      return getArg(i)->getSourceRange();
    return Msg.getArgSourceRange(i);
  }

  SourceRange getReceiverSourceRange() const {
    assert(isObjCMessage());
    return Msg.getReceiverSourceRange();
  }
};

}
}

#endif
