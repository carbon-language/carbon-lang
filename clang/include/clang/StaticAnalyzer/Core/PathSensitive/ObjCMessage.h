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

/// \brief Common wrapper for a call expression or an ObjC message, mainly to
/// provide a common interface for handling their arguments.
class CallOrObjCMessage {
  const CallExpr *CallE;
  ObjCMessage Msg;
  const ProgramState *State;
public:
  CallOrObjCMessage(const CallExpr *callE, const ProgramState *state)
    : CallE(callE), State(state) {}
  CallOrObjCMessage(const ObjCMessage &msg, const ProgramState *state)
    : CallE(0), Msg(msg), State(state) {}

  QualType getResultType(ASTContext &ctx) const;
  
  bool isFunctionCall() const {
    return (bool) CallE;
  }
  
  bool isCXXCall() const {
    return CallE && isa<CXXMemberCallExpr>(CallE);
  }

  const Expr *getOriginExpr() const {
    if (isFunctionCall())
      return CallE;
    return Msg.getOriginExpr();
  }
  
  SVal getFunctionCallee() const;
  SVal getCXXCallee() const;

  unsigned getNumArgs() const {
    if (CallE) return CallE->getNumArgs();
    return Msg.getNumArgs();
  }

  SVal getArgSVal(unsigned i) const {
    assert(i < getNumArgs());
    if (CallE) 
      return State->getSVal(CallE->getArg(i));
    return Msg.getArgSVal(i, State);
  }

  SVal getArgSValAsScalarOrLoc(unsigned i) const;

  const Expr *getArg(unsigned i) const {
    assert(i < getNumArgs());
    if (CallE)
      return CallE->getArg(i);
    return Msg.getArgExpr(i);
  }

  SourceRange getArgSourceRange(unsigned i) const {
    assert(i < getNumArgs());
    if (CallE)
      return CallE->getArg(i)->getSourceRange();
    return Msg.getArgSourceRange(i);
  }
};

}
}

#endif
