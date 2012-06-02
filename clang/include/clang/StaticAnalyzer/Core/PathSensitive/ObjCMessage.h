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

  const Expr *getMessageExpr() const { 
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

/// \brief Common wrapper for a call expression, ObjC message, or C++ 
/// constructor, mainly to provide a common interface for their arguments.
class CallOrObjCMessage {
  llvm::PointerUnion<const CallExpr *, const CXXConstructExpr *> CallE;
  ObjCMessage Msg;
  ProgramStateRef State;
  const LocationContext *LCtx;

  bool isCallbackArg(unsigned Idx, const Type *T) const;

public:
  CallOrObjCMessage(const CallExpr *callE, ProgramStateRef state,
                    const LocationContext *lctx)
    : CallE(callE), State(state), LCtx(lctx) {}
  CallOrObjCMessage(const CXXConstructExpr *consE, ProgramStateRef state,
                    const LocationContext *lctx)
    : CallE(consE), State(state), LCtx(lctx) {}
  CallOrObjCMessage(const ObjCMessage &msg, ProgramStateRef state,
                    const LocationContext *lctx)
    : CallE((CallExpr *)0), Msg(msg), State(state), LCtx(lctx) {}

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

  /// Check if the callee is declared in the system header.
  bool isInSystemHeader() const {
    if (const Decl *FD = getDecl()) {
      const SourceManager &SM =
        State->getStateManager().getContext().getSourceManager();
      return SM.isInSystemHeader(FD->getLocation());
    }
    return false;
  }

  const Expr *getOriginExpr() const {
    if (!CallE)
      return Msg.getMessageExpr();
    if (const CXXConstructExpr *Ctor =
          CallE.dyn_cast<const CXXConstructExpr *>())
      return Ctor;
    return CallE.get<const CallExpr *>();
  }
  
  SVal getFunctionCallee() const;
  SVal getCXXCallee() const;
  SVal getInstanceMessageReceiver(const LocationContext *LC) const;

  /// Get the declaration of the function or method.
  const Decl *getDecl() const;

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
      return Msg.getArgSVal(i, LCtx, State);
    return State->getSVal(getArg(i), LCtx);
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

  /// \brief Check if one of the arguments might be a callback.
  bool hasNonZeroCallbackArg() const;


  /// \brief Check if the name corresponds to a CoreFoundation or CoreGraphics 
  /// function that allows objects to escape.
  ///
  /// Many methods allow a tracked object to escape.  For example:
  ///
  ///   CFMutableDictionaryRef x = CFDictionaryCreateMutable(..., customDeallocator);
  ///   CFDictionaryAddValue(y, key, x);
  ///
  /// We handle this and similar cases with the following heuristic.  If the
  /// function name contains "InsertValue", "SetValue", "AddValue",
  /// "AppendValue", or "SetAttribute", then we assume that arguments may
  /// escape.
  //
  // TODO: To reduce false negatives here, we should track the container
  // allocation site and check if a proper deallocator was set there.
  static bool isCFCGAllowingEscape(StringRef FName);

  // Check if this kind of expression can be inlined by the analyzer.
  static bool canBeInlined(const Stmt *S) {
    return isa<CallExpr>(S);
  }
};

}
}

#endif
