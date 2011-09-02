//=- NSAutoreleasePoolChecker.cpp --------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a NSAutoreleasePoolChecker, a small checker that warns
//  about subpar uses of NSAutoreleasePool.  Note that while the check itself
//  (in its current form) could be written as a flow-insensitive check, in
//  can be potentially enhanced in the future with flow-sensitive information.
//  It is also a good example of the CheckerVisitor interface. 
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ObjCMessage.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Decl.h"

using namespace clang;
using namespace ento;

namespace {
class NSAutoreleasePoolChecker
  : public Checker<check::PreObjCMessage> {
      
  mutable Selector releaseS;

public:
  void checkPreObjCMessage(ObjCMessage msg, CheckerContext &C) const;    
};

} // end anonymous namespace

void NSAutoreleasePoolChecker::checkPreObjCMessage(ObjCMessage msg,
                                                   CheckerContext &C) const {
  
  const Expr *receiver = msg.getInstanceReceiver();
  if (!receiver)
    return;
  
  // FIXME: Enhance with value-tracking information instead of consulting
  // the type of the expression.
  const ObjCObjectPointerType* PT =
    receiver->getType()->getAs<ObjCObjectPointerType>();
  
  if (!PT)
    return;  
  const ObjCInterfaceDecl *OD = PT->getInterfaceDecl();
  if (!OD)
    return;  
  if (!OD->getIdentifier()->getName().equals("NSAutoreleasePool"))
    return;

  if (releaseS.isNull())
    releaseS = GetNullarySelector("release", C.getASTContext());
  // Sending 'release' message?
  if (msg.getSelector() != releaseS)
    return;
                     
  SourceRange R = msg.getSourceRange();

  C.getBugReporter().EmitBasicReport("Use -drain instead of -release",
    "API Upgrade (Apple)",
    "Use -drain instead of -release when using NSAutoreleasePool "
    "and garbage collection", R.getBegin(), &R, 1);
}

void ento::registerNSAutoreleasePoolChecker(CheckerManager &mgr) {
  if (mgr.getLangOptions().getGCMode() != LangOptions::NonGC)
    mgr.registerChecker<NSAutoreleasePoolChecker>();
}
