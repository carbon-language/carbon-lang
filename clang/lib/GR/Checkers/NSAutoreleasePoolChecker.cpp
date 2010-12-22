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
//  (in it's current form) could be written as a flow-insensitive check, in
//  can be potentially enhanced in the future with flow-sensitive information.
//  It is also a good example of the CheckerVisitor interface. 
//
//===----------------------------------------------------------------------===//

#include "clang/GR/BugReporter/BugReporter.h"
#include "clang/GR/PathSensitive/ExprEngine.h"
#include "clang/GR/PathSensitive/CheckerVisitor.h"
#include "BasicObjCFoundationChecks.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Decl.h"

using namespace clang;
using namespace GR;

namespace {
class NSAutoreleasePoolChecker
  : public CheckerVisitor<NSAutoreleasePoolChecker> {
      
  Selector releaseS;

public:
    NSAutoreleasePoolChecker(Selector release_s) : releaseS(release_s) {}
    
  static void *getTag() {
    static int x = 0;
    return &x;
  }

  void PreVisitObjCMessageExpr(CheckerContext &C, const ObjCMessageExpr *ME);    
};

} // end anonymous namespace


void GR::RegisterNSAutoreleasePoolChecks(ExprEngine &Eng) {
  ASTContext &Ctx = Eng.getContext();
  if (Ctx.getLangOptions().getGCMode() != LangOptions::NonGC) {    
    Eng.registerCheck(new NSAutoreleasePoolChecker(GetNullarySelector("release",
                                                                      Ctx)));
  }
}

void
NSAutoreleasePoolChecker::PreVisitObjCMessageExpr(CheckerContext &C,
                                                  const ObjCMessageExpr *ME) {
  
  const Expr *receiver = ME->getInstanceReceiver();
  if (!receiver)
    return;
  
  // FIXME: Enhance with value-tracking information instead of consulting
  // the type of the expression.
  const ObjCObjectPointerType* PT =
    receiver->getType()->getAs<ObjCObjectPointerType>();
  
  if (!PT)
    return;  
  const ObjCInterfaceDecl* OD = PT->getInterfaceDecl();
  if (!OD)
    return;  
  if (!OD->getIdentifier()->getName().equals("NSAutoreleasePool"))
    return;
  
  // Sending 'release' message?
  if (ME->getSelector() != releaseS)
    return;
                     
  SourceRange R = ME->getSourceRange();

  C.getBugReporter().EmitBasicReport("Use -drain instead of -release",
    "API Upgrade (Apple)",
    "Use -drain instead of -release when using NSAutoreleasePool "
    "and garbage collection", ME->getLocStart(), &R, 1);
}
