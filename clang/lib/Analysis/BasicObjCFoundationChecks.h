//== BasicObjCFoundationChecks.h - Simple Apple-Foundation checks -*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicObjCFoundationChecks, a class that encapsulates
//  a set of simple checks to run on Objective-C code using Apple's Foundation
//  classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_BASICOBJCFOUNDATIONCHECKS
#define LLVM_CLANG_ANALYSIS_BASICOBJCFOUNDATIONCHECKS

namespace clang {

class ASTContext;
class BugReporter;
class Decl;
class GRExprEngine;
class GRSimpleAPICheck;

GRSimpleAPICheck *CreateBasicObjCFoundationChecks(ASTContext& Ctx,
                                                  BugReporter& BR);

GRSimpleAPICheck *CreateAuditCFNumberCreate(ASTContext& Ctx,
                                            BugReporter& BR);

GRSimpleAPICheck *CreateAuditCFRetainRelease(ASTContext& Ctx,
                                             BugReporter& BR);

void RegisterNSErrorChecks(BugReporter& BR, GRExprEngine &Eng, const Decl &D);
void RegisterNSAutoreleasePoolChecks(GRExprEngine &Eng);

} // end clang namespace

#endif
