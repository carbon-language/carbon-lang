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

#include "clang/Analysis/PathSensitive/ExplodedGraph.h"
#include "clang/Analysis/PathSensitive/GRSimpleAPICheck.h"
#include "clang/Analysis/PathSensitive/GRState.h"
#include "clang/Analysis/PathDiagnostic.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/Compiler.h"

#ifndef LLVM_CLANG_ANALYSIS_BASICOBJCFOUNDATIONCHECKS
#define LLVM_CLANG_ANALYSIS_BASICOBJCFOUNDATIONCHECKS

namespace clang {
  
class GRSimpleAPICheck;
class ASTContext;
class GRStateManager;  
class BugReporter;
class GRExprEngine;
  
GRSimpleAPICheck *CreateBasicObjCFoundationChecks(ASTContext& Ctx,
                                                  BugReporter& BR);
  
GRSimpleAPICheck *CreateAuditCFNumberCreate(ASTContext& Ctx,
                                            BugReporter& BR);
  
GRSimpleAPICheck *CreateAuditCFRetainRelease(ASTContext& Ctx,
                                             BugReporter& BR);
  
void RegisterNSErrorChecks(BugReporter& BR, GRExprEngine &Eng, const Decl &D);
  
} // end clang namespace

#endif
