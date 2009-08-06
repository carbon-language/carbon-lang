// GRCheckAPI.h - Simple API checks based on GRAuditor ------------*- C++ -*--//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface for building simple, path-sensitive checks
//  that are stateless and only emit warnings at errors that occur at
//  CallExpr or ObjCMessageExpr.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_GRAPICHECKS
#define LLVM_CLANG_ANALYSIS_GRAPICHECKS

#include "clang/Analysis/PathSensitive/GRAuditor.h"
#include "clang/Analysis/PathSensitive/GRState.h"

namespace clang {
  
class Diagnostic;
class BugReporter;
class ASTContext;
class GRExprEngine;
class PathDiagnosticClient;
class ExplodedGraph;
  
  
class GRSimpleAPICheck : public GRAuditor {
public:
  GRSimpleAPICheck() {}
  virtual ~GRSimpleAPICheck() {}
};

} // end namespace clang

#endif
