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

#include "clang/Checker/PathSensitive/GRAuditor.h"

namespace clang {

class GRSimpleAPICheck : public GRAuditor {
public:
  GRSimpleAPICheck() {}
  virtual ~GRSimpleAPICheck() {}
};

} // end namespace clang

#endif
