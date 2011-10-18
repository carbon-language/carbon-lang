//== CheckerContext.cpp - Context info for path-sensitive checkers-----------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerContext that provides contextual info for
//  path-sensitive checkers.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
using namespace clang;
using namespace ento;

CheckerContext::~CheckerContext() {
  // Copy the results into the Dst set.
  for (NodeBuilder::iterator I = NB.results_begin(),
                             E = NB.results_end(); I != E; ++I) {
    Dst.Add(*I);
  }
}
