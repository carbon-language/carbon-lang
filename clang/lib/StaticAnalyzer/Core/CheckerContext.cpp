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
  // Do we need to autotransition?  'Dst' can get populated in a variety of
  // ways, including 'addTransition()' adding the predecessor node to Dst
  // without actually generated a new node.  We also shouldn't autotransition
  // if we are building sinks or we generated a node and decided to not
  // add it as a transition.
  if (Dst.size() == size && !B.BuildSinks && !B.hasGeneratedNode) {
    if (ST && ST != Pred->getState()) {
      static int autoTransitionTag = 0;
      addTransition(ST, &autoTransitionTag);
    }
    else
      Dst.Add(Pred);
  }
}
