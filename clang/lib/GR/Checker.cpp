//== Checker.h - Abstract interface for checkers -----------------*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines Checker and CheckerVisitor, classes used for creating
//  domain-specific checks.
//
//===----------------------------------------------------------------------===//

#include "clang/GR/PathSensitive/Checker.h"
using namespace clang;

Checker::~Checker() {}

CheckerContext::~CheckerContext() {
  // Do we need to autotransition?  'Dst' can get populated in a variety of
  // ways, including 'addTransition()' adding the predecessor node to Dst
  // without actually generated a new node.  We also shouldn't autotransition
  // if we are building sinks or we generated a node and decided to not
  // add it as a transition.
  if (Dst.size() == size && !B.BuildSinks && !B.HasGeneratedNode) {
    if (ST && ST != B.GetState(Pred)) {
      static int autoTransitionTag = 0;
      B.Tag = &autoTransitionTag;
      addTransition(ST);
    }
    else
      Dst.Add(Pred);
  }
}
