//===--- CheckerManager.cpp - Static Analyzer Checker Manager -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the Static Analyzer Checker Manager.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/CheckerProvider.h"

using namespace clang;
using namespace ento;

void CheckerManager::registerCheckersToEngine(ExprEngine &eng) {
  for (unsigned i = 0, e = Funcs.size(); i != e; ++i)
    Funcs[i](eng);
}

// Anchor for the vtable.
CheckerProvider::~CheckerProvider() { }
