//===--- CheckerManager.h - Static Analyzer Checker Manager -----*- C++ -*-===//
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

#ifndef LLVM_CLANG_SA_CORE_CHECKERMANAGER_H
#define LLVM_CLANG_SA_CORE_CHECKERMANAGER_H

#include "llvm/ADT/SmallVector.h"

namespace clang {

namespace ento {
  class ExprEngine;

class CheckerManager {
public:
  typedef void (*RegisterFunc)(ExprEngine &Eng);

  void addCheckerRegisterFunction(RegisterFunc fn) {
    Funcs.push_back(fn);
  }
  
  void registerCheckersToEngine(ExprEngine &eng);

private:
  llvm::SmallVector<RegisterFunc, 8> Funcs;
};

} // end ento namespace

} // end clang namespace

#endif
