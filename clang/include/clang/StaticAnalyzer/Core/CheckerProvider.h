//===--- CheckerProvider.h - Static Analyzer Checkers Provider --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines the Static Analyzer Checker Provider.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_CORE_CHECKERPROVIDER_H
#define LLVM_CLANG_SA_CORE_CHECKERPROVIDER_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {

namespace ento {
  class CheckerManager;

class CheckerOptInfo {
  const char *Name;
  bool Enable;
  bool Claimed;

public:
  CheckerOptInfo(const char *name, bool enable)
    : Name(name), Enable(enable), Claimed(false) { }
  
  const char *getName() const { return Name; }
  bool isEnabled() const { return Enable; }
  bool isDisabled() const { return !isEnabled(); }

  bool isClaimed() const { return Claimed; }
  bool isUnclaimed() const { return !isClaimed(); }
  void claim() { Claimed = true; }
};

class CheckerProvider {
public:
  virtual ~CheckerProvider();
  virtual void registerCheckers(CheckerManager &checkerMgr,
                          CheckerOptInfo *checkOpts, unsigned numCheckOpts) = 0;
  virtual void printHelp(raw_ostream &OS) = 0;
};

} // end ento namespace

} // end clang namespace

#endif
