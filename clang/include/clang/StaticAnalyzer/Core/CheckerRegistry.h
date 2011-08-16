//===--- CheckerRegistry.h - Maintains all available checkers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_CHECKERREGISTRY_H
#define LLVM_CLANG_STATICANALYZER_CORE_CHECKERREGISTRY_H

#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/Basic/LLVM.h"
#include <vector>

namespace clang {
namespace ento {

class CheckerOptInfo;

class CheckerRegistry {
public:
  typedef void (*InitializationFunction)(CheckerManager &);
  struct CheckerInfo {
    InitializationFunction Initialize;
    StringRef FullName;
    StringRef Desc;

    CheckerInfo(InitializationFunction fn, StringRef name, StringRef desc)
    : Initialize(fn), FullName(name), Desc(desc) {}
  };

  typedef std::vector<CheckerInfo> CheckerInfoList;

private:
  template <typename T>
  static void initializeManager(CheckerManager &mgr) {
    mgr.registerChecker<T>();
  }

public:
  void addChecker(InitializationFunction fn, StringRef fullName,
                  StringRef desc);

  template <class T>
  void addChecker(StringRef fullName, StringRef desc) {
    addChecker(&initializeManager<T>, fullName, desc);
  }

  void initializeManager(CheckerManager &mgr,
                         SmallVectorImpl<CheckerOptInfo> &opts) const;
  void printHelp(raw_ostream &out, size_t maxNameChars = 30) const ;

private:
  mutable CheckerInfoList Checkers;
  mutable llvm::StringMap<size_t> Packages;
};

} // end namespace ento
} // end namespace clang

#endif
