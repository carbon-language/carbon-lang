//===--- tools/extra/clang-tidy/ClangTidyModule.cpp - Clang tidy tool -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file Implements classes required to build clang-tidy modules.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyModule.h"

namespace clang {
namespace tidy {

ClangTidyCheckFactories::~ClangTidyCheckFactories() {
  for (const auto &Factory : Factories)
    delete Factory.second;
}

void ClangTidyCheckFactories::addCheckFactory(StringRef Name,
                                              CheckFactoryBase *Factory) {
  Factories[Name] = Factory;
}

void ClangTidyCheckFactories::createChecks(
    ChecksFilter &Filter, SmallVectorImpl<ClangTidyCheck *> &Checks) {
  for (const auto &Factory : Factories) {
    if (Filter.IsCheckEnabled(Factory.first)) {
      ClangTidyCheck *Check = Factory.second->createCheck();
      Check->setName(Factory.first);
      Checks.push_back(Check);
    }
  }
}

} // namespace tidy
} // namespace clang
