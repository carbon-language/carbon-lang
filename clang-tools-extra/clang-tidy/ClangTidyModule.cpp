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
  for (FactoryMap::iterator I = Factories.begin(), E = Factories.end(); I != E;
       ++I) {
    delete I->second;
  }
}
void ClangTidyCheckFactories::addCheckFactory(StringRef Name,
                                              CheckFactoryBase *Factory) {

  Factories[Name] = Factory;
}

void ClangTidyCheckFactories::createChecks(
    ChecksFilter &Filter, SmallVectorImpl<ClangTidyCheck *> &Checks) {
  for (FactoryMap::iterator I = Factories.begin(), E = Factories.end(); I != E;
       ++I) {
    if (Filter.IsCheckEnabled(I->first))
      Checks.push_back(I->second->createCheck());
  }
}

} // namespace tidy
} // namespace clang
