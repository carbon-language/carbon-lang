//===--- tools/extra/clang-tidy/ClangTidyModule.cpp - Clang tidy tool -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file Implements classes required to build clang-tidy modules.
///
//===----------------------------------------------------------------------===//

#include "ClangTidyModule.h"
#include "ClangTidyCheck.h"

namespace clang {
namespace tidy {

void ClangTidyCheckFactories::registerCheckFactory(StringRef Name,
                                                   CheckFactory Factory) {
  Factories.insert_or_assign(Name, std::move(Factory));
}

std::vector<std::unique_ptr<ClangTidyCheck>>
ClangTidyCheckFactories::createChecks(ClangTidyContext *Context) {
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  for (const auto &Factory : Factories) {
    if (Context->isCheckEnabled(Factory.getKey()))
      Checks.emplace_back(Factory.getValue()(Factory.getKey(), Context));
  }
  return Checks;
}

std::vector<std::unique_ptr<ClangTidyCheck>>
ClangTidyCheckFactories::createChecksForLanguage(ClangTidyContext *Context) {
  std::vector<std::unique_ptr<ClangTidyCheck>> Checks;
  const LangOptions &LO = Context->getLangOpts();
  for (const auto &Factory : Factories) {
    if (!Context->isCheckEnabled(Factory.getKey()))
      continue;
    std::unique_ptr<ClangTidyCheck> Check =
        Factory.getValue()(Factory.getKey(), Context);
    if (Check->isLanguageVersionSupported(LO))
      Checks.push_back(std::move(Check));
  }
  return Checks;
}

ClangTidyOptions ClangTidyModule::getModuleOptions() {
  return ClangTidyOptions();
}

} // namespace tidy
} // namespace clang
