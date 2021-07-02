// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_FN_INSERTER_H_
#define MIGRATE_CPP_CPP_REFACTORING_FN_INSERTER_H_

#include "migrate_cpp/cpp_refactoring/matcher.h"

namespace Carbon {

// Inserts `fn` for functions and methods.
class FnInserter : public Matcher {
 public:
  using Matcher::Matcher;
  void Run() override;
};

class FnInserterFactory : public MatcherFactoryBase<FnInserter> {
  virtual auto GetAstMatcher() -> clang::ast_matchers::DeclarationMatcher;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_FN_INSERTER_H_
