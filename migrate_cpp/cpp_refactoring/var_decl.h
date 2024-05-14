// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_MIGRATE_CPP_CPP_REFACTORING_VAR_DECL_H_
#define CARBON_MIGRATE_CPP_CPP_REFACTORING_VAR_DECL_H_

#include "migrate_cpp/cpp_refactoring/matcher.h"

namespace Carbon {

// Updates variable declarations for `var name: type`.
class VarDecl : public Matcher {
 public:
  using Matcher::Matcher;
  void Run() override;

 private:
  auto GetTypeStr(const clang::VarDecl& decl) -> std::string;
};

class VarDeclFactory : public MatcherFactoryBase<VarDecl> {
 public:
  void AddMatcher(
      clang::ast_matchers::MatchFinder* finder,
      clang::ast_matchers::MatchFinder::MatchCallback* callback) override;
};

}  // namespace Carbon

#endif  // CARBON_MIGRATE_CPP_CPP_REFACTORING_VAR_DECL_H_
