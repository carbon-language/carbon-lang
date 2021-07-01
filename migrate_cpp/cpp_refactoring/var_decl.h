// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_VAR_DECL_H_
#define MIGRATE_CPP_CPP_REFACTORING_VAR_DECL_H_

#include "migrate_cpp/cpp_refactoring/matcher.h"

namespace Carbon {

// Updates variable declarations for `var name: Type`.
class VarDecl : public Matcher {
 public:
  static clang::ast_matchers::DeclarationMatcher GetAstMatcher();
  using Matcher::Matcher;
  void Run() override;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_VAR_DECL_H_
