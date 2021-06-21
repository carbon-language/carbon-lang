// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_MATCHER_TEST_BASE_H_
#define MIGRATE_CPP_CPP_REFACTORING_MATCHER_TEST_BASE_H_

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace Carbon {

class MatcherTestBase : public ::testing::Test {
 protected:
  // Expects that that the replacements produced by running the finder result in
  // the specified code transformation.
  void ExpectReplacement(llvm::StringRef before, llvm::StringRef after);

  std::map<std::string, clang::tooling::Replacements> replacements;
  clang::ast_matchers::MatchFinder finder;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_MATCHER_TEST_BASE_H_
