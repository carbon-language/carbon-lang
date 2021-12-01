// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MIGRATE_CPP_CPP_REFACTORING_MATCHER_TEST_BASE_H_
#define MIGRATE_CPP_CPP_REFACTORING_MATCHER_TEST_BASE_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Tooling.h"
#include "migrate_cpp/cpp_refactoring/matcher_manager.h"

namespace Carbon {

// Matcher test framework.
template <typename MatcherFactoryType>
class MatcherTestBase : public ::testing::Test {
 protected:
  MatcherTestBase() : matchers(&replacements) {
    matchers.Register(std::make_unique<MatcherFactoryType>());
  }

  // Expects that that the replacements produced by running the finder result in
  // the specified code transformation.
  void ExpectReplacement(llvm::StringRef before, llvm::StringRef after) {
    auto factory =
        clang::tooling::newFrontendActionFactory(matchers.GetFinder());
    constexpr char Filename[] = "test.cc";
    replacements.clear();
    replacements.insert({Filename, {}});
    ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
        factory->create(), before, {}, Filename, "clang-tool",
        std::make_shared<clang::PCHContainerOperations>(),
        clang::tooling::FileContentMappings()));
    EXPECT_THAT(replacements, testing::ElementsAre(testing::Key(Filename)));
    llvm::Expected<std::string> actual =
        clang::tooling::applyAllReplacements(before, replacements[Filename]);

    // Make a specific note if the matcher didn't make any changes.
    std::string unchanged;
    if (before == *actual) {
      unchanged = "NOTE: Actual matches original text, no changes made.";
    }

    if (after.find('\n') == std::string::npos) {
      EXPECT_THAT(*actual, testing::Eq(after.str())) << unchanged;
    } else {
      // Split lines to get gmock to get an easier-to-read error.
      llvm::SmallVector<llvm::StringRef, 0> actual_lines;
      llvm::SplitString(*actual, actual_lines, "\n");
      llvm::SmallVector<llvm::StringRef, 0> after_lines;
      llvm::SplitString(after, after_lines, "\n");
      EXPECT_THAT(actual_lines, testing::ContainerEq(after_lines)) << unchanged;
    }
  }

  Matcher::ReplacementMap replacements;
  MatcherManager matchers;
};

}  // namespace Carbon

#endif  // MIGRATE_CPP_CPP_REFACTORING_MATCHER_TEST_BASE_H_
