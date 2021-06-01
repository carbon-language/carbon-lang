// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/matcher_test_base.h"

#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace ct = ::clang::tooling;

namespace Carbon {

void MatcherTestBase::ExpectReplacement(llvm::StringRef before,
                                        llvm::StringRef after) {
  auto factory = ct::newFrontendActionFactory(&finder);
  constexpr char Filename[] = "test.cc";
  ASSERT_TRUE(ct::runToolOnCodeWithArgs(
      factory->create(), before, {}, Filename, "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      ct::FileContentMappings()));
  auto it = replacements.find(Filename);
  if (it != replacements.end()) {
    auto actual = ct::applyAllReplacements(before, it->second);
    // Split lines to get gmock to get an easier-to-read error.
    llvm::SmallVector<llvm::StringRef, 0> actual_lines;
    llvm::SplitString(*actual, actual_lines, "\n");
    llvm::SmallVector<llvm::StringRef, 0> after_lines;
    llvm::SplitString(after, after_lines, "\n");
    EXPECT_THAT(actual_lines, testing::ContainerEq(after_lines));
  } else {
    // No replacements; before and after should match.
    EXPECT_THAT(before, testing::Eq(after));
  }
}

}  // namespace Carbon
