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
  replacements.clear();
  replacements.insert({Filename, {}});
  ASSERT_TRUE(ct::runToolOnCodeWithArgs(
      factory->create(), before, {}, Filename, "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      ct::FileContentMappings()));
  EXPECT_THAT(replacements, testing::ElementsAre(testing::Key(Filename)));
  auto actual = ct::applyAllReplacements(before, replacements[Filename]);
  if (after.find('\n') == std::string::npos) {
    EXPECT_THAT(*actual, testing::Eq(after.str()));
  } else {
    // Split lines to get gmock to get an easier-to-read error.
    llvm::SmallVector<llvm::StringRef, 0> actual_lines;
    llvm::SplitString(*actual, actual_lines, "\n");
    llvm::SmallVector<llvm::StringRef, 0> after_lines;
    llvm::SplitString(after, after_lines, "\n");
    EXPECT_THAT(actual_lines, testing::ContainerEq(after_lines));
  }
}

}  // namespace Carbon
