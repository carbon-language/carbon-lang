// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "migrate_cpp/cpp_refactoring/fn_inserter.h"

#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace cam = ::clang::ast_matchers;
namespace ct = ::clang::tooling;

namespace Carbon {
namespace {

TEST(FnInserterTest, StringRep) {
  std::map<std::string, ct::Replacements> replacements;

  cam::MatchFinder finder;
  Carbon::FnInserter fn_inserter(replacements, &finder);

  auto factory = clang::tooling::newFrontendActionFactory(&finder);

  ASSERT_TRUE(ct::runToolOnCodeWithArgs(
      factory->create(), "auto A() -> int;", {}, "test.cc", "clang-tool",
      std::make_shared<clang::PCHContainerOperations>(),
      ct::FileContentMappings()));
}

}  // namespace
}  // namespace Carbon
