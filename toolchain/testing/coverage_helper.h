// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_TESTING_COVERAGE_HELPER_H_
#define CARBON_TOOLCHAIN_TESTING_COVERAGE_HELPER_H_

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "common/set.h"
#include "llvm/ADT/StringExtras.h"
#include "re2/re2.h"

namespace Carbon::Testing {

// Looks for kinds that aren't covered by a file_test in the manifest path.
// Kinds are identified by the provided regular expression kind_pattern.
//
// should_be_covered should return false when a kind is either untestable or not
// yet tested.
//
// TODO: Switch `kind_pattern` to `llvm::StringLiteral` if
// `llvm::StringLiteral::c_str` is added.
template <typename KindT>
auto TestKindCoverage(const std::string& manifest_path,
                      const char* kind_pattern, llvm::ArrayRef<KindT> kinds,
                      llvm::ArrayRef<KindT> untested_kinds) {
  std::ifstream manifest_in(manifest_path.c_str());
  ASSERT_TRUE(manifest_in.good());

  RE2 kind_re(kind_pattern);
  ASSERT_TRUE(kind_re.ok()) << kind_re.error();

  Set<std::string> covered_kinds;

  std::string test_filename;
  while (std::getline(manifest_in, test_filename)) {
    std::ifstream test_in(test_filename);
    ASSERT_TRUE(test_in.good());

    std::string line;
    while (std::getline(test_in, line)) {
      std::string kind;
      if (RE2::PartialMatch(line, kind_re, &kind)) {
        covered_kinds.Insert(kind);
      }
    }
  }

  llvm::SmallVector<llvm::StringRef> missing_kinds;
  for (auto kind : kinds) {
    if (llvm::find(untested_kinds, kind) != untested_kinds.end()) {
      EXPECT_FALSE(covered_kinds.Erase(kind.name()))
          << "Kind " << kind
          << " has coverage even though none was expected. If this has "
             "changed, update the coverage test.";
      continue;
    }
    if (!covered_kinds.Erase(kind.name())) {
      missing_kinds.push_back(kind.name());
    }
  }

  constexpr llvm::StringLiteral Bullet = "\n  - ";

  llvm::sort(missing_kinds);
  EXPECT_TRUE(missing_kinds.empty()) << "Some kinds have no tests:" << Bullet
                                     << llvm::join(missing_kinds, Bullet);

  llvm::SmallVector<std::string> unexpected_matches;
  covered_kinds.ForEach(
      [&](const std::string& match) { unexpected_matches.push_back(match); });
  llvm::sort(unexpected_matches);
  EXPECT_TRUE(unexpected_matches.empty())
      << "Matched things that aren't in the kind list:" << Bullet
      << llvm::join(unexpected_matches, Bullet);
}

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_TESTING_COVERAGE_HELPER_H_
