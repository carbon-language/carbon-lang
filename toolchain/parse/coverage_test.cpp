// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "absl/flags/flag.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/testing/coverage_helper.h"

ABSL_FLAG(std::string, testdata_manifest, "",
          "A path to a file containing repo-relative names of test files.");

namespace Carbon::Parse {
namespace {

constexpr NodeKind NodeKinds[] = {
#define CARBON_PARSE_NODE_KIND(Name) NodeKind::Name,
#include "toolchain/parse/node_kind.def"
};

constexpr NodeKind UntestedNodeKinds[] = {NodeKind::Placeholder};

// Looks for node kinds that aren't covered by a file_test.
TEST(Coverage, NodeKind) {
  Testing::TestKindCoverage(absl::GetFlag(FLAGS_testdata_manifest),
                            R"(kind: '(\w+)')", llvm::ArrayRef(NodeKinds),
                            llvm::ArrayRef(UntestedNodeKinds));
}

}  // namespace
}  // namespace Carbon::Parse
