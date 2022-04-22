// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest-matchers.h>

#include "common/check.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::Testing {

// A singleton for semantics_ir, used by the helpers.
//
// This is done this way mainly so that PrintTo(SemanticsIR::Node) has a
// SemanticsIR to refer back to; PrintTo must be static.
static llvm::Optional<SemanticsIR> g_semantics_ir;

inline auto MappedNode(::testing::Matcher<std::string> key,
                       ::testing::Matcher<SemanticsIR::Node> value)
    -> ::testing::Matcher<llvm::StringMapEntry<SemanticsIR::Node>> {
  return ::testing::AllOf(
      ::testing::Property(
          "key", &llvm::StringMapEntry<SemanticsIR::Node>::getKey, key),
      ::testing::Property(
          "value", &llvm::StringMapEntry<SemanticsIR::Node>::getValue, value));
}

inline auto Block(
    ::testing::Matcher<llvm::ArrayRef<SemanticsIR::Node>> nodes,
    ::testing::Matcher<llvm::StringMap<SemanticsIR::Node>> name_lookup)
    -> ::testing::Matcher<SemanticsIR::Block> {
  return ::testing::AllOf(
      ::testing::Property("nodes", &SemanticsIR::Block::nodes, nodes),
      ::testing::Property("nodes", &SemanticsIR::Block::name_lookup,
                          name_lookup));
}

MATCHER_P(FunctionName, name_matcher,
          llvm::formatv("Function named {0}",
                        ::testing::PrintToString(name_matcher))) {
  CHECK(g_semantics_ir != llvm::None);
  const SemanticsIR::Node& node = arg;
  llvm::Optional<Semantics::Function> function =
      g_semantics_ir->GetFunction(node);
  if (function == llvm::None) {
    *result_listener << "node is not a function";
    return result_listener;
  } else {
    return ExplainMatchResult(name_matcher, function->name(), result_listener);
  }
}

MATCHER_P(Function, body_matcher,
          llvm::formatv("Function body {0}",
                        ::testing::PrintToString(body_matcher))) {
  CHECK(g_semantics_ir != llvm::None);
  const SemanticsIR::Node& node = arg;
  llvm::Optional<Semantics::Function> function =
      g_semantics_ir->GetFunction(node);
  if (function == llvm::None) {
    *result_listener << "node is not a function";
    return result_listener;
  } else {
    return ExplainMatchResult(body_matcher, function->body(), result_listener);
  }
}

}  // namespace Carbon::Testing

namespace Carbon {

// Prints a node for gmock. Needs to be in the matching namespace.
inline void PrintTo(const SemanticsIR::Node& node, std::ostream* output) {
  llvm::Optional<Semantics::Function> function =
      Testing::g_semantics_ir->GetFunction(node);
  CHECK(function != llvm::None);
  *output << *function;
}

// Prints a node for gmock. Needs to be in the matching namespace.
inline void PrintTo(const SemanticsIR::Block& block, std::ostream* output) {
  *output << "Block{";
  llvm::ListSeparator sep;
  for (const auto& entry : block.name_lookup()) {
    *output << llvm::StringRef(sep) << entry.getKey();
  }
  *output << "}";
}

}  // namespace Carbon

namespace llvm {

// Prints a StringMapEntry for gmock. Needs to be in the matching namespace.
inline void PrintTo(
    const llvm::StringMapEntry<Carbon::SemanticsIR::Node>& entry,
    std::ostream* output) {
  *output << "StringMapEntry(" << entry.getKey() << ", ";
  Carbon::PrintTo(entry.getValue(), output);
  *output << ")";
}

}  // namespace llvm

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
