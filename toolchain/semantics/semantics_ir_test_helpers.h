// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest-matchers.h>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/nodes/declared_name.h"
#include "toolchain/semantics/nodes/expression.h"
#include "toolchain/semantics/nodes/pattern_binding.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::Testing {

// A singleton for semantics_ir, used by the helpers.
//
// This is done this way mainly so that PrintTo(SemanticsIR::Node) has a
// SemanticsIR to refer back to; PrintTo must be static.
class SemanticsIRSingleton {
 public:
  static auto GetFunction(SemanticsIR::Node node)
      -> llvm::Optional<Semantics::Function> {
    CARBON_CHECK(semantics_ != llvm::None);
    if (node.kind_ != SemanticsIR::Node::Kind::Function) {
      return llvm::None;
    }
    return semantics_->functions_[node.index_];
  }

  static auto GetNodeText(ParseTree::Node node) -> llvm::StringRef {
    CARBON_CHECK(semantics_ != llvm::None);
    return semantics_->parse_tree_->GetNodeText(node);
  }

  static void Print(llvm::raw_ostream& out, SemanticsIR::Node node) {
    CARBON_CHECK(semantics_ != llvm::None);
    semantics_->Print(out, node);
  }

  static auto semantics() -> const SemanticsIR& {
    CARBON_CHECK(semantics_ != llvm::None);
    return *semantics_;
  }

  static void set_semantics(SemanticsIR semantics) {
    CARBON_CHECK(semantics_ == llvm::None)
        << "Call clear() before setting again.";
    semantics_ = std::move(semantics);
  }

  static void clear() { semantics_ = llvm::None; }

 private:
  static llvm::Optional<SemanticsIR> semantics_;
};

inline auto MappedNode(::testing::Matcher<std::string> key,
                       ::testing::Matcher<SemanticsIR::Node> value)
    -> ::testing::Matcher<llvm::StringMapEntry<SemanticsIR::Node>> {
  return ::testing::AllOf(
      ::testing::Property(
          "key", &llvm::StringMapEntry<SemanticsIR::Node>::getKey, key),
      ::testing::Property(
          "value", &llvm::StringMapEntry<SemanticsIR::Node>::getValue, value));
}

MATCHER_P(DeclaredName, name_matcher,
          llvm::formatv("DeclaredName {0}",
                        ::testing::PrintToString(name_matcher))) {
  const Semantics::DeclaredName& node = arg;
  return ExplainMatchResult(name_matcher,
                            SemanticsIRSingleton::GetNodeText(node.node()),
                            result_listener);
}

MATCHER_P(ExpressionLiteral, text_matcher,
          llvm::formatv("Expression literal {0}",
                        ::testing::PrintToString(text_matcher))) {
  const Semantics::Expression& expr = arg;
  return ExplainMatchResult(
      text_matcher, SemanticsIRSingleton::GetNodeText(expr.literal().node()),
      result_listener);
}

MATCHER_P(FunctionName, name_matcher,
          llvm::formatv("Function named {0}",
                        ::testing::PrintToString(name_matcher))) {
  const SemanticsIR::Node& node = arg;
  if (auto function = SemanticsIRSingleton::GetFunction(node)) {
    return ExplainMatchResult(
        name_matcher,
        SemanticsIRSingleton::GetNodeText(function->name().node()),
        result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

MATCHER_P(FunctionParams, param_matcher,
          llvm::formatv("Function parameters {0}",
                        ::testing::PrintToString(param_matcher))) {
  const SemanticsIR::Node& node = arg;
  if (auto function = SemanticsIRSingleton::GetFunction(node)) {
    return ExplainMatchResult(param_matcher, function->params(),
                              result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

MATCHER_P(FunctionReturnExpr, expression_matcher,
          llvm::formatv("Function return expr {0}",
                        ::testing::PrintToString(expression_matcher))) {
  const SemanticsIR::Node& node = arg;
  if (auto function = SemanticsIRSingleton::GetFunction(node)) {
    return ExplainMatchResult(expression_matcher, function->return_expr(),
                              result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

MATCHER_P(FunctionBody, body_matcher,
          llvm::formatv("Function body {0}",
                        ::testing::PrintToString(body_matcher))) {
  const SemanticsIR::Node& node = arg;
  if (auto function = SemanticsIRSingleton::GetFunction(node)) {
    return ExplainMatchResult(body_matcher, function->body(), result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

inline auto Function(
    ::testing::Matcher<llvm::StringRef> name_matcher,
    ::testing::Matcher<llvm::ArrayRef<Semantics::PatternBinding>>
        params_matcher,
    ::testing::Matcher<llvm::Optional<Semantics::Expression>>
        return_type_matcher) -> ::testing::Matcher<SemanticsIR::Node> {
  return ::testing::AllOf(FunctionName(name_matcher),
                          FunctionParams(params_matcher),
                          FunctionReturnExpr(return_type_matcher));
}

inline auto PatternBinding(
    ::testing::Matcher<llvm::StringRef> name_matcher,
    ::testing::Matcher<Semantics::Expression> type_matcher)
    -> ::testing::Matcher<Semantics::PatternBinding> {
  return ::testing::AllOf(
      ::testing::Property("name", &Semantics::PatternBinding::name,
                          DeclaredName(name_matcher)),
      ::testing::Property("type", &Semantics::PatternBinding::type,
                          type_matcher));
}

}  // namespace Carbon::Testing

namespace Carbon {

// Prints a Node for gmock. Needs to be in the matching namespace.
inline void PrintTo(const SemanticsIR::Node& node, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  Testing::SemanticsIRSingleton::Print(wrapped_out, node);
}

// Prints a Block for gmock. Needs to be in the matching namespace.
inline void PrintTo(const SemanticsIR::Block& block, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  wrapped_out << "Block{";
  llvm::ListSeparator sep;
  for (const auto& entry : block.name_lookup()) {
    wrapped_out << llvm::StringRef(sep) << "`" << entry.getKey() << "`: `";
    Testing::SemanticsIRSingleton::Print(wrapped_out, entry.getValue());
    wrapped_out << "`";
  }
  wrapped_out << "}";
}

}  // namespace Carbon

namespace llvm {

// Prints a StringMapEntry for gmock. Needs to be in the matching namespace.
inline void PrintTo(
    const llvm::StringMapEntry<Carbon::SemanticsIR::Node>& entry,
    std::ostream* out) {
  *out << "StringMapEntry(" << entry.getKey() << ", ";
  Carbon::PrintTo(entry.getValue(), out);
  *out << ")";
}

}  // namespace llvm

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
