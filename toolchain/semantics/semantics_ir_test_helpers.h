// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest-matchers.h>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/semantics/semantics_ir.h"

namespace Carbon::Testing {

// A singleton for g_semanticsir, used by the test helpers.
//
// This is done this way so that calls like PrintTo(Semantics::Declaration) have
// a SemanticsIR to refer back to; PrintTo must be static.
class SemanticsIRForTest {
 public:
  template <typename NodeT>
  static auto GetDeclaration(Semantics::Declaration decl)
      -> llvm::Optional<NodeT> {
    CARBON_CHECK(g_semantics != llvm::None);
    if (decl.kind() != NodeT::MetaNodeKind) {
      return llvm::None;
    }
    return g_semantics->declarations_.Get<NodeT>(decl);
  }

  template <typename NodeT>
  static auto GetExpression(Semantics::Expression expr)
      -> llvm::Optional<NodeT> {
    CARBON_CHECK(g_semantics != llvm::None);
    if (expr.kind() != NodeT::MetaNodeKind) {
      return llvm::None;
    }
    return g_semantics->expressions_.Get<NodeT>(expr);
  }

  template <typename NodeT>
  static auto GetStatement(Semantics::Statement expr) -> llvm::Optional<NodeT> {
    CARBON_CHECK(g_semantics != llvm::None);
    if (expr.kind() != NodeT::MetaNodeKind) {
      return llvm::None;
    }
    return g_semantics->statements_.Get<NodeT>(expr);
  }

  static auto GetNodeText(ParseTree::Node node) -> llvm::StringRef {
    CARBON_CHECK(g_semantics != llvm::None);
    return g_semantics->parse_tree_->GetNodeText(node);
  }

  static void Print(llvm::raw_ostream& out, Semantics::Declaration decl) {
    CARBON_CHECK(g_semantics != llvm::None);
    g_semantics->Print(out, decl);
  }

  static void Print(llvm::raw_ostream& out, Semantics::Expression expr) {
    CARBON_CHECK(g_semantics != llvm::None);
    out << "TODO(expr)";
    // g_semantics->Print(out, expr);
  }

  static void Print(llvm::raw_ostream& out, Semantics::Statement stmt) {
    CARBON_CHECK(g_semantics != llvm::None);
    out << "TODO(stmt)";
    // g_semantics->Print(out, stmt);
  }

  static auto semantics() -> const SemanticsIR& {
    CARBON_CHECK(g_semantics != llvm::None);
    return *g_semantics;
  }

  static void set_semantics(SemanticsIR semantics) {
    CARBON_CHECK(g_semantics == llvm::None)
        << "Call clear() before setting again.";
    g_semantics = std::move(semantics);
  }

  static void clear() { g_semantics = llvm::None; }

 private:
  static llvm::Optional<SemanticsIR> g_semantics;
};

inline auto MappedNode(::testing::Matcher<std::string> key,
                       ::testing::Matcher<Semantics::Declaration> value)
    -> ::testing::Matcher<llvm::StringMapEntry<Semantics::Declaration>> {
  return ::testing::AllOf(
      ::testing::Property(
          "key", &llvm::StringMapEntry<Semantics::Declaration>::getKey, key),
      ::testing::Property(
          "value", &llvm::StringMapEntry<Semantics::Declaration>::getValue,
          value));
}

// Avoids gtest confusion of how to print llvm::None.
MATCHER(IsNone, "is llvm::None") { return arg == llvm::None; }

MATCHER_P(DeclaredName, name_matcher,
          llvm::formatv("DeclaredName {0}",
                        ::testing::PrintToString(name_matcher))) {
  const Semantics::DeclaredName& name = arg;
  return ExplainMatchResult(name_matcher,
                            SemanticsIRForTest::GetNodeText(name.node()),
                            result_listener);
}

MATCHER_P3(InfixOperator, lhs_matcher, op_matcher, rhs_matcher,
           llvm::formatv("InfixOperator {0} {1} {2}",
                         ::testing::PrintToString(lhs_matcher),
                         ::testing::PrintToString(op_matcher),
                         ::testing::PrintToString(rhs_matcher))) {
  const Semantics::Expression& expr = arg;
  if (auto infix =
          SemanticsIRForTest::GetExpression<Semantics::InfixOperator>(expr)) {
    return ExplainMatchResult(op_matcher,
                              SemanticsIRForTest::GetNodeText(infix->node()),
                              result_listener) &&
           ExplainMatchResult(lhs_matcher, infix->lhs(), result_listener) &&
           ExplainMatchResult(rhs_matcher, infix->rhs(), result_listener);
  } else {
    *result_listener << "node is not a literal";
    return result_listener;
  }
}

MATCHER_P(Literal, text_matcher,
          llvm::formatv("Literal {0}",
                        ::testing::PrintToString(text_matcher))) {
  const Semantics::Expression& expr = arg;
  if (auto lit = SemanticsIRForTest::GetExpression<Semantics::Literal>(expr)) {
    return ExplainMatchResult(text_matcher,
                              SemanticsIRForTest::GetNodeText(lit->node()),
                              result_listener);
  } else {
    *result_listener << "node is not a literal";
    return result_listener;
  }
}

MATCHER_P(FunctionName, name_matcher,
          llvm::formatv("Function named {0}",
                        ::testing::PrintToString(name_matcher))) {
  const Semantics::Declaration& decl = arg;
  if (auto function =
          SemanticsIRForTest::GetDeclaration<Semantics::Function>(decl)) {
    return ExplainMatchResult(
        name_matcher, SemanticsIRForTest::GetNodeText(function->name().node()),
        result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

MATCHER_P(FunctionParams, param_matcher,
          llvm::formatv("Function parameters {0}",
                        ::testing::PrintToString(param_matcher))) {
  const Semantics::Declaration& decl = arg;
  if (auto function =
          SemanticsIRForTest::GetDeclaration<Semantics::Function>(decl)) {
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
  const Semantics::Declaration& decl = arg;
  if (auto function =
          SemanticsIRForTest::GetDeclaration<Semantics::Function>(decl)) {
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
  const Semantics::Declaration& decl = arg;
  if (auto function =
          SemanticsIRForTest::GetDeclaration<Semantics::Function>(decl)) {
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
        return_type_matcher,
    ::testing::Matcher<Semantics::StatementBlock> body_matcher)
    -> ::testing::Matcher<Semantics::Declaration> {
  return ::testing::AllOf(
      FunctionName(name_matcher), FunctionParams(params_matcher),
      FunctionReturnExpr(return_type_matcher), FunctionBody(body_matcher));
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

MATCHER_P(Return, expr_matcher,
          llvm::formatv("Return {0}", ::testing::PrintToString(expr_matcher))) {
  const Semantics::Statement& stmt = arg;
  if (auto ret = SemanticsIRForTest::GetStatement<Semantics::Return>(stmt)) {
    return ExplainMatchResult(expr_matcher, ret->expression(), result_listener);
  } else {
    *result_listener << "node is not a function";
    return result_listener;
  }
}

inline auto StatementBlock(
    ::testing::Matcher<llvm::ArrayRef<Semantics::Statement>> nodes_matcher,
    ::testing::Matcher<llvm::StringMap<Semantics::Statement>>
        name_lookup_matcher) -> ::testing::Matcher<Semantics::StatementBlock> {
  return ::testing::AllOf(
      ::testing::Property("nodes", &Semantics::StatementBlock::nodes,
                          nodes_matcher),
      ::testing::Property("name_lookup",
                          &Semantics::StatementBlock::name_lookup,
                          name_lookup_matcher));
}

}  // namespace Carbon::Testing

namespace Carbon::Semantics {

// Prints a Declaration for gmock.
inline void PrintTo(const Declaration& decl, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  Testing::SemanticsIRForTest::Print(wrapped_out, decl);
}

// Prints a DeclarationBlock for gmock.
inline void PrintTo(const DeclarationBlock& block, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  wrapped_out << "DeclarationBlock{";
  llvm::ListSeparator sep;
  for (const auto& entry : block.name_lookup()) {
    wrapped_out << llvm::StringRef(sep) << "`" << entry.getKey() << "`: `";
    Testing::SemanticsIRForTest::Print(wrapped_out, entry.getValue());
    wrapped_out << "`";
  }
  wrapped_out << "}";
}

// Prints an Expression for gmock.
inline void PrintTo(const Expression& expr, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  Testing::SemanticsIRForTest::Print(wrapped_out, expr);
}

// Prints a Statement for gmock.
inline void PrintTo(const Statement& decl, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  Testing::SemanticsIRForTest::Print(wrapped_out, decl);
}

// Prints a StatementBlock for gmock.
inline void PrintTo(const StatementBlock& block, std::ostream* out) {
  llvm::raw_os_ostream wrapped_out(*out);
  wrapped_out << "StatementBlock{";
  llvm::ListSeparator sep;
  for (const auto& node : block.nodes()) {
    wrapped_out << "`";
    Testing::SemanticsIRForTest::Print(wrapped_out, node);
    wrapped_out << "`";
  }
  wrapped_out << "}";
}

}  // namespace Carbon::Semantics

namespace llvm {

// Prints a StringMapEntry for gmock.
inline void PrintTo(
    const llvm::StringMapEntry<Carbon::Semantics::Declaration>& entry,
    std::ostream* out) {
  *out << "StringMapEntry(" << entry.getKey() << ", ";
  Carbon::Semantics::PrintTo(entry.getValue(), out);
  *out << ")";
}

// Prints a StringMapEntry for gmock.
inline void PrintTo(
    const llvm::StringMapEntry<Carbon::Semantics::Statement>& entry,
    std::ostream* out) {
  *out << "StringMapEntry(" << entry.getKey() << ", ";
  Carbon::Semantics::PrintTo(entry.getValue(), out);
  *out << ")";
}

}  // namespace llvm

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
