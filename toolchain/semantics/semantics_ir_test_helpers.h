// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/ostream.h"
#include "toolchain/semantics/semantics_ir_for_test.h"

namespace Carbon::Testing {

template <typename NodeType>
class NodeRefMatcher {
 public:
  using is_gtest_matcher = void;

  NodeRefMatcher(llvm::StringLiteral name,
                 testing::Matcher<const NodeType&> matcher)
      : name_(name), matcher_(matcher) {}

  // Returns true if and only if the matcher matches x; also explains the match
  // result to 'listener'.
  auto MatchAndExplain(const Semantics::NodeRef& node_ref,
                       testing::MatchResultListener* listener) const -> bool {
    if (auto node = SemanticsIRForTest::GetNode<NodeType>(node_ref)) {
      return testing::ExplainMatchResult(matcher_, *node, listener);
    } else {
      *listener << "node is not a " << name_;
      return false;
    }
  }

  // Describes this matcher to an ostream.
  auto DescribeTo(std::ostream* os) const -> void {
    *os << name_.str() << "("
        << ::testing::DescribeMatcher<const NodeType&>(matcher_) << ")";
  }

  // Describes this matcher to an ostream.
  auto DescribeNegationTo(std::ostream* os) const -> void {
    *os << "not ";
    DescribeTo(os);
  }

 private:
  llvm::StringLiteral name_;
  testing::Matcher<const NodeType&> matcher_;
};

auto BinaryOperator(testing::Matcher<int32_t> id_matcher,
                    testing::Matcher<Semantics::BinaryOperator::Op> op_matcher,
                    testing::Matcher<int32_t> lhs_id_matcher,
                    testing::Matcher<int32_t> rhs_id_matcher)
    -> ::testing::Matcher<const Semantics::NodeRef&> {
  using Semantics::BinaryOperator;
  using ::testing::Property;
  return NodeRefMatcher<BinaryOperator>(
      "BinaryOperator",
      testing::AllOf(
          Property("id", &BinaryOperator::id, id_matcher),
          Property("op", &BinaryOperator::op, op_matcher),
          Property("lhs_id", &BinaryOperator::lhs_id, lhs_id_matcher),
          Property("rhs_id", &BinaryOperator::rhs_id, rhs_id_matcher)));
}

auto Function(
    ::testing::Matcher<int32_t> id_matcher,
    ::testing::Matcher<llvm::ArrayRef<Semantics::NodeRef>> body_matcher)
    -> ::testing::Matcher<const Semantics::NodeRef&> {
  using Semantics::Function;
  using testing::Property;
  return NodeRefMatcher<Function>(
      "Function",
      testing::AllOf(Property("id", &Function::id, id_matcher),
                     Property("body", &Function::body, body_matcher)));
}

auto IntegerLiteral(::testing::Matcher<int32_t> id_matcher,
                    ::testing::Matcher<llvm::APInt> value_matcher)
    -> ::testing::Matcher<const Semantics::NodeRef&> {
  using Semantics::IntegerLiteral;
  using testing::Property;
  return NodeRefMatcher<IntegerLiteral>(
      "IntegerLiteral",
      testing::AllOf(Property("id", &IntegerLiteral::id, id_matcher),
                     Property("value", &IntegerLiteral::value, value_matcher)));
}

auto IntegerLiteral(::testing::Matcher<int32_t> id_matcher, int val)
    -> ::testing::Matcher<const Semantics::NodeRef&> {
  return IntegerLiteral(id_matcher, testing::Eq(val));
}

auto Return(
    ::testing::Matcher<llvm::Optional<Semantics::NodeId>> target_id_matcher)
    -> ::testing::Matcher<const Semantics::NodeRef&> {
  using Semantics::Return;
  using testing::Property;
  return NodeRefMatcher<Return>(
      "Return",
      testing::Property("target_id", &Return::target_id, target_id_matcher));
}

auto SetName(::testing::Matcher<llvm::StringRef> name_matcher,
             ::testing::Matcher<int32_t> target_id_matcher)
    -> ::testing::Matcher<const Semantics::NodeRef&> {
  using Semantics::SetName;
  using testing::Property;
  return NodeRefMatcher<SetName>(
      "SetName", testing::AllOf(Property("name", &SetName::name, name_matcher),
                                Property("target_id", &SetName::target_id,
                                         target_id_matcher)));
}

// Avoids gtest confusion of how to print llvm::None.
MATCHER(IsNone, "is llvm::None") { return arg == llvm::None; }

}  // namespace Carbon::Testing

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_TEST_HELPERS_H_
