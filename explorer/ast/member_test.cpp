// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/member.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

#include "explorer/ast/bindings.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/paren_contents.h"
#include "explorer/common/arena.h"
#include "llvm/Support/Casting.h"

// Fake Value classes to avoid depending on /explorer/interpreter
namespace Carbon {
class Value {};
class TestValue : public Value {
 public:
  TestValue(int i) : i_(i) {}
  auto value() const -> int { return i_; }

 private:
  int i_;
};
}  // namespace Carbon

namespace Carbon::Testing {
namespace {

static auto FakeSourceLoc(int line_num) -> SourceLocation {
  return SourceLocation("<test>", line_num);
}

class MemberTest : public ::testing::Test {
 protected:
  Arena arena;
};

TEST_F(MemberTest, NominalMemberType) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  const auto static_type = arena.New<TestValue>(1);
  decl.set_static_type(static_type);
  NominalMember member_decl(&decl);

  EXPECT_EQ(&member_decl.type(), static_type);

  NominalMember member_val(
      arena.New<NamedValue>(NamedValue{"valuename", static_type}));
  EXPECT_EQ(&member_val.type(), static_type);
}

TEST_F(MemberTest, NominalMemberDeclaration) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  const auto static_type = arena.New<TestValue>(1);
  NominalMember member_decl(&decl);

  EXPECT_TRUE(member_decl.declaration());

  NominalMember member_val(
      arena.New<NamedValue>(NamedValue{"valuename", static_type}));
  EXPECT_FALSE(member_val.declaration());
}

TEST_F(MemberTest, NominalMemberIsNamed) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  NominalMember member_decl(&decl);
  EXPECT_TRUE(member_decl.IsNamed("valuename"));
  EXPECT_FALSE(member_decl.IsNamed("anything"));

  NominalMember member_val(
      arena.New<NamedValue>(NamedValue{"valuename", arena.New<TestValue>(1)}));
  EXPECT_TRUE(member_val.IsNamed("valuename"));
  EXPECT_FALSE(member_val.IsNamed("anything"));
}

TEST_F(MemberTest, PositionalMemberIsNamed) {
  PositionalMember member(1, arena.New<TestValue>(1));
  EXPECT_FALSE(member.IsNamed("anything"));
}

TEST_F(MemberTest, BaseClassIsNamed) {
  BaseClass member(arena.New<TestValue>(1));
  EXPECT_FALSE(member.IsNamed("anything"));
}
}  // namespace
}  // namespace Carbon::Testing
