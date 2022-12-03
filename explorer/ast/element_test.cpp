// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/ast/element.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>

#include "explorer/ast/bindings.h"
#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/common/arena.h"
#include "explorer/interpreter/value.h"
#include "llvm/Support/Casting.h"

namespace Carbon::Testing {
namespace {

static auto FakeSourceLoc(int line_num) -> SourceLocation {
  return SourceLocation("<test>", line_num);
}

class ElementTest : public ::testing::Test {
 protected:
  Arena arena;
};

TEST_F(ElementTest, NominalElementType) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  const auto* static_type = arena.New<IntValue>(1);
  decl.set_static_type(static_type);
  MemberElement element_decl(&decl);

  EXPECT_EQ(&element_decl.type(), static_type);

  MemberElement member_val(
      arena.New<NamedValue>(NamedValue{"valuename", static_type}));
  EXPECT_EQ(&member_val.type(), static_type);
}

TEST_F(ElementTest, NominalElementDeclaration) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  const auto* static_type = arena.New<IntValue>(1);
  MemberElement element_decl(&decl);

  EXPECT_TRUE(element_decl.declaration());

  MemberElement member_val(
      arena.New<NamedValue>(NamedValue{"valuename", static_type}));
  EXPECT_FALSE(member_val.declaration());
}

TEST_F(ElementTest, NominalElementIsNamed) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  MemberElement member_decl(&decl);
  EXPECT_TRUE(member_decl.IsNamed("valuename"));
  EXPECT_FALSE(member_decl.IsNamed("anything"));

  MemberElement member_val(
      arena.New<NamedValue>(NamedValue{"valuename", arena.New<IntValue>(1)}));
  EXPECT_TRUE(member_val.IsNamed("valuename"));
  EXPECT_FALSE(member_val.IsNamed("anything"));
}

TEST_F(ElementTest, PositionalElementIsNamed) {
  TupleElement member(1, arena.New<IntValue>(1));
  EXPECT_FALSE(member.IsNamed("anything"));
}

TEST_F(ElementTest, BaseClassElementIsNamed) {
  BaseElement member(arena.New<IntValue>(1));
  EXPECT_FALSE(member.IsNamed("anything"));
}
}  // namespace
}  // namespace Carbon::Testing
