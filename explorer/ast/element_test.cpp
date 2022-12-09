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

TEST_F(ElementTest, NamedElementType) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  const auto* static_type = arena.New<IntValue>(1);
  decl.set_static_type(static_type);
  NamedElement element_decl(&decl);

  EXPECT_EQ(&element_decl.type(), static_type);

  NamedElement named_val(
      arena.New<NamedValue>(NamedValue{"valuename", static_type}));
  EXPECT_EQ(&named_val.type(), static_type);
}

TEST_F(ElementTest, NamedElementDeclaration) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  const auto* static_type = arena.New<IntValue>(1);
  NamedElement element_decl(&decl);

  EXPECT_TRUE(element_decl.declaration());

  NamedElement named_val(
      arena.New<NamedValue>(NamedValue{"valuename", static_type}));
  EXPECT_FALSE(named_val.declaration());
}

TEST_F(ElementTest, NamedElementIsNamed) {
  const auto src_loc = FakeSourceLoc(1);
  VariableDeclaration decl{
      src_loc,
      arena.New<BindingPattern>(src_loc, "valuename",
                                arena.New<AutoPattern>(src_loc),
                                ValueCategory::Var),
      std::nullopt, ValueCategory::Var};
  NamedElement member_decl(&decl);
  EXPECT_TRUE(member_decl.IsNamed("valuename"));
  EXPECT_FALSE(member_decl.IsNamed("anything"));

  NamedElement named_val(
      arena.New<NamedValue>(NamedValue{"valuename", arena.New<IntValue>(1)}));
  EXPECT_TRUE(named_val.IsNamed("valuename"));
  EXPECT_FALSE(named_val.IsNamed("anything"));
}

TEST_F(ElementTest, PositionalElementIsNamed) {
  PositionalElement element(1, arena.New<IntValue>(1));
  EXPECT_FALSE(element.IsNamed("anything"));
}

TEST_F(ElementTest, BaseElementIsNamed) {
  BaseElement element(arena.New<IntValue>(1));
  EXPECT_FALSE(element.IsNamed("anything"));
}
}  // namespace
}  // namespace Carbon::Testing
