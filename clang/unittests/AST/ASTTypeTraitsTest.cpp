//===- unittest/AST/ASTTypeTraits.cpp - AST type traits unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//


#include "clang/AST/ASTTypeTraits.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_type_traits {

TEST(ASTNodeKind, NoKind) {
  EXPECT_FALSE(ASTNodeKind().isBaseOf(ASTNodeKind()));
  EXPECT_FALSE(ASTNodeKind().isSame(ASTNodeKind()));
}

template <typename T> static ASTNodeKind DNT() {
  return ASTNodeKind::getFromNodeKind<T>();
}

TEST(ASTNodeKind, Bases) {
  EXPECT_TRUE(DNT<Decl>().isBaseOf(DNT<VarDecl>()));
  EXPECT_FALSE(DNT<Decl>().isSame(DNT<VarDecl>()));
  EXPECT_FALSE(DNT<VarDecl>().isBaseOf(DNT<Decl>()));

  EXPECT_TRUE(DNT<Decl>().isSame(DNT<Decl>()));
}

TEST(ASTNodeKind, SameBase) {
  EXPECT_TRUE(DNT<Expr>().isBaseOf(DNT<CallExpr>()));
  EXPECT_TRUE(DNT<Expr>().isBaseOf(DNT<BinaryOperator>()));
  EXPECT_FALSE(DNT<CallExpr>().isBaseOf(DNT<BinaryOperator>()));
  EXPECT_FALSE(DNT<BinaryOperator>().isBaseOf(DNT<CallExpr>()));
}

TEST(ASTNodeKind, DiffBase) {
  EXPECT_FALSE(DNT<Expr>().isBaseOf(DNT<ArrayType>()));
  EXPECT_FALSE(DNT<QualType>().isBaseOf(DNT<FunctionDecl>()));
  EXPECT_FALSE(DNT<Type>().isSame(DNT<QualType>()));
}

struct Foo {};

TEST(ASTNodeKind, UnknownKind) {
  // We can construct one, but it is nowhere in the hierarchy.
  EXPECT_FALSE(DNT<Foo>().isSame(DNT<Foo>()));
}

TEST(ASTNodeKind, Name) {
  EXPECT_EQ("Decl", DNT<Decl>().asStringRef());
  EXPECT_EQ("CallExpr", DNT<CallExpr>().asStringRef());
  EXPECT_EQ("ConstantArrayType", DNT<ConstantArrayType>().asStringRef());
  EXPECT_EQ("<None>", ASTNodeKind().asStringRef());
}

}  // namespace ast_type_traits
}  // namespace clang
