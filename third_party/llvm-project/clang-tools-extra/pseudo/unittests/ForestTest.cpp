//===--- ForestTest.cpp - Test Forest dump ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Forest.h"
#include "clang-pseudo/Token.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

namespace clang {
namespace pseudo {
namespace {

// FIXME: extract to a TestGrammar class to allow code sharing among tests.
class ForestTest : public ::testing::Test {
public:
  void build(llvm::StringRef BNF) {
    Diags.clear();
    G = Grammar::parseBNF(BNF, Diags);
  }

  SymbolID symbol(llvm::StringRef Name) const {
    for (unsigned I = 0; I < NumTerminals; ++I)
      if (G->table().Terminals[I] == Name)
        return tokenSymbol(static_cast<tok::TokenKind>(I));
    for (SymbolID ID = 0; ID < G->table().Nonterminals.size(); ++ID)
      if (G->table().Nonterminals[ID].Name == Name)
        return ID;
    ADD_FAILURE() << "No such symbol found: " << Name;
    return 0;
  }

  RuleID ruleFor(llvm::StringRef NonterminalName) const {
    auto RuleRange = G->table().Nonterminals[symbol(NonterminalName)].RuleRange;
    if (RuleRange.End - RuleRange.Start == 1)
      return G->table().Nonterminals[symbol(NonterminalName)].RuleRange.Start;
    ADD_FAILURE() << "Expected a single rule for " << NonterminalName
                  << ", but it has " << RuleRange.End - RuleRange.Start
                  << " rule!\n";
    return 0;
  }

protected:
  std::unique_ptr<Grammar> G;
  std::vector<std::string> Diags;
};

TEST_F(ForestTest, DumpBasic) {
  build(R"cpp(
    _ := add-expression
    add-expression := id-expression + id-expression
    id-expression := IDENTIFIER
  )cpp");
  ASSERT_TRUE(Diags.empty());
  ForestArena Arena;
  const auto &TS =
      cook(lex("a + b", clang::LangOptions()), clang::LangOptions());

  auto T = Arena.createTerminals(TS);
  ASSERT_EQ(T.size(), 3u);
  const auto *Left = &Arena.createSequence(
      symbol("id-expression"), ruleFor("id-expression"), {&T.front()});
  const auto *Right = &Arena.createSequence(symbol("id-expression"),
                                            ruleFor("id-expression"), {&T[2]});

  const auto *Add =
      &Arena.createSequence(symbol("add-expression"), ruleFor("add-expression"),
                            {Left, &T[1], Right});
  EXPECT_EQ(Add->dumpRecursive(*G, true),
            "[  0, end) add-expression := id-expression + id-expression\n"
            "[  0,   1) ├─id-expression~IDENTIFIER := tok[0]\n"
            "[  1,   2) ├─+ := tok[1]\n"
            "[  2, end) └─id-expression~IDENTIFIER := tok[2]\n");
  EXPECT_EQ(Add->dumpRecursive(*G, false),
            "[  0, end) add-expression := id-expression + id-expression\n"
            "[  0,   1) ├─id-expression := IDENTIFIER\n"
            "[  0,   1) │ └─IDENTIFIER := tok[0]\n"
            "[  1,   2) ├─+ := tok[1]\n"
            "[  2, end) └─id-expression := IDENTIFIER\n"
            "[  2, end)   └─IDENTIFIER := tok[2]\n");
}

TEST_F(ForestTest, DumpAmbiguousAndRefs) {
  build(R"cpp(
    _ := type
    type := class-type # rule 3
    type := enum-type # rule 4
    class-type := shared-type
    enum-type := shared-type
    shared-type := IDENTIFIER)cpp");
  ASSERT_TRUE(Diags.empty());
  ForestArena Arena;
  const auto &TS = cook(lex("abc", clang::LangOptions()), clang::LangOptions());

  auto Terminals = Arena.createTerminals(TS);
  ASSERT_EQ(Terminals.size(), 1u);

  const auto *SharedType = &Arena.createSequence(
      symbol("shared-type"), ruleFor("shared-type"), {Terminals.begin()});
  const auto *ClassType = &Arena.createSequence(
      symbol("class-type"), ruleFor("class-type"), {SharedType});
  const auto *EnumType = &Arena.createSequence(
      symbol("enum-type"), ruleFor("enum-type"), {SharedType});
  const auto *Alternative1 =
      &Arena.createSequence(symbol("type"), /*RuleID=*/3, {ClassType});
  const auto *Alternative2 =
      &Arena.createSequence(symbol("type"), /*RuleID=*/4, {EnumType});
  const auto *Type =
      &Arena.createAmbiguous(symbol("type"), {Alternative1, Alternative2});
  EXPECT_EQ(Type->dumpRecursive(*G),
            "[  0, end) type := <ambiguous>\n"
            "[  0, end) ├─type := class-type\n"
            "[  0, end) │ └─class-type := shared-type\n"
            "[  0, end) │   └─shared-type := IDENTIFIER #1\n"
            "[  0, end) │     └─IDENTIFIER := tok[0]\n"
            "[  0, end) └─type := enum-type\n"
            "[  0, end)   └─enum-type := shared-type\n"
            "[  0, end)     └─shared-type := IDENTIFIER =#1\n"
            "[  0, end)       └─IDENTIFIER := tok[0]\n");
}

} // namespace
} // namespace pseudo
} // namespace clang
