//===-- DexTests.cpp  ---------------------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "TestIndex.h"
#include "index/Index.h"
#include "index/SymbolID.h"
#include "index/dex/Dex.h"
#include "index/dex/Iterator.h"
#include "index/dex/Token.h"
#include "index/dex/Trigram.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using ::testing::AnyOf;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

namespace clang {
namespace clangd {
namespace dex {
namespace {

//===----------------------------------------------------------------------===//
// Query iterator tests.
//===----------------------------------------------------------------------===//

std::vector<DocID> consumeIDs(Iterator &It) {
  auto IDAndScore = consume(It);
  std::vector<DocID> IDs(IDAndScore.size());
  for (size_t I = 0; I < IDAndScore.size(); ++I)
    IDs[I] = IDAndScore[I].first;
  return IDs;
}

TEST(DexIterators, DocumentIterator) {
  const PostingList L({4, 7, 8, 20, 42, 100});
  auto DocIterator = L.iterator();

  EXPECT_EQ(DocIterator->peek(), 4U);
  EXPECT_FALSE(DocIterator->reachedEnd());

  DocIterator->advance();
  EXPECT_EQ(DocIterator->peek(), 7U);
  EXPECT_FALSE(DocIterator->reachedEnd());

  DocIterator->advanceTo(20);
  EXPECT_EQ(DocIterator->peek(), 20U);
  EXPECT_FALSE(DocIterator->reachedEnd());

  DocIterator->advanceTo(65);
  EXPECT_EQ(DocIterator->peek(), 100U);
  EXPECT_FALSE(DocIterator->reachedEnd());

  DocIterator->advanceTo(420);
  EXPECT_TRUE(DocIterator->reachedEnd());
}

TEST(DexIterators, AndTwoLists) {
  Corpus C{10000};
  const PostingList L0({0, 5, 7, 10, 42, 320, 9000});
  const PostingList L1({0, 4, 7, 10, 30, 60, 320, 9000});

  auto And = C.intersect(L1.iterator(), L0.iterator());

  EXPECT_FALSE(And->reachedEnd());
  EXPECT_THAT(consumeIDs(*And), ElementsAre(0U, 7U, 10U, 320U, 9000U));

  And = C.intersect(L0.iterator(), L1.iterator());

  And->advanceTo(0);
  EXPECT_EQ(And->peek(), 0U);
  And->advanceTo(5);
  EXPECT_EQ(And->peek(), 7U);
  And->advanceTo(10);
  EXPECT_EQ(And->peek(), 10U);
  And->advanceTo(42);
  EXPECT_EQ(And->peek(), 320U);
  And->advanceTo(8999);
  EXPECT_EQ(And->peek(), 9000U);
  And->advanceTo(9001);
}

TEST(DexIterators, AndThreeLists) {
  Corpus C{10000};
  const PostingList L0({0, 5, 7, 10, 42, 320, 9000});
  const PostingList L1({0, 4, 7, 10, 30, 60, 320, 9000});
  const PostingList L2({1, 4, 7, 11, 30, 60, 320, 9000});

  auto And = C.intersect(L0.iterator(), L1.iterator(), L2.iterator());
  EXPECT_EQ(And->peek(), 7U);
  And->advanceTo(300);
  EXPECT_EQ(And->peek(), 320U);
  And->advanceTo(100000);

  EXPECT_TRUE(And->reachedEnd());
}

TEST(DexIterators, AndEmpty) {
  Corpus C{10000};
  const PostingList L1{1};
  const PostingList L2{2};
  // These iterators are empty, but the optimizer can't tell.
  auto Empty1 = C.intersect(L1.iterator(), L2.iterator());
  auto Empty2 = C.intersect(L1.iterator(), L2.iterator());
  // And syncs iterators on construction, and used to fail on empty children.
  auto And = C.intersect(std::move(Empty1), std::move(Empty2));
  EXPECT_TRUE(And->reachedEnd());
}

TEST(DexIterators, OrTwoLists) {
  Corpus C{10000};
  const PostingList L0({0, 5, 7, 10, 42, 320, 9000});
  const PostingList L1({0, 4, 7, 10, 30, 60, 320, 9000});

  auto Or = C.unionOf(L0.iterator(), L1.iterator());

  EXPECT_FALSE(Or->reachedEnd());
  EXPECT_EQ(Or->peek(), 0U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 4U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 5U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 7U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 10U);
  Or->advance();
  EXPECT_EQ(Or->peek(), 30U);
  Or->advanceTo(42);
  EXPECT_EQ(Or->peek(), 42U);
  Or->advanceTo(300);
  EXPECT_EQ(Or->peek(), 320U);
  Or->advanceTo(9000);
  EXPECT_EQ(Or->peek(), 9000U);
  Or->advanceTo(9001);
  EXPECT_TRUE(Or->reachedEnd());

  Or = C.unionOf(L0.iterator(), L1.iterator());

  EXPECT_THAT(consumeIDs(*Or),
              ElementsAre(0U, 4U, 5U, 7U, 10U, 30U, 42U, 60U, 320U, 9000U));
}

TEST(DexIterators, OrThreeLists) {
  Corpus C{10000};
  const PostingList L0({0, 5, 7, 10, 42, 320, 9000});
  const PostingList L1({0, 4, 7, 10, 30, 60, 320, 9000});
  const PostingList L2({1, 4, 7, 11, 30, 60, 320, 9000});

  auto Or = C.unionOf(L0.iterator(), L1.iterator(), L2.iterator());

  EXPECT_FALSE(Or->reachedEnd());
  EXPECT_EQ(Or->peek(), 0U);

  Or->advance();
  EXPECT_EQ(Or->peek(), 1U);

  Or->advance();
  EXPECT_EQ(Or->peek(), 4U);

  Or->advanceTo(7);

  Or->advanceTo(59);
  EXPECT_EQ(Or->peek(), 60U);

  Or->advanceTo(9001);
  EXPECT_TRUE(Or->reachedEnd());
}

// FIXME(kbobyrev): The testcase below is similar to what is expected in real
// queries. It should be updated once new iterators (such as boosting, limiting,
// etc iterators) appear. However, it is not exhaustive and it would be
// beneficial to implement automatic generation (e.g. fuzzing) of query trees
// for more comprehensive testing.
TEST(DexIterators, QueryTree) {
  //
  //                      +-----------------+
  //                      |And Iterator:1, 5|
  //                      +--------+--------+
  //                               |
  //                               |
  //                 +-------------+----------------------+
  //                 |                                    |
  //                 |                                    |
  //      +----------v----------+              +----------v------------+
  //      |And Iterator: 1, 5, 9|              |Or Iterator: 0, 1, 3, 5|
  //      +----------+----------+              +----------+------------+
  //                 |                                    |
  //          +------+-----+                        ------------+
  //          |            |                        |           |
  //  +-------v-----+ +----+---+                +---v----+ +----v---+
  //  |1, 3, 5, 8, 9| |Boost: 2|                |Boost: 3| |Boost: 4|
  //  +-------------+ +----+---+                +---+----+ +----+---+
  //                       |                        |           |
  //                  +----v-----+                +-v--+    +---v---+
  //                  |1, 5, 7, 9|                |1, 5|    |0, 3, 5|
  //                  +----------+                +----+    +-------+
  //
  Corpus C{10};
  const PostingList L0({1, 3, 5, 8, 9});
  const PostingList L1({1, 5, 7, 9});
  const PostingList L2({1, 5});
  const PostingList L3({0, 3, 5});

  // Root of the query tree: [1, 5]
  auto Root = C.intersect(
      // Lower And Iterator: [1, 5, 9]
      C.intersect(L0.iterator(), C.boost(L1.iterator(), 2U)),
      // Lower Or Iterator: [0, 1, 5]
      C.unionOf(C.boost(L2.iterator(), 3U), C.boost(L3.iterator(), 4U)));

  EXPECT_FALSE(Root->reachedEnd());
  EXPECT_EQ(Root->peek(), 1U);
  Root->advanceTo(0);
  // Advance multiple times. Shouldn't do anything.
  Root->advanceTo(1);
  Root->advanceTo(0);
  EXPECT_EQ(Root->peek(), 1U);
  auto ElementBoost = Root->consume();
  EXPECT_THAT(ElementBoost, 6);
  Root->advance();
  EXPECT_EQ(Root->peek(), 5U);
  Root->advanceTo(5);
  EXPECT_EQ(Root->peek(), 5U);
  ElementBoost = Root->consume();
  EXPECT_THAT(ElementBoost, 8);
  Root->advanceTo(9000);
  EXPECT_TRUE(Root->reachedEnd());
}

TEST(DexIterators, StringRepresentation) {
  Corpus C{10};
  const PostingList L1({1, 3, 5});
  const PostingList L2({1, 7, 9});

  // No token given, prints full posting list.
  auto I1 = L1.iterator();
  EXPECT_EQ(llvm::to_string(*I1), "[1 3 5]");

  // Token given, uses token's string representation.
  Token Tok(Token::Kind::Trigram, "L2");
  auto I2 = L1.iterator(&Tok);
  EXPECT_EQ(llvm::to_string(*I2), "T=L2");

  auto Tree = C.limit(C.intersect(std::move(I1), std::move(I2)), 10);
  // AND reorders its children, we don't care which order it prints.
  EXPECT_THAT(llvm::to_string(*Tree), AnyOf("(LIMIT 10 (& [1 3 5] T=L2))",
                                            "(LIMIT 10 (& T=L2 [1 3 5]))"));
}

TEST(DexIterators, Limit) {
  Corpus C{10000};
  const PostingList L0({3, 6, 7, 20, 42, 100});
  const PostingList L1({1, 3, 5, 6, 7, 30, 100});
  const PostingList L2({0, 3, 5, 7, 8, 100});

  auto DocIterator = C.limit(L0.iterator(), 42);
  EXPECT_THAT(consumeIDs(*DocIterator), ElementsAre(3, 6, 7, 20, 42, 100));

  DocIterator = C.limit(L0.iterator(), 3);
  EXPECT_THAT(consumeIDs(*DocIterator), ElementsAre(3, 6, 7));

  DocIterator = C.limit(L0.iterator(), 0);
  EXPECT_THAT(consumeIDs(*DocIterator), ElementsAre());

  auto AndIterator =
      C.intersect(C.limit(C.all(), 343), C.limit(L0.iterator(), 2),
                  C.limit(L1.iterator(), 3), C.limit(L2.iterator(), 42));
  EXPECT_THAT(consumeIDs(*AndIterator), ElementsAre(3, 7));
}

TEST(DexIterators, True) {
  EXPECT_TRUE(Corpus{0}.all()->reachedEnd());
  EXPECT_THAT(consumeIDs(*Corpus{4}.all()), ElementsAre(0, 1, 2, 3));
}

TEST(DexIterators, Boost) {
  Corpus C{5};
  auto BoostIterator = C.boost(C.all(), 42U);
  EXPECT_FALSE(BoostIterator->reachedEnd());
  auto ElementBoost = BoostIterator->consume();
  EXPECT_THAT(ElementBoost, 42U);

  const PostingList L0({2, 4});
  const PostingList L1({1, 4});
  auto Root = C.unionOf(C.all(), C.boost(L0.iterator(), 2U),
                        C.boost(L1.iterator(), 3U));

  ElementBoost = Root->consume();
  EXPECT_THAT(ElementBoost, 1);
  Root->advance();
  EXPECT_THAT(Root->peek(), 1U);
  ElementBoost = Root->consume();
  EXPECT_THAT(ElementBoost, 3);

  Root->advance();
  EXPECT_THAT(Root->peek(), 2U);
  ElementBoost = Root->consume();
  EXPECT_THAT(ElementBoost, 2);

  Root->advanceTo(4);
  ElementBoost = Root->consume();
  EXPECT_THAT(ElementBoost, 3);
}

TEST(DexIterators, Optimizations) {
  Corpus C{5};
  const PostingList L1{1};
  const PostingList L2{2};
  const PostingList L3{3};

  // empty and/or yield true/false
  EXPECT_EQ(llvm::to_string(*C.intersect()), "true");
  EXPECT_EQ(llvm::to_string(*C.unionOf()), "false");

  // true/false inside and/or short-circuit
  EXPECT_EQ(llvm::to_string(*C.intersect(L1.iterator(), C.all())), "[1]");
  EXPECT_EQ(llvm::to_string(*C.intersect(L1.iterator(), C.none())), "false");
  // Not optimized to avoid breaking boosts.
  EXPECT_EQ(llvm::to_string(*C.unionOf(L1.iterator(), C.all())),
            "(| [1] true)");
  EXPECT_EQ(llvm::to_string(*C.unionOf(L1.iterator(), C.none())), "[1]");

  // and/or nested inside and/or are flattened
  EXPECT_EQ(llvm::to_string(*C.intersect(
                L1.iterator(), C.intersect(L1.iterator(), L1.iterator()))),
            "(& [1] [1] [1])");
  EXPECT_EQ(llvm::to_string(*C.unionOf(
                L1.iterator(), C.unionOf(L2.iterator(), L3.iterator()))),
            "(| [1] [2] [3])");

  // optimizations combine over multiple levels
  EXPECT_EQ(llvm::to_string(*C.intersect(
                C.intersect(L1.iterator(), C.intersect()), C.unionOf(C.all()))),
            "[1]");
}

//===----------------------------------------------------------------------===//
// Search token tests.
//===----------------------------------------------------------------------===//

::testing::Matcher<std::vector<Token>>
tokensAre(std::initializer_list<std::string> Strings, Token::Kind Kind) {
  std::vector<Token> Tokens;
  for (const auto &TokenData : Strings) {
    Tokens.push_back(Token(Kind, TokenData));
  }
  return ::testing::UnorderedElementsAreArray(Tokens);
}

::testing::Matcher<std::vector<Token>>
trigramsAre(std::initializer_list<std::string> Trigrams) {
  return tokensAre(Trigrams, Token::Kind::Trigram);
}

std::vector<Token> identifierTrigramTokens(llvm::StringRef S) {
  std::vector<Trigram> Trigrams;
  generateIdentifierTrigrams(S, Trigrams);
  std::vector<Token> Tokens;
  for (Trigram T : Trigrams)
    Tokens.emplace_back(Token::Kind::Trigram, T.str());
  return Tokens;
}

TEST(DexTrigrams, IdentifierTrigrams) {
  EXPECT_THAT(identifierTrigramTokens("X86"), trigramsAre({"x86", "x", "x8"}));

  EXPECT_THAT(identifierTrigramTokens("nl"), trigramsAre({"nl", "n"}));

  EXPECT_THAT(identifierTrigramTokens("n"), trigramsAre({"n"}));

  EXPECT_THAT(identifierTrigramTokens("clangd"),
              trigramsAre({"c", "cl", "cla", "lan", "ang", "ngd"}));

  EXPECT_THAT(identifierTrigramTokens("abc_def"),
              trigramsAre({"a", "d", "ab", "ad", "de", "abc", "abd", "ade",
                           "bcd", "bde", "cde", "def"}));

  EXPECT_THAT(identifierTrigramTokens("a_b_c_d_e_"),
              trigramsAre({"a", "b", "ab", "bc", "abc", "bcd", "cde"}));

  EXPECT_THAT(identifierTrigramTokens("unique_ptr"),
              trigramsAre({"u",   "p",   "un",  "up",  "pt",  "uni", "unp",
                           "upt", "niq", "nip", "npt", "iqu", "iqp", "ipt",
                           "que", "qup", "qpt", "uep", "ept", "ptr"}));

  EXPECT_THAT(identifierTrigramTokens("TUDecl"),
              trigramsAre({"t", "d", "tu", "td", "de", "tud", "tde", "ude",
                           "dec", "ecl"}));

  EXPECT_THAT(identifierTrigramTokens("IsOK"),
              trigramsAre({"i", "o", "is", "ok", "io", "iso", "iok", "sok"}));

  EXPECT_THAT(identifierTrigramTokens("_pb"),
              trigramsAre({"_", "_p", "p", "pb"}));
  EXPECT_THAT(identifierTrigramTokens("__pb"),
              trigramsAre({"_", "_p", "p", "pb"}));

  EXPECT_THAT(identifierTrigramTokens("abc_defGhij__klm"),
              trigramsAre({"a",   "d",   "ab",  "ad",  "dg",  "de",  "abc",
                           "abd", "ade", "adg", "bcd", "bde", "bdg", "cde",
                           "cdg", "def", "deg", "dgh", "dgk", "efg", "egh",
                           "egk", "fgh", "fgk", "ghi", "ghk", "gkl", "hij",
                           "hik", "hkl", "ijk", "ikl", "jkl", "klm"}));
  EXPECT_THAT(identifierTrigramTokens(""), IsEmpty());
}

TEST(DexTrigrams, QueryTrigrams) {
  EXPECT_THAT(generateQueryTrigrams("c"), trigramsAre({"c"}));
  EXPECT_THAT(generateQueryTrigrams("cl"), trigramsAre({"cl"}));
  EXPECT_THAT(generateQueryTrigrams("cla"), trigramsAre({"cla"}));

  EXPECT_THAT(generateQueryTrigrams(""), trigramsAre({}));
  EXPECT_THAT(generateQueryTrigrams("_"), trigramsAre({"_"}));
  EXPECT_THAT(generateQueryTrigrams("__"), trigramsAre({"_"}));
  EXPECT_THAT(generateQueryTrigrams("___"), trigramsAre({"_"}));

  EXPECT_THAT(generateQueryTrigrams("m_"), trigramsAre({"m"}));

  EXPECT_THAT(generateQueryTrigrams("p_b"), trigramsAre({"pb"}));
  EXPECT_THAT(generateQueryTrigrams("pb_"), trigramsAre({"pb"}));
  EXPECT_THAT(generateQueryTrigrams("_p"), trigramsAre({"_p"}));
  EXPECT_THAT(generateQueryTrigrams("_pb_"), trigramsAre({"pb"}));
  EXPECT_THAT(generateQueryTrigrams("__pb"), trigramsAre({"pb"}));

  EXPECT_THAT(generateQueryTrigrams("X86"), trigramsAre({"x86"}));

  EXPECT_THAT(generateQueryTrigrams("clangd"),
              trigramsAre({"cla", "lan", "ang", "ngd"}));

  EXPECT_THAT(generateQueryTrigrams("abc_def"),
              trigramsAre({"abc", "bcd", "cde", "def"}));

  EXPECT_THAT(generateQueryTrigrams("a_b_c_d_e_"),
              trigramsAre({"abc", "bcd", "cde"}));

  EXPECT_THAT(generateQueryTrigrams("unique_ptr"),
              trigramsAre({"uni", "niq", "iqu", "que", "uep", "ept", "ptr"}));

  EXPECT_THAT(generateQueryTrigrams("TUDecl"),
              trigramsAre({"tud", "ude", "dec", "ecl"}));

  EXPECT_THAT(generateQueryTrigrams("IsOK"), trigramsAre({"iso", "sok"}));

  EXPECT_THAT(generateQueryTrigrams("abc_defGhij__klm"),
              trigramsAre({"abc", "bcd", "cde", "def", "efg", "fgh", "ghi",
                           "hij", "ijk", "jkl", "klm"}));
}

TEST(DexSearchTokens, SymbolPath) {
  EXPECT_THAT(generateProximityURIs(
                  "unittest:///clang-tools-extra/clangd/index/Token.h"),
              ElementsAre("unittest:///clang-tools-extra/clangd/index/Token.h",
                          "unittest:///clang-tools-extra/clangd/index",
                          "unittest:///clang-tools-extra/clangd",
                          "unittest:///clang-tools-extra", "unittest:///"));

  EXPECT_THAT(generateProximityURIs("unittest:///a/b/c.h"),
              ElementsAre("unittest:///a/b/c.h", "unittest:///a/b",
                          "unittest:///a", "unittest:///"));
}

//===----------------------------------------------------------------------===//
// Index tests.
//===----------------------------------------------------------------------===//

TEST(Dex, Lookup) {
  auto I = Dex::build(generateSymbols({"ns::abc", "ns::xyz"}), RefSlab(),
                      RelationSlab());
  EXPECT_THAT(lookup(*I, SymbolID("ns::abc")), UnorderedElementsAre("ns::abc"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::abc"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::abc", "ns::xyz"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::nonono"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::xyz"));
  EXPECT_THAT(lookup(*I, SymbolID("ns::nonono")), UnorderedElementsAre());
}

TEST(Dex, FuzzyFind) {
  auto Index =
      Dex::build(generateSymbols({"ns::ABC", "ns::BCD", "::ABC",
                                  "ns::nested::ABC", "other::ABC", "other::A"}),
                 RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "ABC";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*Index, Req), UnorderedElementsAre("ns::ABC"));
  Req.Scopes = {"ns::", "ns::nested::"};
  EXPECT_THAT(match(*Index, Req),
              UnorderedElementsAre("ns::ABC", "ns::nested::ABC"));
  Req.Query = "A";
  Req.Scopes = {"other::"};
  EXPECT_THAT(match(*Index, Req),
              UnorderedElementsAre("other::A", "other::ABC"));
  Req.Query = "";
  Req.Scopes = {};
  Req.AnyScope = true;
  EXPECT_THAT(match(*Index, Req),
              UnorderedElementsAre("ns::ABC", "ns::BCD", "::ABC",
                                   "ns::nested::ABC", "other::ABC",
                                   "other::A"));
}

TEST(DexTest, DexLimitedNumMatches) {
  auto I = Dex::build(generateNumSymbols(0, 100), RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "5";
  Req.AnyScope = true;
  Req.Limit = 3;
  bool Incomplete;
  auto Matches = match(*I, Req, &Incomplete);
  EXPECT_TRUE(Req.Limit);
  EXPECT_EQ(Matches.size(), *Req.Limit);
  EXPECT_TRUE(Incomplete);
}

TEST(DexTest, FuzzyMatch) {
  auto I = Dex::build(
      generateSymbols({"LaughingOutLoud", "LionPopulation", "LittleOldLady"}),
      RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "lol";
  Req.AnyScope = true;
  Req.Limit = 2;
  EXPECT_THAT(match(*I, Req),
              UnorderedElementsAre("LaughingOutLoud", "LittleOldLady"));
}

TEST(DexTest, ShortQuery) {
  auto I = Dex::build(generateSymbols({"_OneTwoFourSix"}), RefSlab(),
                      RelationSlab());
  FuzzyFindRequest Req;
  Req.AnyScope = true;
  bool Incomplete;

  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_FALSE(Incomplete) << "Empty string is not a short query";

  Req.Query = "o";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_TRUE(Incomplete) << "Using first head as unigram";

  Req.Query = "_o";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_TRUE(Incomplete) << "Using delimiter and first head as bigram";

  Req.Query = "on";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_TRUE(Incomplete) << "Using first head and tail as bigram";

  Req.Query = "ot";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_TRUE(Incomplete) << "Using first two heads as bigram";

  Req.Query = "tw";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_TRUE(Incomplete) << "Using second head and tail as bigram";

  Req.Query = "tf";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_TRUE(Incomplete) << "Using second and third heads as bigram";

  Req.Query = "fo";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre());
  EXPECT_TRUE(Incomplete) << "Short queries have different semantics";

  Req.Query = "tfs";
  EXPECT_THAT(match(*I, Req, &Incomplete), ElementsAre("_OneTwoFourSix"));
  EXPECT_FALSE(Incomplete) << "3-char string is not a short query";
}

TEST(DexTest, MatchQualifiedNamesWithoutSpecificScope) {
  auto I = Dex::build(generateSymbols({"a::y1", "b::y2", "y3"}), RefSlab(),
                      RelationSlab());
  FuzzyFindRequest Req;
  Req.AnyScope = true;
  Req.Query = "y";
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "b::y2", "y3"));
}

TEST(DexTest, MatchQualifiedNamesWithGlobalScope) {
  auto I = Dex::build(generateSymbols({"a::y1", "b::y2", "y3"}), RefSlab(),
                      RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {""};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("y3"));
}

TEST(DexTest, MatchQualifiedNamesWithOneScope) {
  auto I =
      Dex::build(generateSymbols({"a::y1", "a::y2", "a::x", "b::y2", "y3"}),
                 RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "a::y2"));
}

TEST(DexTest, MatchQualifiedNamesWithMultipleScopes) {
  auto I =
      Dex::build(generateSymbols({"a::y1", "a::y2", "a::x", "b::y3", "y3"}),
                 RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::", "b::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1", "a::y2", "b::y3"));
}

TEST(DexTest, NoMatchNestedScopes) {
  auto I = Dex::build(generateSymbols({"a::y1", "a::b::y2"}), RefSlab(),
                      RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("a::y1"));
}

TEST(DexTest, WildcardScope) {
  auto I = Dex::build(generateSymbols({"a::y1", "a::b::y2", "c::y3"}),
                      RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.AnyScope = true;
  Req.Query = "y";
  Req.Scopes = {"a::"};
  EXPECT_THAT(match(*I, Req),
              UnorderedElementsAre("a::y1", "a::b::y2", "c::y3"));
}

TEST(DexTest, IgnoreCases) {
  auto I = Dex::build(generateSymbols({"ns::ABC", "ns::abc"}), RefSlab(),
                      RelationSlab());
  FuzzyFindRequest Req;
  Req.Query = "AB";
  Req.Scopes = {"ns::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre("ns::ABC", "ns::abc"));
}

TEST(DexTest, UnknownPostingList) {
  // Regression test: we used to ignore unknown scopes and accept any symbol.
  auto I = Dex::build(generateSymbols({"ns::ABC", "ns::abc"}), RefSlab(),
                      RelationSlab());
  FuzzyFindRequest Req;
  Req.Scopes = {"ns2::"};
  EXPECT_THAT(match(*I, Req), UnorderedElementsAre());
}

TEST(DexTest, Lookup) {
  auto I = Dex::build(generateSymbols({"ns::abc", "ns::xyz"}), RefSlab(),
                      RelationSlab());
  EXPECT_THAT(lookup(*I, SymbolID("ns::abc")), UnorderedElementsAre("ns::abc"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::abc"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::abc", "ns::xyz"));
  EXPECT_THAT(lookup(*I, {SymbolID("ns::nonono"), SymbolID("ns::xyz")}),
              UnorderedElementsAre("ns::xyz"));
  EXPECT_THAT(lookup(*I, SymbolID("ns::nonono")), UnorderedElementsAre());
}

TEST(DexTest, SymbolIndexOptionsFilter) {
  auto CodeCompletionSymbol = symbol("Completion");
  auto NonCodeCompletionSymbol = symbol("NoCompletion");
  CodeCompletionSymbol.Flags = Symbol::SymbolFlag::IndexedForCodeCompletion;
  NonCodeCompletionSymbol.Flags = Symbol::SymbolFlag::None;
  std::vector<Symbol> Symbols{CodeCompletionSymbol, NonCodeCompletionSymbol};
  Dex I(Symbols, RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.AnyScope = true;
  Req.RestrictForCodeCompletion = false;
  EXPECT_THAT(match(I, Req), ElementsAre("Completion", "NoCompletion"));
  Req.RestrictForCodeCompletion = true;
  EXPECT_THAT(match(I, Req), ElementsAre("Completion"));
}

TEST(DexTest, ProximityPathsBoosting) {
  auto RootSymbol = symbol("root::abc");
  RootSymbol.CanonicalDeclaration.FileURI = "unittest:///file.h";
  auto CloseSymbol = symbol("close::abc");
  CloseSymbol.CanonicalDeclaration.FileURI = "unittest:///a/b/c/d/e/f/file.h";

  std::vector<Symbol> Symbols{CloseSymbol, RootSymbol};
  Dex I(Symbols, RefSlab(), RelationSlab());

  FuzzyFindRequest Req;
  Req.AnyScope = true;
  Req.Query = "abc";
  // The best candidate can change depending on the proximity paths.
  Req.Limit = 1;

  // FuzzyFind request comes from the file which is far from the root: expect
  // CloseSymbol to come out.
  Req.ProximityPaths = {testPath("a/b/c/d/e/f/file.h")};
  EXPECT_THAT(match(I, Req), ElementsAre("close::abc"));

  // FuzzyFind request comes from the file which is close to the root: expect
  // RootSymbol to come out.
  Req.ProximityPaths = {testPath("file.h")};
  EXPECT_THAT(match(I, Req), ElementsAre("root::abc"));
}

TEST(DexTests, Refs) {
  llvm::DenseMap<SymbolID, std::vector<Ref>> Refs;
  auto AddRef = [&](const Symbol &Sym, const char *Filename, RefKind Kind) {
    auto &SymbolRefs = Refs[Sym.ID];
    SymbolRefs.emplace_back();
    SymbolRefs.back().Kind = Kind;
    SymbolRefs.back().Location.FileURI = Filename;
  };
  auto Foo = symbol("foo");
  auto Bar = symbol("bar");
  AddRef(Foo, "foo.h", RefKind::Declaration);
  AddRef(Foo, "foo.cc", RefKind::Definition);
  AddRef(Foo, "reffoo.h", RefKind::Reference);
  AddRef(Bar, "bar.h", RefKind::Declaration);

  RefsRequest Req;
  Req.IDs.insert(Foo.ID);
  Req.Filter = RefKind::Declaration | RefKind::Definition;

  std::vector<std::string> Files;
  EXPECT_FALSE(Dex(std::vector<Symbol>{Foo, Bar}, Refs, RelationSlab())
                   .refs(Req, [&](const Ref &R) {
                     Files.push_back(R.Location.FileURI);
                   }));
  EXPECT_THAT(Files, UnorderedElementsAre("foo.h", "foo.cc"));

  Req.Limit = 1;
  Files.clear();
  EXPECT_TRUE(Dex(std::vector<Symbol>{Foo, Bar}, Refs, RelationSlab())
                  .refs(Req, [&](const Ref &R) {
                    Files.push_back(R.Location.FileURI);
                  }));
  EXPECT_THAT(Files, ElementsAre(AnyOf("foo.h", "foo.cc")));
}

TEST(DexTests, Relations) {
  auto Parent = symbol("Parent");
  auto Child1 = symbol("Child1");
  auto Child2 = symbol("Child2");

  std::vector<Symbol> Symbols{Parent, Child1, Child2};

  std::vector<Relation> Relations{{Parent.ID, RelationKind::BaseOf, Child1.ID},
                                  {Parent.ID, RelationKind::BaseOf, Child2.ID}};

  Dex I{Symbols, RefSlab(), Relations};

  std::vector<SymbolID> Results;
  RelationsRequest Req;
  Req.Subjects.insert(Parent.ID);
  Req.Predicate = RelationKind::BaseOf;
  I.relations(Req, [&](const SymbolID &Subject, const Symbol &Object) {
    Results.push_back(Object.ID);
  });
  EXPECT_THAT(Results, UnorderedElementsAre(Child1.ID, Child2.ID));
}

TEST(DexIndex, IndexedFiles) {
  SymbolSlab Symbols;
  RefSlab Refs;
  auto Size = Symbols.bytes() + Refs.bytes();
  auto Data = std::make_pair(std::move(Symbols), std::move(Refs));
  llvm::StringSet<> Files = {"unittest:///foo.cc", "unittest:///bar.cc"};
  Dex I(std::move(Data.first), std::move(Data.second), RelationSlab(),
        std::move(Files), IndexContents::All, std::move(Data), Size);
  auto ContainsFile = I.indexedFiles();
  EXPECT_EQ(ContainsFile("unittest:///foo.cc"), IndexContents::All);
  EXPECT_EQ(ContainsFile("unittest:///bar.cc"), IndexContents::All);
  EXPECT_EQ(ContainsFile("unittest:///foobar.cc"), IndexContents::None);
}

TEST(DexTest, PreferredTypesBoosting) {
  auto Sym1 = symbol("t1");
  Sym1.Type = "T1";
  auto Sym2 = symbol("t2");
  Sym2.Type = "T2";

  std::vector<Symbol> Symbols{Sym1, Sym2};
  Dex I(Symbols, RefSlab(), RelationSlab());

  FuzzyFindRequest Req;
  Req.AnyScope = true;
  Req.Query = "t";
  // The best candidate can change depending on the preferred type.
  Req.Limit = 1;

  Req.PreferredTypes = {std::string(Sym1.Type)};
  EXPECT_THAT(match(I, Req), ElementsAre("t1"));

  Req.PreferredTypes = {std::string(Sym2.Type)};
  EXPECT_THAT(match(I, Req), ElementsAre("t2"));
}

TEST(DexTest, TemplateSpecialization) {
  SymbolSlab::Builder B;

  Symbol S = symbol("TempSpec");
  S.ID = SymbolID("0");
  B.insert(S);

  S = symbol("TempSpec");
  S.ID = SymbolID("1");
  S.TemplateSpecializationArgs = "<int, bool>";
  S.SymInfo.Properties = static_cast<index::SymbolPropertySet>(
      index::SymbolProperty::TemplateSpecialization);
  B.insert(S);

  S = symbol("TempSpec");
  S.ID = SymbolID("2");
  S.TemplateSpecializationArgs = "<int, U>";
  S.SymInfo.Properties = static_cast<index::SymbolPropertySet>(
      index::SymbolProperty::TemplatePartialSpecialization);
  B.insert(S);

  auto I = dex::Dex::build(std::move(B).build(), RefSlab(), RelationSlab());
  FuzzyFindRequest Req;
  Req.AnyScope = true;

  Req.Query = "TempSpec";
  EXPECT_THAT(match(*I, Req),
              UnorderedElementsAre("TempSpec", "TempSpec<int, bool>",
                                   "TempSpec<int, U>"));

  // FIXME: Add filtering for template argument list.
  Req.Query = "TempSpec<int";
  EXPECT_THAT(match(*I, Req), IsEmpty());
}

} // namespace
} // namespace dex
} // namespace clangd
} // namespace clang
