//===- unittest/ASTMatchers/Dynamic/ParserTest.cpp - Parser unit tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include <string>
#include <vector>

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/Parser.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "gtest/gtest.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

class DummyDynTypedMatcher : public DynTypedMatcher {
public:
  DummyDynTypedMatcher(uint64_t ID) : ID(ID) {}

  typedef ast_matchers::internal::ASTMatchFinder ASTMatchFinder;
  typedef ast_matchers::internal::BoundNodesTreeBuilder BoundNodesTreeBuilder;
  virtual bool matches(const ast_type_traits::DynTypedNode DynNode,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return false;
  }

  /// \brief Makes a copy of this matcher object.
  virtual DynTypedMatcher *clone() const {
    return new DummyDynTypedMatcher(ID);
  }

  /// \brief Returns a unique ID for the matcher.
  virtual uint64_t getID() const { return ID; }

private:
  uint64_t ID;
};

class MockSema : public Parser::Sema {
public:
  virtual ~MockSema() {}

  uint64_t expectMatcher(StringRef MatcherName) {
    uint64_t ID = ExpectedMatchers.size() + 1;
    ExpectedMatchers[MatcherName] = ID;
    return ID;
  }

  void parse(StringRef Code) {
    Diagnostics Error;
    VariantValue Value;
    Parser::parseExpression(Code, this, &Value, &Error);
    Values.push_back(Value);
    Errors.push_back(Error.ToStringFull());
  }

  DynTypedMatcher *actOnMatcherExpression(StringRef MatcherName,
                                          const SourceRange &NameRange,
                                          ArrayRef<ParserValue> Args,
                                          Diagnostics *Error) {
    MatcherInfo ToStore = { MatcherName, NameRange, Args };
    Matchers.push_back(ToStore);
    return new DummyDynTypedMatcher(ExpectedMatchers[MatcherName]);
  }

  struct MatcherInfo {
    StringRef MatcherName;
    SourceRange NameRange;
    std::vector<ParserValue> Args;
  };

  std::vector<std::string> Errors;
  std::vector<VariantValue> Values;
  std::vector<MatcherInfo> Matchers;
  llvm::StringMap<uint64_t> ExpectedMatchers;
};

TEST(ParserTest, ParseString) {
  MockSema Sema;
  Sema.parse("\"Foo\"");
  Sema.parse("\"\"");
  Sema.parse("\"Baz");
  EXPECT_EQ(3ULL, Sema.Values.size());
  EXPECT_EQ("Foo", Sema.Values[0].getString());
  EXPECT_EQ("", Sema.Values[1].getString());
  EXPECT_EQ("1:1: Error parsing string token: <\"Baz>", Sema.Errors[2]);
}

bool matchesRange(const SourceRange &Range, unsigned StartLine,
                  unsigned EndLine, unsigned StartColumn, unsigned EndColumn) {
  EXPECT_EQ(StartLine, Range.Start.Line);
  EXPECT_EQ(EndLine, Range.End.Line);
  EXPECT_EQ(StartColumn, Range.Start.Column);
  EXPECT_EQ(EndColumn, Range.End.Column);
  return Range.Start.Line == StartLine && Range.End.Line == EndLine &&
         Range.Start.Column == StartColumn && Range.End.Column == EndColumn;
}

TEST(ParserTest, ParseMatcher) {
  MockSema Sema;
  const uint64_t ExpectedFoo = Sema.expectMatcher("Foo");
  const uint64_t ExpectedBar = Sema.expectMatcher("Bar");
  const uint64_t ExpectedBaz = Sema.expectMatcher("Baz");
  Sema.parse(" Foo ( Bar (), Baz( \n \"B A,Z\") )  ");
  for (size_t i = 0, e = Sema.Errors.size(); i != e; ++i) {
    EXPECT_EQ("", Sema.Errors[i]);
  }

  EXPECT_EQ(1ULL, Sema.Values.size());
  EXPECT_EQ(ExpectedFoo, Sema.Values[0].getMatcher().getID());

  EXPECT_EQ(3ULL, Sema.Matchers.size());
  const MockSema::MatcherInfo Bar = Sema.Matchers[0];
  EXPECT_EQ("Bar", Bar.MatcherName);
  EXPECT_TRUE(matchesRange(Bar.NameRange, 1, 1, 8, 14));
  EXPECT_EQ(0ULL, Bar.Args.size());

  const MockSema::MatcherInfo Baz = Sema.Matchers[1];
  EXPECT_EQ("Baz", Baz.MatcherName);
  EXPECT_TRUE(matchesRange(Baz.NameRange, 1, 2, 16, 10));
  EXPECT_EQ(1ULL, Baz.Args.size());
  EXPECT_EQ("B A,Z", Baz.Args[0].Value.getString());

  const MockSema::MatcherInfo Foo = Sema.Matchers[2];
  EXPECT_EQ("Foo", Foo.MatcherName);
  EXPECT_TRUE(matchesRange(Foo.NameRange, 1, 2, 2, 12));
  EXPECT_EQ(2ULL, Foo.Args.size());
  EXPECT_EQ(ExpectedBar, Foo.Args[0].Value.getMatcher().getID());
  EXPECT_EQ(ExpectedBaz, Foo.Args[1].Value.getMatcher().getID());
}

using ast_matchers::internal::Matcher;

TEST(ParserTest, FullParserTest) {
  OwningPtr<DynTypedMatcher> Matcher(Parser::parseMatcherExpression(
      "hasInitializer(binaryOperator(hasLHS(integerLiteral())))", NULL));
  EXPECT_TRUE(matchesDynamic("int x = 1 + false;", *Matcher));
  EXPECT_FALSE(matchesDynamic("int x = true + 1;", *Matcher));

  Diagnostics Error;
  EXPECT_TRUE(Parser::parseMatcherExpression(
      "hasInitializer(\n    binaryOperator(hasLHS(\"A\")))", &Error) == NULL);
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher hasInitializer.\n"
            "2:5: Error parsing argument 1 for matcher binaryOperator.\n"
            "2:20: Error building matcher hasLHS.\n"
            "2:27: Incorrect type on function hasLHS for arg 1.",
            Error.ToStringFull());
}

std::string ParseWithError(StringRef Code) {
  Diagnostics Error;
  VariantValue Value;
  Parser::parseExpression(Code, &Value, &Error);
  return Error.ToStringFull();
}

std::string ParseMatcherWithError(StringRef Code) {
  Diagnostics Error;
  Parser::parseMatcherExpression(Code, &Error);
  return Error.ToStringFull();
}

TEST(ParserTest, Errors) {
  EXPECT_EQ(
      "1:5: Error parsing matcher. Found token <123> while looking for '('.",
      ParseWithError("Foo 123"));
  EXPECT_EQ(
      "1:9: Error parsing matcher. Found token <123> while looking for ','.",
      ParseWithError("Foo(\"A\" 123)"));
  EXPECT_EQ(
      "1:4: Error parsing matcher. Found end-of-code while looking for ')'.",
      ParseWithError("Foo("));
  EXPECT_EQ("1:1: End of code found while looking for token.",
            ParseWithError(""));
  EXPECT_EQ("Input value is not a matcher expression.",
            ParseMatcherWithError("\"A\""));
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher Foo.\n"
            "1:5: Invalid token <(> found when looking for a value.",
            ParseWithError("Foo(("));
}

}  // end anonymous namespace
}  // end namespace dynamic
}  // end namespace ast_matchers
}  // end namespace clang
