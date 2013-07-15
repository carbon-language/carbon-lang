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
  DummyDynTypedMatcher(uint64_t ID, StringRef BoundID)
      : ID(ID), BoundID(BoundID) {}

  typedef ast_matchers::internal::ASTMatchFinder ASTMatchFinder;
  typedef ast_matchers::internal::BoundNodesTreeBuilder BoundNodesTreeBuilder;
  virtual bool matches(const ast_type_traits::DynTypedNode DynNode,
                       ASTMatchFinder *Finder,
                       BoundNodesTreeBuilder *Builder) const {
    return false;
  }

  /// \brief Makes a copy of this matcher object.
  virtual DynTypedMatcher *clone() const {
    return new DummyDynTypedMatcher(*this);
  }

  /// \brief Returns a unique ID for the matcher.
  virtual uint64_t getID() const { return ID; }

  virtual DynTypedMatcher* tryBind(StringRef BoundID) const {
    return new DummyDynTypedMatcher(ID, BoundID);
  }

  StringRef boundID() const { return BoundID; }

  virtual ast_type_traits::ASTNodeKind getSupportedKind() const {
    return ast_type_traits::ASTNodeKind();
  }

private:
  uint64_t ID;
  std::string BoundID;
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

  MatcherList actOnMatcherExpression(StringRef MatcherName,
                                     const SourceRange &NameRange,
                                     StringRef BindID,
                                     ArrayRef<ParserValue> Args,
                                     Diagnostics *Error) {
    MatcherInfo ToStore = { MatcherName, NameRange, Args, BindID };
    Matchers.push_back(ToStore);
    DummyDynTypedMatcher Matcher(ExpectedMatchers[MatcherName]);
    OwningPtr<DynTypedMatcher> Out(Matcher.tryBind(BindID));
    return *Out;
  }

  struct MatcherInfo {
    StringRef MatcherName;
    SourceRange NameRange;
    std::vector<ParserValue> Args;
    std::string BoundID;
  };

  std::vector<std::string> Errors;
  std::vector<VariantValue> Values;
  std::vector<MatcherInfo> Matchers;
  llvm::StringMap<uint64_t> ExpectedMatchers;
};

TEST(ParserTest, ParseUnsigned) {
  MockSema Sema;
  Sema.parse("0");
  Sema.parse("123");
  Sema.parse("0x1f");
  Sema.parse("12345678901");
  Sema.parse("1a1");
  EXPECT_EQ(5U, Sema.Values.size());
  EXPECT_EQ(0U, Sema.Values[0].getUnsigned());
  EXPECT_EQ(123U, Sema.Values[1].getUnsigned());
  EXPECT_EQ(31U, Sema.Values[2].getUnsigned());
  EXPECT_EQ("1:1: Error parsing unsigned token: <12345678901>", Sema.Errors[3]);
  EXPECT_EQ("1:1: Error parsing unsigned token: <1a1>", Sema.Errors[4]);
}

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
  Sema.parse(" Foo ( Bar ( 17), Baz( \n \"B A,Z\") ) .bind( \"Yo!\") ");
  for (size_t i = 0, e = Sema.Errors.size(); i != e; ++i) {
    EXPECT_EQ("", Sema.Errors[i]);
  }

  EXPECT_EQ(1ULL, Sema.Values.size());
  EXPECT_EQ(ExpectedFoo, Sema.Values[0].getMatchers().matchers()[0]->getID());
  EXPECT_EQ("Yo!", static_cast<const DummyDynTypedMatcher *>(
                       Sema.Values[0].getMatchers().matchers()[0])->boundID());

  EXPECT_EQ(3ULL, Sema.Matchers.size());
  const MockSema::MatcherInfo Bar = Sema.Matchers[0];
  EXPECT_EQ("Bar", Bar.MatcherName);
  EXPECT_TRUE(matchesRange(Bar.NameRange, 1, 1, 8, 17));
  EXPECT_EQ(1ULL, Bar.Args.size());
  EXPECT_EQ(17U, Bar.Args[0].Value.getUnsigned());

  const MockSema::MatcherInfo Baz = Sema.Matchers[1];
  EXPECT_EQ("Baz", Baz.MatcherName);
  EXPECT_TRUE(matchesRange(Baz.NameRange, 1, 2, 19, 10));
  EXPECT_EQ(1ULL, Baz.Args.size());
  EXPECT_EQ("B A,Z", Baz.Args[0].Value.getString());

  const MockSema::MatcherInfo Foo = Sema.Matchers[2];
  EXPECT_EQ("Foo", Foo.MatcherName);
  EXPECT_TRUE(matchesRange(Foo.NameRange, 1, 2, 2, 12));
  EXPECT_EQ(2ULL, Foo.Args.size());
  EXPECT_EQ(ExpectedBar,
            Foo.Args[0].Value.getMatchers().matchers()[0]->getID());
  EXPECT_EQ(ExpectedBaz,
            Foo.Args[1].Value.getMatchers().matchers()[0]->getID());
  EXPECT_EQ("Yo!", Foo.BoundID);
}

using ast_matchers::internal::Matcher;

TEST(ParserTest, FullParserTest) {
  Diagnostics Error;
  OwningPtr<DynTypedMatcher> VarDecl(Parser::parseMatcherExpression(
      "varDecl(hasInitializer(binaryOperator(hasLHS(integerLiteral()),"
      "                                      hasOperatorName(\"+\"))))",
      &Error));
  EXPECT_EQ("", Error.ToStringFull());
  Matcher<Decl> M = Matcher<Decl>::constructFrom(*VarDecl);
  EXPECT_TRUE(matches("int x = 1 + false;", M));
  EXPECT_FALSE(matches("int x = true + 1;", M));
  EXPECT_FALSE(matches("int x = 1 - false;", M));
  EXPECT_FALSE(matches("int x = true - 1;", M));

  OwningPtr<DynTypedMatcher> HasParameter(Parser::parseMatcherExpression(
      "functionDecl(hasParameter(1, hasName(\"x\")))", &Error));
  EXPECT_EQ("", Error.ToStringFull());
  M = Matcher<Decl>::constructFrom(*HasParameter);

  EXPECT_TRUE(matches("void f(int a, int x);", M));
  EXPECT_FALSE(matches("void f(int x, int a);", M));

  EXPECT_TRUE(Parser::parseMatcherExpression(
      "hasInitializer(\n    binaryOperator(hasLHS(\"A\")))", &Error) == NULL);
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher hasInitializer.\n"
            "2:5: Error parsing argument 1 for matcher binaryOperator.\n"
            "2:20: Error building matcher hasLHS.\n"
            "2:27: Incorrect type for arg 1. "
            "(Expected = Matcher<Expr>) != (Actual = String)",
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
  EXPECT_EQ("1:7: Expected end of code.", ParseWithError("expr()a"));
  EXPECT_EQ("1:11: Malformed bind() expression.",
            ParseWithError("isArrow().biind"));
  EXPECT_EQ("1:15: Malformed bind() expression.",
            ParseWithError("isArrow().bind"));
  EXPECT_EQ("1:16: Malformed bind() expression.",
            ParseWithError("isArrow().bind(foo"));
  EXPECT_EQ("1:21: Malformed bind() expression.",
            ParseWithError("isArrow().bind(\"foo\""));
  EXPECT_EQ("1:1: Error building matcher isArrow.\n"
            "1:1: Matcher does not support binding.",
            ParseWithError("isArrow().bind(\"foo\")"));
  EXPECT_EQ("Input value has unresolved overloaded type: "
            "Matcher<DoStmt|ForStmt|WhileStmt>",
            ParseMatcherWithError("hasBody(stmt())"));
}

}  // end anonymous namespace
}  // end namespace dynamic
}  // end namespace ast_matchers
}  // end namespace clang
