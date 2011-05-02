//===- unittest/Tooling/JsonCompileCommandLineDatabaseTest ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../../lib/Tooling/JsonCompileCommandLineDatabase.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

TEST(UnescapeJsonCommandLine, ReturnsEmptyArrayOnEmptyString) {
  std::vector<std::string> Result = UnescapeJsonCommandLine("");
  EXPECT_TRUE(Result.empty());
}

TEST(UnescapeJsonCommandLine, SplitsOnSpaces) {
  std::vector<std::string> Result = UnescapeJsonCommandLine("a b c");
  ASSERT_EQ(3ul, Result.size());
  EXPECT_EQ("a", Result[0]);
  EXPECT_EQ("b", Result[1]);
  EXPECT_EQ("c", Result[2]);
}

TEST(UnescapeJsonCommandLine, MungesMultipleSpaces) {
  std::vector<std::string> Result = UnescapeJsonCommandLine("   a   b   ");
  ASSERT_EQ(2ul, Result.size());
  EXPECT_EQ("a", Result[0]);
  EXPECT_EQ("b", Result[1]);
}

TEST(UnescapeJsonCommandLine, UnescapesBackslashCharacters) {
  std::vector<std::string> Backslash = UnescapeJsonCommandLine("a\\\\\\\\");
  ASSERT_EQ(1ul, Backslash.size());
  EXPECT_EQ("a\\", Backslash[0]);
  std::vector<std::string> Quote = UnescapeJsonCommandLine("a\\\\\\\"");
  ASSERT_EQ(1ul, Quote.size());
  EXPECT_EQ("a\"", Quote[0]);
}

TEST(UnescapeJsonCommandLine, DoesNotMungeSpacesBetweenQuotes) {
  std::vector<std::string> Result = UnescapeJsonCommandLine("\\\"  a  b  \\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("  a  b  ", Result[0]);
}

TEST(UnescapeJsonCommandLine, AllowsMultipleQuotedArguments) {
  std::vector<std::string> Result = UnescapeJsonCommandLine(
      "  \\\" a \\\"  \\\" b \\\"  ");
  ASSERT_EQ(2ul, Result.size());
  EXPECT_EQ(" a ", Result[0]);
  EXPECT_EQ(" b ", Result[1]);
}

TEST(UnescapeJsonCommandLine, AllowsEmptyArgumentsInQuotes) {
  std::vector<std::string> Result = UnescapeJsonCommandLine(
      "\\\"\\\"\\\"\\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_TRUE(Result[0].empty()) << Result[0];
}

TEST(UnescapeJsonCommandLine, ParsesEscapedQuotesInQuotedStrings) {
  std::vector<std::string> Result = UnescapeJsonCommandLine(
      "\\\"\\\\\\\"\\\"");
  ASSERT_EQ(1ul, Result.size());
  EXPECT_EQ("\"", Result[0]);
}

TEST(UnescapeJsonCommandLine, ParsesMultipleArgumentsWithEscapedCharacters) {
  std::vector<std::string> Result = UnescapeJsonCommandLine(
      "  \\\\\\\"  \\\"a \\\\\\\" b \\\"     \\\"and\\\\\\\\c\\\"   \\\\\\\"");
  ASSERT_EQ(4ul, Result.size());
  EXPECT_EQ("\"", Result[0]);
  EXPECT_EQ("a \" b ", Result[1]);
  EXPECT_EQ("and\\c", Result[2]);
  EXPECT_EQ("\"", Result[3]);
}

TEST(UnescapeJsonCommandLine, ParsesStringsWithoutSpacesIntoSingleArgument) {
  std::vector<std::string> QuotedNoSpaces = UnescapeJsonCommandLine(
      "\\\"a\\\"\\\"b\\\"");
  ASSERT_EQ(1ul, QuotedNoSpaces.size());
  EXPECT_EQ("ab", QuotedNoSpaces[0]);

  std::vector<std::string> MixedNoSpaces = UnescapeJsonCommandLine(
      "\\\"a\\\"bcd\\\"ef\\\"\\\"\\\"\\\"g\\\"");
  ASSERT_EQ(1ul, MixedNoSpaces.size());
  EXPECT_EQ("abcdefg", MixedNoSpaces[0]);
}

TEST(UnescapeJsonCommandLine, ParsesQuotedStringWithoutClosingQuote) {
  std::vector<std::string> Unclosed = UnescapeJsonCommandLine("\"abc");
  ASSERT_EQ(1ul, Unclosed.size());
  EXPECT_EQ("abc", Unclosed[0]);

  std::vector<std::string> EndsInBackslash = UnescapeJsonCommandLine("\"a\\");
  ASSERT_EQ(1ul, EndsInBackslash.size());
  EXPECT_EQ("a", EndsInBackslash[0]);

  std::vector<std::string> Empty = UnescapeJsonCommandLine("\"");
  ASSERT_EQ(1ul, Empty.size());
  EXPECT_EQ("", Empty[0]);
}

TEST(JsonCompileCommandLineParser, FailsOnEmptyString) {
  JsonCompileCommandLineParser Parser("", NULL);
  EXPECT_FALSE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, DoesNotReadAfterInput) {
  JsonCompileCommandLineParser Parser(llvm::StringRef(NULL, 0), NULL);
  EXPECT_FALSE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesEmptyArray) {
  JsonCompileCommandLineParser Parser("[]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsIfNotClosingArray) {
  JsonCompileCommandLineParser JustOpening("[", NULL);
  EXPECT_FALSE(JustOpening.Parse()) << JustOpening.GetErrorMessage();
  JsonCompileCommandLineParser WithSpaces("  [  ", NULL);
  EXPECT_FALSE(WithSpaces.Parse()) << WithSpaces.GetErrorMessage();
  JsonCompileCommandLineParser WithGarbage("  [x", NULL);
  EXPECT_FALSE(WithGarbage.Parse()) << WithGarbage.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesEmptyArrayWithWhitespace) {
  JsonCompileCommandLineParser Spaces("   [   ]   ", NULL);
  EXPECT_TRUE(Spaces.Parse()) << Spaces.GetErrorMessage();
  JsonCompileCommandLineParser AllWhites("\t\r\n[\t\n \t\r ]\t\r \n\n", NULL);
  EXPECT_TRUE(AllWhites.Parse()) << AllWhites.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsIfNotStartingArray) {
  JsonCompileCommandLineParser ObjectStart("{", NULL);
  EXPECT_FALSE(ObjectStart.Parse()) << ObjectStart.GetErrorMessage();
  // We don't implement a full JSON parser, and thus parse only a subset
  // of valid JSON.
  JsonCompileCommandLineParser Object("{}", NULL);
  EXPECT_FALSE(Object.Parse()) << Object.GetErrorMessage();
  JsonCompileCommandLineParser Character("x", NULL);
  EXPECT_FALSE(Character.Parse()) << Character.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesEmptyObject) {
  JsonCompileCommandLineParser Parser("[{}]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesObject) {
  JsonCompileCommandLineParser Parser("[{\"a\":\"/b\"}]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesMultipleKeyValuePairsInObject) {
  JsonCompileCommandLineParser Parser(
      "[{\"a\":\"/b\",\"c\":\"d\",\"e\":\"f\"}]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsIfNotClosingObject) {
  JsonCompileCommandLineParser MissingCloseOnEmpty("[{]", NULL);
  EXPECT_FALSE(MissingCloseOnEmpty.Parse())
      << MissingCloseOnEmpty.GetErrorMessage();
  JsonCompileCommandLineParser MissingCloseAfterPair("[{\"a\":\"b\"]", NULL);
  EXPECT_FALSE(MissingCloseAfterPair.Parse())
      << MissingCloseAfterPair.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsIfMissingColon) {
  JsonCompileCommandLineParser StringString("[{\"a\"\"/b\"}]", NULL);
  EXPECT_FALSE(StringString.Parse()) << StringString.GetErrorMessage();
  JsonCompileCommandLineParser StringSpaceString("[{\"a\" \"b\"}]", NULL);
  EXPECT_FALSE(StringSpaceString.Parse())
      << StringSpaceString.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsOnMissingQuote) {
  JsonCompileCommandLineParser OpenQuote("[{a\":\"b\"}]", NULL);
  EXPECT_FALSE(OpenQuote.Parse()) << OpenQuote.GetErrorMessage();
  JsonCompileCommandLineParser CloseQuote("[{\"a\":\"b}]", NULL);
  EXPECT_FALSE(CloseQuote.Parse()) << CloseQuote.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesEscapedQuotes) {
  JsonCompileCommandLineParser Parser(
      "[{\"a\":\"\\\"b\\\"  \\\" \\\"\"}]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesEmptyString) {
  JsonCompileCommandLineParser Parser("[{\"a\":\"\"}]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsOnMissingString) {
  JsonCompileCommandLineParser MissingValue("[{\"a\":}]", NULL);
  EXPECT_FALSE(MissingValue.Parse()) << MissingValue.GetErrorMessage();
  JsonCompileCommandLineParser MissingKey("[{:\"b\"}]", NULL);
  EXPECT_FALSE(MissingKey.Parse()) << MissingKey.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesMultipleObjects) {
  JsonCompileCommandLineParser Parser(
      "["
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" }"
      "]", NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsOnMissingComma) {
  JsonCompileCommandLineParser Parser(
      "["
      " { \"a\" : \"b\" }"
      " { \"a\" : \"b\" }"
      "]", NULL);
  EXPECT_FALSE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, FailsOnSuperfluousComma) {
  JsonCompileCommandLineParser Parser(
      "[ { \"a\" : \"b\" }, ]", NULL);
  EXPECT_FALSE(Parser.Parse()) << Parser.GetErrorMessage();
}

TEST(JsonCompileCommandLineParser, ParsesSpacesInBetweenTokens) {
  JsonCompileCommandLineParser Parser(
      " \t \n\n \r [ \t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r,\t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r]\t \n\n \r",
      NULL);
  EXPECT_TRUE(Parser.Parse()) << Parser.GetErrorMessage();
}

} // end namespace tooling
} // end namespace clang
