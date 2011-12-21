//===- unittest/Tooling/JSONParserTest ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Casting.h"
#include "llvm/Support/JSONParser.h"
#include "llvm/ADT/Twine.h"
#include "gtest/gtest.h"

namespace llvm {

// Checks that the given input gives a parse error. Makes sure that an error
// text is available and the parse fails.
static void ExpectParseError(StringRef Message, StringRef Input) {
  SourceMgr SM;
  JSONParser Parser(Input, &SM);
  EXPECT_FALSE(Parser.validate()) << Message << ": " << Input;
  EXPECT_TRUE(Parser.failed()) << Message << ": " << Input;
}

// Checks that the given input can be parsed without error.
static void ExpectParseSuccess(StringRef Message, StringRef Input) {
  SourceMgr SM;
  JSONParser Parser(Input, &SM);
  EXPECT_TRUE(Parser.validate()) << Message << ": " << Input;
}

TEST(JSONParser, FailsOnEmptyString) {
  ExpectParseError("Empty JSON text", "");
}
 
TEST(JSONParser, FailsIfStartsWithString) {
  ExpectParseError("Top-level string", "\"x\"");
}

TEST(JSONParser, ParsesEmptyArray) {
  ExpectParseSuccess("Empty array", "[]");
}

TEST(JSONParser, FailsIfNotClosingArray) {
  ExpectParseError("Not closing array", "[");
  ExpectParseError("Not closing array", "  [  ");
  ExpectParseError("Not closing array", "  [x");
}

TEST(JSONParser, ParsesEmptyArrayWithWhitespace) {
  ExpectParseSuccess("Array with spaces", "  [  ]  ");
  ExpectParseSuccess("All whitespaces", "\t\r\n[\t\n \t\r ]\t\r \n\n");
}

TEST(JSONParser, ParsesEmptyObject) {
  ExpectParseSuccess("Empty object", "[{}]");
}

TEST(JSONParser, ParsesObject) {
  ExpectParseSuccess("Object with an entry", "[{\"a\":\"/b\"}]");
}

TEST(JSONParser, ParsesMultipleKeyValuePairsInObject) {
  ExpectParseSuccess("Multiple key, value pairs",
                     "[{\"a\":\"/b\",\"c\":\"d\",\"e\":\"f\"}]");
}

TEST(JSONParser, FailsIfNotClosingObject) {
  ExpectParseError("Missing close on empty", "[{]");
  ExpectParseError("Missing close after pair", "[{\"a\":\"b\"]");
}

TEST(JSONParser, FailsIfMissingColon) {
  ExpectParseError("Missing colon between key and value", "[{\"a\"\"/b\"}]");
  ExpectParseError("Missing colon between key and value", "[{\"a\" \"b\"}]");
}

TEST(JSONParser, FailsOnMissingQuote) {
  ExpectParseError("Missing open quote", "[{a\":\"b\"}]");
  ExpectParseError("Missing closing quote", "[{\"a\":\"b}]");
}

TEST(JSONParser, ParsesEscapedQuotes) {
  ExpectParseSuccess("Parses escaped string in key and value",
                     "[{\"a\":\"\\\"b\\\"  \\\" \\\"\"}]");
}

TEST(JSONParser, ParsesEmptyString) {
  ExpectParseSuccess("Parses empty string in value", "[{\"a\":\"\"}]");
}

TEST(JSONParser, FailsOnMissingString) {
  ExpectParseError("Missing value", "[{\"a\":}]");
  ExpectParseError("Missing key", "[{:\"b\"}]");
}

TEST(JSONParser, ParsesMultipleObjects) {
  ExpectParseSuccess(
      "Multiple objects in array",
      "["
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" }"
      "]");
}

TEST(JSONParser, FailsOnMissingComma) {
  ExpectParseError(
      "Missing comma",
      "["
      " { \"a\" : \"b\" }"
      " { \"a\" : \"b\" }"
      "]");
}

TEST(JSONParser, FailsOnSuperfluousComma) {
  ExpectParseError("Superfluous comma in array", "[ { \"a\" : \"b\" }, ]");
  ExpectParseError("Superfluous comma in object", "{ \"a\" : \"b\", }");
}

TEST(JSONParser, ParsesSpacesInBetweenTokens) {
  ExpectParseSuccess(
      "Various whitespace between tokens",
      " \t \n\n \r [ \t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r,\t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r]\t \n\n \r");
}

TEST(JSONParser, ParsesArrayOfArrays) {
  ExpectParseSuccess("Array of arrays", "[[]]");
}

TEST(JSONParser, HandlesEndOfFileGracefully) {
  ExpectParseError("In string starting with EOF", "[\"");
  ExpectParseError("In string hitting EOF", "[\"   ");
  ExpectParseError("In string escaping EOF", "[\"  \\");
  ExpectParseError("In array starting with EOF", "[");
  ExpectParseError("In array element starting with EOF", "[[], ");
  ExpectParseError("In array hitting EOF", "[[] ");
  ExpectParseError("In array hitting EOF", "[[]");
  ExpectParseError("In object hitting EOF", "{\"\"");
}

// Checks that the given string can be parsed into an identical string inside
// of an array.
static void ExpectCanParseString(StringRef String) {
  std::string StringInArray = (llvm::Twine("[\"") + String + "\"]").str();
  SourceMgr SM;
  JSONParser Parser(StringInArray, &SM);
  const JSONArray *ParsedArray = dyn_cast<JSONArray>(Parser.parseRoot());
  StringRef ParsedString =
      dyn_cast<JSONString>(*ParsedArray->begin())->getRawText();
  EXPECT_EQ(String, ParsedString.str());
}

// Checks that parsing the given string inside an array fails.
static void ExpectCannotParseString(StringRef String) {
  std::string StringInArray = (llvm::Twine("[\"") + String + "\"]").str();
  ExpectParseError((Twine("When parsing string \"") + String + "\"").str(),
                   StringInArray);
}

TEST(JSONParser, ParsesStrings) {
  ExpectCanParseString("");
  ExpectCannotParseString("\\");
  ExpectCannotParseString("\"");
  ExpectCanParseString(" ");
  ExpectCanParseString("\\ ");
  ExpectCanParseString("\\\"");
  ExpectCannotParseString("\"\\");
  ExpectCannotParseString(" \\");
  ExpectCanParseString("\\\\");
  ExpectCannotParseString("\\\\\\");
  ExpectCanParseString("\\\\\\\\");
  ExpectCanParseString("\\\" ");
  ExpectCannotParseString("\\\\\" ");
  ExpectCanParseString("\\\\\\\" ");
  ExpectCanParseString("    \\\\  \\\"  \\\\\\\"   ");
}

TEST(JSONParser, WorksWithIteratorAlgorithms) {
  SourceMgr SM;
  JSONParser Parser("[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\"]", &SM);
  const JSONArray *Array = dyn_cast<JSONArray>(Parser.parseRoot());
  EXPECT_EQ(6, std::distance(Array->begin(), Array->end()));
}

} // end namespace llvm
