//===--- JSONParser.cpp - Simple JSON parser ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a JSON parser.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/JSONParser.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

JSONParser::JSONParser(StringRef Input)
  : Input(Input), Position(Input.begin()) {}

JSONValue *JSONParser::parseRoot() {
  if (Position != Input.begin())
    report_fatal_error("Cannot resuse JSONParser.");
  if (isWhitespace())
    nextNonWhitespace();
  if (errorIfAtEndOfFile("'[' or '{' at start of JSON text"))
    return 0;
  switch (*Position) {
    case '[':
      return new (ValueAllocator.Allocate<JSONArray>(1)) JSONArray(this);
    case '{':
      return new (ValueAllocator.Allocate<JSONObject>(1)) JSONObject(this);
    default:
      setExpectedError("'[' or '{' at start of JSON text", *Position);
      return 0;
  }
}

bool JSONParser::validate() {
  return parseRoot()->skip();
}

// Sets the current error to:
// "Error while parsing JSON: expected <Expected>, but found <Found>".
void JSONParser::setExpectedError(StringRef Expected, StringRef Found) {
  ErrorMessage = ("Error while parsing JSON: expected " +
    Expected + ", but found " + Found + ".").str();
}

// Sets the current error to:
// "Error while parsing JSON: expected <Expected>, but found <Found>".
void JSONParser::setExpectedError(StringRef Expected, char Found) {
  setExpectedError(Expected, StringRef(&Found, 1));
}

// If there is no character available, returns true and sets the current error
// to: "Error while parsing JSON: expected <Expected>, but found EOF.".
bool JSONParser::errorIfAtEndOfFile(StringRef Expected) {
  if (Position == Input.end()) {
    setExpectedError(Expected, "EOF");
    return true;
  }
  return false;
}

// Sets the current error if the current character is not C to:
// "Error while parsing JSON: expected 'C', but got <current character>".
bool JSONParser::errorIfNotAt(char C, StringRef Message) {
  if (Position == Input.end() || *Position != C) {
    std::string Expected =
      ("'" + StringRef(&C, 1) + "' " + Message).str();
    if (Position == Input.end())
      setExpectedError(Expected, "EOF");
    else
      setExpectedError(Expected, *Position);
    return true;
  }
  return false;
}

// Forbidding inlining improves performance by roughly 20%.
// FIXME: Remove once llvm optimizes this to the faster version without hints.
LLVM_ATTRIBUTE_NOINLINE static bool
wasEscaped(StringRef::iterator First, StringRef::iterator Position);

// Returns whether a character at 'Position' was escaped with a leading '\'.
// 'First' specifies the position of the first character in the string.
static bool wasEscaped(StringRef::iterator First,
                       StringRef::iterator Position) {
  assert(Position - 1 >= First);
  StringRef::iterator I = Position - 1;
  // We calulate the number of consecutive '\'s before the current position
  // by iterating backwards through our string.
  while (I >= First && *I == '\\') --I;
  // (Position - 1 - I) now contains the number of '\'s before the current
  // position. If it is odd, the character at 'Positon' was escaped.
  return (Position - 1 - I) % 2 == 1;
}

// Parses a JSONString, assuming that the current position is on a quote.
JSONString *JSONParser::parseString() {
  assert(Position != Input.end());
  assert(!isWhitespace());
  if (errorIfNotAt('"', "at start of string"))
    return 0;
  StringRef::iterator First = Position + 1;

  // Benchmarking shows that this loop is the hot path of the application with
  // about 2/3rd of the runtime cycles. Since escaped quotes are not the common
  // case, and multiple escaped backslashes before escaped quotes are very rare,
  // we pessimize this case to achieve a smaller inner loop in the common case.
  // We're doing that by having a quick inner loop that just scans for the next
  // quote. Once we find the quote we check the last character to see whether
  // the quote might have been escaped. If the last character is not a '\', we
  // know the quote was not escaped and have thus found the end of the string.
  // If the immediately preceding character was a '\', we have to scan backwards
  // to see whether the previous character was actually an escaped backslash, or
  // an escape character for the quote. If we find that the current quote was
  // escaped, we continue parsing for the next quote and repeat.
  // This optimization brings around 30% performance improvements.
  do {
    // Step over the current quote.
    ++Position;
    // Find the next quote.
    while (Position != Input.end() && *Position != '"')
      ++Position;
    if (errorIfAtEndOfFile("\" at end of string"))
      return 0;
    // Repeat until the previous character was not a '\' or was an escaped
    // backslash.
  } while (*(Position - 1) == '\\' && wasEscaped(First, Position));

  return new (ValueAllocator.Allocate<JSONString>())
      JSONString(StringRef(First, Position - First));
}


// Advances the position to the next non-whitespace position.
void JSONParser::nextNonWhitespace() {
  do {
    ++Position;
  } while (isWhitespace());
}

// Checks if there is a whitespace character at the current position.
bool JSONParser::isWhitespace() {
  return Position != Input.end() && (*Position == ' ' || *Position == '\t' ||
                                     *Position == '\n' || *Position == '\r');
}

bool JSONParser::failed() const {
  return !ErrorMessage.empty();
}

std::string JSONParser::getErrorMessage() const {
  return ErrorMessage;
}

bool JSONAtom::skip() const {
  switch (MyKind) {
  case JK_Array:        return cast<JSONArray>(this)->skip();
  case JK_Object:       return cast<JSONObject>(this)->skip();
  case JK_String:       return cast<JSONString>(this)->skip();
  case JK_KeyValuePair: return cast<JSONKeyValuePair>(this)->skip();
  }
  llvm_unreachable("Impossible enum value.");
}

// Parses a JSONValue, assuming that the current position is at the first
// character of the value.
JSONValue *JSONParser::parseValue() {
  assert(Position != Input.end());
  assert(!isWhitespace());
  switch (*Position) {
    case '[':
      return new (ValueAllocator.Allocate<JSONArray>(1)) JSONArray(this);
    case '{':
      return new (ValueAllocator.Allocate<JSONObject>(1)) JSONObject(this);
    case '"':
      return parseString();
    default:
      setExpectedError("'[', '{' or '\"' at start of value", *Position);
      return 0;
  }
}

// Parses a JSONKeyValuePair, assuming that the current position is at the first
// character of the key, value pair.
JSONKeyValuePair *JSONParser::parseKeyValuePair() {
  assert(Position != Input.end());
  assert(!isWhitespace());

  JSONString *Key = parseString();
  if (Key == 0)
    return 0;

  nextNonWhitespace();
  if (errorIfNotAt(':', "between key and value"))
    return 0;

  nextNonWhitespace();
  const JSONValue *Value = parseValue();
  if (Value == 0)
    return 0;

  return new (ValueAllocator.Allocate<JSONKeyValuePair>(1))
    JSONKeyValuePair(Key, Value);
}

template <> JSONValue *JSONParser::parseElement() {
  return parseValue();
}

template <> JSONKeyValuePair *JSONParser::parseElement() {
  return parseKeyValuePair();
}
