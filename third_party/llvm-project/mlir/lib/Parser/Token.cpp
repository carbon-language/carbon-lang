//===- Token.cpp - MLIR Token Implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Token class for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "Token.h"
#include "llvm/ADT/StringExtras.h"
using namespace mlir;
using llvm::SMLoc;
using llvm::SMRange;

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// For an integer token, return its value as an unsigned.  If it doesn't fit,
/// return None.
Optional<unsigned> Token::getUnsignedIntegerValue() const {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  unsigned result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return None;
  return result;
}

/// For an integer token, return its value as a uint64_t.  If it doesn't fit,
/// return None.
Optional<uint64_t> Token::getUInt64IntegerValue(StringRef spelling) {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  uint64_t result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return None;
  return result;
}

/// For a floatliteral, return its value as a double. Return None if the value
/// underflows or overflows.
Optional<double> Token::getFloatingPointValue() const {
  double result = 0;
  if (spelling.getAsDouble(result))
    return None;
  return result;
}

/// For an inttype token, return its bitwidth.
Optional<unsigned> Token::getIntTypeBitwidth() const {
  assert(getKind() == inttype);
  unsigned bitwidthStart = (spelling[0] == 'i' ? 1 : 2);
  unsigned result = 0;
  if (spelling.drop_front(bitwidthStart).getAsInteger(10, result))
    return None;
  return result;
}

Optional<bool> Token::getIntTypeSignedness() const {
  assert(getKind() == inttype);
  if (spelling[0] == 'i')
    return llvm::None;
  if (spelling[0] == 's')
    return true;
  assert(spelling[0] == 'u');
  return false;
}

/// Given a token containing a string literal, return its value, including
/// removing the quote characters and unescaping the contents of the string. The
/// lexer has already verified that this token is valid.
std::string Token::getStringValue() const {
  assert(getKind() == string ||
         (getKind() == at_identifier && getSpelling()[1] == '"'));
  // Start by dropping the quotes.
  StringRef bytes = getSpelling().drop_front().drop_back();
  if (getKind() == at_identifier)
    bytes = bytes.drop_front();

  std::string result;
  result.reserve(bytes.size());
  for (unsigned i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    switch (c1) {
    case '"':
    case '\\':
      result.push_back(c1);
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    default:
      break;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c2 = bytes[i++];

    assert(llvm::isHexDigit(c1) && llvm::isHexDigit(c2) && "invalid escape");
    result.push_back((llvm::hexDigitValue(c1) << 4) | llvm::hexDigitValue(c2));
  }

  return result;
}

/// Given a token containing a hex string literal, return its value or None if
/// the token does not contain a valid hex string.
Optional<std::string> Token::getHexStringValue() const {
  assert(getKind() == string);

  // Get the internal string data, without the quotes.
  StringRef bytes = getSpelling().drop_front().drop_back();

  // Try to extract the binary data from the hex string.
  std::string hex;
  if (!bytes.consume_front("0x") || !llvm::tryGetFromHex(bytes, hex))
    return llvm::None;
  return hex;
}

/// Given a token containing a symbol reference, return the unescaped string
/// value.
std::string Token::getSymbolReference() const {
  assert(is(Token::at_identifier) && "expected valid @-identifier");
  StringRef nameStr = getSpelling().drop_front();

  // Check to see if the reference is a string literal, or a bare identifier.
  if (nameStr.front() == '"')
    return getStringValue();
  return std::string(nameStr);
}

/// Given a hash_identifier token like #123, try to parse the number out of
/// the identifier, returning None if it is a named identifier like #x or
/// if the integer doesn't fit.
Optional<unsigned> Token::getHashIdentifierNumber() const {
  assert(getKind() == hash_identifier);
  unsigned result = 0;
  if (spelling.drop_front().getAsInteger(10, result))
    return None;
  return result;
}

/// Given a punctuation or keyword token kind, return the spelling of the
/// token as a string.  Warning: This will abort on markers, identifiers and
/// literal tokens since they have no fixed spelling.
StringRef Token::getTokenSpelling(Kind kind) {
  switch (kind) {
  default:
    llvm_unreachable("This token kind has no fixed spelling");
#define TOK_PUNCTUATION(NAME, SPELLING)                                        \
  case NAME:                                                                   \
    return SPELLING;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return #SPELLING;
#include "TokenKinds.def"
  }
}

/// Return true if this is one of the keyword token kinds (e.g. kw_if).
bool Token::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "TokenKinds.def"
  }
}
