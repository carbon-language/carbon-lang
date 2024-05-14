// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_CHARACTER_SET_H_
#define CARBON_TOOLCHAIN_LEX_CHARACTER_SET_H_

#include "llvm/ADT/StringExtras.h"

namespace Carbon::Lex {

// TODO: These definitions need to be updated to match whatever Unicode lexical
// rules we pick. The function interfaces will need to change to accommodate
// multi-byte characters.

// Is this an alphabetical character according to Carbon's lexical rules?
//
// Alphabetical characters are permitted at the start of identifiers. This
// currently includes 'A'..'Z' and 'a'..'z'.
inline auto IsAlpha(char c) -> bool { return llvm::isAlpha(c); }

// Is this a decimal digit according to Carbon's lexical rules?
//
// This currently includes '0'..'9'.
inline auto IsDecimalDigit(char c) -> bool { return llvm::isDigit(c); }

// Is this an alphanumeric character according to Carbon's lexical rules?
//
// Alphanumeric characters are permitted as trailing characters in identifiers
// and numeric literals. This includes alphabetical characters plus decimal
// digits.
//
// Note that '_' is not considered alphanumeric, despite in most circumstances
// being a valid continuation character of an identifier or numeric literal.
inline auto IsAlnum(char c) -> bool { return llvm::isAlnum(c); }

// Is this a hexadecimal digit according to Carbon's lexical rules?
//
// Hexadecimal digits are permitted in `0x`-prefixed literals, as well as after
// a `\x` escape sequence.
//
// Note that lowercase 'a'..'f' are currently not considered hexadecimal digits
// in any context.
inline auto IsUpperHexDigit(char c) -> bool {
  return ('0' <= c && c <= '9') || ('A' <= c && c <= 'F');
}

// Is this a lowercase letter?
//
// Lowercase letters in numeric literals can be followed by `+` or `-` to
// extend the literal.
inline auto IsLower(char c) -> bool { return 'a' <= c && c <= 'z'; }

// Is this character considered to be horizontal whitespace?
//
// Such characters can appear in the indentation of a line.
inline auto IsHorizontalWhitespace(char c) -> bool {
  return c == ' ' || c == '\t';
}

// Is this character considered to be vertical whitespace?
//
// Such characters are considered to terminate lines.
inline auto IsVerticalWhitespace(char c) -> bool { return c == '\n'; }

// Is this character considered to be whitespace?
//
// Changes here will need matching changes in
// `TokenizedBuffer::Lexer::SkipWhitespace`.
inline auto IsSpace(char c) -> bool {
  return IsHorizontalWhitespace(c) || IsVerticalWhitespace(c);
}

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_CHARACTER_SET_H_
