// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FORTRAN_PARSER_CHARACTERS_H_
#define FORTRAN_PARSER_CHARACTERS_H_

// Define some character classification predicates and
// conversions here to avoid dependences upon <cctype> and
// also to accomodate Fortran tokenization.
// TODO: EBCDIC?

#include <cstddef>
#include <optional>
#include <string>

namespace Fortran {
namespace parser {

enum class Encoding { UTF8, EUC_JP };

inline constexpr bool IsUpperCaseLetter(char ch) {
  return ch >= 'A' && ch <= 'Z';
}

inline constexpr bool IsLowerCaseLetter(char ch) {
  return ch >= 'a' && ch <= 'z';
}

inline constexpr bool IsLetter(char ch) {
  return IsUpperCaseLetter(ch) || IsLowerCaseLetter(ch);
}

inline constexpr bool IsDecimalDigit(char ch) { return ch >= '0' && ch <= '9'; }

inline constexpr bool IsHexadecimalDigit(char ch) {
  return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') ||
      (ch >= 'a' && ch <= 'f');
}

inline constexpr bool IsOctalDigit(char ch) { return ch >= '0' && ch <= '7'; }

inline constexpr bool IsLegalIdentifierStart(char ch) {
  return IsLetter(ch) || ch == '_' || ch == '@' || ch == '$';
}

inline constexpr bool IsLegalInIdentifier(char ch) {
  return IsLegalIdentifierStart(ch) || IsDecimalDigit(ch);
}

inline constexpr char ToLowerCaseLetter(char ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 'a' : ch;
}

inline constexpr char ToLowerCaseLetter(char &&ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 'a' : ch;
}

inline std::string ToLowerCaseLetters(const std::string &str) {
  std::string lowered{str};
  for (char &ch : lowered) {
    ch = ToLowerCaseLetter(ch);
  }
  return lowered;
}

inline constexpr char ToUpperCaseLetter(char ch) {
  return IsLowerCaseLetter(ch) ? ch - 'a' + 'A' : ch;
}

inline constexpr char ToUpperCaseLetter(char &&ch) {
  return IsLowerCaseLetter(ch) ? ch - 'a' + 'A' : ch;
}

inline std::string ToUpperCaseLetters(const std::string &str) {
  std::string raised{str};
  for (char &ch : raised) {
    ch = ToUpperCaseLetter(ch);
  }
  return raised;
}

inline constexpr bool IsSameApartFromCase(char x, char y) {
  return ToLowerCaseLetter(x) == ToLowerCaseLetter(y);
}

inline constexpr char DecimalDigitValue(char ch) { return ch - '0'; }

inline constexpr char HexadecimalDigitValue(char ch) {
  return IsUpperCaseLetter(ch)
      ? ch - 'A' + 10
      : IsLowerCaseLetter(ch) ? ch - 'a' + 10 : DecimalDigitValue(ch);
}

inline constexpr std::optional<char> BackslashEscapeValue(char ch) {
  switch (ch) {
  // case 'a': return {'\a'};  pgf90 doesn't know about \a
  case 'b': return {'\b'};
  case 'f': return {'\f'};
  case 'n': return {'\n'};
  case 'r': return {'\r'};
  case 't': return {'\t'};
  case 'v': return {'\v'};
  case '"':
  case '\'':
  case '\\': return {ch};
  default: return {};
  }
}

inline constexpr std::optional<char> BackslashEscapeChar(char ch) {
  switch (ch) {
  // case '\a': return {'a'};  pgf90 doesn't know about \a
  case '\b': return {'b'};
  case '\f': return {'f'};
  case '\n': return {'n'};
  case '\r': return {'r'};
  case '\t': return {'t'};
  case '\v': return {'v'};
  case '"':
  case '\'':
  case '\\': return {ch};
  default: return {};
  }
}

template<typename NORMAL, typename INSERTED>
void EmitQuotedChar(char ch, const NORMAL &emit, const INSERTED &insert,
    bool doubleDoubleQuotes = true, bool doubleBackslash = true) {
  if (ch == '"') {
    if (doubleDoubleQuotes) {
      insert('"');
    }
    emit('"');
  } else if (ch == '\\') {
    if (doubleBackslash) {
      insert('\\');
    }
    emit('\\');
  } else if (ch < ' ') {
    insert('\\');
    if (std::optional escape{BackslashEscapeChar(ch)}) {
      emit(*escape);
    } else {
      // octal escape sequence
      insert('0' + ((ch >> 6) & 3));
      insert('0' + ((ch >> 3) & 7));
      insert('0' + (ch & 7));
    }
  } else {
    emit(ch);
  }
}

std::optional<int> UTF8CharacterBytes(const char *);
std::optional<int> EUC_JPCharacterBytes(const char *);
std::optional<std::size_t> CountCharacters(
    const char *, std::size_t bytes, std::optional<int> (*)(const char *));
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_CHARACTERS_H_
