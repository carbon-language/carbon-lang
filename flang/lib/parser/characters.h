// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

namespace Fortran::parser {

// We can easily support Fortran program source in any character
// set whose first 128 code points correspond to ASCII codes 0-127 (ISO/IEC646).
// The specific encodings that we can handle include:
//   LATIN_1: ISO 8859-1 Latin-1
//   UTF_8: Multi-byte encoding of Unicode (ISO/IEC 10646)
//   EUC_JP: 1-3 byte encoding of JIS X 0208 / 0212
enum class Encoding { LATIN_1, UTF_8, EUC_JP };

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
  // case 'a': return {'\a'};  // pgf90 has no \a
  case 'b': return {'\b'};
  case 'f': return {'\f'};
  case 'n': return {'\n'};
  case 'r': return {'\r'};
  case 't': return {'\t'};
  case 'v': return {'\v'};
  case '"':
  case '\'':
  case '\\': return {ch};
  default: return std::nullopt;
  }
}

inline constexpr std::optional<char> BackslashEscapeChar(char ch) {
  switch (ch) {
  // case '\a': return {'a'};  // pgf90 has no \a
  case '\b': return {'b'};
  case '\f': return {'f'};
  case '\n': return {'n'};
  case '\r': return {'r'};
  case '\t': return {'t'};
  case '\v': return {'v'};
  case '"':
  case '\'':
  case '\\': return {ch};
  default: return std::nullopt;
  }
}

struct EncodedCharacter {
  char buffer[4];
  int bytes{0};
};

EncodedCharacter EncodeLATIN_1(char);
EncodedCharacter EncodeUTF_8(char32_t);
EncodedCharacter EncodeEUC_JP(char16_t);
EncodedCharacter EncodeCharacter(Encoding, char32_t);

template<typename NORMAL, typename INSERTED>
void EmitQuotedChar(char32_t ch, const NORMAL &emit, const INSERTED &insert,
    bool doubleDoubleQuotes = true, bool doubleBackslash = true,
    Encoding encoding = Encoding::UTF_8) {
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
  } else if (ch < ' ' || (encoding == Encoding::LATIN_1 && ch >= 0x7f)) {
    insert('\\');
    if (std::optional<char> escape{BackslashEscapeChar(ch)}) {
      emit(*escape);
    } else {
      // octal escape sequence
      insert('0' + ((ch >> 6) & 3));
      insert('0' + ((ch >> 3) & 7));
      insert('0' + (ch & 7));
    }
  } else {
    EncodedCharacter encoded{EncodeCharacter(encoding, ch)};
    for (int j{0}; j < encoded.bytes; ++j) {
      emit(encoded.buffer[j]);
    }
  }
}

std::string QuoteCharacterLiteral(const std::string &,
    bool doubleDoubleQuotes = true, bool doubleBackslash = true,
    Encoding = Encoding::LATIN_1);
std::string QuoteCharacterLiteral(const std::u16string &,
    bool doubleDoubleQuotes = true, bool doubleBackslash = true,
    Encoding = Encoding::EUC_JP);
std::string QuoteCharacterLiteral(const std::u32string &,
    bool doubleDoubleQuotes = true, bool doubleBackslash = true,
    Encoding = Encoding::UTF_8);

std::optional<int> UTF_8CharacterBytes(const char *);
std::optional<int> EUC_JPCharacterBytes(const char *);
std::optional<int> CharacterBytes(const char *, Encoding);
std::optional<int> CountCharacters(const char *, std::size_t bytes, Encoding);

struct DecodedCharacter {
  char32_t unicode{0};
  int bytes{0};  // signifying failure
};

DecodedCharacter DecodeUTF_8Character(const char *, std::size_t);
DecodedCharacter DecodeEUC_JPCharacter(const char *, std::size_t);
DecodedCharacter DecodeLATIN1Character(const char *);
DecodedCharacter DecodeCharacter(Encoding, const char *, std::size_t);

std::u32string DecodeUTF_8(const std::string &);
std::u16string DecodeEUC_JP(const std::string &);
}
#endif  // FORTRAN_PARSER_CHARACTERS_H_
