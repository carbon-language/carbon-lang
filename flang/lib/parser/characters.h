#ifndef FORTRAN_PARSER_CHARACTERS_H_
#define FORTRAN_PARSER_CHARACTERS_H_

// Define some character classification predicates and
// conversions here to avoid dependences upon <cctype> and
// also to accomodate Fortran tokenization.

#include <optional>
#include <string>

namespace Fortran {
namespace parser {

enum class Encoding { UTF8, EUC_JP };

static constexpr bool IsUpperCaseLetter(char ch) {
  if constexpr ('A' == static_cast<char>(0xc1)) {
    // EBCDIC
    return (ch >= 'A' && ch <= 'I') || (ch >= 'J' && ch <= 'R') ||
        (ch >= 'S' && ch <= 'Z');
  } else {
    return ch >= 'A' && ch <= 'Z';
  }
}

static constexpr bool IsLowerCaseLetter(char ch) {
  if constexpr ('a' == static_cast<char>(0x81)) {
    // EBCDIC
    return (ch >= 'a' && ch <= 'i') || (ch >= 'j' && ch <= 'r') ||
        (ch >= 's' && ch <= 'z');
  } else {
    return ch >= 'a' && ch <= 'z';
  }
}

static constexpr bool IsLetter(char ch) {
  return IsUpperCaseLetter(ch) || IsLowerCaseLetter(ch);
}

static constexpr bool IsDecimalDigit(char ch) { return ch >= '0' && ch <= '9'; }

static constexpr bool IsHexadecimalDigit(char ch) {
  return (ch >= '0' && ch <= '9') || (ch >= 'A' && ch <= 'F') ||
      (ch >= 'a' && ch <= 'f');
}

static constexpr bool IsOctalDigit(char ch) { return ch >= '0' && ch <= '7'; }

static constexpr bool IsLegalIdentifierStart(char ch) {
  return IsLetter(ch) || ch == '_' || ch == '@' || ch == '$';
}

static constexpr bool IsLegalInIdentifier(char ch) {
  return IsLegalIdentifierStart(ch) || IsDecimalDigit(ch);
}

static constexpr char ToLowerCaseLetter(char ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 'a' : ch;
}

static constexpr char ToLowerCaseLetter(char &&ch) {
  return IsUpperCaseLetter(ch) ? ch - 'A' + 'a' : ch;
}

static inline std::string ToLowerCaseLetters(const std::string &str) {
  std::string lowered{str};
  for (char &ch : lowered) {
    ch = ToLowerCaseLetter(ch);
  }
  return lowered;
}

static constexpr char ToUpperCaseLetter(char ch) {
  return IsLowerCaseLetter(ch) ? ch - 'a' + 'A' : ch;
}

static constexpr char ToUpperCaseLetter(char &&ch) {
  return IsLowerCaseLetter(ch) ? ch - 'a' + 'A' : ch;
}

static inline std::string ToUpperCaseLetters(const std::string &str) {
  std::string raised{str};
  for (char &ch : raised) {
    ch = ToUpperCaseLetter(ch);
  }
  return raised;
}

static constexpr bool IsSameApartFromCase(char x, char y) {
  return ToLowerCaseLetter(x) == ToLowerCaseLetter(y);
}

static constexpr char DecimalDigitValue(char ch) { return ch - '0'; }

static constexpr char HexadecimalDigitValue(char ch) {
  return IsUpperCaseLetter(ch)
      ? ch - 'A' + 10
      : IsLowerCaseLetter(ch) ? ch - 'a' + 10 : DecimalDigitValue(ch);
}

static constexpr std::optional<char> BackslashEscapeValue(char ch) {
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

static constexpr std::optional<char> BackslashEscapeChar(char ch) {
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
std::optional<size_t> CountCharacters(
    const char *, size_t bytes, std::optional<int> (*)(const char *));
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_CHARACTERS_H_
