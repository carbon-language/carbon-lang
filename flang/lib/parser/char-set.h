#ifndef FORTRAN_PARSER_CHAR_SET_H_
#define FORTRAN_PARSER_CHAR_SET_H_

// Sets of distinct characters that are valid in Fortran programs outside
// character literals are encoded as 64-bit integers by mapping them to a 6-bit
// character set encoding in which the case of letters is lost.  These sets
// need to be suitable for constexprs, so std::bitset<> was not eligible.

#include <cinttypes>
#include <string>

namespace Fortran {
namespace parser {

using SetOfChars = std::uint64_t;

static constexpr char SixBitEncoding(char c) {
  if (c <= 32 /*space*/) {
    // map control characters, incl. LF (newline), to '?'
    c = '?';
  } else if (c >= 127) {
    // map DEL and 8-bit characters to '^'
    c = '^';
  } else if (c >= 96) {
    // map lower-case letters to upper-case
    c -= 32;
  }
  // range is now [32..95]; reduce to [0..63]
  return c - 32;
}

static constexpr char SixBitDecoding(char c) {
  c += 32;
  if (c == '?') {
    return '\n';
  }
  return c;
}

static constexpr SetOfChars SingletonChar(char c) {
  return static_cast<SetOfChars>(1) << SixBitEncoding(c);
}

static constexpr SetOfChars CharsToSet(const char str[], std::size_t n = 256) {
  SetOfChars chars{0};
  for (std::size_t j{0}; j < n; ++j) {
    if (str[j] == '\0') {
      break;
    }
    chars |= SingletonChar(str[j]);
  }
  return chars;
}

static const SetOfChars emptySetOfChars{0};
static const SetOfChars fullSetOfChars{~static_cast<SetOfChars>(0)};
static const SetOfChars setOfLetters{
    CharsToSet("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")};
static const SetOfChars setOfDecimalDigits{CharsToSet("0123456789")};
static const SetOfChars setOfIdentifierStarts{setOfLetters | CharsToSet("_@$")};
static const SetOfChars setOfIdentifierChars{
    setOfIdentifierStarts | setOfDecimalDigits};

// sanity check
static_assert(setOfLetters == 0x7fffffe00000000);
static_assert(setOfDecimalDigits == 0x3ff0000);

static inline constexpr bool IsCharInSet(SetOfChars set, char c) {
  return (set & SingletonChar(c)) != 0;
}

static inline constexpr bool IsSingleton(SetOfChars set) {
  return (set & (set - 1)) == emptySetOfChars;
}

std::string SetOfCharsToString(SetOfChars);
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_CHAR_SET_H_
