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

struct SetOfChars {
  constexpr SetOfChars() {}
  constexpr SetOfChars(std::uint64_t b) : bits_{b} {}
  constexpr SetOfChars(const SetOfChars &) = default;
  constexpr SetOfChars(SetOfChars &&) = default;
  constexpr SetOfChars &operator=(const SetOfChars &) = default;
  constexpr SetOfChars &operator=(SetOfChars &&) = default;
  std::string ToString() const;
  std::uint64_t bits_{0};
};

static constexpr std::uint64_t EncodeChar(char c) {
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
  // range is now [32..95]; reduce to [0..63] and use as a shift count
  return static_cast<std::uint64_t>(1) << (c - 32);
}

static constexpr SetOfChars SingletonChar(char c) { return {EncodeChar(c)}; }

static constexpr SetOfChars CharsToSet(const char str[], std::size_t n = 256) {
  SetOfChars chars;
  for (std::size_t j{0}; j < n; ++j) {
    if (str[j] == '\0') {
      break;
    }
    chars.bits_ |= EncodeChar(str[j]);
  }
  return chars;
}

static inline constexpr bool IsCharInSet(SetOfChars set, char c) {
  return (set.bits_ & EncodeChar(c)) != 0;
}

std::string SetOfCharsToString(SetOfChars);
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_CHAR_SET_H_
