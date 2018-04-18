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
  constexpr SetOfChars(char c) {
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
    bits_ = static_cast<std::uint64_t>(1) << (c - 32);
  }
  constexpr SetOfChars(const char str[], std::size_t n = 256) {
    for (std::size_t j{0}; j < n; ++j) {
      bits_ |= SetOfChars{str[j]}.bits_;
    }
  }
  constexpr SetOfChars(std::uint64_t b) : bits_{b} {}
  constexpr SetOfChars(const SetOfChars &) = default;
  constexpr SetOfChars(SetOfChars &&) = default;
  constexpr SetOfChars &operator=(const SetOfChars &) = default;
  constexpr SetOfChars &operator=(SetOfChars &&) = default;
  std::string ToString() const;
  std::uint64_t bits_{0};
};

static inline constexpr bool IsCharInSet(SetOfChars set, char c) {
  return (set.bits_ & SetOfChars{c}.bits_) != 0;
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_CHAR_SET_H_
