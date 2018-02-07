#ifndef FORTRAN_CHAR_PARSERS_H_
#define FORTRAN_CHAR_PARSERS_H_

// Defines simple character-level parsers for use by the tokenizing
// parsers in cooked-chars.h.

#include "basic-parsers.h"
#include "parse-state.h"
#include <optional>

namespace Fortran {
namespace parser {

template<char goal> struct ExactRaw {
  using resultType = char;
  constexpr ExactRaw() {}
  constexpr ExactRaw(const ExactRaw &) {}
  static std::optional<char> Parse(ParseState *state) {
    if (std::optional<char> ch{state->GetNextRawChar()}) {
      if (*ch == goal) {
        state->Advance();
        return ch;
      }
    }
    return {};
  }
};

template<char a, char z> struct ExactRawRange {
  using resultType = char;
  constexpr ExactRawRange() {}
  constexpr ExactRawRange(const ExactRawRange &){};
  static std::optional<char> Parse(ParseState *state) {
    if (std::optional<char> ch{state->GetNextRawChar()}) {
      if (*ch >= a && *ch <= z) {
        state->Advance();
        return ch;
      }
    }
    return {};
  }
};

template<char unwanted> struct AnyCharExcept {
  using resultType = char;
  constexpr AnyCharExcept() {}
  constexpr AnyCharExcept(const AnyCharExcept &) {}
  static std::optional<char> Parse(ParseState *state) {
    if (std::optional<char> ch{state->GetNextRawChar()}) {
      if (*ch != unwanted) {
        state->Advance();
        return ch;
      }
    }
    return {};
  }
};

template<char goal> struct SkipPast {
  using resultType = Success;
  constexpr SkipPast() {}
  constexpr SkipPast(const SkipPast &) {}
  static std::optional<Success> Parse(ParseState *state) {
    while (std::optional<char> ch{state->GetNextRawChar()}) {
      state->Advance();
      if (*ch == goal) {
        return {Success{}};
      }
    }
    return {};
  }
};

// Line endings have been previously normalized to simple newlines.
constexpr auto eoln = ExactRaw<'\n'>{};

static inline bool InCharLiteral(const ParseState &state) {
  return state.inCharLiteral();
}

constexpr StatePredicateGuardParser inCharLiteral{InCharLiteral};

class RawStringMatch {
public:
  using resultType = Success;
  constexpr RawStringMatch(const RawStringMatch &) = default;
  constexpr RawStringMatch(const char *str, size_t n) : str_{str}, length_{n} {}
  std::optional<Success> Parse(ParseState *state) const {
    const char *p{str_};
    for (size_t j{0}; j < length_ && *p != '\0'; ++j, ++p) {
      if (std::optional<char> ch{state->GetNextRawChar()}) {
        if (tolower(*ch) != *p) {
          return {};
        }
        state->Advance();
      } else {
        return {};
      }
    }
    return {Success{}};
  }

private:
  const char *const str_;
  const size_t length_;
};

constexpr RawStringMatch operator""_raw(const char str[], size_t n) {
  return RawStringMatch{str, n};
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_CHAR_PARSERS_H_
