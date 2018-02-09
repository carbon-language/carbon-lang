#ifndef FORTRAN_COOKED_TOKENS_H_
#define FORTRAN_COOKED_TOKENS_H_

// These parsers are driven by the Fortran grammar (grammar.h) to consume
// the cooked character stream from cookedNextChar (cooked-chars.h) and
// partition it into a context-sensitive token stream.

#include "basic-parsers.h"
#include "cooked-chars.h"
#include "idioms.h"
#include "provenance.h"
#include <cctype>
#include <cstring>
#include <functional>
#include <limits>
#include <list>
#include <optional>
#include <string>

namespace Fortran {
namespace parser {

class CharPredicateGuardParser {
public:
  using resultType = char;
  constexpr CharPredicateGuardParser(
      const CharPredicateGuardParser &) = default;
  constexpr CharPredicateGuardParser(bool (*f)(char), const char *msg)
    : predicate_{f}, message_{msg} {}
  std::optional<char> Parse(ParseState *state) const {
    auto at = state->GetLocation();
    if (std::optional<char> result{cookedNextChar.Parse(state)}) {
      if (predicate_(*result)) {
        return result;
      }
    }
    state->PutMessage(at, message_);
    return {};
  }

private:
  bool (*const predicate_)(char);
  const char *const message_;
};

static inline constexpr bool IsDecimalDigit(char ch) { return isdigit(ch); }

static inline constexpr bool IsOctalDigit(char ch) {
  return ch >= '0' && ch <= '7';
}

static inline constexpr bool IsHexadecimalDigit(char ch) {
  return isxdigit(ch);
}

static inline constexpr bool IsLetter(char ch) { return isalpha(ch); }

static inline constexpr char ToLower(char &&ch) { return tolower(ch); }

constexpr CharPredicateGuardParser digit{IsDecimalDigit, "expected digit"};

constexpr auto letter = applyFunction(
    ToLower, CharPredicateGuardParser{IsLetter, "expected letter"});

template<char good> class CharMatch {
public:
  using resultType = char;
  constexpr CharMatch() {}
  static std::optional<char> Parse(ParseState *state) {
    auto at = state->GetLocation();
    std::optional<char> result{cookedNextChar.Parse(state)};
    if (result && *result != good) {
      result.reset();
    }
    if (!result) {
      state->PutMessage(at, "expected '"s + good + '\'');
    }
    return {result};
  }
};

constexpr struct Space {
  using resultType = Success;
  constexpr Space() {}
  static std::optional<Success> Parse(ParseState *state) {
    std::optional<char> ch{cookedNextChar.Parse(state)};
    if (ch) {
      if (ch == ' ' || ch == '\t') {
        return {Success{}};
      }
    }
    return {};
  }
} space;

constexpr auto spaces = skipMany(space);

class TokenStringMatch {
public:
  using resultType = Success;
  constexpr TokenStringMatch(const TokenStringMatch &) = default;
  constexpr TokenStringMatch(const char *str, size_t n)
    : str_{str}, length_{n} {}
  constexpr TokenStringMatch(const char *str) : str_{str} {}
  std::optional<Success> Parse(ParseState *state) const {
    auto at = state->GetLocation();
    if (!spaces.Parse(state)) {
      return {};
    }
    const char *p{str_};
    std::optional<char> ch;  // initially empty
    for (size_t j{0}; j < length_ && *p != '\0'; ++j, ++p) {
      const auto spaceSkipping{*p == ' '};
      if (spaceSkipping) {
        if (j + 1 == length_ || p[1] == ' ' || p[1] == '\0') {
          continue;  // redundant; ignore
        }
      }
      if (!ch && !(ch = cookedNextChar.Parse(state))) {
        return {};
      }
      if (spaceSkipping) {
        // medial space: 0 or more spaces/tabs accepted, none required
        while (*ch == ' ' || *ch == '\t') {
          if (!(ch = cookedNextChar.Parse(state))) {
            return {};
          }
        }
        // ch remains full for next iteration
      } else if (*ch == tolower(*p)) {
        ch.reset();
      } else {
        state->PutMessage(at, "expected '"s + str_ + '\'');
        return {};
      }
    }
    return spaces.Parse(state);
  }

private:
  const char *const str_;
  const size_t length_{std::numeric_limits<size_t>::max()};
};

constexpr TokenStringMatch operator""_tok(const char str[], size_t n) {
  return TokenStringMatch{str, n};
}

template<class PA, std::enable_if_t<std::is_class<PA>::value, int> = 0>
inline constexpr SequenceParser<TokenStringMatch, PA> operator>>(
    const char *str, const PA &p) {
  return SequenceParser<TokenStringMatch, PA>{TokenStringMatch{str}, p};
}

template<class PA, std::enable_if_t<std::is_class<PA>::value, int> = 0>
inline constexpr InvertedSequenceParser<PA, TokenStringMatch> operator/(
    const PA &p, const char *str) {
  return InvertedSequenceParser<PA, TokenStringMatch>{p, TokenStringMatch{str}};
}

template<class PA>
inline constexpr SequenceParser<TokenStringMatch,
    InvertedSequenceParser<PA, TokenStringMatch>>
parenthesized(const PA &p) {
  return "(" >> p / ")";
}

template<class PA>
inline constexpr SequenceParser<TokenStringMatch,
    InvertedSequenceParser<PA, TokenStringMatch>>
bracketed(const PA &p) {
  return "[" >> p / "]";
}

static inline int HexadecimalDigitValue(char ch) {
  if (IsDecimalDigit(ch)) {
    return ch - '0';
  }
  return toupper(ch) - 'A' + 10;
}

// Quoted character literal constants.
struct CharLiteralChar {
  struct Result {
    Result(char c, bool esc) : ch{c}, wasEscaped{esc} {}
    static Result Bare(char c) { return Result{c, false}; }
    static Result Escaped(char c) { return Result{c, true}; }
    char ch;
    bool wasEscaped;
  };
  using resultType = Result;
  static std::optional<Result> Parse(ParseState *state) {
    auto at = state->GetLocation();
    std::optional<char> och{cookedNextChar.Parse(state)};
    if (!och.has_value()) {
      return {};
    }
    char ch{*och};
    if (ch == '\n') {
      state->PutMessage(at, "unclosed character constant");
      return {};
    }
    if (ch != '\\' || !state->enableBackslashEscapesInCharLiterals()) {
      return {Result::Bare(ch)};
    }
    if (!(och = cookedNextChar.Parse(state)).has_value()) {
      return {};
    }
    switch ((ch = *och)) {
    case 'a': return {Result::Escaped('\a')};
    case 'b': return {Result::Escaped('\b')};
    case 'f': return {Result::Escaped('\f')};
    case 'n': return {Result::Escaped('\n')};
    case 'r': return {Result::Escaped('\r')};
    case 't': return {Result::Escaped('\t')};
    case 'v': return {Result::Escaped('\v')};
    case '"':
    case '\'':
    case '\\': return {Result::Escaped(ch)};
    case '\n': state->PutMessage(at, "unclosed character constant"); return {};
    default:
      if (IsOctalDigit(ch)) {
        ch -= '0';
        for (int j = (ch > 3 ? 1 : 2); j-- > 0;) {
          static constexpr auto octalDigit = attempt(
              CharPredicateGuardParser{IsOctalDigit, "expected octal digit"});
          if ((och = octalDigit.Parse(state)).has_value()) {
            ch = 8 * ch + *och - '0';
          }
        }
      } else if (ch == 'x' || ch == 'X') {
        ch = 0;
        for (int j = 0; j++ < 2;) {
          static constexpr auto hexDigit = attempt(CharPredicateGuardParser{
              IsHexadecimalDigit, "expected hexadecimal digit"});
          if ((och = hexDigit.Parse(state)).has_value()) {
            ch = 16 * ch + HexadecimalDigitValue(*och);
          }
        }
      } else {
        state->PutMessage(at, "bad escaped character");
      }
      return {Result::Escaped(ch)};
    }
  }
};

template<char quote> struct CharLiteral {
  using resultType = std::string;
  static std::optional<std::string> Parse(ParseState *state) {
    std::string str;
    CHECK(!state->set_inCharLiteral(true));
    static constexpr auto nextch = attempt(CharLiteralChar{});
    while (std::optional<CharLiteralChar::Result> ch{nextch.Parse(state)}) {
      if (ch->ch == quote && !ch->wasEscaped) {
        static constexpr auto doubled = attempt(CharMatch<quote>{});
        if (!doubled.Parse(state).has_value()) {
          state->set_inCharLiteral(false);
          return {str};
        }
      }
      str += ch->ch;
    }
    return {};
  }
};

// Parse "BOZ" binary literal quoted constants.
// As extensions, support X as an alternate hexadecimal marker, and allow
// BOZX markers to appear as synonyms.
struct BOZLiteral {
  using resultType = std::uint64_t;
  static std::optional<std::uint64_t> Parse(ParseState *state) {
    std::optional<int> shift;
    auto baseChar = [&shift](char ch) -> bool {
      switch (toupper(ch)) {
      case 'B': shift = 1; return true;
      case 'O': shift = 3; return true;
      case 'Z': shift = 4; return true;
      case 'X': shift = 4; return true;
      default: return false;
      }
    };

    if (!spaces.Parse(state)) {
      return {};
    }

    auto ch = cookedNextChar.Parse(state);
    if (!ch) {
      return {};
    }
    if (toupper(*ch) == 'X' && state->strictConformance()) {
      return {};
    }
    if (baseChar(*ch) && !(ch = cookedNextChar.Parse(state))) {
      return {};
    }

    char quote = *ch;
    if (quote != '\'' && quote != '"') {
      return {};
    }

    auto at = state->GetLocation();
    std::string content;
    while (true) {
      if (!(ch = cookedNextChar.Parse(state))) {
        return {};
      }
      if (*ch == quote) {
        break;
      }
      if (!isxdigit(*ch)) {
        return {};
      }
      content += *ch;
    }

    if (!shift && !state->strictConformance()) {
      // extension: base allowed to appear as suffix
      if (!(ch = cookedNextChar.Parse(state)) || !baseChar(*ch)) {
        return {};
      }
    }

    if (content.empty()) {
      state->PutMessage(at, "no digit in BOZ literal");
      return {};
    }

    std::uint64_t value{0};
    for (auto digit : content) {
      digit = HexadecimalDigitValue(digit);
      if ((digit >> *shift) > 0) {
        state->PutMessage(at, "bad digit in BOZ literal");
        return {};
      }
      std::uint64_t was{value};
      value <<= *shift;
      if ((value >> *shift) != was) {
        state->PutMessage(at, "excessive digits in BOZ literal");
        return {};
      }
      value |= digit;
    }
    return {value};
  }
};

// Unsigned decimal digit string; no space skipping
struct DigitString {
  using resultType = std::uint64_t;
  static std::optional<std::uint64_t> Parse(ParseState *state) {
    static constexpr auto getDigit = attempt(digit);
    auto at = state->GetLocation();
    std::optional<char> firstDigit{getDigit.Parse(state)};
    if (!firstDigit) {
      return {};
    }
    std::uint64_t value = *firstDigit - '0';
    bool overflow{false};
    while (auto nextDigit{getDigit.Parse(state)}) {
      if (value > std::numeric_limits<std::uint64_t>::max() / 10) {
        overflow = true;
      }
      value *= 10;
      int digitValue = *nextDigit - '0';
      if (value > std::numeric_limits<std::uint64_t>::max() - digitValue) {
        overflow = true;
      }
      value += digitValue;
    }
    if (overflow) {
      state->PutMessage(at, "overflow in decimal literal");
    }
    return {value};
  }
};

// Legacy feature: Hollerith literal constants
struct HollerithLiteral {
  using resultType = std::string;
  static std::optional<std::string> Parse(ParseState *state) {
    if (!spaces.Parse(state)) {
      return {};
    }
    auto at = state->GetLocation();
    std::optional<std::uint64_t> charCount{DigitString{}.Parse(state)};
    if (!charCount || *charCount < 1) {
      return {};
    }
    std::optional<char> h{letter.Parse(state)};
    if (!h || (*h != 'h' && *h != 'H')) {
      return {};
    }
    std::string content;
    CHECK(!state->set_inCharLiteral(true));
    for (auto j = *charCount; j-- > 0;) {
      std::optional<char> ch{cookedNextChar.Parse(state)};
      if (!ch || !isprint(*ch)) {
        state->PutMessage(at, "insufficient or bad characters in Hollerith");
        state->set_inCharLiteral(false);
        return {};
      }
      content += *ch;
    }
    state->set_inCharLiteral(false);
    return {content};
  }
};

// A common idiom in the Fortran grammar is an optional item (usually
// a nonempty comma-separated list) that, if present, must follow a comma
// and precede a doubled colon.  When the item is absent, the comma must
// not appear, and the doubled colons are optional.
//   [[, xyz] ::]     is  optionalBeforeColons(xyz)
//   [[, xyz]... ::]  is  optionalBeforeColons(nonemptyList(xyz))
template<typename PA> inline constexpr auto optionalBeforeColons(const PA &p) {
  return "," >> p / "::" || "::" >> construct<typename PA::resultType>{} ||
      !","_tok >> construct<typename PA::resultType>{};
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_COOKED_TOKENS_H_
