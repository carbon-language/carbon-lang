#ifndef FORTRAN_COOKED_CHARS_H_
#define FORTRAN_COOKED_CHARS_H_

// Defines the parser cookedNextChar, which supplies all of the input to
// the next stage of parsing, viz. the tokenization parsers in cooked-tokens.h.
// It consumes the stream of raw characters and removes Fortran comments,
// continuation line markers, and characters that appear in the right margin
// of fixed form source after the column limit.  It inserts spaces to
// pad out source card images to fixed form's right margin when necessary.
// These parsers are largely bypassed when the prescanner is used, but still
// serve as the definition of correct character cooking, apart from
// preprocessing and file inclusion, which are not supported here.

#include "basic-parsers.h"
#include "char-parsers.h"
#include "idioms.h"
#include "parse-state.h"
#include <optional>

namespace Fortran {
namespace parser {

constexpr struct FixedFormPadding {
  using resultType = char;
  static std::optional<char> Parse(ParseState *state) {
    if (state->inCharLiteral() && state->inFortran() && state->inFixedForm() &&
        state->position().column() <= state->columns()) {
      if (std::optional<char> ch{state->GetNextRawChar()}) {
        if (*ch == '\n') {
          state->AdvancePositionForPadding();
          return {' '};
        }
      }
    }
    return {};
  }
} fixedFormPadding;

static inline void IncrementSkippedNewLines(ParseState *state) {
  state->set_skippedNewLines(state->skippedNewLines() + 1);
}

constexpr StateUpdateParser noteSkippedNewLine{IncrementSkippedNewLines};

static inline bool InRightMargin(const ParseState &state) {
  if (state.inFortran() && state.inFixedForm() &&
      state.position().column() > state.columns() &&
      !state.tabInCurrentLine()) {
    if (std::optional<char> ch{state.GetNextRawChar()}) {
      return *ch != '\n';
    }
  }
  return false;
}

constexpr StatePredicateGuardParser inRightMargin{InRightMargin};

template<int col> struct AtFixedFormColumn {
  using resultType = Success;
  constexpr AtFixedFormColumn() {}
  constexpr AtFixedFormColumn(const AtFixedFormColumn &) {}
  static std::optional<Success> Parse(ParseState *state) {
    if (state->inFortran() && state->inFixedForm() && !state->IsAtEnd() &&
        state->position().column() == col) {
      return {Success{}};
    }
    return {};
  }
};

template<int col> struct AtColumn {
  using resultType = Success;
  constexpr AtColumn() {}
  constexpr AtColumn(const AtColumn &) {}
  static std::optional<Success> Parse(ParseState *state) {
    if (!state->IsAtEnd() && state->position().column() == col) {
      return {Success{}};
    }
    return {};
  }
};

static inline bool AtOldDebugLineMarker(const ParseState &state) {
  if (state.inFortran() && state.inFixedForm() &&
      state.position().column() == 1) {
    if (std::optional<char> ch{state.GetNextRawChar()}) {
      return toupper(*ch) == 'D';
    }
  }
  return false;
}

static inline bool AtDisabledOldDebugLine(const ParseState &state) {
  return AtOldDebugLineMarker(state) && !state.enableOldDebugLines();
}

static inline bool AtEnabledOldDebugLine(const ParseState &state) {
  return AtOldDebugLineMarker(state) && state.enableOldDebugLines();
}

static constexpr StatePredicateGuardParser atDisabledOldDebugLine{
    AtDisabledOldDebugLine},
    atEnabledOldDebugLine{AtEnabledOldDebugLine};

constexpr auto skipPastNewLine = SkipPast<'\n'>{} / noteSkippedNewLine;

// constexpr auto rawSpace =
//  (ExactRaw<' '>{} || ExactRaw<'\t'>{} ||
//   atEnabledOldDebugLine >> rawNextChar) >> ok;
constexpr struct FastRawSpaceParser {
  using resultType = Success;
  constexpr FastRawSpaceParser() {}
  constexpr FastRawSpaceParser(const FastRawSpaceParser &) {}
  static std::optional<Success> Parse(ParseState *state) {
    if (std::optional<char> ch{state->GetNextRawChar()}) {
      if (*ch == ' ' || *ch == '\t' ||
          (toupper(*ch) == 'D' && state->position().column() == 1 &&
              state->enableOldDebugLines() && state->inFortran() &&
              state->inFixedForm())) {
        state->Advance();
        return {Success{}};
      }
    }
    return {};
  }
} rawSpace;

constexpr auto skipAnyRawSpaces = skipManyFast(rawSpace);

constexpr auto commentBang =
    !inCharLiteral >> !AtFixedFormColumn<6>{} >> ExactRaw<'!'>{} >> ok;

constexpr auto fixedComment = AtFixedFormColumn<1>{} >>
    ((ExactRaw<'*'>{} || ExactRaw<'C'>{} || ExactRaw<'c'>{}) >> ok ||
        atDisabledOldDebugLine ||
        extension(ExactRaw<'%'>{} /* VAX %list, %eject, &c. */) >> ok);

constexpr auto comment =
    (skipAnyRawSpaces >> (commentBang || inRightMargin) || fixedComment) >>
    skipPastNewLine;

constexpr auto blankLine = skipAnyRawSpaces >> eoln >> ok;

inline bool InFortran(const ParseState &state) { return state.inFortran(); }

constexpr StatePredicateGuardParser inFortran{InFortran};

inline bool FixedFormFortran(const ParseState &state) {
  return state.inFortran() && state.inFixedForm();
}

constexpr StatePredicateGuardParser fixedFormFortran{FixedFormFortran};

inline bool FreeFormFortran(const ParseState &state) {
  return state.inFortran() && !state.inFixedForm();
}

constexpr StatePredicateGuardParser freeFormFortran{FreeFormFortran};

constexpr auto lineEnd = comment || blankLine;
constexpr auto skippedLineEnd = lineEnd / noteSkippedNewLine;
constexpr auto someSkippedLineEnds = skippedLineEnd >> skipMany(skippedLineEnd);

constexpr auto fixedFormContinuation = fixedFormFortran >>
    someSkippedLineEnds >>
    (extension(AtColumn<1>{} >>
         (ExactRaw<'&'>{} ||  // extension: & in column 1
             (ExactRaw<'\t'>{} >>  // VAX Fortran: tab and then 1-9
                 ExactRawRange<'1', '9'>{}))) ||
        (skipAnyRawSpaces >> AtColumn<6>{} >> AnyCharExcept<'0'>{})) >>
    ok;

constexpr auto freeFormContinuation = freeFormFortran >>
    ((ExactRaw<'&'>{} >> blankLine >> skipMany(skippedLineEnd) >>
         skipAnyRawSpaces >> ExactRaw<'&'>{} >> ok) ||
        (ExactRaw<'&'>{} >> !inCharLiteral >> someSkippedLineEnds >>
            maybe(skipAnyRawSpaces >> ExactRaw<'&'>{}) >> ok) ||
        // PGI-only extension: don't need '&' on initial line if it's on later
        // one
        extension(eoln >> skipMany(skippedLineEnd) >> skipAnyRawSpaces >>
            ExactRaw<'&'>{} >> ok));

constexpr auto skippable = freeFormContinuation ||
    fixedFormFortran >> (fixedFormContinuation || !inCharLiteral >> rawSpace ||
                            AtColumn<6>{} >> ExactRaw<'0'>{} >> ok);

char toLower(char &&ch) { return tolower(ch); }

// TODO: skip \\ \n in C mode, increment skipped newline count;
// drain skipped newlines.

constexpr auto slowCookedNextChar = fixedFormPadding ||
    skipMany(skippable) >>
        (inCharLiteral >> rawNextChar || lineEnd >> pure('\n') ||
            rawSpace >> skipAnyRawSpaces >> pure(' ') ||
            // TODO: detect and report non-digit in fixed form label field
            inFortran >> applyFunction(toLower, rawNextChar) || rawNextChar);

constexpr struct CookedChar {
  using resultType = char;
  static std::optional<char> Parse(ParseState *state) {
    if (state->prescanned()) {
      return rawNextChar.Parse(state);
    }
    return slowCookedNextChar.Parse(state);
  }
} cookedNextChar;

static inline bool ConsumedAllInput(const ParseState &state) {
  return state.IsAtEnd();
}

constexpr StatePredicateGuardParser consumedAllInput{ConsumedAllInput};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_COOKED_CHARS_H_
