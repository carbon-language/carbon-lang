//===-- runtime/format.cc ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "format.h"
#include "../lib/common/format.h"
#include "../lib/decimal/decimal.h"
#include <limits>

namespace Fortran::runtime {

template<typename CHAR>
FormatControl<CHAR>::FormatControl(FormatContext &context, const CHAR *format,
    std::size_t formatLength, const MutableModes &modes, int maxHeight)
  : context_{context}, modes_{modes}, maxHeight_{static_cast<std::uint8_t>(
                                          maxHeight)},
    format_{format}, formatLength_{static_cast<int>(formatLength)} {
  // The additional two items are for the whole string and a
  // repeated non-parenthesized edit descriptor.
  if (maxHeight > std::numeric_limits<std::int8_t>::max()) {
    context_.terminator.Crash(
        "internal Fortran runtime error: maxHeight %d", maxHeight);
  }
  stack_[0].start = offset_;
  stack_[0].remaining = Iteration::unlimited;  // 13.4(8)
}

template<typename CHAR>
int FormatControl<CHAR>::GetMaxParenthesisNesting(
    Terminator &terminator, const CHAR *format, std::size_t formatLength) {
  using Validator = common::FormatValidator<CHAR>;
  typename Validator::Reporter reporter{
      [&](const common::FormatMessage &message) {
        terminator.Crash(message.text, message.arg);
        return false;  // crashes on error above
      }};
  Validator validator{format, formatLength, reporter};
  validator.Check();
  return validator.maxNesting();
}

static void HandleCharacterLiteral(
    FormatContext &context, const char *str, std::size_t chars) {
  if (context.handleCharacterLiteral1) {
    context.handleCharacterLiteral1(str, chars);
  }
}

static void HandleCharacterLiteral(
    FormatContext &context, const char16_t *str, std::size_t chars) {
  if (context.handleCharacterLiteral2) {
    context.handleCharacterLiteral2(str, chars);
  }
}

static void HandleCharacterLiteral(
    FormatContext &context, const char32_t *str, std::size_t chars) {
  if (context.handleCharacterLiteral4) {
    context.handleCharacterLiteral4(str, chars);
  }
}

template<typename CHAR> int FormatControl<CHAR>::GetIntField(CHAR firstCh) {
  CHAR ch{firstCh ? firstCh : PeekNext()};
  if (ch < '0' || ch > '9') {
    context_.terminator.Crash(
        "Invalid FORMAT: integer expected at '%c'", static_cast<char>(ch));
  }
  int result{0};
  while (ch >= '0' && ch <= '9') {
    if (result >
        std::numeric_limits<int>::max() / 10 - (static_cast<int>(ch) - '0')) {
      context_.terminator.Crash("FORMAT integer field out of range");
    }
    result = 10 * result + ch - '0';
    if (firstCh) {
      firstCh = '\0';
    } else {
      ++offset_;
    }
    ch = PeekNext();
  }
  return result;
}

static void HandleControl(MutableModes &modes, std::uint16_t &scale,
    FormatContext &context, char ch, char next, int n) {
  switch (ch) {
  case 'B':
    if (next == 'Z') {
      modes.editingFlags |= blankZero;
      return;
    }
    if (next == 'N') {
      modes.editingFlags &= ~blankZero;
      return;
    }
    break;
  case 'D':
    if (next == 'C') {
      modes.editingFlags |= decimalComma;
      return;
    }
    if (next == 'P') {
      modes.editingFlags &= ~decimalComma;
      return;
    }
    break;
  case 'P':
    if (!next) {
      scale = n;  // kP - decimal scaling by 10**k (TODO)
      return;
    }
    break;
  case 'R':
    switch (next) {
    case 'N': modes.roundingMode = common::RoundingMode::TiesToEven; return;
    case 'Z': modes.roundingMode = common::RoundingMode::ToZero; return;
    case 'U': modes.roundingMode = common::RoundingMode::Up; return;
    case 'D': modes.roundingMode = common::RoundingMode::Down; return;
    case 'C':
      modes.roundingMode = common::RoundingMode::TiesAwayFromZero;
      return;
    default: break;
    }
    break;
  case 'X':
    if (!next) {
      if (context.handleRelativePosition) {
        context.handleRelativePosition(n);
      }
      return;
    }
    break;
  case 'S':
    if (next == 'P') {
      modes.editingFlags |= signPlus;
      return;
    }
    if (!next || next == 'S') {
      modes.editingFlags &= ~signPlus;
      return;
    }
    break;
  case 'T': {
    if (!next) {  // Tn
      if (context.handleAbsolutePosition) {
        context.handleAbsolutePosition(n);
      }
      return;
    }
    if (next == 'L' || next == 'R') {  // TLn & TRn
      if (context.handleRelativePosition) {
        context.handleRelativePosition(next == 'L' ? -n : n);
      }
      return;
    }
  } break;
  default: break;
  }
  if (next) {
    context.terminator.Crash(
        "Unknown '%c%c' edit descriptor in FORMAT", ch, next);
  } else {
    context.terminator.Crash("Unknown '%c' edit descriptor in FORMAT", ch);
  }
}

// Locates the next data edit descriptor in the format.
// Handles all repetition counts and control edit descriptors.
// Generally assumes that the format string has survived the common
// format validator gauntlet.
template<typename CHAR> int FormatControl<CHAR>::CueUpNextDataEdit(bool stop) {
  int unlimitedLoopCheck{-1};
  while (true) {
    std::optional<int> repeat;
    bool unlimited{false};
    CHAR ch{Capitalize(GetNextChar())};
    while (ch == ',' || ch == ':') {
      // Skip commas, and don't complain if they're missing; the format
      // validator does that.
      if (stop && ch == ':') {
        return 0;
      }
      ch = Capitalize(GetNextChar());
    }
    if (ch >= '0' && ch <= '9') {  // repeat count
      repeat = GetIntField(ch);
      ch = GetNextChar();
    } else if (ch == '*') {
      unlimited = true;
      ch = GetNextChar();
      if (ch != '(') {
        context_.terminator.Crash(
            "Invalid FORMAT: '*' may appear only before '('");
      }
    }
    if (ch == '(') {
      if (height_ >= maxHeight_) {
        context_.terminator.Crash(
            "FORMAT stack overflow: too many nested parentheses");
      }
      stack_[height_].start = offset_ - 1;  // the '('
      if (unlimited || height_ == 0) {
        stack_[height_].remaining = Iteration::unlimited;
        unlimitedLoopCheck = offset_ - 1;
      } else if (repeat) {
        if (*repeat <= 0) {
          *repeat = 1;  // error recovery
        }
        stack_[height_].remaining = *repeat - 1;
      } else {
        stack_[height_].remaining = 0;
      }
      ++height_;
    } else if (height_ == 0) {
      context_.terminator.Crash("FORMAT lacks initial '('");
    } else if (ch == ')') {
      if (height_ == 1 && stop) {
        return 0;  // end of FORMAT and no data items remain
      }
      if (stack_[height_ - 1].remaining == Iteration::unlimited) {
        offset_ = stack_[height_ - 1].start + 1;
        if (offset_ == unlimitedLoopCheck) {
          context_.terminator.Crash(
              "Unlimited repetition in FORMAT lacks data edit descriptors");
        }
      } else if (stack_[height_ - 1].remaining-- > 0) {
        offset_ = stack_[height_ - 1].start + 1;
      } else {
        --height_;
      }
    } else if (ch == '\'' || ch == '"') {
      // Quoted 'character literal'
      CHAR quote{ch};
      auto start{offset_};
      while (offset_ < formatLength_ && format_[offset_] != quote) {
        ++offset_;
      }
      if (offset_ >= formatLength_) {
        context_.terminator.Crash(
            "FORMAT missing closing quote on character literal");
      }
      ++offset_;
      std::size_t chars{
          static_cast<std::size_t>(&format_[offset_] - &format_[start])};
      if (PeekNext() == quote) {
        // subtle: handle doubled quote character in a literal by including
        // the first in the output, then treating the second as the start
        // of another character literal.
      } else {
        --chars;
      }
      HandleCharacterLiteral(context_, format_ + start, chars);
    } else if (ch == 'H') {
      // 9HHOLLERITH
      if (!repeat || *repeat < 1 || offset_ + *repeat > formatLength_) {
        context_.terminator.Crash("Invalid width on Hollerith in FORMAT");
      }
      HandleCharacterLiteral(
          context_, format_ + offset_, static_cast<std::size_t>(*repeat));
      offset_ += *repeat;
    } else if (ch >= 'A' && ch <= 'Z') {
      int start{offset_ - 1};
      CHAR next{Capitalize(PeekNext())};
      if (next < 'A' || next > 'Z') {
        next = '\0';
      }
      if (ch == 'E' ||
          (!next &&
              (ch == 'A' || ch == 'I' || ch == 'B' || ch == 'O' || ch == 'Z' ||
                  ch == 'F' || ch == 'D' || ch == 'G'))) {
        // Data edit descriptor found
        offset_ = start;
        return repeat ? *repeat : 1;
      } else {
        // Control edit descriptor
        if (ch == 'T') {  // Tn, TLn, TRn
          repeat = GetIntField();
        }
        HandleControl(modes_, scale_, context_, static_cast<char>(ch),
            static_cast<char>(next), repeat ? *repeat : 1);
      }
    } else if (ch == '/') {
      if (context_.handleSlash) {
        context_.handleSlash();
      }
    } else {
      context_.terminator.Crash(
          "Invalid character '%c' in FORMAT", static_cast<char>(ch));
    }
  }
}

template<typename CHAR>
void FormatControl<CHAR>::GetNext(DataEdit &edit, int maxRepeat) {

  // TODO: DT editing

  // Return the next data edit descriptor
  int repeat{CueUpNextDataEdit()};
  auto start{offset_};
  edit.descriptor = static_cast<char>(Capitalize(GetNextChar()));
  if (edit.descriptor == 'E') {
    edit.variation = static_cast<char>(Capitalize(PeekNext()));
    if (edit.variation >= 'A' && edit.variation <= 'Z') {
      ++offset_;
    } else {
      edit.variation = '\0';
    }
  } else {
    edit.variation = '\0';
  }

  edit.width = GetIntField();
  edit.modes = modes_;
  if (PeekNext() == '.') {
    ++offset_;
    edit.digits = GetIntField();
    CHAR ch{PeekNext()};
    if (ch == 'e' || ch == 'E' || ch == 'd' || ch == 'D') {
      ++offset_;
      edit.expoDigits = GetIntField();
    } else {
      edit.expoDigits.reset();
    }
  } else {
    edit.digits.reset();
    edit.expoDigits.reset();
  }

  // Handle repeated nonparenthesized edit descriptors
  if (repeat > 1) {
    stack_[height_].start = start;  // after repeat count
    stack_[height_].remaining = repeat;  // full count
    ++height_;
  }
  edit.repeat = 1;
  if (height_ > 1) {
    int start{stack_[height_ - 1].start};
    if (format_[start] != '(') {
      if (stack_[height_ - 1].remaining > maxRepeat) {
        edit.repeat = maxRepeat;
        stack_[height_ - 1].remaining -= maxRepeat;
        offset_ = start;  // repeat same edit descriptor next time
      } else {
        edit.repeat = stack_[height_ - 1].remaining;
        --height_;
      }
    }
  }
}

template<typename CHAR> void FormatControl<CHAR>::FinishOutput() {
  CueUpNextDataEdit(true /* stop at colon or end of FORMAT */);
}

template class FormatControl<char>;
template class FormatControl<char16_t>;
template class FormatControl<char32_t>;
}
