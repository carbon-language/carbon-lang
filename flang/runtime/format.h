//===-- runtime/format.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// FORMAT string processing

#ifndef FORTRAN_RUNTIME_FORMAT_H_
#define FORTRAN_RUNTIME_FORMAT_H_

#include "terminator.h"
#include "../lib/common/Fortran.h"
#include <cinttypes>
#include <optional>

namespace Fortran::runtime {

enum EditingFlags {
  blankZero = 1,  // BLANK=ZERO or BZ edit
  decimalComma = 2,  // DECIMAL=COMMA or DC edit
  signPlus = 4,  // SIGN=PLUS or SP edit
};

struct MutableModes {
  std::uint8_t editingFlags{0};  // BN, DP, SS
  common::RoundingMode roundingMode{common::RoundingMode::TiesToEven};  // RN
};

// A single edit descriptor extracted from a FORMAT
struct DataEdit {
  char descriptor;  // capitalized: one of A, I, B, O, Z, F, E(N/S/X), D, G
  char variation{'\0'};  // N, S, or X for EN, ES, EX
  int width;  // the 'w' field
  std::optional<int> digits;  // the 'm' or 'd' field
  std::optional<int> expoDigits;  // 'Ee' field
  MutableModes modes;
  int repeat{1};
};

struct FormatContext {
  Terminator &terminator;
  void (*handleCharacterLiteral1)(const char *, std::size_t){nullptr};
  void (*handleCharacterLiteral2)(const char16_t *, std::size_t){nullptr};
  void (*handleCharacterLiteral4)(const char32_t *, std::size_t){nullptr};
  void (*handleSlash)(){nullptr};
  void (*handleAbsolutePosition)(int){nullptr};  // Tn
  void (*handleRelativePosition)(int){nullptr};  // nX, TRn, TLn (negated)
};

// Generates a sequence of DataEdits from a FORMAT statement or
// default-CHARACTER string.  Driven by I/O item list processing.
// Errors are fatal.  See clause 13.4 in Fortran 2018 for background.
template<typename CHAR = char> class FormatControl {
public:
  FormatControl(FormatContext &, const CHAR *format, std::size_t formatLength,
      const MutableModes &initialModes, int maxHeight = maxMaxHeight);

  // Determines the max parenthesis nesting level by scanning and validating
  // the FORMAT string.
  static int GetMaxParenthesisNesting(
      Terminator &, const CHAR *format, std::size_t formatLength);

  // For attempting to allocate in a user-supplied stack area
  static std::size_t GetNeededSize(int maxHeight) {
    return sizeof(FormatControl) -
        sizeof(Iteration) * (maxMaxHeight - maxHeight);
  }

  // Extracts the next data edit descriptor, handling control edit descriptors
  // along the way.
  void GetNext(DataEdit &, int maxRepeat = 1);

  // Emit any remaining character literals after the last data item.
  void FinishOutput();

private:
  static constexpr std::uint8_t maxMaxHeight{100};

  struct Iteration {
    static constexpr int unlimited{-1};
    int start{0};  // offset in format_ of '(' or a repeated edit descriptor
    int remaining{0};  // while >0, decrement and iterate
  };

  void SkipBlanks() {
    while (offset_ < formatLength_ && format_[offset_] == ' ') {
      ++offset_;
    }
  }
  CHAR PeekNext() {
    SkipBlanks();
    return offset_ < formatLength_ ? format_[offset_] : '\0';
  }
  CHAR GetNextChar() {
    SkipBlanks();
    if (offset_ >= formatLength_) {
      context_.terminator.Crash("FORMAT missing at least one ')'");
    }
    return format_[offset_++];
  }
  int GetIntField(CHAR firstCh = '\0');

  // Advances through the FORMAT until the next data edit
  // descriptor has been found; handles control edit descriptors
  // along the way.  Returns the repeat count that appeared
  // before the descriptor (defaulting to 1) and leaves offset_
  // pointing to the data edit.
  int CueUpNextDataEdit(bool stop = false);

  static constexpr CHAR Capitalize(CHAR ch) {
    return ch >= 'a' && ch <= 'z' ? ch + 'A' - 'a' : ch;
  }

  // Data members are arranged and typed so as to reduce size.
  // This structure may be allocated in stack space loaned by the
  // user program for internal I/O.
  FormatContext &context_;
  MutableModes modes_;
  std::uint16_t scale_{0};  // kP
  const std::uint8_t maxHeight_{maxMaxHeight};
  std::uint8_t height_{0};
  const CHAR *format_;
  int formatLength_;
  int offset_{0};  // next item is at format_[offset_]

  // must be last, may be incomplete
  Iteration stack_[maxMaxHeight];
};

extern template class FormatControl<char>;
extern template class FormatControl<char16_t>;
extern template class FormatControl<char32_t>;
}
#endif  // FORTRAN_RUNTIME_FORMAT_H_
