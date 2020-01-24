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

#include "environment.h"
#include "terminator.h"
#include "flang/common/Fortran.h"
#include <cinttypes>
#include <optional>

namespace Fortran::runtime::io {

enum EditingFlags {
  blankZero = 1,  // BLANK=ZERO or BZ edit
  decimalComma = 2,  // DECIMAL=COMMA or DC edit
  signPlus = 4,  // SIGN=PLUS or SP edit
};

struct MutableModes {
  std::uint8_t editingFlags{0};  // BN, DP, SS
  common::RoundingMode roundingMode{
      executionEnvironment
          .defaultOutputRoundingMode};  // RP/ROUND='PROCESSOR_DEFAULT'
  bool pad{false};  // PAD= mode on READ
  char delim{'\0'};  // DELIM=
  short scale{0};  // kP
};

// A single edit descriptor extracted from a FORMAT
struct DataEdit {
  char descriptor;  // capitalized: one of A, I, B, O, Z, F, E(N/S/X), D, G
  char variation{'\0'};  // N, S, or X for EN, ES, EX
  std::optional<int> width;  // the 'w' field; optional for A
  std::optional<int> digits;  // the 'm' or 'd' field
  std::optional<int> expoDigits;  // 'Ee' field
  MutableModes modes;
  int repeat{1};
};

class FormatContext : virtual public Terminator {
public:
  FormatContext() {}
  virtual ~FormatContext() {}
  explicit FormatContext(const MutableModes &modes) : mutableModes_{modes} {}
  virtual bool Emit(const char *, std::size_t) = 0;
  virtual bool Emit(const char16_t *, std::size_t) = 0;
  virtual bool Emit(const char32_t *, std::size_t) = 0;
  virtual bool HandleSlash(int = 1) = 0;
  virtual bool HandleRelativePosition(std::int64_t) = 0;
  virtual bool HandleAbsolutePosition(std::int64_t) = 0;
  MutableModes &mutableModes() { return mutableModes_; }

private:
  MutableModes mutableModes_;
};

// Generates a sequence of DataEdits from a FORMAT statement or
// default-CHARACTER string.  Driven by I/O item list processing.
// Errors are fatal.  See clause 13.4 in Fortran 2018 for background.
template<typename CHAR = char> class FormatControl {
public:
  FormatControl() {}
  // TODO: make 'format' a reference here and below
  FormatControl(Terminator &, const CHAR *format, std::size_t formatLength,
      int maxHeight = maxMaxHeight);

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
  void GetNext(FormatContext &, DataEdit &, int maxRepeat = 1);

  // Emit any remaining character literals after the last data item.
  void FinishOutput(FormatContext &);

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
  CHAR GetNextChar(Terminator &terminator) {
    SkipBlanks();
    if (offset_ >= formatLength_) {
      terminator.Crash("FORMAT missing at least one ')'");
    }
    return format_[offset_++];
  }
  int GetIntField(Terminator &, CHAR firstCh = '\0');

  // Advances through the FORMAT until the next data edit
  // descriptor has been found; handles control edit descriptors
  // along the way.  Returns the repeat count that appeared
  // before the descriptor (defaulting to 1) and leaves offset_
  // pointing to the data edit.
  int CueUpNextDataEdit(FormatContext &, bool stop = false);

  static constexpr CHAR Capitalize(CHAR ch) {
    return ch >= 'a' && ch <= 'z' ? ch + 'A' - 'a' : ch;
  }

  // Data members are arranged and typed so as to reduce size.
  // This structure may be allocated in stack space loaned by the
  // user program for internal I/O.
  const std::uint8_t maxHeight_{maxMaxHeight};
  std::uint8_t height_{0};
  const CHAR *format_{nullptr};
  int formatLength_{0};
  int offset_{0};  // next item is at format_[offset_]

  // must be last, may be incomplete
  Iteration stack_[maxMaxHeight];
};

extern template class FormatControl<char>;
extern template class FormatControl<char16_t>;
extern template class FormatControl<char32_t>;
}
#endif  // FORTRAN_RUNTIME_FORMAT_H_
