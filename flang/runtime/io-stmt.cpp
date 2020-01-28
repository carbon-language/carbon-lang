//===-- runtime/io-stmt.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-stmt.h"
#include "memory.h"
#include <algorithm>
#include <cstring>

namespace Fortran::runtime::io {

int IoStatementState::EndIoStatement() { return GetIoStat(); }

int InternalIoStatementState::EndIoStatement() {
  auto result{GetIoStat()};
  if (free_) {
    FreeMemory(this);
  }
  return result;
}

InternalIoStatementState::InternalIoStatementState(
    const char *sourceFile, int sourceLine)
  : IoStatementState(sourceFile, sourceLine) {}

template<bool isInput, typename CHAR>
InternalFormattedIoStatementState<isInput,
    CHAR>::InternalFormattedIoStatementState(Buffer internal,
    std::size_t internalLength, const CHAR *format, std::size_t formatLength,
    const char *sourceFile, int sourceLine)
  : InternalIoStatementState{sourceFile, sourceLine}, FormatContext{},
    internal_{internal}, internalLength_{internalLength}, format_{*this, format,
                                                              formatLength} {
  std::fill_n(internal_, internalLength_, static_cast<CHAR>(' '));
}

template<bool isInput, typename CHAR>
void InternalFormattedIoStatementState<isInput, CHAR>::Emit(
    const CHAR *data, std::size_t chars) {
  if constexpr (isInput) {
    FormatContext::Emit(data, chars);  // default Crash()
  } else if (at_ + chars > internalLength_) {
    SignalEor();
  } else {
    std::memcpy(internal_ + at_, data, chars * sizeof(CHAR));
    at_ += chars;
  }
}

template<bool isInput, typename CHAR>
void InternalFormattedIoStatementState<isInput, CHAR>::HandleAbsolutePosition(
    int n) {
  if (n < 0 || static_cast<std::size_t>(n) >= internalLength_) {
    Crash("T%d control edit descriptor is out of range", n);
  } else {
    at_ = n;
  }
}

template<bool isInput, typename CHAR>
void InternalFormattedIoStatementState<isInput, CHAR>::HandleRelativePosition(
    int n) {
  if (n < 0) {
    at_ -= std::min(at_, -static_cast<std::size_t>(n));
  } else {
    at_ += n;
    if (at_ > internalLength_) {
      Crash("TR%d control edit descriptor is out of range", n);
    }
  }
}

template<bool isInput, typename CHAR>
int InternalFormattedIoStatementState<isInput, CHAR>::EndIoStatement() {
  format_.FinishOutput(*this);
  auto result{GetIoStat()};
  if (free_) {
    FreeMemory(this);
  }
  return result;
}

template class InternalFormattedIoStatementState<false>;
}
