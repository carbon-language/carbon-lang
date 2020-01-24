//===-- runtime/io-stmt.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-stmt.h"
#include "memory.h"
#include "unit.h"
#include <algorithm>
#include <cstring>

namespace Fortran::runtime::io {

IoStatementState::IoStatementState(const char *sourceFile, int sourceLine)
  : IoErrorHandler{sourceFile, sourceLine} {}

int IoStatementState::EndIoStatement() { return GetIoStat(); }

// Defaults
void IoStatementState::GetNext(DataEdit &, int) {
  Crash("GetNext() called for I/O statement that is not a formatted data "
        "transfer statement");
}
bool IoStatementState::Emit(const char *, std::size_t) {
  Crash("Emit() called for I/O statement that is not an output statement");
  return false;
}
bool IoStatementState::Emit(const char16_t *, std::size_t) {
  Crash("Emit() called for I/O statement that is not an output statement");
  return false;
}
bool IoStatementState::Emit(const char32_t *, std::size_t) {
  Crash("Emit() called for I/O statement that is not an output statement");
  return false;
}
bool IoStatementState::HandleSlash(int) {
  Crash("HandleSlash() called for I/O statement that is not a formatted data "
        "transfer statement");
  return false;
}
bool IoStatementState::HandleRelativePosition(std::int64_t) {
  Crash("HandleRelativePosition() called for I/O statement that is not a "
        "formatted data transfer statement");
  return false;
}
bool IoStatementState::HandleAbsolutePosition(std::int64_t) {
  Crash("HandleAbsolutePosition() called for I/O statement that is not a "
        "formatted data transfer statement");
  return false;
}

template<bool isInput, typename CHAR>
FixedRecordIoStatementState<isInput, CHAR>::FixedRecordIoStatementState(
    Buffer buffer, std::size_t length, const char *sourceFile, int sourceLine)
  : IoStatementState{sourceFile, sourceLine}, buffer_{buffer}, length_{length} {
}

template<bool isInput, typename CHAR>
bool FixedRecordIoStatementState<isInput, CHAR>::Emit(
    const CHAR *data, std::size_t chars) {
  if constexpr (isInput) {
    IoStatementState::Emit(data, chars);  // default Crash()
    return false;
  } else if (at_ + chars > length_) {
    SignalEor();
    if (at_ < length_) {
      std::memcpy(buffer_ + at_, data, (length_ - at_) * sizeof(CHAR));
      at_ = furthest_ = length_;
    }
    return false;
  } else {
    std::memcpy(buffer_ + at_, data, chars * sizeof(CHAR));
    at_ += chars;
    furthest_ = std::max(furthest_, at_);
    return true;
  }
}

template<bool isInput, typename CHAR>
bool FixedRecordIoStatementState<isInput, CHAR>::HandleAbsolutePosition(
    std::int64_t n) {
  if (n < 0) {
    n = 0;
  }
  n += leftTabLimit_;
  bool ok{true};
  if (static_cast<std::size_t>(n) > length_) {
    SignalEor();
    n = length_;
    ok = false;
  }
  if constexpr (!isInput) {
    if (static_cast<std::size_t>(n) > furthest_) {
      std::fill_n(buffer_ + furthest_, n - furthest_, static_cast<CHAR>(' '));
    }
  }
  at_ = n;
  furthest_ = std::max(furthest_, at_);
  return ok;
}

template<bool isInput, typename CHAR>
bool FixedRecordIoStatementState<isInput, CHAR>::HandleRelativePosition(
    std::int64_t n) {
  return HandleAbsolutePosition(n + at_ - leftTabLimit_);
}

template<bool isInput, typename CHAR>
int FixedRecordIoStatementState<isInput, CHAR>::EndIoStatement() {
  if constexpr (!isInput) {
    HandleAbsolutePosition(length_ - leftTabLimit_);  // fill
  }
  return GetIoStat();
}

template<bool isInput, typename CHAR>
int InternalIoStatementState<isInput, CHAR>::EndIoStatement() {
  auto result{FixedRecordIoStatementState<isInput, CHAR>::EndIoStatement()};
  if (free_) {
    FreeMemory(this);
  }
  return result;
}

template<bool isInput, typename CHAR>
InternalIoStatementState<isInput, CHAR>::InternalIoStatementState(
    Buffer buffer, std::size_t length, const char *sourceFile, int sourceLine)
  : FixedRecordIoStatementState<isInput, CHAR>(
        buffer, length, sourceFile, sourceLine) {}

template<bool isInput, typename CHAR>
InternalFormattedIoStatementState<isInput,
    CHAR>::InternalFormattedIoStatementState(Buffer buffer, std::size_t length,
    const CHAR *format, std::size_t formatLength, const char *sourceFile,
    int sourceLine)
  : InternalIoStatementState<isInput, CHAR>{buffer, length, sourceFile,
        sourceLine},
    format_{*this, format, formatLength} {}

template<bool isInput, typename CHAR>
int InternalFormattedIoStatementState<isInput, CHAR>::EndIoStatement() {
  format_.FinishOutput(*this);
  return InternalIoStatementState<isInput, CHAR>::EndIoStatement();
}

template<bool isInput, typename CHAR>
ExternalFormattedIoStatementState<isInput,
    CHAR>::ExternalFormattedIoStatementState(ExternalFile &file,
    const CHAR *format, std::size_t formatLength, const char *sourceFile,
    int sourceLine)
  : IoStatementState{sourceFile, sourceLine}, file_{file}, format_{*this,
                                                               format,
                                                               formatLength} {}

template<bool isInput, typename CHAR>
bool ExternalFormattedIoStatementState<isInput, CHAR>::Emit(
    const CHAR *data, std::size_t chars) {
  // TODO: UTF-8 encoding of 2- and 4-byte characters
  return file_.Emit(data, chars * sizeof(CHAR), *this);
}

template<bool isInput, typename CHAR>
bool ExternalFormattedIoStatementState<isInput, CHAR>::HandleSlash(int n) {
  while (n-- > 0) {
    if (!file_.NextOutputRecord(*this)) {
      return false;
    }
  }
  return true;
}

template<bool isInput, typename CHAR>
bool ExternalFormattedIoStatementState<isInput, CHAR>::HandleAbsolutePosition(
    std::int64_t n) {
  return file_.HandleAbsolutePosition(n, *this);
}

template<bool isInput, typename CHAR>
bool ExternalFormattedIoStatementState<isInput, CHAR>::HandleRelativePosition(
    std::int64_t n) {
  return file_.HandleRelativePosition(n, *this);
}

template<bool isInput, typename CHAR>
int ExternalFormattedIoStatementState<isInput, CHAR>::EndIoStatement() {
  format_.FinishOutput(*this);
  if constexpr (!isInput) {
    file_.NextOutputRecord(*this);  // TODO: non-advancing I/O
  }
  int result{GetIoStat()};
  file_.EndIoStatement();  // annihilates *this in file_.u_
  return result;
}

template class InternalFormattedIoStatementState<false>;
template class ExternalFormattedIoStatementState<false>;
}
