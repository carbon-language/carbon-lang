//===-- runtime/io-stmt.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-stmt.h"
#include "connection.h"
#include "format.h"
#include "memory.h"
#include "tools.h"
#include "unit.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace Fortran::runtime::io {

int IoStatementBase::EndIoStatement() { return GetIoStat(); }

DataEdit IoStatementBase::GetNextDataEdit(int) {
  Crash("IoStatementBase::GetNextDataEdit() called for non-formatted I/O "
        "statement");
}

template<bool isInput, typename CHAR>
InternalIoStatementState<isInput, CHAR>::InternalIoStatementState(
    Buffer scalar, std::size_t length, const char *sourceFile, int sourceLine)
  : IoStatementBase{sourceFile, sourceLine}, unit_{scalar, length} {}

template<bool isInput, typename CHAR>
InternalIoStatementState<isInput, CHAR>::InternalIoStatementState(
    const Descriptor &d, const char *sourceFile, int sourceLine)
  : IoStatementBase{sourceFile, sourceLine}, unit_{d, *this} {}

template<bool isInput, typename CHAR>
bool InternalIoStatementState<isInput, CHAR>::Emit(
    const CharType *data, std::size_t chars) {
  if constexpr (isInput) {
    Crash("InternalIoStatementState<true>::Emit() called for input statement");
    return false;
  }
  return unit_.Emit(data, chars, *this);
}

template<bool isInput, typename CHAR>
bool InternalIoStatementState<isInput, CHAR>::AdvanceRecord(int n) {
  while (n-- > 0) {
    if (!unit_.AdvanceRecord(*this)) {
      return false;
    }
  }
  return true;
}

template<bool isInput, typename CHAR>
int InternalIoStatementState<isInput, CHAR>::EndIoStatement() {
  if constexpr (!isInput) {
    unit_.EndIoStatement();  // fill
  }
  auto result{IoStatementBase::EndIoStatement()};
  if (free_) {
    FreeMemory(this);
  }
  return result;
}

template<bool isInput, typename CHAR>
InternalFormattedIoStatementState<isInput,
    CHAR>::InternalFormattedIoStatementState(Buffer buffer, std::size_t length,
    const CHAR *format, std::size_t formatLength, const char *sourceFile,
    int sourceLine)
  : InternalIoStatementState<isInput, CHAR>{buffer, length, sourceFile,
        sourceLine},
    ioStatementState_{*this}, format_{*this, format, formatLength} {}

template<bool isInput, typename CHAR>
InternalFormattedIoStatementState<isInput,
    CHAR>::InternalFormattedIoStatementState(const Descriptor &d,
    const CHAR *format, std::size_t formatLength, const char *sourceFile,
    int sourceLine)
  : InternalIoStatementState<isInput, CHAR>{d, sourceFile, sourceLine},
    ioStatementState_{*this}, format_{*this, format, formatLength} {}

template<bool isInput, typename CHAR>
int InternalFormattedIoStatementState<isInput, CHAR>::EndIoStatement() {
  if constexpr (!isInput) {
    format_.FinishOutput(*this);
  }
  return InternalIoStatementState<isInput, CHAR>::EndIoStatement();
}

template<bool isInput, typename CHAR>
bool InternalFormattedIoStatementState<isInput, CHAR>::HandleAbsolutePosition(
    std::int64_t n) {
  return unit_.HandleAbsolutePosition(n, *this);
}

template<bool isInput, typename CHAR>
bool InternalFormattedIoStatementState<isInput, CHAR>::HandleRelativePosition(
    std::int64_t n) {
  return unit_.HandleRelativePosition(n, *this);
}

template<bool isInput, typename CHAR>
InternalListIoStatementState<isInput, CHAR>::InternalListIoStatementState(
    Buffer buffer, std::size_t length, const char *sourceFile, int sourceLine)
  : InternalIoStatementState<isInput, CharType>{buffer, length, sourceFile,
        sourceLine},
    ioStatementState_{*this} {}

template<bool isInput, typename CHAR>
InternalListIoStatementState<isInput, CHAR>::InternalListIoStatementState(
    const Descriptor &d, const char *sourceFile, int sourceLine)
  : InternalIoStatementState<isInput, CharType>{d, sourceFile, sourceLine},
    ioStatementState_{*this} {}

ExternalIoStatementBase::ExternalIoStatementBase(
    ExternalFileUnit &unit, const char *sourceFile, int sourceLine)
  : IoStatementBase{sourceFile, sourceLine}, unit_{unit} {}

MutableModes &ExternalIoStatementBase::mutableModes() { return unit_.modes; }

ConnectionState &ExternalIoStatementBase::GetConnectionState() { return unit_; }

int ExternalIoStatementBase::EndIoStatement() {
  if (unit_.nonAdvancing) {
    unit_.leftTabLimit = unit_.furthestPositionInRecord;
    unit_.nonAdvancing = false;
  } else {
    unit_.leftTabLimit.reset();
  }
  auto result{IoStatementBase::EndIoStatement()};
  unit_.EndIoStatement();  // annihilates *this in unit_.u_
  return result;
}

void OpenStatementState::set_path(
    const char *path, std::size_t length, int kind) {
  if (kind != 1) {  // TODO
    Crash("OPEN: FILE= with unimplemented: CHARACTER(KIND=%d)", kind);
  }
  std::size_t bytes{length * kind};  // TODO: UTF-8 encoding of Unicode path
  path_ = SaveDefaultCharacter(path, bytes, *this);
  pathLength_ = length;
}

int OpenStatementState::EndIoStatement() {
  if (wasExtant_ && status_ != OpenStatus::Old) {
    Crash("OPEN statement for connected unit must have STATUS='OLD'");
  }
  unit().OpenUnit(status_, position_, std::move(path_), pathLength_, *this);
  return IoStatementBase::EndIoStatement();
}

int CloseStatementState::EndIoStatement() {
  unit().CloseUnit(status_, *this);
  return IoStatementBase::EndIoStatement();
}

int NoopCloseStatementState::EndIoStatement() {
  auto result{IoStatementBase::EndIoStatement()};
  FreeMemory(this);
  return result;
}

template<bool isInput> int ExternalIoStatementState<isInput>::EndIoStatement() {
  if constexpr (!isInput) {
    if (!unit().nonAdvancing) {
      unit().AdvanceRecord(*this);
    }
    unit().FlushIfTerminal(*this);
  }
  return ExternalIoStatementBase::EndIoStatement();
}

template<bool isInput>
bool ExternalIoStatementState<isInput>::Emit(
    const char *data, std::size_t chars) {
  if (isInput) {
    Crash("ExternalIoStatementState::Emit called for input statement");
  }
  return unit().Emit(data, chars * sizeof(*data), *this);
}

template<bool isInput>
bool ExternalIoStatementState<isInput>::Emit(
    const char16_t *data, std::size_t chars) {
  if (isInput) {
    Crash("ExternalIoStatementState::Emit called for input statement");
  }
  // TODO: UTF-8 encoding
  return unit().Emit(
      reinterpret_cast<const char *>(data), chars * sizeof(*data), *this);
}

template<bool isInput>
bool ExternalIoStatementState<isInput>::Emit(
    const char32_t *data, std::size_t chars) {
  if (isInput) {
    Crash("ExternalIoStatementState::Emit called for input statement");
  }
  // TODO: UTF-8 encoding
  return unit().Emit(
      reinterpret_cast<const char *>(data), chars * sizeof(*data), *this);
}

template<bool isInput>
bool ExternalIoStatementState<isInput>::AdvanceRecord(int n) {
  while (n-- > 0) {
    if (!unit().AdvanceRecord(*this)) {
      return false;
    }
  }
  return true;
}

template<bool isInput>
bool ExternalIoStatementState<isInput>::HandleAbsolutePosition(std::int64_t n) {
  return unit().HandleAbsolutePosition(n, *this);
}

template<bool isInput>
bool ExternalIoStatementState<isInput>::HandleRelativePosition(std::int64_t n) {
  return unit().HandleRelativePosition(n, *this);
}

template<bool isInput, typename CHAR>
ExternalFormattedIoStatementState<isInput,
    CHAR>::ExternalFormattedIoStatementState(ExternalFileUnit &unit,
    const CHAR *format, std::size_t formatLength, const char *sourceFile,
    int sourceLine)
  : ExternalIoStatementState<isInput>{unit, sourceFile, sourceLine},
    mutableModes_{unit.modes}, format_{*this, format, formatLength} {}

template<bool isInput, typename CHAR>
int ExternalFormattedIoStatementState<isInput, CHAR>::EndIoStatement() {
  format_.FinishOutput(*this);
  return ExternalIoStatementState<isInput>::EndIoStatement();
}

DataEdit IoStatementState::GetNextDataEdit(int n) {
  return std::visit([&](auto &x) { return x.get().GetNextDataEdit(n); }, u_);
}

bool IoStatementState::Emit(const char *data, std::size_t n) {
  return std::visit([=](auto &x) { return x.get().Emit(data, n); }, u_);
}

bool IoStatementState::AdvanceRecord(int n) {
  return std::visit([=](auto &x) { return x.get().AdvanceRecord(n); }, u_);
}

int IoStatementState::EndIoStatement() {
  return std::visit([](auto &x) { return x.get().EndIoStatement(); }, u_);
}

ConnectionState &IoStatementState::GetConnectionState() {
  return std::visit(
      [](auto &x) -> ConnectionState & { return x.get().GetConnectionState(); },
      u_);
}

MutableModes &IoStatementState::mutableModes() {
  return std::visit(
      [](auto &x) -> MutableModes & { return x.get().mutableModes(); }, u_);
}

IoErrorHandler &IoStatementState::GetIoErrorHandler() const {
  return std::visit(
      [](auto &x) -> IoErrorHandler & {
        return static_cast<IoErrorHandler &>(x.get());
      },
      u_);
}

bool IoStatementState::EmitRepeated(char ch, std::size_t n) {
  return std::visit(
      [=](auto &x) {
        for (std::size_t j{0}; j < n; ++j) {
          if (!x.get().Emit(&ch, 1)) {
            return false;
          }
        }
        return true;
      },
      u_);
}

bool IoStatementState::EmitField(
    const char *p, std::size_t length, std::size_t width) {
  if (width <= 0) {
    width = static_cast<int>(length);
  }
  if (length > static_cast<std::size_t>(width)) {
    return EmitRepeated('*', width);
  } else {
    return EmitRepeated(' ', static_cast<int>(width - length)) &&
        Emit(p, length);
  }
}

bool ListDirectedStatementState<false>::NeedAdvance(
    const ConnectionState &connection, std::size_t width) const {
  return connection.positionInRecord > 0 &&
      width > connection.RemainingSpaceInRecord();
}

bool ListDirectedStatementState<false>::EmitLeadingSpaceOrAdvance(
    IoStatementState &io, std::size_t length, bool isCharacter) {
  if (length == 0) {
    return true;
  }
  const ConnectionState &connection{io.GetConnectionState()};
  int space{connection.positionInRecord == 0 ||
      !(isCharacter && lastWasUndelimitedCharacter)};
  lastWasUndelimitedCharacter = false;
  if (NeedAdvance(connection, space + length)) {
    return io.AdvanceRecord();
  }
  if (space) {
    return io.Emit(" ", 1);
  }
  return true;
}

template<bool isInput>
int UnformattedIoStatementState<isInput>::EndIoStatement() {
  auto &ext{static_cast<ExternalIoStatementState<isInput> &>(*this)};
  ExternalFileUnit &unit{ext.unit()};
  if (unit.access == Access::Sequential && !unit.recordLength.has_value()) {
    // Overwrite the first four bytes of the record with its length,
    // and also append the length.  These four bytes were skipped over
    // in BeginUnformattedOutput().
    // TODO: Break very large records up into subrecords with negative
    // headers &/or footers
    union {
      std::uint32_t u;
      char c[sizeof u];
    } u;
    u.u = unit.furthestPositionInRecord - sizeof u.c;
    // TODO: Convert record length to little-endian on big-endian host?
    if (!(ext.Emit(u.c, sizeof u.c) && ext.HandleAbsolutePosition(0) &&
            ext.Emit(u.c, sizeof u.c) && ext.AdvanceRecord())) {
      return false;
    }
  }
  return ext.EndIoStatement();
}

template class InternalIoStatementState<false>;
template class InternalIoStatementState<true>;
template class InternalFormattedIoStatementState<false>;
template class InternalFormattedIoStatementState<true>;
template class InternalListIoStatementState<false>;
template class ExternalIoStatementState<false>;
template class ExternalFormattedIoStatementState<false>;
template class ExternalListIoStatementState<false>;
template class UnformattedIoStatementState<false>;
}
