//===-- runtime/io-stmt.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Represents state of an I/O statement in progress

#ifndef FORTRAN_RUNTIME_IO_STMT_H_
#define FORTRAN_RUNTIME_IO_STMT_H_

#include "descriptor.h"
#include "format.h"
#include "io-error.h"
#include <type_traits>

namespace Fortran::runtime::io {

class ExternalFile;

class IoStatementState : public IoErrorHandler, public FormatContext {
public:
  IoStatementState(const char *sourceFile, int sourceLine);
  virtual ~IoStatementState() {}

  virtual int EndIoStatement();

  // Default (crashing) callback overrides for FormatContext
  virtual void GetNext(DataEdit &, int maxRepeat = 1);
  virtual bool Emit(const char *, std::size_t);
  virtual bool Emit(const char16_t *, std::size_t);
  virtual bool Emit(const char32_t *, std::size_t);
  virtual bool HandleSlash(int);
  virtual bool HandleRelativePosition(std::int64_t);
  virtual bool HandleAbsolutePosition(std::int64_t);
};

template<bool IsInput, typename CHAR = char>
class FixedRecordIoStatementState : public IoStatementState {
protected:
  using Buffer = std::conditional_t<IsInput, const CHAR *, CHAR *>;

public:
  FixedRecordIoStatementState(
      Buffer, std::size_t, const char *sourceFile, int sourceLine);

  virtual bool Emit(const CHAR *, std::size_t chars /* not bytes */);
  // TODO virtual void HandleSlash(int);
  virtual bool HandleRelativePosition(std::int64_t);
  virtual bool HandleAbsolutePosition(std::int64_t);
  virtual int EndIoStatement();

private:
  Buffer buffer_{nullptr};
  std::size_t length_;  // RECL= or internal I/O character variable length
  std::size_t leftTabLimit_{0};  // nonzero only when non-advancing
  std::size_t at_{0};
  std::size_t furthest_{0};
};

template<bool isInput, typename CHAR = char>
class InternalIoStatementState
  : public FixedRecordIoStatementState<isInput, CHAR> {
public:
  using typename FixedRecordIoStatementState<isInput, CHAR>::Buffer;
  InternalIoStatementState(Buffer, std::size_t,
      const char *sourceFile = nullptr, int sourceLine = 0);
  virtual int EndIoStatement();

protected:
  bool free_{true};
};

template<bool isInput, typename CHAR = char>
class InternalFormattedIoStatementState
  : public InternalIoStatementState<isInput, CHAR> {
public:
  using typename InternalIoStatementState<isInput, CHAR>::Buffer;
  InternalFormattedIoStatementState(Buffer internal, std::size_t internalLength,
      const CHAR *format, std::size_t formatLength,
      const char *sourceFile = nullptr, int sourceLine = 0);
  void GetNext(DataEdit &edit, int maxRepeat = 1) {
    format_.GetNext(*this, edit, maxRepeat);
  }
  int EndIoStatement();

private:
  FormatControl<CHAR> format_;  // must be last, may be partial
};

template<bool isInput, typename CHAR = char>
class ExternalFormattedIoStatementState : public IoStatementState {
public:
  ExternalFormattedIoStatementState(ExternalFile &, const CHAR *format,
      std::size_t formatLength, const char *sourceFile = nullptr,
      int sourceLine = 0);
  void GetNext(DataEdit &edit, int maxRepeat = 1) {
    format_.GetNext(*this, edit, maxRepeat);
  }
  bool Emit(const CHAR *, std::size_t chars /* not bytes */);
  bool HandleSlash(int);
  bool HandleRelativePosition(std::int64_t);
  bool HandleAbsolutePosition(std::int64_t);
  int EndIoStatement();

private:
  ExternalFile &file_;
  FormatControl<CHAR> format_;
};

extern template class InternalFormattedIoStatementState<false>;
extern template class ExternalFormattedIoStatementState<false>;

}
#endif  // FORTRAN_RUNTIME_IO_STMT_H_
