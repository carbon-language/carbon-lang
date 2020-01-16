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

class IoStatementState : public IoErrorHandler {
public:
  using IoErrorHandler::IoErrorHandler;
  virtual int EndIoStatement();

protected:
};

class InternalIoStatementState : public IoStatementState {
public:
  InternalIoStatementState(const char *sourceFile, int sourceLine);
  virtual int EndIoStatement();

protected:
  bool free_{true};
};

template<bool IsInput, typename CHAR = char>
class InternalFormattedIoStatementState : public InternalIoStatementState,
                                          private FormatContext {
private:
  using Buffer = std::conditional_t<IsInput, const CHAR *, CHAR *>;

public:
  InternalFormattedIoStatementState(Buffer internal, std::size_t internalLength,
      const CHAR *format, std::size_t formatLength,
      const char *sourceFile = nullptr, int sourceLine = 0);
  void Emit(const CHAR *, std::size_t chars);
  // TODO pmk: void HandleSlash(int);
  void HandleRelativePosition(int);
  void HandleAbsolutePosition(int);
  int EndIoStatement();

private:
  Buffer internal_;
  std::size_t internalLength_;
  std::size_t at_{0};
  FormatControl<CHAR> format_;  // must be last, may be partial
};

extern template class InternalFormattedIoStatementState<false>;

}
#endif  // FORTRAN_RUNTIME_IO_STMT_H_
