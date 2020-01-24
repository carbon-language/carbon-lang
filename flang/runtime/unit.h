//===-- runtime/unit.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fortran I/O units

#ifndef FORTRAN_RUNTIME_IO_UNIT_H_
#define FORTRAN_RUNTIME_IO_UNIT_H_

#include "buffer.h"
#include "descriptor.h"
#include "file.h"
#include "format.h"
#include "io-error.h"
#include "io-stmt.h"
#include "lock.h"
#include "memory.h"
#include "terminator.h"
#include <cstdlib>
#include <cstring>
#include <optional>
#include <variant>

namespace Fortran::runtime::io {

enum class Access { Sequential, Direct, Stream };

inline bool IsRecordFile(Access a) { return a != Access::Stream; }

// These characteristics of a connection are immutable after being
// established in an OPEN statement.
struct ConnectionAttributes {
  Access access{Access::Sequential};  // ACCESS='SEQUENTIAL', 'DIRECT', 'STREAM'
  std::optional<std::int64_t> recordLength;  // RECL= when fixed-length
  bool unformatted{false};  // FORM='UNFORMATTED'
  bool isUTF8{false};  // ENCODING='UTF-8'
  bool asynchronousAllowed{false};  // ASYNCHRONOUS='YES'
};

struct ConnectionState : public ConnectionAttributes {
  // Positions in a record file (sequential or direct, but not stream)
  std::int64_t recordOffsetInFile{0};
  std::int64_t currentRecordNumber{1};  // 1 is first
  std::int64_t positionInRecord{0};  // offset in current record
  std::int64_t furthestPositionInRecord{0};  // max(positionInRecord)
  std::optional<std::int64_t> leftTabLimit;  // offset in current record
  // nextRecord value captured after ENDFILE/REWIND/BACKSPACE statement
  // on a sequential access file
  std::optional<std::int64_t> endfileRecordNumber;
  // Mutable modes set at OPEN() that can be overridden in READ/WRITE & FORMAT
  MutableModes modes;  // BLANK=, DECIMAL=, SIGN=, ROUND=, PAD=, DELIM=, kP
};

class InternalUnit : public ConnectionState, public IoErrorHandler {
public:
  InternalUnit(Descriptor &, const char *sourceFile, int sourceLine)
    : IoErrorHandler{sourceFile, sourceLine} {
// TODO pmk    descriptor_.Establish(...);
    descriptor_.GetLowerBounds(at_);
    recordLength = descriptor_.ElementBytes();
    endfileRecordNumber = descriptor_.Elements();
  }
  ~InternalUnit() {
    if (!doNotFree_) {
      std::free(this);
    }
  }

private:
  bool doNotFree_{false};
  Descriptor descriptor_;
  SubscriptValue at_[maxRank];
};

class ExternalFile : public ConnectionState,  // TODO: privatize these
                     public OpenFile,
                     public FileFrame<ExternalFile> {
public:
  explicit ExternalFile(int unitNumber) : unitNumber_{unitNumber} {}
  static ExternalFile *LookUp(int unit);
  static ExternalFile &LookUpOrCrash(int unit, Terminator &);
  static ExternalFile &Create(int unit, Terminator &);
  static void InitializePredefinedUnits(Terminator &);
  static void CloseAll(IoErrorHandler &);

  void CloseUnit(IoErrorHandler &);

  // TODO: accessors & mutators for many OPEN() specifiers
  template<typename A, typename... X> A &BeginIoStatement(X&&... xs) {
    // TODO: lock_.Take() here, and keep it until EndIoStatement()?
    // Nested I/O from derived types wouldn't work, though.
    return u_.emplace<A>(std::forward<X>(xs)...);
  }
  void EndIoStatement();

  bool SetPositionInRecord(std::int64_t, IoErrorHandler &);
  bool Emit(const char *, std::size_t bytes, IoErrorHandler &);
  void SetLeftTabLimit();
  bool NextOutputRecord(IoErrorHandler &);
  bool HandleAbsolutePosition(std::int64_t, IoErrorHandler &);
  bool HandleRelativePosition(std::int64_t, IoErrorHandler &);
private:
  int unitNumber_{-1};
  Lock lock_;
  bool isReading_{false};
  std::variant<std::monostate, ExternalFormattedIoStatementState<false>> u_;
};

}
#endif  // FORTRAN_RUNTIME_IO_UNIT_H_
