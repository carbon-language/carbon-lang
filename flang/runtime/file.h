//===-- runtime/file.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Raw system I/O wrappers

#ifndef FORTRAN_RUNTIME_FILE_H_
#define FORTRAN_RUNTIME_FILE_H_

#include "io-error.h"
#include "lock.h"
#include "memory.h"
#include "terminator.h"
#include <cinttypes>
#include <optional>

namespace Fortran::runtime::io {

class OpenFile {
public:
  using Offset = std::uint64_t;

  Offset position() const { return position_; }

  void Open(const char *path, std::size_t pathLength, const char *status,
      std::size_t statusLength, const char *action, std::size_t actionLength,
      IoErrorHandler &);
  void Close(const char *action, std::size_t actionLength, IoErrorHandler &);

  // Reads data into memory; returns amount acquired.  Synchronous.
  // Partial reads (less than minBytes) signify end-of-file.  If the
  // buffer is larger than minBytes, and extra returned data will be
  // preserved for future consumption, set maxBytes larger than minBytes
  // to reduce system calls  This routine handles EAGAIN/EWOULDBLOCK and EINTR.
  std::size_t Read(Offset, char *, std::size_t minBytes, std::size_t maxBytes,
      IoErrorHandler &);

  // Writes data.  Synchronous.  Partial writes indicate program-handled
  // error conditions.
  std::size_t Write(Offset, const char *, std::size_t, IoErrorHandler &);

  // Truncates the file
  void Truncate(Offset, IoErrorHandler &);

  // Asynchronous transfers
  int ReadAsynchronously(Offset, char *, std::size_t, IoErrorHandler &);
  int WriteAsynchronously(Offset, const char *, std::size_t, IoErrorHandler &);
  void Wait(int id, IoErrorHandler &);
  void WaitAll(IoErrorHandler &);

private:
  struct Pending {
    int id;
    int ioStat{0};
    OwningPtr<Pending> next;
  };

  // lock_ must be held for these
  void CheckOpen(Terminator &);
  bool Seek(Offset, IoErrorHandler &);
  bool RawSeek(Offset);
  int PendingResult(Terminator &, int);

  Lock lock_;
  int fd_{-1};
  OwningPtr<char> path_;
  Offset position_{0};
  std::optional<Offset> knownSize_;
  int nextId_;
  OwningPtr<Pending> pending_;
};
}
#endif  // FORTRAN_RUNTIME_FILE_H_
