//===-- runtime/file.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"
#include "magic-numbers.h"
#include "memory.h"
#include "tools.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

namespace Fortran::runtime::io {

void OpenFile::Open(const char *path, std::size_t pathLength,
    const char *status, std::size_t statusLength, const char *action,
    std::size_t actionLength, IoErrorHandler &handler) {
  CriticalSection criticalSection{lock_};
  RUNTIME_CHECK(handler, fd_ < 0);  // TODO handle re-openings
  int flags{0};
  static const char *actions[]{"READ", "WRITE", "READWRITE", nullptr};
  switch (IdentifyValue(action, actionLength, actions)) {
  case 0:
    flags = O_RDONLY;
    mayRead_ = true;
    mayWrite_ = false;
    break;
  case 1:
    flags = O_WRONLY;
    mayRead_ = false;
    mayWrite_ = true;
    break;
  case 2:
    mayRead_ = true;
    mayWrite_ = true;
    flags = O_RDWR;
    break;
  default:
    handler.Crash(
        "Invalid ACTION='%.*s'", action, static_cast<int>(actionLength));
  }
  if (!status) {
    status = "UNKNOWN", statusLength = 7;
  }
  static const char *statuses[]{
      "OLD", "NEW", "SCRATCH", "REPLACE", "UNKNOWN", nullptr};
  switch (IdentifyValue(status, statusLength, statuses)) {
  case 0:  // STATUS='OLD'
    if (!path && fd_ >= 0) {
      // TODO: Update OpenFile in situ; can ACTION be changed?
      return;
    }
    break;
  case 1:  // STATUS='NEW'
    flags |= O_CREAT | O_EXCL;
    break;
  case 2:  // STATUS='SCRATCH'
    if (path_.get()) {
      handler.Crash("FILE= must not appear with STATUS='SCRATCH'");
      path_.reset();
    }
    {
      char path[]{"/tmp/Fortran-Scratch-XXXXXX"};
      fd_ = ::mkstemp(path);
      if (fd_ < 0) {
        handler.SignalErrno();
      }
      ::unlink(path);
    }
    return;
  case 3:  // STATUS='REPLACE'
    flags |= O_CREAT | O_TRUNC;
    break;
  case 4:  // STATUS='UNKNOWN'
    if (fd_ >= 0) {
      return;
    }
    flags |= O_CREAT;
    break;
  default:
    handler.Crash(
        "Invalid STATUS='%.*s'", status, static_cast<int>(statusLength));
  }
  // If we reach this point, we're opening a new file
  if (fd_ >= 0) {
    if (::close(fd_) != 0) {
      handler.SignalErrno();
    }
  }
  path_ = SaveDefaultCharacter(path, pathLength, handler);
  pathLength_ = pathLength;
  if (!path_.get()) {
    handler.Crash(
        "FILE= is required unless STATUS='OLD' and unit is connected");
  }
  fd_ = ::open(path_.get(), flags, 0600);
  if (fd_ < 0) {
    handler.SignalErrno();
  }
  pending_.reset();
  knownSize_.reset();
}

void OpenFile::Predefine(int fd) {
  CriticalSection criticalSection{lock_};
  fd_ = fd;
  path_.reset();
  pathLength_ = 0;
  position_ = 0;
  knownSize_.reset();
  nextId_ = 0;
  pending_.reset();
}

void OpenFile::Close(
    const char *status, std::size_t statusLength, IoErrorHandler &handler) {
  CriticalSection criticalSection{lock_};
  CheckOpen(handler);
  pending_.reset();
  knownSize_.reset();
  static const char *statuses[]{"KEEP", "DELETE", nullptr};
  switch (IdentifyValue(status, statusLength, statuses)) {
  case 0: break;
  case 1:
    if (path_.get()) {
      ::unlink(path_.get());
    }
    break;
  default:
    if (status) {
      handler.Crash(
          "Invalid STATUS='%.*s'", status, static_cast<int>(statusLength));
    }
  }
  path_.reset();
  if (fd_ >= 0) {
    if (::close(fd_) != 0) {
      handler.SignalErrno();
    }
    fd_ = -1;
  }
}

std::size_t OpenFile::Read(FileOffset at, char *buffer, std::size_t minBytes,
    std::size_t maxBytes, IoErrorHandler &handler) {
  if (maxBytes == 0) {
    return 0;
  }
  CriticalSection criticalSection{lock_};
  CheckOpen(handler);
  if (!Seek(at, handler)) {
    return 0;
  }
  if (maxBytes < minBytes) {
    minBytes = maxBytes;
  }
  std::size_t got{0};
  while (got < minBytes) {
    auto chunk{::read(fd_, buffer + got, maxBytes - got)};
    if (chunk == 0) {
      handler.SignalEnd();
      break;
    }
    if (chunk < 0) {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        handler.SignalError(err);
        break;
      }
    } else {
      position_ += chunk;
      got += chunk;
    }
  }
  return got;
}

std::size_t OpenFile::Write(FileOffset at, const char *buffer,
    std::size_t bytes, IoErrorHandler &handler) {
  if (bytes == 0) {
    return 0;
  }
  CriticalSection criticalSection{lock_};
  CheckOpen(handler);
  if (!Seek(at, handler)) {
    return 0;
  }
  std::size_t put{0};
  while (put < bytes) {
    auto chunk{::write(fd_, buffer + put, bytes - put)};
    if (chunk >= 0) {
      position_ += chunk;
      put += chunk;
    } else {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        handler.SignalError(err);
        break;
      }
    }
  }
  if (knownSize_ && position_ > *knownSize_) {
    knownSize_ = position_;
  }
  return put;
}

void OpenFile::Truncate(FileOffset at, IoErrorHandler &handler) {
  CriticalSection criticalSection{lock_};
  CheckOpen(handler);
  if (!knownSize_ || *knownSize_ != at) {
    if (::ftruncate(fd_, at) != 0) {
      handler.SignalErrno();
    }
    knownSize_ = at;
  }
}

// The operation is performed immediately; the results are saved
// to be claimed by a later WAIT statement.
// TODO: True asynchronicity
int OpenFile::ReadAsynchronously(
    FileOffset at, char *buffer, std::size_t bytes, IoErrorHandler &handler) {
  CriticalSection criticalSection{lock_};
  CheckOpen(handler);
  int iostat{0};
  for (std::size_t got{0}; got < bytes;) {
#if _XOPEN_SOURCE >= 500 || _POSIX_C_SOURCE >= 200809L
    auto chunk{::pread(fd_, buffer + got, bytes - got, at)};
#else
    auto chunk{Seek(at, handler) ? ::read(fd_, buffer + got, bytes - got) : -1};
#endif
    if (chunk == 0) {
      iostat = FORTRAN_RUNTIME_IOSTAT_END;
      break;
    }
    if (chunk < 0) {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        iostat = err;
        break;
      }
    } else {
      at += chunk;
      got += chunk;
    }
  }
  return PendingResult(handler, iostat);
}

// TODO: True asynchronicity
int OpenFile::WriteAsynchronously(FileOffset at, const char *buffer,
    std::size_t bytes, IoErrorHandler &handler) {
  CriticalSection criticalSection{lock_};
  CheckOpen(handler);
  int iostat{0};
  for (std::size_t put{0}; put < bytes;) {
#if _XOPEN_SOURCE >= 500 || _POSIX_C_SOURCE >= 200809L
    auto chunk{::pwrite(fd_, buffer + put, bytes - put, at)};
#else
    auto chunk{
        Seek(at, handler) ? ::write(fd_, buffer + put, bytes - put) : -1};
#endif
    if (chunk >= 0) {
      at += chunk;
      put += chunk;
    } else {
      auto err{errno};
      if (err != EAGAIN && err != EWOULDBLOCK && err != EINTR) {
        iostat = err;
        break;
      }
    }
  }
  return PendingResult(handler, iostat);
}

void OpenFile::Wait(int id, IoErrorHandler &handler) {
  std::optional<int> ioStat;
  {
    CriticalSection criticalSection{lock_};
    Pending *prev{nullptr};
    for (Pending *p{pending_.get()}; p; p = (prev = p)->next.get()) {
      if (p->id == id) {
        ioStat = p->ioStat;
        if (prev) {
          prev->next.reset(p->next.release());
        } else {
          pending_.reset(p->next.release());
        }
        break;
      }
    }
  }
  if (ioStat) {
    handler.SignalError(*ioStat);
  }
}

void OpenFile::WaitAll(IoErrorHandler &handler) {
  while (true) {
    int ioStat;
    {
      CriticalSection criticalSection{lock_};
      if (pending_) {
        ioStat = pending_->ioStat;
        pending_.reset(pending_->next.release());
      } else {
        return;
      }
    }
    handler.SignalError(ioStat);
  }
}

void OpenFile::CheckOpen(Terminator &terminator) {
  RUNTIME_CHECK(terminator, fd_ >= 0);
}

bool OpenFile::Seek(FileOffset at, IoErrorHandler &handler) {
  if (at == position_) {
    return true;
  } else if (RawSeek(at)) {
    position_ = at;
    return true;
  } else {
    handler.SignalErrno();
    return false;
  }
}

bool OpenFile::RawSeek(FileOffset at) {
#ifdef _LARGEFILE64_SOURCE
  return ::lseek64(fd_, at, SEEK_SET) == 0;
#else
  return ::lseek(fd_, at, SEEK_SET) == 0;
#endif
}

int OpenFile::PendingResult(Terminator &terminator, int iostat) {
  int id{nextId_++};
  pending_.reset(&New<Pending>{}(terminator, id, iostat, std::move(pending_)));
  return id;
}
}
