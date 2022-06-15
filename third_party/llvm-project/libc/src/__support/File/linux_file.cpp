//===--- Linux specialization of the File data structure ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "file.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.

#include <errno.h>
#include <fcntl.h> // For mode_t and other flags to the open syscall
#include <stdio.h>
#include <stdlib.h>      // For malloc
#include <sys/syscall.h> // For syscall numbers

namespace __llvm_libc {

namespace {

size_t write_func(File *, const void *, size_t);
size_t read_func(File *, void *, size_t);
int seek_func(File *, long, int);
int close_func(File *);
int flush_func(File *);

} // anonymous namespace

class LinuxFile : public File {
  int fd;

public:
  constexpr LinuxFile(int file_descriptor, void *buffer, size_t buffer_size,
                      int buffer_mode, bool owned, File::ModeFlags modeflags)
      : File(&write_func, &read_func, &seek_func, &close_func, flush_func,
             buffer, buffer_size, buffer_mode, owned, modeflags),
        fd(file_descriptor) {}

  static void init(LinuxFile *f, int file_descriptor, void *buffer,
                   size_t buffer_size, int buffer_mode, bool owned,
                   File::ModeFlags modeflags) {
    File::init(f, &write_func, &read_func, &seek_func, &close_func, &flush_func,
               buffer, buffer_size, buffer_mode, owned, modeflags);
    f->fd = file_descriptor;
  }

  int get_fd() const { return fd; }
};

namespace {

size_t write_func(File *f, const void *data, size_t size) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  long ret = __llvm_libc::syscall(SYS_write, lf->get_fd(), data, size);
  if (ret < 0) {
    errno = -ret;
    return 0;
  }
  return ret;
}

size_t read_func(File *f, void *buf, size_t size) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  long ret = __llvm_libc::syscall(SYS_read, lf->get_fd(), buf, size);
  if (ret < 0) {
    errno = -ret;
    return 0;
  }
  return ret;
}

int seek_func(File *f, long offset, int whence) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
#ifdef SYS_lseek
  long ret = __llvm_libc::syscall(SYS_lseek, lf->get_fd(), offset, whence);
#elif defined(SYS__llseek)
  long result;
  long ret = __llvm_libc::syscall(SYS__lseek, lf->get_fd(), offset >> 32,
                                  offset, &result, whence);
#else
#error "lseek and _llseek syscalls not available to perform a seek operation."
#endif

  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

int close_func(File *f) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  long ret = __llvm_libc::syscall(SYS_close, lf->get_fd());
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

int flush_func(File *f) {
  auto *lf = reinterpret_cast<LinuxFile *>(f);
  long ret = __llvm_libc::syscall(SYS_fsync, lf->get_fd());
  if (ret < 0) {
    errno = -ret;
    return -1;
  }
  return 0;
}

} // anonymous namespace

File *openfile(const char *path, const char *mode) {
  using ModeFlags = File::ModeFlags;
  auto modeflags = File::mode_flags(mode);
  if (modeflags == 0) {
    errno = EINVAL;
    return nullptr;
  }
  long open_flags = 0;
  if (modeflags & ModeFlags(File::OpenMode::APPEND)) {
    open_flags = O_CREAT | O_APPEND;
    if (modeflags & ModeFlags(File::OpenMode::PLUS))
      open_flags |= O_RDWR;
    else
      open_flags |= O_WRONLY;
  } else if (modeflags & ModeFlags(File::OpenMode::WRITE)) {
    open_flags = O_CREAT | O_TRUNC;
    if (modeflags & ModeFlags(File::OpenMode::PLUS))
      open_flags |= O_RDWR;
    else
      open_flags |= O_WRONLY;
  } else {
    if (modeflags & ModeFlags(File::OpenMode::PLUS))
      open_flags |= O_RDWR;
    else
      open_flags |= O_RDONLY;
  }

  // File created will have 0666 permissions.
  constexpr long OPEN_MODE =
      S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;

#ifdef SYS_open
  int fd = __llvm_libc::syscall(SYS_open, path, open_flags, OPEN_MODE);
#elif defined(SYS_openat)
  int fd =
      __llvm_libc::syscall(SYS_openat, AT_FDCWD, path, open_flags, OPEN_MODE);
#else
#error "SYS_open and SYS_openat syscalls not available to perform a file open."
#endif

  if (fd < 0) {
    errno = -fd;
    return nullptr;
  }

  void *buffer = malloc(File::DEFAULT_BUFFER_SIZE);
  auto *file = reinterpret_cast<LinuxFile *>(malloc(sizeof(LinuxFile)));
  LinuxFile::init(file, fd, buffer, File::DEFAULT_BUFFER_SIZE, _IOFBF, true,
                  modeflags);
  return file;
}

constexpr size_t STDOUT_BUFFER_SIZE = 1024;
char stdout_buffer[STDOUT_BUFFER_SIZE];
static LinuxFile StdOut(1, stdout_buffer, STDOUT_BUFFER_SIZE, _IOLBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stdout = &StdOut;

constexpr size_t STDERR_BUFFER_SIZE = 0;
static LinuxFile StdErr(2, nullptr, STDERR_BUFFER_SIZE, _IONBF, false,
                        File::ModeFlags(File::OpenMode::APPEND));
File *stderr = &StdErr;

} // namespace __llvm_libc
