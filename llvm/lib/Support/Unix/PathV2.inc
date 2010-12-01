//===- llvm/Support/Unix/PathV2.cpp - Unix Path Implementation --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific implementation of the PathV2 API.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"
#if HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#if HAVE_FCNTL_H
#include <fcntl.h>
#endif
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

namespace {
  struct AutoFD {
    int FileDescriptor;

    AutoFD(int fd) : FileDescriptor(fd) {}
    ~AutoFD() {
      if (FileDescriptor >= 0)
        ::close(FileDescriptor);
    }

    int take() {
      int ret = FileDescriptor;
      FileDescriptor = -1;
      return ret;
    }

    operator int() const {return FileDescriptor;}
  };
}

namespace llvm {
namespace sys  {
namespace path {

error_code current_path(SmallVectorImpl<char> &result) {
  long size = ::pathconf(".", _PC_PATH_MAX);
  result.reserve(size + 1);
  result.set_size(size + 1);

  if (::getcwd(result.data(), result.size()) == 0)
    return error_code(errno, system_category());

  result.set_size(strlen(result.data()));
  return make_error_code(errc::success);
}

} // end namespace path

namespace fs{

error_code copy_file(const Twine &from, const Twine &to, copy_option copt) {
 // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toStringRef(from_storage);
  StringRef t = to.toStringRef(to_storage);

  const size_t buf_sz = 32768;
  char buffer[buf_sz];
  int from_file = -1, to_file = -1;

  // Open from.
  if ((from_file = ::open(f.begin(), O_RDONLY)) < 0)
    return error_code(errno, system_category());
  AutoFD from_fd(from_file);

  // Stat from.
  struct stat from_stat;
  if (::stat(f.begin(), &from_stat) != 0)
    return error_code(errno, system_category());

  // Setup to flags.
  int to_flags = O_CREAT | O_WRONLY;
  if (copt == copy_option::fail_if_exists)
    to_flags |= O_EXCL;

  // Open to.
  if ((to_file = ::open(t.begin(), to_flags, from_stat.st_mode)) < 0)
    return error_code(errno, system_category());
  AutoFD to_fd(to_file);

  // Copy!
  ssize_t sz, sz_read = 1, sz_write;
  while (sz_read > 0 &&
         (sz_read = ::read(from_fd, buffer, buf_sz)) > 0) {
    // Allow for partial writes - see Advanced Unix Programming (2nd Ed.),
    // Marc Rochkind, Addison-Wesley, 2004, page 94
    sz_write = 0;
    do {
      if ((sz = ::write(to_fd, buffer + sz_write, sz_read - sz_write)) < 0) {
        sz_read = sz;  // cause read loop termination.
        break;         // error.
      }
      sz_write += sz;
    } while (sz_write < sz_read);
  }

  // After all the file operations above the return value of close actually
  // matters.
  if (::close(from_fd.take()) < 0) sz_read = -1;
  if (::close(to_fd.take()) < 0) sz_read = -1;

  // Check for errors.
  if (sz_read < 0)
    return error_code(errno, system_category());

  return make_error_code(errc::success);
}

} // end namespace fs
} // end namespace sys
} // end namespace llvm
