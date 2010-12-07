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
#if HAVE_STDIO_H
#include <stdio.h>
#endif

using namespace llvm;

namespace {
  /// This class automatically closes the given file descriptor when it goes out
  /// of scope. You can take back explicit ownership of the file descriptor by
  /// calling take(). The destructor does not verify that close was successful.
  /// Therefore, never allow this class to call close on a file descriptor that
  /// has been read from or written to.
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

  error_code TempDir(SmallVectorImpl<char> &result) {
    // FIXME: Don't use TMPDIR if program is SUID or SGID enabled.
    const char *dir = 0;
    (dir = std::getenv("TMPDIR" )) ||
    (dir = std::getenv("TMP"    )) ||
    (dir = std::getenv("TEMP"   )) ||
    (dir = std::getenv("TEMPDIR")) ||
#ifdef P_tmpdir
    (dir = P_tmpdir) ||
#endif
    (dir = "/tmp");

    result.clear();
    StringRef d(dir);
    result.append(d.begin(), d.end());
    return success;
  }
}

namespace llvm {
namespace sys  {
namespace fs {

error_code current_path(SmallVectorImpl<char> &result) {
  result.reserve(MAXPATHLEN);

  while (true) {
    if (::getcwd(result.data(), result.capacity()) == 0) {
      // See if there was a real error.
      if (errno != errc::not_enough_memory)
        return error_code(errno, system_category());
      // Otherwise there just wasn't enough space.
      result.reserve(result.capacity() * 2);
    } else
      break;
  }

  result.set_size(strlen(result.data()));
  return success;
}

error_code copy_file(const Twine &from, const Twine &to, copy_option copt) {
 // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

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

  return success;
}

error_code create_directory(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::mkdir(p.begin(), S_IRWXU | S_IRWXG) == -1) {
    if (errno != errc::file_exists)
      return error_code(errno, system_category());
    existed = true;
  } else
    existed = false;

  return success;
}

error_code create_hard_link(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::link(t.begin(), f.begin()) == -1)
    return error_code(errno, system_category());

  return success;
}

error_code create_symlink(const Twine &to, const Twine &from) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::symlink(t.begin(), f.begin()) == -1)
    return error_code(errno, system_category());

  return success;
}

error_code remove(const Twine &path, bool &existed) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::remove(p.begin()) == -1) {
    if (errno != errc::no_such_file_or_directory)
      return error_code(errno, system_category());
    existed = false;
  } else
    existed = true;

  return success;
}

error_code rename(const Twine &from, const Twine &to) {
  // Get arguments.
  SmallString<128> from_storage;
  SmallString<128> to_storage;
  StringRef f = from.toNullTerminatedStringRef(from_storage);
  StringRef t = to.toNullTerminatedStringRef(to_storage);

  if (::rename(f.begin(), t.begin()) == -1)
    return error_code(errno, system_category());

  return success;
}

error_code resize_file(const Twine &path, uint64_t size) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  if (::truncate(p.begin(), size) == -1)
    return error_code(errno, system_category());

  return success;
}

error_code exists(const Twine &path, bool &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat status;
  if (::stat(p.begin(), &status) == -1) {
    if (errno != errc::no_such_file_or_directory)
      return error_code(errno, system_category());
    result = false;
  } else
    result = true;

  return success;
}

error_code equivalent(const Twine &A, const Twine &B, bool &result) {
  // Get arguments.
  SmallString<128> a_storage;
  SmallString<128> b_storage;
  StringRef a = A.toNullTerminatedStringRef(a_storage);
  StringRef b = B.toNullTerminatedStringRef(b_storage);

  struct stat stat_a, stat_b;
  int error_b = ::stat(b.begin(), &stat_b);
  int error_a = ::stat(a.begin(), &stat_a);

  // If both are invalid, it's an error. If only one is, the result is false.
  if (error_a != 0 || error_b != 0) {
    if (error_a == error_b)
      return error_code(errno, system_category());
    result = false;
  } else {
    result =
      stat_a.st_dev == stat_b.st_dev &&
      stat_a.st_ino == stat_b.st_ino;
  }

  return success;
}

error_code file_size(const Twine &path, uint64_t &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat status;
  if (::stat(p.begin(), &status) == -1)
    return error_code(errno, system_category());
  if (!S_ISREG(status.st_mode))
    return make_error_code(errc::operation_not_permitted);

  result = status.st_size;
  return success;
}

error_code status(const Twine &path, file_status &result) {
  SmallString<128> path_storage;
  StringRef p = path.toNullTerminatedStringRef(path_storage);

  struct stat status;
  if (::stat(p.begin(), &status) != 0) {
    error_code ec(errno, system_category());
    if (ec == errc::no_such_file_or_directory)
      result = file_status(file_type::file_not_found);
    else
      result = file_status(file_type::status_error);
    return ec;
  }

  if (S_ISDIR(status.st_mode))
    result = file_status(file_type::directory_file);
  else if (S_ISREG(status.st_mode))
    result = file_status(file_type::regular_file);
  else if (S_ISBLK(status.st_mode))
    result = file_status(file_type::block_file);
  else if (S_ISCHR(status.st_mode))
    result = file_status(file_type::character_file);
  else if (S_ISFIFO(status.st_mode))
    result = file_status(file_type::fifo_file);
  else if (S_ISSOCK(status.st_mode))
    result = file_status(file_type::socket_file);
  else
    result = file_status(file_type::type_unknown);

  return success;
}

error_code unique_file(const Twine &model, int &result_fd,
                             SmallVectorImpl<char> &result_path) {
  SmallString<128> Model;
  model.toVector(Model);
  // Null terminate.
  Model.c_str();

  // Make model absolute by prepending a temp directory if it's not already.
  bool absolute;
  if (error_code ec = path::is_absolute(Twine(Model), absolute)) return ec;
  if (!absolute) {
    SmallString<128> TDir;
    if (error_code ec = TempDir(TDir)) return ec;
    if (error_code ec = path::append(TDir, Twine(Model))) return ec;
    Model.swap(TDir);
  }

  // Replace '%' with random chars. From here on, DO NOT modify model. It may be
  // needed if the randomly chosen path already exists.
  SmallString<128> RandomPath;
  RandomPath.reserve(Model.size() + 1);
  ::srand(::time(NULL));

retry_random_path:
  // This is opened here instead of above to make it easier to track when to
  // close it. Collisions should be rare enough for the possible extra syscalls
  // not to matter.
  FILE *RandomSource = ::fopen("/dev/urandom", "r");
  RandomPath.set_size(0);
  for (SmallVectorImpl<char>::const_iterator i = Model.begin(),
                                             e = Model.end(); i != e; ++i) {
    if (*i == '%') {
      char val = 0;
      if (RandomSource)
        val = fgetc(RandomSource);
      else
        val = ::rand();
      RandomPath.push_back("0123456789abcdef"[val & 15]);
    } else
      RandomPath.push_back(*i);
  }

  if (RandomSource)
    ::fclose(RandomSource);

  // Try to open + create the file.
rety_open_create:
  int RandomFD = ::open(RandomPath.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
  if (RandomFD == -1) {
    // If the file existed, try again, otherwise, error.
    if (errno == errc::file_exists)
      goto retry_random_path;
    // The path prefix doesn't exist.
    if (errno == errc::no_such_file_or_directory) {
      StringRef p(RandomPath.begin(), RandomPath.size());
      SmallString<64> dir_to_create;
      for (path::const_iterator i = path::begin(p),
                                e = --path::end(p); i != e; ++i) {
        if (error_code ec = path::append(dir_to_create, *i)) return ec;
        bool Exists;
        if (error_code ec = exists(Twine(dir_to_create), Exists)) return ec;
        if (!Exists) {
          // Don't try to create network paths.
          if (i->size() > 2 && (*i)[0] == '/' &&
                               (*i)[1] == '/' &&
                               (*i)[2] != '/')
            return make_error_code(errc::no_such_file_or_directory);
          if (::mkdir(dir_to_create.c_str(), 0700) == -1)
            return error_code(errno, system_category());
        }
      }
      goto rety_open_create;
    }
    return error_code(errno, system_category());
  }

   // Make the path absolute.
  char real_path_buff[PATH_MAX + 1];
  if (realpath(RandomPath.c_str(), real_path_buff) == NULL) {
    ::close(RandomFD);
    ::unlink(RandomPath.c_str());
    return error_code(errno, system_category());
  }

  result_path.clear();
  StringRef d(real_path_buff);
  result_path.append(d.begin(), d.end());

  result_fd = RandomFD;
  return success;
}

} // end namespace fs
} // end namespace sys
} // end namespace llvm
