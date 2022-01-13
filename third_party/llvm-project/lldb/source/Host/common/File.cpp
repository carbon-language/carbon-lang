//===-- File.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/File.h"

#include <cerrno>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <fcntl.h>

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <termios.h>
#include <unistd.h>
#endif

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;
using llvm::Expected;

Expected<const char *>
File::GetStreamOpenModeFromOptions(File::OpenOptions options) {
  File::OpenOptions rw =
      options & (File::eOpenOptionReadOnly | File::eOpenOptionWriteOnly |
                 File::eOpenOptionReadWrite);

  if (options & File::eOpenOptionAppend) {
    if (rw == File::eOpenOptionReadWrite) {
      if (options & File::eOpenOptionCanCreateNewOnly)
        return "a+x";
      else
        return "a+";
    } else if (rw == File::eOpenOptionWriteOnly) {
      if (options & File::eOpenOptionCanCreateNewOnly)
        return "ax";
      else
        return "a";
    }
  } else if (rw == File::eOpenOptionReadWrite) {
    if (options & File::eOpenOptionCanCreate) {
      if (options & File::eOpenOptionCanCreateNewOnly)
        return "w+x";
      else
        return "w+";
    } else
      return "r+";
  } else if (rw == File::eOpenOptionWriteOnly) {
    return "w";
  } else if (rw == File::eOpenOptionReadOnly) {
    return "r";
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "invalid options, cannot convert to mode string");
}

Expected<File::OpenOptions> File::GetOptionsFromMode(llvm::StringRef mode) {
  OpenOptions opts =
      llvm::StringSwitch<OpenOptions>(mode)
          .Cases("r", "rb", eOpenOptionReadOnly)
          .Cases("w", "wb", eOpenOptionWriteOnly)
          .Cases("a", "ab",
                 eOpenOptionWriteOnly | eOpenOptionAppend |
                 eOpenOptionCanCreate)
          .Cases("r+", "rb+", "r+b", eOpenOptionReadWrite)
          .Cases("w+", "wb+", "w+b",
                 eOpenOptionReadWrite | eOpenOptionCanCreate |
                 eOpenOptionTruncate)
          .Cases("a+", "ab+", "a+b",
                 eOpenOptionReadWrite | eOpenOptionAppend |
                     eOpenOptionCanCreate)
          .Default(eOpenOptionInvalid);
  if (opts != eOpenOptionInvalid)
    return opts;
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "invalid mode, cannot convert to File::OpenOptions");
}

int File::kInvalidDescriptor = -1;
FILE *File::kInvalidStream = nullptr;

Status File::Read(void *buf, size_t &num_bytes) {
  return std::error_code(ENOTSUP, std::system_category());
}
Status File::Write(const void *buf, size_t &num_bytes) {
  return std::error_code(ENOTSUP, std::system_category());
}

bool File::IsValid() const { return false; }

Status File::Close() { return Flush(); }

IOObject::WaitableHandle File::GetWaitableHandle() {
  return IOObject::kInvalidHandleValue;
}

Status File::GetFileSpec(FileSpec &file_spec) const {
  file_spec.Clear();
  return std::error_code(ENOTSUP, std::system_category());
}

int File::GetDescriptor() const { return kInvalidDescriptor; }

FILE *File::GetStream() { return nullptr; }

off_t File::SeekFromStart(off_t offset, Status *error_ptr) {
  if (error_ptr)
    *error_ptr = std::error_code(ENOTSUP, std::system_category());
  return -1;
}

off_t File::SeekFromCurrent(off_t offset, Status *error_ptr) {
  if (error_ptr)
    *error_ptr = std::error_code(ENOTSUP, std::system_category());
  return -1;
}

off_t File::SeekFromEnd(off_t offset, Status *error_ptr) {
  if (error_ptr)
    *error_ptr = std::error_code(ENOTSUP, std::system_category());
  return -1;
}

Status File::Read(void *dst, size_t &num_bytes, off_t &offset) {
  return std::error_code(ENOTSUP, std::system_category());
}

Status File::Write(const void *src, size_t &num_bytes, off_t &offset) {
  return std::error_code(ENOTSUP, std::system_category());
}

Status File::Flush() { return Status(); }

Status File::Sync() { return Flush(); }

void File::CalculateInteractiveAndTerminal() {
  const int fd = GetDescriptor();
  if (!DescriptorIsValid(fd)) {
    m_is_interactive = eLazyBoolNo;
    m_is_real_terminal = eLazyBoolNo;
    m_supports_colors = eLazyBoolNo;
    return;
  }
  m_is_interactive = eLazyBoolNo;
  m_is_real_terminal = eLazyBoolNo;
#if defined(_WIN32)
  if (_isatty(fd)) {
    m_is_interactive = eLazyBoolYes;
    m_is_real_terminal = eLazyBoolYes;
#if defined(ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    m_supports_colors = eLazyBoolYes;
#endif
  }
#else
  if (isatty(fd)) {
    m_is_interactive = eLazyBoolYes;
    struct winsize window_size;
    if (::ioctl(fd, TIOCGWINSZ, &window_size) == 0) {
      if (window_size.ws_col > 0) {
        m_is_real_terminal = eLazyBoolYes;
        if (llvm::sys::Process::FileDescriptorHasColors(fd))
          m_supports_colors = eLazyBoolYes;
      }
    }
  }
#endif
}

bool File::GetIsInteractive() {
  if (m_is_interactive == eLazyBoolCalculate)
    CalculateInteractiveAndTerminal();
  return m_is_interactive == eLazyBoolYes;
}

bool File::GetIsRealTerminal() {
  if (m_is_real_terminal == eLazyBoolCalculate)
    CalculateInteractiveAndTerminal();
  return m_is_real_terminal == eLazyBoolYes;
}

bool File::GetIsTerminalWithColors() {
  if (m_supports_colors == eLazyBoolCalculate)
    CalculateInteractiveAndTerminal();
  return m_supports_colors == eLazyBoolYes;
}

size_t File::Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  size_t result = PrintfVarArg(format, args);
  va_end(args);
  return result;
}

size_t File::PrintfVarArg(const char *format, va_list args) {
  size_t result = 0;
  char *s = nullptr;
  result = vasprintf(&s, format, args);
  if (s != nullptr) {
    if (result > 0) {
      size_t s_len = result;
      Write(s, s_len);
      result = s_len;
    }
    free(s);
  }
  return result;
}

Expected<File::OpenOptions> File::GetOptions() const {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "GetOptions() not implemented for this File class");
}

uint32_t File::GetPermissions(Status &error) const {
  int fd = GetDescriptor();
  if (!DescriptorIsValid(fd)) {
    error = std::error_code(ENOTSUP, std::system_category());
    return 0;
  }
  struct stat file_stats;
  if (::fstat(fd, &file_stats) == -1) {
    error.SetErrorToErrno();
    return 0;
  }
  error.Clear();
  return file_stats.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
}

Expected<File::OpenOptions> NativeFile::GetOptions() const { return m_options; }

int NativeFile::GetDescriptor() const {
  if (DescriptorIsValid())
    return m_descriptor;

  // Don't open the file descriptor if we don't need to, just get it from the
  // stream if we have one.
  if (StreamIsValid()) {
#if defined(_WIN32)
    return _fileno(m_stream);
#else
    return fileno(m_stream);
#endif
  }

  // Invalid descriptor and invalid stream, return invalid descriptor.
  return kInvalidDescriptor;
}

IOObject::WaitableHandle NativeFile::GetWaitableHandle() {
  return GetDescriptor();
}

FILE *NativeFile::GetStream() {
  if (!StreamIsValid()) {
    if (DescriptorIsValid()) {
      auto mode = GetStreamOpenModeFromOptions(m_options);
      if (!mode)
        llvm::consumeError(mode.takeError());
      else {
        if (!m_own_descriptor) {
// We must duplicate the file descriptor if we don't own it because when you
// call fdopen, the stream will own the fd
#ifdef _WIN32
          m_descriptor = ::_dup(GetDescriptor());
#else
          m_descriptor = dup(GetDescriptor());
#endif
          m_own_descriptor = true;
        }

        m_stream = llvm::sys::RetryAfterSignal(nullptr, ::fdopen, m_descriptor,
                                               mode.get());

        // If we got a stream, then we own the stream and should no longer own
        // the descriptor because fclose() will close it for us

        if (m_stream) {
          m_own_stream = true;
          m_own_descriptor = false;
        }
      }
    }
  }
  return m_stream;
}

Status NativeFile::Close() {
  Status error;
  if (StreamIsValid()) {
    if (m_own_stream) {
      if (::fclose(m_stream) == EOF)
        error.SetErrorToErrno();
    } else {
      File::OpenOptions rw =
          m_options & (File::eOpenOptionReadOnly | File::eOpenOptionWriteOnly |
                       File::eOpenOptionReadWrite);

      if (rw == eOpenOptionWriteOnly || rw == eOpenOptionReadWrite) {
        if (::fflush(m_stream) == EOF)
          error.SetErrorToErrno();
      }
    }
  }
  if (DescriptorIsValid() && m_own_descriptor) {
    if (::close(m_descriptor) != 0)
      error.SetErrorToErrno();
  }
  m_descriptor = kInvalidDescriptor;
  m_stream = kInvalidStream;
  m_options = OpenOptions(0);
  m_own_stream = false;
  m_own_descriptor = false;
  m_is_interactive = eLazyBoolCalculate;
  m_is_real_terminal = eLazyBoolCalculate;
  return error;
}

Status NativeFile::GetFileSpec(FileSpec &file_spec) const {
  Status error;
#ifdef F_GETPATH
  if (IsValid()) {
    char path[PATH_MAX];
    if (::fcntl(GetDescriptor(), F_GETPATH, path) == -1)
      error.SetErrorToErrno();
    else
      file_spec.SetFile(path, FileSpec::Style::native);
  } else {
    error.SetErrorString("invalid file handle");
  }
#elif defined(__linux__)
  char proc[64];
  char path[PATH_MAX];
  if (::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", GetDescriptor()) < 0)
    error.SetErrorString("cannot resolve file descriptor");
  else {
    ssize_t len;
    if ((len = ::readlink(proc, path, sizeof(path) - 1)) == -1)
      error.SetErrorToErrno();
    else {
      path[len] = '\0';
      file_spec.SetFile(path, FileSpec::Style::native);
    }
  }
#else
  error.SetErrorString(
      "NativeFile::GetFileSpec is not supported on this platform");
#endif

  if (error.Fail())
    file_spec.Clear();
  return error;
}

off_t NativeFile::SeekFromStart(off_t offset, Status *error_ptr) {
  off_t result = 0;
  if (DescriptorIsValid()) {
    result = ::lseek(m_descriptor, offset, SEEK_SET);

    if (error_ptr) {
      if (result == -1)
        error_ptr->SetErrorToErrno();
      else
        error_ptr->Clear();
    }
  } else if (StreamIsValid()) {
    result = ::fseek(m_stream, offset, SEEK_SET);

    if (error_ptr) {
      if (result == -1)
        error_ptr->SetErrorToErrno();
      else
        error_ptr->Clear();
    }
  } else if (error_ptr) {
    error_ptr->SetErrorString("invalid file handle");
  }
  return result;
}

off_t NativeFile::SeekFromCurrent(off_t offset, Status *error_ptr) {
  off_t result = -1;
  if (DescriptorIsValid()) {
    result = ::lseek(m_descriptor, offset, SEEK_CUR);

    if (error_ptr) {
      if (result == -1)
        error_ptr->SetErrorToErrno();
      else
        error_ptr->Clear();
    }
  } else if (StreamIsValid()) {
    result = ::fseek(m_stream, offset, SEEK_CUR);

    if (error_ptr) {
      if (result == -1)
        error_ptr->SetErrorToErrno();
      else
        error_ptr->Clear();
    }
  } else if (error_ptr) {
    error_ptr->SetErrorString("invalid file handle");
  }
  return result;
}

off_t NativeFile::SeekFromEnd(off_t offset, Status *error_ptr) {
  off_t result = -1;
  if (DescriptorIsValid()) {
    result = ::lseek(m_descriptor, offset, SEEK_END);

    if (error_ptr) {
      if (result == -1)
        error_ptr->SetErrorToErrno();
      else
        error_ptr->Clear();
    }
  } else if (StreamIsValid()) {
    result = ::fseek(m_stream, offset, SEEK_END);

    if (error_ptr) {
      if (result == -1)
        error_ptr->SetErrorToErrno();
      else
        error_ptr->Clear();
    }
  } else if (error_ptr) {
    error_ptr->SetErrorString("invalid file handle");
  }
  return result;
}

Status NativeFile::Flush() {
  Status error;
  if (StreamIsValid()) {
    if (llvm::sys::RetryAfterSignal(EOF, ::fflush, m_stream) == EOF)
      error.SetErrorToErrno();
  } else if (!DescriptorIsValid()) {
    error.SetErrorString("invalid file handle");
  }
  return error;
}

Status NativeFile::Sync() {
  Status error;
  if (DescriptorIsValid()) {
#ifdef _WIN32
    int err = FlushFileBuffers((HANDLE)_get_osfhandle(m_descriptor));
    if (err == 0)
      error.SetErrorToGenericError();
#else
    if (llvm::sys::RetryAfterSignal(-1, ::fsync, m_descriptor) == -1)
      error.SetErrorToErrno();
#endif
  } else {
    error.SetErrorString("invalid file handle");
  }
  return error;
}

#if defined(__APPLE__)
// Darwin kernels only can read/write <= INT_MAX bytes
#define MAX_READ_SIZE INT_MAX
#define MAX_WRITE_SIZE INT_MAX
#endif

Status NativeFile::Read(void *buf, size_t &num_bytes) {
  Status error;

#if defined(MAX_READ_SIZE)
  if (num_bytes > MAX_READ_SIZE) {
    uint8_t *p = (uint8_t *)buf;
    size_t bytes_left = num_bytes;
    // Init the num_bytes read to zero
    num_bytes = 0;

    while (bytes_left > 0) {
      size_t curr_num_bytes;
      if (bytes_left > MAX_READ_SIZE)
        curr_num_bytes = MAX_READ_SIZE;
      else
        curr_num_bytes = bytes_left;

      error = Read(p + num_bytes, curr_num_bytes);

      // Update how many bytes were read
      num_bytes += curr_num_bytes;
      if (bytes_left < curr_num_bytes)
        bytes_left = 0;
      else
        bytes_left -= curr_num_bytes;

      if (error.Fail())
        break;
    }
    return error;
  }
#endif

  ssize_t bytes_read = -1;
  if (DescriptorIsValid()) {
    bytes_read = llvm::sys::RetryAfterSignal(-1, ::read, m_descriptor, buf, num_bytes);
    if (bytes_read == -1) {
      error.SetErrorToErrno();
      num_bytes = 0;
    } else
      num_bytes = bytes_read;
  } else if (StreamIsValid()) {
    bytes_read = ::fread(buf, 1, num_bytes, m_stream);

    if (bytes_read == 0) {
      if (::feof(m_stream))
        error.SetErrorString("feof");
      else if (::ferror(m_stream))
        error.SetErrorString("ferror");
      num_bytes = 0;
    } else
      num_bytes = bytes_read;
  } else {
    num_bytes = 0;
    error.SetErrorString("invalid file handle");
  }
  return error;
}

Status NativeFile::Write(const void *buf, size_t &num_bytes) {
  Status error;

#if defined(MAX_WRITE_SIZE)
  if (num_bytes > MAX_WRITE_SIZE) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t bytes_left = num_bytes;
    // Init the num_bytes written to zero
    num_bytes = 0;

    while (bytes_left > 0) {
      size_t curr_num_bytes;
      if (bytes_left > MAX_WRITE_SIZE)
        curr_num_bytes = MAX_WRITE_SIZE;
      else
        curr_num_bytes = bytes_left;

      error = Write(p + num_bytes, curr_num_bytes);

      // Update how many bytes were read
      num_bytes += curr_num_bytes;
      if (bytes_left < curr_num_bytes)
        bytes_left = 0;
      else
        bytes_left -= curr_num_bytes;

      if (error.Fail())
        break;
    }
    return error;
  }
#endif

  ssize_t bytes_written = -1;
  if (DescriptorIsValid()) {
    bytes_written =
        llvm::sys::RetryAfterSignal(-1, ::write, m_descriptor, buf, num_bytes);
    if (bytes_written == -1) {
      error.SetErrorToErrno();
      num_bytes = 0;
    } else
      num_bytes = bytes_written;
  } else if (StreamIsValid()) {
    bytes_written = ::fwrite(buf, 1, num_bytes, m_stream);

    if (bytes_written == 0) {
      if (::feof(m_stream))
        error.SetErrorString("feof");
      else if (::ferror(m_stream))
        error.SetErrorString("ferror");
      num_bytes = 0;
    } else
      num_bytes = bytes_written;

  } else {
    num_bytes = 0;
    error.SetErrorString("invalid file handle");
  }

  return error;
}

Status NativeFile::Read(void *buf, size_t &num_bytes, off_t &offset) {
  Status error;

#if defined(MAX_READ_SIZE)
  if (num_bytes > MAX_READ_SIZE) {
    uint8_t *p = (uint8_t *)buf;
    size_t bytes_left = num_bytes;
    // Init the num_bytes read to zero
    num_bytes = 0;

    while (bytes_left > 0) {
      size_t curr_num_bytes;
      if (bytes_left > MAX_READ_SIZE)
        curr_num_bytes = MAX_READ_SIZE;
      else
        curr_num_bytes = bytes_left;

      error = Read(p + num_bytes, curr_num_bytes, offset);

      // Update how many bytes were read
      num_bytes += curr_num_bytes;
      if (bytes_left < curr_num_bytes)
        bytes_left = 0;
      else
        bytes_left -= curr_num_bytes;

      if (error.Fail())
        break;
    }
    return error;
  }
#endif

#ifndef _WIN32
  int fd = GetDescriptor();
  if (fd != kInvalidDescriptor) {
    ssize_t bytes_read =
        llvm::sys::RetryAfterSignal(-1, ::pread, fd, buf, num_bytes, offset);
    if (bytes_read < 0) {
      num_bytes = 0;
      error.SetErrorToErrno();
    } else {
      offset += bytes_read;
      num_bytes = bytes_read;
    }
  } else {
    num_bytes = 0;
    error.SetErrorString("invalid file handle");
  }
#else
  std::lock_guard<std::mutex> guard(offset_access_mutex);
  long cur = ::lseek(m_descriptor, 0, SEEK_CUR);
  SeekFromStart(offset);
  error = Read(buf, num_bytes);
  if (!error.Fail())
    SeekFromStart(cur);
#endif
  return error;
}

Status NativeFile::Write(const void *buf, size_t &num_bytes, off_t &offset) {
  Status error;

#if defined(MAX_WRITE_SIZE)
  if (num_bytes > MAX_WRITE_SIZE) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t bytes_left = num_bytes;
    // Init the num_bytes written to zero
    num_bytes = 0;

    while (bytes_left > 0) {
      size_t curr_num_bytes;
      if (bytes_left > MAX_WRITE_SIZE)
        curr_num_bytes = MAX_WRITE_SIZE;
      else
        curr_num_bytes = bytes_left;

      error = Write(p + num_bytes, curr_num_bytes, offset);

      // Update how many bytes were read
      num_bytes += curr_num_bytes;
      if (bytes_left < curr_num_bytes)
        bytes_left = 0;
      else
        bytes_left -= curr_num_bytes;

      if (error.Fail())
        break;
    }
    return error;
  }
#endif

  int fd = GetDescriptor();
  if (fd != kInvalidDescriptor) {
#ifndef _WIN32
    ssize_t bytes_written =
        llvm::sys::RetryAfterSignal(-1, ::pwrite, m_descriptor, buf, num_bytes, offset);
    if (bytes_written < 0) {
      num_bytes = 0;
      error.SetErrorToErrno();
    } else {
      offset += bytes_written;
      num_bytes = bytes_written;
    }
#else
    std::lock_guard<std::mutex> guard(offset_access_mutex);
    long cur = ::lseek(m_descriptor, 0, SEEK_CUR);
    SeekFromStart(offset);
    error = Write(buf, num_bytes);
    long after = ::lseek(m_descriptor, 0, SEEK_CUR);

    if (!error.Fail())
      SeekFromStart(cur);

    offset = after;
#endif
  } else {
    num_bytes = 0;
    error.SetErrorString("invalid file handle");
  }
  return error;
}

size_t NativeFile::PrintfVarArg(const char *format, va_list args) {
  if (StreamIsValid()) {
    return ::vfprintf(m_stream, format, args);
  } else {
    return File::PrintfVarArg(format, args);
  }
}

mode_t File::ConvertOpenOptionsForPOSIXOpen(OpenOptions open_options) {
  mode_t mode = 0;
  File::OpenOptions rw =
      open_options & (File::eOpenOptionReadOnly | File::eOpenOptionWriteOnly |
                      File::eOpenOptionReadWrite);
  if (rw == eOpenOptionReadWrite)
    mode |= O_RDWR;
  else if (rw == eOpenOptionWriteOnly)
    mode |= O_WRONLY;
  else if (rw == eOpenOptionReadOnly)
    mode |= O_RDONLY;

  if (open_options & eOpenOptionAppend)
    mode |= O_APPEND;

  if (open_options & eOpenOptionTruncate)
    mode |= O_TRUNC;

  if (open_options & eOpenOptionNonBlocking)
    mode |= O_NONBLOCK;

  if (open_options & eOpenOptionCanCreateNewOnly)
    mode |= O_CREAT | O_EXCL;
  else if (open_options & eOpenOptionCanCreate)
    mode |= O_CREAT;

  return mode;
}

char File::ID = 0;
char NativeFile::ID = 0;
