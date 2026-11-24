// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>

// Undefine common Win32 macros that can collide with C++ identifiers.
#undef CopyFile
#undef CreateDirectory
#undef DeleteFile
#undef ERROR
#undef Yield

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include "common/build_data.h"
#include "common/filesystem.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/WindowsError.h"

namespace Carbon::Filesystem {

static auto ErrnoToErrorCode(int errnum) -> std::error_code {
  return std::error_code(errnum, std::generic_category());
}

static auto HasFlag(OpenFlags flags, OpenFlags bit) -> bool {
  return (static_cast<int>(flags) & static_cast<int>(bit)) != 0;
}

static auto HasAccessFlag(AccessCheckFlags flags, AccessCheckFlags bit)
    -> bool {
  return (static_cast<int>(flags) & static_cast<int>(bit)) != 0;
}

// Resizes the file referred to by CRT descriptor `fd` to `new_size` without
// changing the current file position.
static auto ResizeFileByFd(int fd, int64_t new_size) -> std::error_code {
  HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (handle == INVALID_HANDLE_VALUE) {
    return std::make_error_code(std::errc::bad_file_descriptor);
  }
  FILE_END_OF_FILE_INFO eof_info = {};
  eof_info.EndOfFile.QuadPart = new_size;
  if (!::SetFileInformationByHandle(handle, FileEndOfFileInfo, &eof_info,
                                    sizeof(eof_info))) {
    return llvm::mapWindowsError(::GetLastError());
  }
  return {};
}

// Converts a UTF-8 path string to UTF-16, adding the `\\?\` extended-length
// prefix when the resolved path exceeds `MAX_PATH`.
static auto ToWidePath(llvm::StringRef path_utf8, std::wstring& out)
    -> std::error_code {
  llvm::SmallString<256> path_str(path_utf8);
  if (path_str.starts_with("//?/")) {
    llvm::sys::path::native(path_str,
                            llvm::sys::path::Style::windows_backslash);
  }
  if (!llvm::ConvertUTF8toWide(path_str, out)) {
    return std::make_error_code(std::errc::invalid_argument);
  }
  if (path_str.starts_with(R"(\\?\)")) {
    return {};
  }
  size_t cur_dir_len = 0;
  if (!llvm::sys::path::is_absolute(path_str)) {
    cur_dir_len = ::GetCurrentDirectoryW(0, nullptr);
    if (cur_dir_len == 0) {
      return llvm::mapWindowsError(::GetLastError());
    }
  }
  if (out.size() + cur_dir_len < MAX_PATH) {
    return {};
  }
  if (std::error_code ec = llvm::sys::fs::make_absolute(path_str)) {
    return ec;
  }
  llvm::sys::path::native(path_str, llvm::sys::path::Style::windows_backslash);
  llvm::sys::path::remove_dots(path_str, /*remove_dot_dot=*/true,
                               llvm::sys::path::Style::windows_backslash);
  std::string prefixed;
  if (path_str.starts_with(R"(\\)")) {
    prefixed = R"(\\?\UNC\)";
    prefixed.append(path_str.data() + 2, path_str.size() - 2);
  } else {
    prefixed = R"(\\?\)";
    prefixed.append(path_str.data(), path_str.size());
  }
  if (!llvm::ConvertUTF8toWide(prefixed, out)) {
    return std::make_error_code(std::errc::invalid_argument);
  }
  return {};
}

// Returns true if `path_utf8` is a reparse point (such as a symlink or
// junction).
static auto IsReparsePoint(llvm::StringRef path_utf8) -> bool {
  std::wstring wide_path;
  if (ToWidePath(path_utf8, wide_path)) {
    return false;
  }
  DWORD attrs = ::GetFileAttributesW(wide_path.c_str());
  return attrs != INVALID_FILE_ATTRIBUTES &&
         (attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
}

auto FdError::Print(llvm::raw_ostream& out) const -> void {
  // The `format_` member is a `StringLiteral` that is null terminated, so
  // `.data()` is safe here.
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
  out << llvm::formatv(format_.data(), target_)
      << " failed: " << error_code().message();
}

auto PathError::Print(llvm::raw_ostream& out) const -> void {
  // The `format_` member is a `StringLiteral` that is null terminated, so
  // `.data()` is safe here.
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
  out << llvm::formatv(
             format_.data(), path_.string(),
             dir_path_.empty() ? std::string("CWD") : dir_path_.string())
      << " failed: " << error_code().message();
}

auto FileLock::Destroy() -> void {
  if (fd_ < 0) {
    fd_ = -1;
    return;
  }

  std::error_code ec = llvm::sys::fs::unlockFile(fd_);
  CARBON_CHECK(!ec, "{0}",
               FdError(ec, "Unexpected error while _unlocking_ '{0}'", fd_));
  fd_ = -1;
}

auto Internal::FileRefBase::Stat() -> ErrorOr<FileStatus, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::Stat on '{0}'", fd_);
  }
  FileStatus status;
  if (std::error_code ec = llvm::sys::fs::status(fd_, status.status_)) {
    return FdError(ec, "File::Stat on '{0}'", fd_);
  }
  return status;
}

auto Internal::FileRefBase::UpdateTimes(std::optional<TimePoint> time_point)
    -> ErrorOr<Success, FdError> {
  llvm::StringLiteral error_format =
      time_point
          ? llvm::StringLiteral("File::UpdateTimes to a specific time on '{0}'")
          : llvm::StringLiteral("File::UpdateTimes to now on '{0}'");
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   error_format, fd_);
  }
  TimePoint target_time =
      time_point ? *time_point
                 : std::chrono::time_point_cast<Duration>(Clock::now());
  if (std::error_code ec =
          llvm::sys::fs::setLastAccessAndModificationTime(fd_, target_time)) {
    return FdError(ec, error_format, fd_);
  }
  return Success();
}

auto Internal::FileRefBase::Seek(int64_t delta) -> ErrorOr<int64_t, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::Seek on '{0}'", fd_);
  }
  int64_t byte_offset = _lseeki64(fd_, delta, SEEK_CUR);
  if (byte_offset == -1) {
    return FdError(ErrnoToErrorCode(errno), "File::Seek on '{0}'", fd_);
  }
  return byte_offset;
}

auto Internal::FileRefBase::SeekFromBeginning(int64_t delta_from_beginning)
    -> ErrorOr<int64_t, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::SeekTo on '{0}'", fd_);
  }
  int64_t byte_offset = _lseeki64(fd_, delta_from_beginning, SEEK_SET);
  if (byte_offset == -1) {
    return FdError(ErrnoToErrorCode(errno), "File::SeekTo on '{0}'", fd_);
  }
  return byte_offset;
}

auto Internal::FileRefBase::SeekFromEnd(int64_t delta_from_end)
    -> ErrorOr<int64_t, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::SeekFromEnd on '{0}'", fd_);
  }
  int64_t byte_offset = _lseeki64(fd_, delta_from_end, SEEK_END);
  if (byte_offset == -1) {
    return FdError(ErrnoToErrorCode(errno), "File::SeekFromEnd on '{0}'", fd_);
  }
  return byte_offset;
}

auto Internal::FileRefBase::Truncate(int64_t new_size)
    -> ErrorOr<Success, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::Truncate on '{0}'", fd_);
  }
  if (new_size < 0) {
    return FdError(std::make_error_code(std::errc::invalid_argument),
                   "File::Truncate on '{0}'", fd_);
  }
  if (std::error_code ec = ResizeFileByFd(fd_, new_size)) {
    return FdError(ec, "File::Truncate on '{0}'", fd_);
  }
  return Success();
}

auto Internal::FileRefBase::ReadToBuffer(
    llvm::MutableArrayRef<std::byte> buffer)
    -> ErrorOr<llvm::MutableArrayRef<std::byte>, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::Read on '{0}'", fd_);
  }
  if (buffer.empty()) {
    return buffer;
  }
  unsigned int to_read = static_cast<unsigned int>(std::min<size_t>(
      buffer.size(), static_cast<size_t>(std::numeric_limits<int>::max())));
  int read_bytes = _read(fd_, buffer.data(), to_read);
  if (read_bytes == -1) {
    return FdError(ErrnoToErrorCode(errno), "File::Read on '{0}'", fd_);
  }
  return buffer.slice(0, read_bytes);
}

auto Internal::FileRefBase::WriteFromBuffer(llvm::ArrayRef<std::byte> buffer)
    -> ErrorOr<llvm::ArrayRef<std::byte>, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::Write on '{0}'", fd_);
  }
  if (buffer.empty()) {
    return buffer;
  }
  unsigned int to_write = static_cast<unsigned int>(std::min<size_t>(
      buffer.size(), static_cast<size_t>(std::numeric_limits<int>::max())));
  int written_bytes = _write(fd_, buffer.data(), to_write);
  if (written_bytes == -1) {
    return FdError(ErrnoToErrorCode(errno), "File::Write on '{0}'", fd_);
  }
  return buffer.drop_front(written_bytes);
}

auto Internal::FileRefBase::TryLock(FileLock::Kind kind, Duration deadline,
                                    Duration poll_interval)
    -> ErrorOr<FileLock, FdError> {
  if (fd_ < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::TryLock on '{0}'", fd_);
  }
  CARBON_CHECK(poll_interval <= deadline);
  if (deadline != Duration(0) && poll_interval == Duration(0)) {
    poll_interval =
        std::max(Duration(std::chrono::microseconds(1)), deadline / 1000);
  }
  if (deadline != Duration(0)) {
    CARBON_CHECK(
        deadline >= std::chrono::microseconds(10),
        "A deadline for a file lock shorter than 10 microseconds is not "
        "supported, callers can implement their own polling logic.");
    CARBON_CHECK(poll_interval >= std::chrono::microseconds(1),
                 "Polling for a file lock faster than every microsecond is not "
                 "supported, callers can implement their own polling logic.");
  }

  llvm::sys::fs::LockKind llvm_kind = kind == FileLock::Exclusive
                                          ? llvm::sys::fs::LockKind::Exclusive
                                          : llvm::sys::fs::LockKind::Shared;
  auto stop = Clock::now() + deadline;
  for (;;) {
    std::error_code ec = llvm::sys::fs::tryLockFile(
        fd_, std::chrono::milliseconds(0), llvm_kind);
    if (!ec) {
      return FileLock(fd_);
    }

    if (ec != std::errc::no_lock_available || deadline == Duration(0) ||
        Clock::now() >= stop) {
      return FdError(ec, "File::TryLock on '{0}'", fd_);
    }

    std::this_thread::sleep_for(poll_interval);
  }
}

auto Internal::FileRefBase::Close() && -> ErrorOr<Success, FdError> {
  int fd = std::exchange(fd_, -1);
  if (fd < 0) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "File::Close on '{0}'", fd);
  }
  // `_open_osfhandle` transferred ownership of the Win32 `HANDLE` to the CRT
  // descriptor, so `_close` releases both the CRT fd and the underlying handle.
  if (_close(fd) == 0) {
    return Success();
  }
  return FdError(ErrnoToErrorCode(errno), "File::Close on '{0}'", fd);
}

auto DirRef::ResolvePath(const std::filesystem::path& path,
                         llvm::StringLiteral format) const
    -> ErrorOr<std::filesystem::path, PathError> {
  if (path.empty()) {
    return PathError(std::make_error_code(std::errc::no_such_file_or_directory),
                     format, path, DirPathForError());
  }
  if (path.is_absolute()) {
    return path;
  }
  if (!is_valid()) {
    return PathError(std::make_error_code(std::errc::bad_file_descriptor),
                     format, path, DirPathForError());
  }
  if (is_cwd_) {
    return path;
  }
  return *dir_path_ / path;
}

auto DirRef::DirPathForError() const -> std::filesystem::path {
  if (dir_path_ != nullptr) {
    return *dir_path_;
  }
  return {};
}

auto DirRef::Read() -> ErrorOr<Reader, FdError> {
  if (!is_valid()) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "Dir::Read on '{0}'", DirPathForError());
  }
  llvm::sys::fs::file_status st;
  std::string dir_str = is_cwd_ ? std::string(".") : dir_path_->string();
  if (std::error_code ec = llvm::sys::fs::status(dir_str, st)) {
    return FdError(ec, "Dir::Read on '{0}'", DirPathForError());
  }
  if (!llvm::sys::fs::is_directory(st)) {
    return FdError(std::make_error_code(std::errc::not_a_directory),
                   "Dir::Read on '{0}'", DirPathForError());
  }
  if (is_cwd_) {
    return Reader(/*dir_path=*/nullptr, /*is_cwd=*/true);
  }
  return Reader(new std::filesystem::path(*dir_path_), /*is_cwd=*/false);
}

auto DirRef::Access(const std::filesystem::path& path, AccessCheckFlags check)
    -> ErrorOr<bool, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Access on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  std::string full = resolved.string();
  if (std::error_code ec =
          llvm::sys::fs::access(full, llvm::sys::fs::AccessMode::Exist)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  if (HasAccessFlag(check, AccessCheckFlags::Write)) {
    if (std::error_code ec =
            llvm::sys::fs::access(full, llvm::sys::fs::AccessMode::Write)) {
      return PathError(ec, ErrorFormat, path, DirPathForError());
    }
  }
  if (HasAccessFlag(check, AccessCheckFlags::Execute)) {
    // On Unix, `X_OK` on a directory checks search/traversal permission.
    // `llvm::sys::fs::access(..., AccessMode::Execute)` returns
    // `permission_denied` for directories on Windows, so allow directories.
    if (!llvm::sys::fs::is_directory(full)) {
      if (std::error_code ec =
              llvm::sys::fs::access(full, llvm::sys::fs::AccessMode::Execute)) {
        return PathError(ec, ErrorFormat, path, DirPathForError());
      }
    }
  }
  return true;
}

auto DirRef::Stat() -> ErrorOr<FileStatus, FdError> {
  if (!is_valid()) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "Dir::Stat on '{0}': ", DirPathForError());
  }
  FileStatus status;
  std::string dir_str = is_cwd_ ? std::string(".") : dir_path_->string();
  if (std::error_code ec = llvm::sys::fs::status(dir_str, status.status_)) {
    return FdError(ec, "Dir::Stat on '{0}': ", DirPathForError());
  }
  return status;
}

auto DirRef::Stat(const std::filesystem::path& path)
    -> ErrorOr<FileStatus, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Stat on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  FileStatus status;
  if (std::error_code ec = llvm::sys::fs::status(
          resolved.string(), status.status_, /*follow=*/true)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return status;
}

auto DirRef::Lstat(const std::filesystem::path& path)
    -> ErrorOr<FileStatus, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Lstat on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  FileStatus status;
  std::string full = resolved.string();
  if (std::error_code ec =
          llvm::sys::fs::status(full, status.status_, /*follow=*/false)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  if (IsReparsePoint(full)) {
    llvm::SmallString<256> target;
    if (!llvm::sys::fs::readlink(full, target)) {
      status.is_symlink_ = true;
      status.symlink_size_ = static_cast<int64_t>(target.size());
    }
  }
  return status;
}

auto DirRef::UpdateTimes(const std::filesystem::path& path,
                         std::optional<TimePoint> time_point)
    -> ErrorOr<Success, PathError> {
  llvm::StringLiteral error_format =
      time_point ? llvm::StringLiteral(
                       "Dir::UpdateTimes to a specific time on '{0}' relative "
                       "to '{1}'")
                 : llvm::StringLiteral(
                       "Dir::UpdateTimes to now on '{0}' relative to '{1}'");
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, error_format));
  TimePoint target_time =
      time_point ? *time_point
                 : std::chrono::time_point_cast<Duration>(Clock::now());
  if (std::error_code ec = llvm::sys::fs::setLastAccessAndModificationTime(
          resolved.string(), target_time)) {
    return PathError(ec, error_format, path, DirPathForError());
  }
  return Success();
}

auto DirRef::Readlink(const std::filesystem::path& path)
    -> ErrorOr<std::string, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Readlink on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  llvm::SmallString<256> target;
  if (std::error_code ec = llvm::sys::fs::readlink(resolved.string(), target)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return std::string(target);
}

auto DirRef::OpenFileImpl(const std::filesystem::path& path, OpenAccess access,
                          CreationOptions creation_options,
                          ModeType creation_mode, OpenFlags flags)
    -> ErrorOr<int, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Open on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  std::string full = resolved.string();
  bool has_truncate = HasFlag(flags, OpenFlags::Truncate);
  bool has_append = HasFlag(flags, OpenFlags::Append);

  if (HasFlag(flags, OpenFlags::NoFollow) && IsReparsePoint(full)) {
    return PathError(
        std::make_error_code(std::errc::too_many_symbolic_link_levels),
        ErrorFormat, path, DirPathForError());
  }

  llvm::sys::fs::FileAccess llvm_access = llvm::sys::fs::FA_Read;
  switch (access) {
    case OpenAccess::ReadOnly:
      llvm_access = llvm::sys::fs::FA_Read;
      break;
    case OpenAccess::WriteOnly:
      llvm_access = llvm::sys::fs::FA_Write;
      break;
    case OpenAccess::ReadWrite:
      llvm_access = llvm::sys::fs::FA_Read | llvm::sys::fs::FA_Write;
      break;
  }

  llvm::sys::fs::CreationDisposition disp = llvm::sys::fs::CD_OpenExisting;
  bool truncate_after_open = false;
  switch (creation_options) {
    case OpenExisting:
      disp = llvm::sys::fs::CD_OpenExisting;
      truncate_after_open = has_truncate;
      break;
    case OpenAlways:
      disp = has_truncate ? llvm::sys::fs::CD_CreateAlways
                          : llvm::sys::fs::CD_OpenAlways;
      break;
    case CreateAlways:
      disp = llvm::sys::fs::CD_CreateAlways;
      break;
    case CreateNew:
      disp = llvm::sys::fs::CD_CreateNew;
      break;
  }

  bool existed_before =
      (creation_options == OpenAlways || creation_options == CreateAlways) &&
      llvm::sys::fs::exists(full);

  llvm::Expected<llvm::sys::fs::file_t> handle = llvm::sys::fs::openNativeFile(
      full, disp, llvm_access, llvm::sys::fs::OF_None, creation_mode);
  if (!handle) {
    return PathError(llvm::errorToErrorCode(handle.takeError()), ErrorFormat,
                     path, DirPathForError());
  }

  int crt_flags = _O_BINARY;
  if (access == OpenAccess::ReadOnly) {
    crt_flags |= _O_RDONLY;
  }
  if (has_append) {
    crt_flags |= _O_APPEND;
  }
  int fd =
      _open_osfhandle(reinterpret_cast<intptr_t>(handle->get()), crt_flags);
  if (fd == -1) {
    int errnum = errno;
    llvm::sys::fs::file_t f = *handle;
    (void)llvm::sys::fs::closeFile(f);
    return PathError(ErrnoToErrorCode(errnum), ErrorFormat, path,
                     DirPathForError());
  }

  if (truncate_after_open) {
    if (std::error_code ec = ResizeFileByFd(fd, 0)) {
      _close(fd);
      return PathError(ec, "Dir::Open truncate on '{0}' relative to '{1}'",
                       path, DirPathForError());
    }
  }

  // Apply read-only attribute if the file was newly created without write bits.
  bool created =
      creation_options == CreateNew ||
      ((creation_options == OpenAlways || creation_options == CreateAlways) &&
       !existed_before);
  if (created && (creation_mode & 0222) == 0) {
    (void)llvm::sys::fs::setPermissions(
        full, static_cast<llvm::sys::fs::perms>(creation_mode));
  }

  return fd;
}

auto DirRef::OpenDir(const std::filesystem::path& path,
                     CreationOptions creation_options, ModeType creation_mode,
                     OpenFlags open_flags) -> ErrorOr<Dir, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Opening directory '{0}' relative to '{1}' during DirRef::OpenDir";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  std::string full = resolved.string();

  if (creation_options != OpenExisting) {
    CARBON_CHECK(creation_options != CreateAlways,
                 "Invalid `creation_options` value of `CreateAlways`: there is "
                 "no support for truncating directories, and so they cannot be "
                 "created in an analogous way to files if they already exist.");

    bool ignore_existing = creation_options == OpenAlways;
    if (std::error_code ec = llvm::sys::fs::create_directory(
            full, ignore_existing,
            static_cast<llvm::sys::fs::perms>(creation_mode))) {
      return PathError(ec,
                       "Creating directory '{0}' relative to '{1}' during "
                       "DirRef::OpenDir",
                       path, DirPathForError());
    }
  }

  bool follow = !HasFlag(open_flags, OpenFlags::NoFollow);
  if (!follow && IsReparsePoint(full)) {
    return PathError(
        std::make_error_code(std::errc::too_many_symbolic_link_levels),
        ErrorFormat, path, DirPathForError());
  }
  llvm::sys::fs::file_status st;
  if (std::error_code ec = llvm::sys::fs::status(full, st, follow)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  if (!llvm::sys::fs::is_directory(st)) {
    return PathError(std::make_error_code(std::errc::not_a_directory),
                     ErrorFormat, path, DirPathForError());
  }

  // Resolve to an absolute path so the opened `Dir` remains stable across
  // subsequent `Chdir` calls.
  llvm::SmallString<256> abs_path(full);
  if (std::error_code ec = llvm::sys::fs::make_absolute(abs_path)) {
    return PathError(ec,
                     "Resolving directory '{0}' relative to '{1}' during "
                     "DirRef::OpenDir",
                     path, DirPathForError());
  }

  return Dir(new std::filesystem::path(std::string(abs_path)));
}

auto DirRef::Rename(const std::filesystem::path& path, DirRef target_dir,
                    const std::filesystem::path& target_path)
    -> ErrorOr<Success, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Rename on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved_from,
                          ResolvePath(path, ErrorFormat));
  CARBON_ASSIGN_OR_RETURN(
      std::filesystem::path resolved_to,
      target_dir.ResolvePath(target_path,
                             "Dir::Rename target '{0}' relative to '{1}'"));
  std::string from = resolved_from.string();
  std::string to = resolved_to.string();
  if (std::error_code ec = llvm::sys::fs::rename(from, to)) {
    // On Windows, renaming a directory over an existing directory fails with
    // `permission_denied` rather than `directory_not_empty`. Normalize that so
    // callers checking `error.not_empty()` (such as concurrent runtime cache
    // commits) behave consistently with POSIX.
    if (ec == std::errc::permission_denied &&
        llvm::sys::fs::is_directory(from) && llvm::sys::fs::is_directory(to)) {
      ec = std::make_error_code(std::errc::directory_not_empty);
    }
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return Success();
}

auto DirRef::Chdir() -> ErrorOr<Success, FdError> {
  if (is_cwd_) {
    return Success();
  }
  if (dir_path_ == nullptr) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "Dir::Chdir on '{0}'", DirPathForError());
  }
  if (std::error_code ec =
          llvm::sys::fs::set_current_path(dir_path_->string())) {
    return FdError(ec, "Dir::Chdir on '{0}'", DirPathForError());
  }
  return Success();
}

auto DirRef::Chdir(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Chdir on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  std::string full = resolved.string();
  llvm::sys::fs::file_status st;
  if (std::error_code ec = llvm::sys::fs::status(full, st)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  if (!llvm::sys::fs::is_directory(st)) {
    return PathError(std::make_error_code(std::errc::not_a_directory),
                     ErrorFormat, path, DirPathForError());
  }
  if (std::error_code ec = llvm::sys::fs::set_current_path(full)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return Success();
}

auto DirRef::Symlink(const std::filesystem::path& path,
                     const std::string& target) -> ErrorOr<Success, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Symlink on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  // Match POSIX semantics: ignore everything after the first embedded null.
  if (std::error_code ec =
          llvm::sys::fs::create_symlink(target.c_str(), resolved.string())) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return Success();
}

auto DirRef::Unlink(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Unlink on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  std::string full = resolved.string();
  std::wstring wide_path;
  if (std::error_code ec = ToWidePath(full, wide_path)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  DWORD attrs = ::GetFileAttributesW(wide_path.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    return PathError(llvm::mapWindowsError(::GetLastError()), ErrorFormat, path,
                     DirPathForError());
  }
  // `Unlink` must not remove non-symlink directories (callers such as `Rmtree`
  // rely on `is_dir()` to distinguish directories from files).
  if ((attrs & FILE_ATTRIBUTE_DIRECTORY) != 0 &&
      (attrs & FILE_ATTRIBUTE_REPARSE_POINT) == 0) {
    return PathError(std::make_error_code(std::errc::is_a_directory),
                     ErrorFormat, path, DirPathForError());
  }
  // Windows refuses to delete read-only files unless the read-only attribute is
  // cleared first.
  if ((attrs & FILE_ATTRIBUTE_READONLY) != 0) {
    DWORD new_attrs = attrs & ~FILE_ATTRIBUTE_READONLY;
    ::SetFileAttributesW(wide_path.c_str(),
                         new_attrs ? new_attrs : FILE_ATTRIBUTE_NORMAL);
  }
  if (std::error_code ec =
          llvm::sys::fs::remove(full, /*IgnoreNonExisting=*/false)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return Success();
}

auto DirRef::Rmdir(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  constexpr llvm::StringLiteral ErrorFormat =
      "Dir::Rmdir on '{0}' relative to '{1}'";
  CARBON_ASSIGN_OR_RETURN(std::filesystem::path resolved,
                          ResolvePath(path, ErrorFormat));
  std::string full = resolved.string();
  std::wstring wide_path;
  if (std::error_code ec = ToWidePath(full, wide_path)) {
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  if (!::RemoveDirectoryW(wide_path.c_str())) {
    DWORD err = ::GetLastError();
    std::error_code ec = err == ERROR_DIRECTORY
                             ? std::make_error_code(std::errc::not_a_directory)
                             : llvm::mapWindowsError(err);
    return PathError(ec, ErrorFormat, path, DirPathForError());
  }
  return Success();
}

auto Dir::TakeAndRead() && -> ErrorOr<Reader, FdError> {
  if (!is_valid()) {
    return FdError(std::make_error_code(std::errc::bad_file_descriptor),
                   "Dir::Read on '{0}'", DirPathForError());
  }
  std::filesystem::path* dir_path = std::exchange(dir_path_, nullptr);
  bool is_cwd = std::exchange(is_cwd_, false);
  return Reader(dir_path, is_cwd);
}

DirRef::Iterator::Iterator(const std::filesystem::path* dir_path, bool is_cwd) {
  CARBON_CHECK(is_cwd || dir_path != nullptr, "Invalid directory reader");
  std::string path_str = is_cwd ? std::string(".") : dir_path->string();
  std::error_code ec;
  iter_ = llvm::sys::fs::directory_iterator(path_str, ec,
                                            /*follow_symlinks=*/false);
  CARBON_CHECK(!ec, "Failed to iterate directory '{0}': {1}", path_str,
               ec.message());
  PopulateEntry();
}

auto DirRef::Iterator::PopulateEntry() -> void {
  if (iter_ == llvm::sys::fs::directory_iterator()) {
    entry_ = Entry();
    return;
  }
  llvm::StringRef full_path = iter_->path();
  entry_.name_ = llvm::sys::path::filename(full_path).str();
  entry_.type_ = iter_->type();
  entry_.is_symlink_ = IsReparsePoint(full_path);
}

auto DirRef::Iterator::operator++() -> Iterator& {
  CARBON_CHECK(iter_ != llvm::sys::fs::directory_iterator(),
               "Cannot increment an end-iterator");
  std::error_code ec;
  iter_.increment(ec);
  CARBON_CHECK(!ec, "Error advancing directory iterator: {0}", ec.message());
  PopulateEntry();
  return *this;
}

auto DirRef::Reader::begin() -> Iterator {
  return Iterator(dir_path_, is_cwd_);
}

auto MakeTmpDir() -> ErrorOr<RemovingDir, Error> {
  std::filesystem::path tmpdir_path;
  if (const char* test_tmpdir = std::getenv("TEST_TMPDIR");
      test_tmpdir != nullptr && test_tmpdir[0] != '\0') {
    tmpdir_path = test_tmpdir;
  } else {
    llvm::SmallString<128> sys_tmp;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, sys_tmp);
    tmpdir_path = std::string(sys_tmp);
  }

  std::filesystem::path target = BuildData::BuildTarget.str();
  tmpdir_path /= target.filename();
  return MakeTmpDirWithPrefix(std::move(tmpdir_path));
}

auto MakeTmpDirWithPrefix(std::filesystem::path prefix)
    -> ErrorOr<RemovingDir, Error> {
  constexpr int MaxAttempts = 128;
  constexpr char HexDigits[] = "0123456789abcdef";
  std::string prefix_str = prefix.string();
  std::string candidate;
  std::error_code last_ec;
  for (int attempt = 0; attempt < MaxAttempts; ++attempt) {
    uint32_t rand_val = llvm::sys::Process::GetRandomNumber();
    candidate = prefix_str;
    candidate.push_back('.');
    for (int i = 0; i < 6; ++i) {
      candidate.push_back(HexDigits[(rand_val >> (i * 4)) & 0xf]);
    }
    last_ec = llvm::sys::fs::create_directory(candidate,
                                              /*IgnoreExisting=*/false);
    if (!last_ec) {
      std::filesystem::path tmpdir_path(candidate);
      CARBON_ASSIGN_OR_RETURN(
          Dir tmp, Cwd().OpenDir(tmpdir_path, OpenExisting, /*creation_mode=*/0,
                                 OpenFlags::NoFollow));
      return RemovingDir(std::move(tmp), std::move(tmpdir_path));
    }
    if (last_ec != std::errc::file_exists) {
      break;
    }
  }
  return Error(llvm::formatv("Creating temporary directory with prefix '{0}' "
                             "failed: {1}",
                             prefix_str, last_ec.message())
                   .str());
}

}  // namespace Carbon::Filesystem

#endif  // _WIN32
