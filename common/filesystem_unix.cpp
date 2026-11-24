// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _WIN32

#include <fcntl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <utility>

#include "common/build_data.h"
#include "common/filesystem.h"
#include "llvm/Support/MathExtras.h"

namespace Carbon::Filesystem {

// Render an error number from `errno` to the provided stream using the richest
// rendering available on the platform.
static auto PrintErrorNumber(llvm::raw_ostream& out, int errnum) -> void {
#if defined(_GNU_SOURCE) && \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32))
  // For sufficiently recent glibc versions, use GNU-specific routines to
  // compute the error name and description.
  llvm::StringRef name = strerrordesc_np(errnum);
  llvm::StringRef desc = strerrorname_np(errnum);

  out << llvm::formatv("{0}: {1}", name, desc);
#elif defined(__APPLE__) || defined(_GNU_SOURCE) || defined(_POSIX_SOURCE)
  // Broadly portable fallback for Unix-like systems.
  char buffer[4096];
#ifdef _GNU_SOURCE
  const char* str = strerror_r(errnum, buffer, sizeof(buffer));
  // The GNU version doesn't report a meta-error.
  int meta_error = 0;
#else
  int meta_error = strerror_r(errnum, buffer, sizeof(buffer));
  const char* str = buffer;
#endif
  if (meta_error == 0) {
    out << llvm::formatv("errno {0}: {1}", errnum, llvm::StringRef(str));
  } else {
    out << llvm::formatv(
        "error number {0}; encountered meta-error number {1} while rendering "
        "an error message",
        errnum, meta_error);
  }
#else
#error TODO: Implement this for other platforms.
#endif
}

auto FdError::Print(llvm::raw_ostream& out) const -> void {
  // The `format_` member is a `StringLiteral` that is null terminated, so
  // `.data()` is safe here.
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
  out << llvm::formatv(format_.data(), fd_) << " failed: ";
  PrintErrorNumber(out, unix_errnum());
}

auto PathError::Print(llvm::raw_ostream& out) const -> void {
  // The `format_` member is a `StringLiteral` that is null terminated, so
  // `.data()` is safe here.
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
  out << llvm::formatv(format_.data(), path_.string(),
                       dir_fd_ == AT_FDCWD ? std::string("AT_FDCWD")
                                           : std::to_string(dir_fd_))
      << " failed: ";
  PrintErrorNumber(out, unix_errnum());
}

// A macOS specific sleep routine that builds on more standard utilities. This
// is technically a portable implementation so we always compile it but only use
// it on macOS where the more efficient direct use of `clock_nanosleep` isn't
// available.
[[maybe_unused]]
static auto SleepMacos(Duration sleep) -> void {
  TimePoint stop = Clock::now() + sleep;

  timespec sleep_ts = Internal::DurationToTimespec(sleep);
  for (;;) {
    timespec rem_sleep_ts = {};
    int result = nanosleep(&sleep_ts, &rem_sleep_ts);
    if (result == 0) {
      return;
    }

    // Continue sleeping if we get interrupted by a resumable signal. For
    // everything else report it.
    if (errno != EINTR) {
      int errnum = errno;
      RawStringOstream error_os;
      PrintErrorNumber(error_os, errnum);
      CARBON_FATAL("Unexpected error while sleeping: {0}", error_os.TakeStr());
    }

    // Update to the remaining sleep time for the next attempt at sleeping.
    sleep_ts = rem_sleep_ts;

    // Also check if the clock has passed our stop time as a fallback to avoid
    // too much clock skew.
    if (Clock::now() > stop) {
      return;
    }
  }
}

static auto Sleep(Duration sleep) -> void {
  // For every platform but macOS we can sleep directly on an absolute time.
#if __APPLE__
  // On Apple platforms, dispatch to a specialized routine.
  SleepMacos(sleep);
#else

  // We use `clock_gettime` instead of the filesystem `Clock` or some other
  // `std::chrono` clock because we want to use the exact same clock that we'll
  // use for sleeping below, and we'll need the time in a `timespec` for that
  // call anyways. We do use a monotonic clock to try and avoid sleeps being
  // interrupted by clock changes.
  timespec ts = {};
  int result = clock_gettime(CLOCK_MONOTONIC, &ts);
  CARBON_CHECK(result == 0, "Error getting the time: {0}", strerror(errno));

  // Add the sleep duration to the current monotonic timestamp with carry on
  // the nanosecond component.
  timespec delta = Internal::DurationToTimespec(sleep);
  ts.tv_sec += delta.tv_sec;
  ts.tv_nsec += delta.tv_nsec;
  constexpr long NsecPerSec = 1'000'000'000;
  if (ts.tv_nsec >= NsecPerSec) {
    ts.tv_sec += ts.tv_nsec / NsecPerSec;
    ts.tv_nsec %= NsecPerSec;
  }

  do {
    result = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);

    // Continue sleeping if we get interrupted by a resumable signal. Because
    // we're using a monotonic clock and an absolute deadline time we will
    // eventually progress past that deadline.
  } while (result != 0 && (errno == EINTR));

  if (result != 0) {
    int errnum = errno;
    RawStringOstream error_os;
    PrintErrorNumber(error_os, errnum);
    CARBON_FATAL("Unexpected error while sleeping: {0}", error_os.TakeStr());
  }
#endif
}

auto Internal::FileRefBase::TryLock(FileLock::Kind kind, Duration deadline,
                                    Duration poll_interval)
    -> ErrorOr<FileLock, FdError> {
  CARBON_CHECK(poll_interval <= deadline);
  if (deadline != Duration(0) && poll_interval == Duration(0)) {
    // If the caller didn't provide a poll interval but did provide a deadline,
    // pick a poll interval to roughly be 1/1000th of the deadline but at least
    // 1 microsecond. We don't support polling faster than 1 microsecond given
    // how expensive file locking is.
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
  auto stop = Clock::now() + deadline;
  for (;;) {
    int result = flock(fd_, static_cast<int>(kind) | LOCK_NB);
    if (result == 0) {
      return FileLock(fd_);
    }

    // Return an error if this is something other than blocking for the lock to
    // be available, or we didn't get a deadline for continuing to try and
    // acquire the lock, or we've reached our deadline.
    if (errno != EWOULDBLOCK || deadline == Duration(0) ||
        Clock::now() >= stop) {
      return FdError(errno, "File::TryLock on '{0}'", fd_);
    }

    // The caller requested attempting to wait up to a deadline to acquire the
    // lock with a specific poll interval. Try to sleep for that poll interval
    // before trying the lock again.
    Sleep(poll_interval);
  }
}

auto DirRef::OpenDir(const std::filesystem::path& path,
                     CreationOptions creation_options, ModeType creation_mode,
                     OpenFlags open_flags) -> ErrorOr<Dir, PathError> {
  // If we potentially need to create a directory, we have to do that
  // separately as no systems support `O_CREAT | O_DIRECTORY`, even though
  // that would be (much) nicer.

  if (creation_options == CreateNew) {
    // If we are required to be the one that created the directory, disable
    // following the last symlink when we open that directory. The last symlink
    // is the only one that matters for security here because it is only valid
    // to create the last component. It is that directory component that we want
    // to ensure has not been replaced with a symlink by an adversarial
    // concurrent process.
    open_flags |= OpenFlags::NoFollow;
  }

  if (creation_options != OpenExisting) {
    CARBON_CHECK(creation_options != CreateAlways,
                 "Invalid `creation_options` value of `CreateAlways`: there is "
                 "no support for truncating directories, and so they cannot be "
                 "created in an analogous way to files if they already exist.");

    if (mkdirat(dfd_, path.c_str(), creation_mode) != 0) {
      // Unless the error is just that the path already exists, and that is
      // allowed for the requested creation flags, report any error here as part
      // of opening just like we would if the error originated from `openat`
      // with `O_CREAT`.
      if (creation_options == CreateNew || errno != EEXIST) {
        return PathError(errno,
                         "Calling `mkdirat` on '{0}' relative to '{1}' during "
                         "DirRef::OpenDir",
                         path, dfd_);
      }
    }
  }

  // Open this path as a directory. Note that this has to succeed, and when we
  // created the directory we require the last component to not be a symlink in
  // case it was _replaced_ with a symlink while running.
  int result_fd =
      openat(dfd_, path.c_str(), static_cast<int>(open_flags) | O_DIRECTORY);
  if (result_fd == -1) {
    // No need for `EINTR` handling here as if this is a FIFO it would be an
    // error with `O_DIRECTORY`.
    return PathError(
        errno,
        "Calling `openat` on '{0}' relative to '{1}' during DirRef::OpenDir",
        path, dfd_);
  }
  Dir result(result_fd);

  // If we were required to create the directory, we also need to verify that
  // the opened file descriptor continues to have the same permissions and the
  // correct owner as we couldn't do the creation atomically with the open. This
  // defends against an adversarial removal of the created directory and
  // creation of a new directory with the same name but either with wider
  // permissions such as all-write, or with a different owner.
  //
  // We don't defend against replacement with a directory of the same name, same
  // permissions, same owner, but different group. There is no good way to do
  // this defense given the complexity of group assignment, and there appears to
  // be no need. Achieving such a replacement without superuser power would
  // require a parent directory with `setgid` bit, and a group that gives the
  // attacker access -- but such a parent directory would make *any* creation
  // vulnerable without any need for a replacement, so we can't defend against
  // that here. The caller has ample tools to defend against this including
  // taking care with the parent directory and restricting the group permission
  // bits which we *do* verify.
  if (creation_options == CreateNew) {
    auto stat_result = result.Stat();
    if (!stat_result.ok()) {
      // Manually propagate this error so we can attach it back to the opened
      // path and relative directory.
      return PathError(stat_result.error().unix_errnum(),
                       "DirRef::Stat after opening '{0}' relative to '{1}'",
                       path, dfd_);
    }

    // Check that the owning UID is the current effective UID.
    if (stat_result->unix_uid() != geteuid()) {
      // Model this as `EPERM`, which is a bit awkward, but should be fine.
      return PathError(EPERM,
                       "Unexpected UID change after creating '{0}' relative to "
                       "'{1}' during DirRef::OpenDir",
                       path, dfd_);
    }

    // Check that the permissions are a subset of the requested ones. They may
    // have been masked down by `umask`, but if there are *new* permissions,
    // that would be a security issue.
    if ((stat_result->permissions() & creation_mode) !=
        stat_result->permissions()) {
      // Model this with `EPERM` and a custom message.
      return PathError(EPERM,
                       "Unexpected permissions after creating '{0}' relative "
                       "to '{1}' during DirRef::OpenDir",
                       path, dfd_);
    }
  }

  return result;
}

auto DirRef::ReadlinkSlow(const std::filesystem::path& path)
    -> ErrorOr<std::string, PathError> {
  constexpr ssize_t MinBufferSize =
#ifdef PATH_MAX
      PATH_MAX
#else
      1024
#endif
      ;
  // Read directly into a string to avoid allocating two large buffers.
  std::string large_buffer;
  // Stat the symlink to get an initial guess at the size.
  CARBON_ASSIGN_OR_RETURN(FileStatus status, Lstat(path));
  // We try to use the size from the `lstat` unless it is empty, in which case
  // we try to use our minimum buffer size which is `PATH_MAX` or a constant
  // value. We have a fallback to dynamically discover an adequate buffer size
  // below that will handle any inaccuracy.
  ssize_t buffer_size = status.size();
  if (buffer_size == 0) {
    buffer_size = MinBufferSize;
  }
  large_buffer.resize(buffer_size);
  ssize_t result =
      readlinkat(dfd_, path.c_str(), large_buffer.data(), large_buffer.size());
  if (result == -1) {
    return PathError(errno, "Readlink on '{0}' relative to '{1}'", path, dfd_);
  }

  // Now the really bad fallback case: if there are racing writes to the
  // symlink, the guessed size may not have been large enough. As a last-ditch
  // effort, begin doubling (from the next power of two >= our min buffer size)
  // the length until it fits. We cap this at 10 MiB to prevent egregious file
  // system contents (or some bug somewhere) from exhausting memory.
  constexpr ssize_t MaxBufferSize = 10 << 20;
  while (result == static_cast<ssize_t>(large_buffer.size())) {
    int64_t next_buffer_size = std::max<ssize_t>(
        MinBufferSize, llvm::NextPowerOf2(large_buffer.size()));
    if (next_buffer_size > MaxBufferSize) {
      return PathError(ENOMEM, "Readlink on '{0}' relative to '{1}'", path,
                       dfd_);
    }
    large_buffer.resize(next_buffer_size);
    result = readlinkat(dfd_, path.c_str(), large_buffer.data(),
                        large_buffer.size());
    if (result == -1) {
      return PathError(errno, "Readlink on '{0}' relative to '{1}'", path,
                       dfd_);
    }
  }

  // Fix-up the size of the string and return it.
  large_buffer.resize(result);
  return large_buffer;
}

auto MakeTmpDir() -> ErrorOr<RemovingDir, Error> {
  std::filesystem::path tmpdir_path = "/tmp";
  // We use both `TEST_TMPDIR` and `TMPDIR`. The `TEST_TMPDIR` is set by Bazel
  // and preferred to keep tests using the expected output tree rather than
  // the system temporary directory.
  for (const char* tmpdir_env_name : {"TEST_TMPDIR", "TMPDIR"}) {
    const char* tmpdir_env_cstr = getenv(tmpdir_env_name);
    if (tmpdir_env_cstr == nullptr) {
      continue;
    }
    std::filesystem::path tmpdir_env = tmpdir_env_cstr;
    tmpdir_path = std::move(tmpdir_env);
    break;
  }

  std::filesystem::path target = BuildData::BuildTarget.str();
  tmpdir_path /= target.filename();
  return MakeTmpDirWithPrefix(std::move(tmpdir_path));
}

auto MakeTmpDirWithPrefix(std::filesystem::path prefix)
    -> ErrorOr<RemovingDir, Error> {
  std::filesystem::path tmpdir_path = std::move(prefix);
  tmpdir_path += ".XXXXXX";

  std::string tmpdir_path_buffer = tmpdir_path.string();
  char* result = mkdtemp(tmpdir_path_buffer.data());
  if (result == nullptr) {
    RawStringOstream os;
    os << llvm::formatv("Calling mkdtemp on '{0}' failed: ",
                        tmpdir_path.string());
    PrintErrorNumber(os, errno);
    return Error(os.TakeStr());
  }
  CARBON_CHECK(result == tmpdir_path_buffer.data(),
               "`mkdtemp` used a modified path");
  tmpdir_path = std::move(tmpdir_path_buffer);

  // Because `mkdtemp` doesn't return an open directory atomically, open the
  // created directory and perform safety checks similar to `OpenDir` when
  // creating a new directory.
  CARBON_ASSIGN_OR_RETURN(
      Dir tmp, Cwd().OpenDir(tmpdir_path, OpenExisting, /*creation_mode=*/0,
                             OpenFlags::NoFollow));
  // Make sure we try to remove the directory from here on out.
  RemovingDir result_dir(std::move(tmp), tmpdir_path);

  // It's a bit awkward to report `fstat` errors as `Error`s, but we
  // don't have much choice. The stat failing here would be very weird.
  CARBON_ASSIGN_OR_RETURN(FileStatus stat, result_dir.Stat());

  // The permissions must be exactly 0700 for a temporary directory, and the UID
  // should be ours.
  if (stat.permissions() != 0700 && stat.unix_uid() != geteuid()) {
    return Error(
        llvm::formatv("Found incorrect permissions or UID on tmpdir '{0}'",
                      tmpdir_path.string())
            .str());
  }

  return result_dir;
}

}  // namespace Carbon::Filesystem

#endif  // !_WIN32
