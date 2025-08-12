// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/filesystem.h"

#include <fcntl.h>
#include <unistd.h>

#include "common/build_data.h"
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
  out << llvm::formatv(format_.data(), path_, dir_fd_) << " failed: ";
  PrintErrorNumber(out, unix_errnum());
}

auto Internal::FileRefBase::ReadToString() -> ErrorOr<std::string, FdError> {
  std::string result;

  // Read a buffer at a time until we reach the end. We use the pipe buffer
  // length as our max buffer size as it is likely to be small but reasonable
  // for the OS, and in the case of pipes the same chunking in which the data
  // will arrive.
  //
  // TODO: Replace this with a smaller buffer and using `resize_and_overwrite`
  // to read into the string in-place for larger strings. Unclear if that will
  // be any faster, but it will be much more friendly to callers with
  // constrained stack sizes and use less memory overall.
  std::byte buffer[PIPE_BUF];
  for (;;) {
    auto read_result = ReadToBuffer(buffer);
    if (!read_result.ok()) {
      return std::move(read_result).error();
    }
    if (read_result->empty()) {
      // EOF
      break;
    }
    result.append(reinterpret_cast<const char*>(read_result->data()),
                  read_result->size());
  }

  return result;
}

auto Internal::FileRefBase::WriteFromString(llvm::StringRef str)
    -> ErrorOr<Success, FdError> {
  auto bytes = llvm::ArrayRef<std::byte>(
      reinterpret_cast<const std::byte*>(str.data()), str.size());
  while (!bytes.empty()) {
    auto write_result = WriteFromBuffer(bytes);
    if (!write_result.ok()) {
      return std::move(write_result).error();
    }
    bytes = *write_result;
  }
  return Success();
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

auto DirRef::ReadFileToString(const std::filesystem::path& path)
    -> ErrorOr<std::string, PathError> {
  CARBON_ASSIGN_OR_RETURN(ReadFile f, OpenReadOnly(path));
  auto result = f.ReadToString();
  if (result.ok()) {
    return *std::move(result);
  }
  return PathError(result.error().unix_errnum(),
                   "Dir::ReadFileToString on '{0}' relative to '{1}'", path,
                   dfd_);
}

auto DirRef::WriteFileFromString(const std::filesystem::path& path,
                                 llvm::StringRef content,
                                 CreationOptions creation_options)
    -> ErrorOr<Success, PathError> {
  CARBON_ASSIGN_OR_RETURN(WriteFile f, OpenWriteOnly(path, creation_options));
  auto write_result = f.WriteFromString(content);
  if (!write_result.ok()) {
    return PathError(
        write_result.error().unix_errnum(),
        "Write error in Dir::WriteFileFromString on '{0}' relative to '{1}'",
        path, dfd_);
  }
  auto close_result = std::move(f).Close();
  if (!close_result.ok()) {
    return PathError(
        close_result.error().unix_errnum(),
        "Close error in Dir::WriteFileFromString on '{0}' relative to '{1}'",
        path, dfd_);
  }
  return Success();
}

auto DirRef::CreateDirectories(const std::filesystem::path& path,
                               ModeType creation_mode)
    -> ErrorOr<Dir, PathError> {
  // Avoid having to handle an empty path by immediately rejecting it as
  // invalid.
  if (path.empty()) {
    return PathError(EINVAL,
                     "DirRef::CreateDirectories on '{0}' relative to '{1}'",
                     path, dfd_);
  }
  // Try directly opening the directory and use that if successful. This is an
  // important hot path case of users essentially doing an "open-always" form of
  // creating multiple steps of directories.
  auto open_result = OpenDir(path, OpenExisting);
  if (open_result.ok()) {
    return std::move(*open_result);
  } else if (!open_result.error().no_entity()) {
    return std::move(open_result).error();
  }

  // Walk from the full path towards this directory (or the root) to find the
  // first existing directory. This is faster than walking down as no file
  // descriptors have to be allocated for any intervening directories, etc. We
  // keep the path components that are missing as we pop them off for easy
  // traversal back down.
  std::optional<Dir> work_dir;
  // Paths typically consist of relatively few components
  // and so we can use a bit of stack and avoid allocating and moving the paths
  // in common cases. We use `8` as an arbitrary but likely good for all of the
  // hottest cases.
  llvm::SmallVector<std::filesystem::path, 8> missing_components;
  missing_components.push_back(path.filename());
  for (std::filesystem::path parent_path = path.parent_path();
       !parent_path.empty(); parent_path = parent_path.parent_path()) {
    auto open_result = OpenDir(parent_path, OpenExisting);
    if (open_result.ok()) {
      work_dir = std::move(*open_result);
      break;
    }
    missing_components.push_back(parent_path.filename());
  }
  CARBON_CHECK(!missing_components.empty());

  // If we haven't yet opened an intermediate directory, start by creating one
  // relative to this directory. We can't do this as part of the loop below as
  // `this` and the newly opened directory have different types.
  if (!work_dir) {
    std::filesystem::path component = missing_components.pop_back_val();
    CARBON_ASSIGN_OR_RETURN(
        Dir component_dir,
        OpenDir(component, CreationOptions::OpenAlways, creation_mode));
    // Move this component into our temporary directory slot.
    work_dir = std::move(component_dir);
  }

  // Now walk through the remaining components opening and creating each
  // relative to the previous.
  while (!missing_components.empty()) {
    std::filesystem::path component = missing_components.pop_back_val();
    CARBON_ASSIGN_OR_RETURN(
        Dir component_dir,
        work_dir->OpenDir(component, CreationOptions::OpenAlways,
                          creation_mode));

    // Close the current temporary directory and move the new component
    // directory object into its place.
    work_dir = std::move(component_dir);
  }

  CARBON_CHECK(work_dir,
               "Should always have created at least one directory for a "
               "non-empty path!");
  return std::move(work_dir).value();
}

auto DirRef::Rmtree(const std::filesystem::path& path)
    -> ErrorOr<Success, PathError> {
  struct DirAndIterator {
    DirRef::Reader dir;
    ssize_t dir_entry_start;
  };
  llvm::SmallVector<DirAndIterator> dir_stack;

  llvm::SmallVector<std::filesystem::path> dir_entries;
  llvm::SmallVector<std::filesystem::path> unknown_entries;

  dir_entries.push_back(path);
  for (;;) {
    // When we bottom out, we're removing the initial tree path and doing so
    // relative to `this` directory.
    DirRef current = dir_stack.empty() ? *this : dir_stack.back().dir;
    ssize_t dir_entry_start =
        dir_stack.empty() ? 0 : dir_stack.back().dir_entry_start;

    // If we've finished all the child directories of the current entry in the
    // stack, pop it off and continue.
    if (dir_entry_start == static_cast<ssize_t>(dir_entries.size())) {
      dir_stack.pop_back();
      continue;
    }
    CARBON_CHECK(dir_entry_start < static_cast<ssize_t>(dir_entries.size()));

    // Take the last entry under the current directory and try removing it.
    const std::filesystem::path& entry_path = dir_entries.back();
    auto rmdir_result = current.Rmdir(entry_path);
    if (rmdir_result.ok() || rmdir_result.error().no_entity()) {
      // Removed here or elsewhere already, so pop the entry.
      dir_entries.pop_back();
      if (dir_entries.empty()) {
        // The last entry is the input path with an empty stack, so we've
        // finished at this point.
        CARBON_CHECK(dir_stack.empty());
        return Success();
      }
      continue;
    }
    // If we get any error other than not-empty, just return that.
    if (!rmdir_result.error().not_empty()) {
      return std::move(rmdir_result).error();
    }

    // Recurse into the subdirectory since it isn't empty, opening it, getting a
    // reader, and pushing it onto our stack.
    CARBON_ASSIGN_OR_RETURN(Dir subdir, current.OpenDir(entry_path));
    auto read_result = std::move(subdir).TakeAndRead();
    if (!read_result.ok()) {
      return PathError(
          read_result.error().unix_errnum(),
          "Dir::Read on '{0}' relative to '{1}' during RmdirRecursively",
          entry_path, current.dfd_);
    }
    dir_stack.push_back(
        {*std::move(read_result), static_cast<ssize_t>(dir_entries.size())});

    // Now read the directory entries. It would be nice to be able to directly
    // remove the files and empty directories as we find them when reading, and
    // the POSIX spec appears to require that to work, but testing shows at
    // least some Linux environments don't work reliably in this case and will
    // fail to visit some entries entirely. As a consequence, we walk the entire
    // directory and collect the entries into data structures before beginning
    // to remove them.
    DirRef::Reader& subdir_reader = dir_stack.back().dir;
    for (const auto& entry : subdir_reader) {
      llvm::StringRef name = entry.name();
      if (name == "." || name == "..") {
        continue;
      }
      if (entry.is_known_dir()) {
        dir_entries.push_back(name.str());
      } else {
        // We end up here for entries known to be regular files, other kinds of
        // non-directory entries, or when the entry kind isn't known.
        //
        // Unless we *know* the entry is a directory, we put it into the unknown
        // entries. For these, we unlink them first in case they are
        // non-directory entries and use the failure of that to move any
        // directories that end up here to the directory entries list.
        unknown_entries.push_back(name.str());
      }
    }

    // We can immediately try to unlink all the unknown entries, which will
    // include any regular files, and use an error on directories that were
    // unknown above to switch them to the `dir_entries` list.
    while (!unknown_entries.empty()) {
      std::filesystem::path name = unknown_entries.pop_back_val();
      auto unlink_result = subdir_reader.Unlink(name);
      if (unlink_result.ok() || unlink_result.error().no_entity()) {
        continue;
      } else if (!unlink_result.error().is_dir()) {
        return std::move(unlink_result).error();
      }
      dir_entries.push_back(std::move(name));
    }

    // We'll handle the directory entries we've queued here in the next
    // iteration, removing them or recursing as needed.
  }
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
  large_buffer.resize(status.size());
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
    if (!tmpdir_env.is_absolute()) {
      continue;
    }
    tmpdir_path = std::move(tmpdir_env);
    break;
  }

  std::filesystem::path target = BuildData::BuildTarget.str();
  tmpdir_path /= target.filename();
  tmpdir_path += ".XXXXXX";

  std::string tmpdir_path_buffer = tmpdir_path.native();
  char* result = mkdtemp(tmpdir_path_buffer.data());
  if (result == nullptr) {
    RawStringOstream os;
    os << llvm::formatv("Calling mkdtemp on '{0}' failed: ",
                        tmpdir_path.native());
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
                      tmpdir_path.native())
            .str());
  }

  return result_dir;
}

}  // namespace Carbon::Filesystem
