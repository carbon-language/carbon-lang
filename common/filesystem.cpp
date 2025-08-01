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
#ifdef _GNU_SOURCE
  // Use GNU-specific routines to compute the error name and description.
  llvm::StringRef name = strerrordesc_np(errnum);
  llvm::StringRef desc = strerrorname_np(errnum);

  out << llvm::formatv("{0}: {1}", name, desc);
#elif defined(__APPLE__) || defined(_POSIX_SOURCE)
  char buffer[4096];
  int meta_error = strerror_r(errnum, buffer, sizeof(buffer));
  if (meta_error == 0) {
    out << llvm::formatv("errno {0}: {1}", errnum, llvm::StringRef(buffer));
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
  PrintErrorNumber(out, number());
}

auto PathError::Print(llvm::raw_ostream& out) const -> void {
  // The `format_` member is a `StringLiteral` that is null terminated, so
  // `.data()` is safe here.
  // NOLINTNEXTLINE(bugprone-suspicious-stringview-data-usage)
  out << llvm::formatv(format_.data(), path_, dir_fd_) << " failed: ";
  PrintErrorNumber(out, number());
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

auto DirRef::ReadFileToString(std::filesystem::path path)
    -> ErrorOr<std::string, PathError> {
  CARBON_ASSIGN_OR_RETURN(ReadFile f, OpenReadOnly(path));
  auto result = f.ReadToString();
  if (result.ok()) {
    return *std::move(result);
  }
  return PathError(result.error().number(),
                   "Dir::ReadFileToString on '{0}' relative to '{1}'",
                   std::move(path), dfd_);
}

auto DirRef::WriteFileFromString(std::filesystem::path path,
                                 llvm::StringRef content,
                                 CreationOptions creation_flags)
    -> ErrorOr<Success, PathError> {
  CARBON_ASSIGN_OR_RETURN(WriteFile f, OpenWriteOnly(path, creation_flags));
  auto result = f.WriteFromString(content);
  if (result.ok()) {
    return Success();
  }
  return PathError(result.error().number(),
                   "Dir::WriteFileFromString on '{0}' relative to '{1}'",
                   std::move(path), dfd_);
}

auto DirRef::CreateDirectories(std::filesystem::path path,
                               ModeType creation_mode)
    -> ErrorOr<Dir, PathError> {
  // Avoid having to handle an empty path by immediately rejecting it as
  // invalid.
  if (path.empty()) {
    return PathError(EINVAL,
                     "DirRef::CreateDirectories on '{0}' relative to '{1}'",
                     std::move(path), dfd_);
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
  llvm::SmallVector<std::filesystem::path, 32> missing_components;
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

auto DirRef::Rmtree(std::filesystem::path path) -> ErrorOr<Success, PathError> {
  struct DirAndWorklists {
    DirRef::Reader dir;
    ssize_t dir_entries_start;
  };
  llvm::SmallVector<DirAndWorklists> dir_stack;
  llvm::SmallVector<std::filesystem::path> dir_entries;

  auto push_dir = [&](DirRef current_dir,
                      const std::filesystem::path& entry_path)
      -> ErrorOr<Success, PathError> {
    CARBON_ASSIGN_OR_RETURN(Dir subdir, current_dir.OpenDir(entry_path));

    auto read_result = std::move(subdir).TakeAndRead();
    if (!read_result.ok()) {
      return PathError(
          read_result.error().number(),
          "Dir::Read on '{0}' relative to '{1}' during RmdirRecursively",
          entry_path, current_dir.dfd_);
    }
    ssize_t dir_entries_start = dir_entries.size();

    for (const Dir::Entry& entry : *read_result) {
      llvm::StringRef name = entry.name();
      if (name == "." || name == "..") {
        continue;
      }

      // If we don't know this is a directory form the entry, try unlinking. For
      // unknown entries, the failure will tell us to fall through to the
      // directory case if needed without an extra stat.
      if (auto is_dir = entry.is_dir(); is_dir.has_value() && !*is_dir) {
        auto unlink_result = read_result->Unlink(name.str());
        if (unlink_result.ok() || unlink_result.error().no_entity()) {
          continue;
        } else if (!unlink_result.error().is_dir()) {
          return std::move(unlink_result).error();
        }
      }

      // This is a directory, so try to speculatively remove it. This will fail
      // for non-empty directories, but avoids opening, reading, and closing the
      // directory when already empty.
      auto rmdir_result = read_result->Rmdir(name.str());
      if (rmdir_result.ok() || rmdir_result.error().no_entity()) {
        // Removed here or by something else.
        continue;
      }
      if (rmdir_result.error().not_empty()) {
        // Found a non-empty directory, add it to our list and continue.
        dir_entries.push_back(name.str());
        continue;
      }

      // Otherwise, some unknown error, so propagate that.
      return std::move(rmdir_result).error();
    }

    dir_stack.push_back({*std::move(read_result), dir_entries_start});
    return Success();
  };

  CARBON_RETURN_IF_ERROR(push_dir(*this, path));

  while (true) {
    auto& current = dir_stack.back();
    if (current.dir_entries_start != static_cast<ssize_t>(dir_entries.size())) {
      CARBON_CHECK(current.dir_entries_start <
                   static_cast<ssize_t>(dir_entries.size()));
      CARBON_RETURN_IF_ERROR(push_dir(current.dir, dir_entries.back()));
      continue;
    }

    dir_stack.pop_back();
    if (dir_stack.empty()) {
      break;
    }

    // Pop this entry of the parent and remove it.
    auto& parent = dir_stack.back();
    CARBON_CHECK(parent.dir_entries_start <
                 static_cast<ssize_t>(dir_entries.size()));
    std::filesystem::path subdir_path = dir_entries.pop_back_val();
    CARBON_RETURN_IF_ERROR(parent.dir.Rmdir(std::move(subdir_path)));
  }

  return Rmdir(std::move(path));
}

auto DirRef::ReadlinkSlow(std::filesystem::path path)
    -> ErrorOr<std::string, PathError> {
  // Read directly into a string to avoid allocating two large buffers.
  std::string large_buffer;
  // Stat the symlink to get an initial guess at the size.
  CARBON_ASSIGN_OR_RETURN(FileStatus status, Lstat(path));
  // We try to use the size from the `lstat` unless it is empty, in which case
  // we try to use `PATH_MAX` or a constant value. We have a fallback to
  // dynamically discover an adequate buffer size below that will handle any
  // inaccuracy.
  ssize_t buffer_size = status.size();
  if (buffer_size == 0) {
#ifdef PATH_MAX
    buffer_size = PATH_MAX;
#else
    buffer_size = 1024;
#endif
  }
  large_buffer.resize(status.size());
  ssize_t result =
      readlinkat(dfd_, path.c_str(), large_buffer.data(), large_buffer.size());
  if (result == -1) {
    return PathError(errno, "Readlink on '{0}' relative to '{1}'",
                     std::move(path), dfd_);
  }

  // Now the really bad fallback case: if there are racing writes to the
  // symlink, the guessed size may not have been large enough. As a last-ditch
  // effort, begin doubling (from the next power of two >= PATH_MAX) the length
  // until it fits. We cap this at 10 MiB to prevent egregious file system
  // contents (or some bug somewhere) from exhausting memory.
  constexpr ssize_t MaxSize = 10 << 20;
  constexpr ssize_t MinSize =
#ifdef PATH_MAX
      PATH_MAX
#else
      1024
#endif
      ;
  while (result == static_cast<ssize_t>(large_buffer.size())) {
    if (large_buffer.size() >= MaxSize) {
      return PathError(errno, "Readlink on '{0}' relative to '{1}'",
                       std::move(path), dfd_);
    }
    large_buffer.resize(
        std::max<ssize_t>(MinSize, llvm::NextPowerOf2(large_buffer.size())));
    result = readlinkat(dfd_, path.c_str(), large_buffer.data(),
                        large_buffer.size());
    if (result == -1) {
      return PathError(errno, "Readlink on '{0}' relative to '{1}'",
                       std::move(path), dfd_);
    }
  }

  // Fix-up the size of the string and return it.
  large_buffer.resize(result);
  return large_buffer;
}

auto DirRef::OpenDir(std::filesystem::path path, CreationOptions creation_flags,
                     ModeType creation_mode) -> ErrorOr<Dir, PathError> {
  // If we potentially need to create a directory, we have to do that
  // separately as no systems support `O_CREAT | O_DIRECTORY`, even though
  // that would be (much) nicer.
  bool created = false;
  int open_flags = O_DIRECTORY;
  if (creation_flags != OpenExisting) {
    CARBON_CHECK(creation_flags != CreateAlways,
                 "Invalid `creation_flags` value of `CreateAlways`: there is "
                 "no support for truncating directories, and so they cannot be "
                 "created in an analogous way to files if they already exist.");

    if (mkdirat(dfd_, path.c_str(), creation_mode) == 0) {
      created = true;

      // If we created the directory, we also disable following the last
      // symlink. The last symlink is the only one that matters for security
      // here because `mkdirat` above is only valid to create a single directory
      // component. It is that directory component that we want to ensure has
      // not been replaced with a symlink by an adversarial concurrent process.
      open_flags |= O_NOFOLLOW;
    } else {
      // Unless the error is just that the path already exists, and that is
      // allowed for the requested creation flags, report any error here as part
      // of opening just like we would if the error originated from `openat`
      // with `O_CREAT`.
      if (creation_flags == CreateNew || errno != EEXIST) {
        return PathError(errno,
                         "Calling `mkdirat` on '{0}' relative to '{1}' during "
                         "DirRef::OpenDir",
                         std::move(path), dfd_);
      }
    }
  }

  // Open this path as a directory. Note that this has to succeed, and when we
  // created the directory we require the last component to not be a symlink in
  // case it was _replaced_ with a symlink while running.
  int result_fd = openat(dfd_, path.c_str(), open_flags);
  if (result_fd == -1) {
    // No need for `EINTR` handling here as if this is a FIFO it would be an
    // error with `O_DIRECTORY`.
    return PathError(
        errno,
        "Calling `openat` on '{0}' relative to '{1}' during DirRef::OpenDir",
        std::move(path), dfd_);
  }
  Dir result(result_fd);

  // If we actually created the directory, we also need to verify that the
  // opened file descriptor continues to have the same permissions and the
  // correct owner and group as we couldn't do the creation atomically with the
  // open.
  if (created) {
    auto stat_result = result.Stat();
    if (!stat_result.ok()) {
      // Manually propagate this error so we can attach it back to the opened
      // path and relative directory.
      return PathError(stat_result.error().number(),
                       "DirRef::Stat after opening '{0}' relative to '{1}'",
                       std::move(path), dfd_);
    }

    // Check that the permissions are a subset of the requested ones. They may
    // have been masked down by `umask`, but if there are *new* permissions,
    // that would be a security issue. We first need to extract the permission
    // bits from the mode, as other bits are separately controlled.
    if ((stat_result->permissions() & creation_mode) !=
        stat_result->permissions()) {
      // Model this `EPERM`.
      return PathError(EPERM,
                       "Setting permissions when creating '{0}' relative to "
                       "'{1}' during DirRef::OpenDir",
                       std::move(path), dfd_);
    }
    // Also check that the UID is the current effective UID. We don't currently
    // verify the GID because it could come from the parent directory, so
    // callers that need to should instead validate this themselves.
    if (stat_result->uid() != geteuid()) {
      // Model this as `EPERM`, which is a bit awkward, but should be fine.
      return PathError(EPERM,
                       "Setting UID when creating '{0}' relative to "
                       "'{1}' during DirRef::OpenDir",
                       std::move(path), dfd_);
    }
  }

  return result;
}

auto MakeTmpDir() -> ErrorOr<RemovingDir, Error> {
  std::filesystem::path tmpdir_path = "/tmp";
  for (const char* tmpdir_env_name : {"TEST_TMPDIR", "TMPDIR"}) {
    const char* tmpdir_env_cstr = getenv(tmpdir_env_name);
    if (tmpdir_env_cstr == nullptr) {
      continue;
    }
    std::filesystem::path tmpdir_env = std::string(tmpdir_env_cstr);
    if (!tmpdir_env.is_absolute()) {
      continue;
    }
    tmpdir_path = std::move(tmpdir_env);
    break;
  }

  std::filesystem::path target = BuildData::BuildTarget.str();
  std::string dir_name = target.filename().native();
  dir_name += ".XXXXXX";

  tmpdir_path /= dir_name;

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
  // created directory and perform safety checks. We can be more strict here as
  // there can't be correct racing creation of the *same* new temporary
  // directory.
  CARBON_ASSIGN_OR_RETURN(Dir tmp, Cwd().OpenDir(tmpdir_path));
  // Make sure we try to remove the directory from here on out.
  RemovingDir result_dir(std::move(tmp), tmpdir_path);

  // It's a bit awkward to report `fstat` errors as `Error`s, but we
  // don't have much choice. The stat failing here would be very weird.
  CARBON_ASSIGN_OR_RETURN(FileStatus stat, result_dir.Stat());

  // The permissions must be exactly 0700 for a temporary directory, and the UID
  // should be ours.
  if (stat.permissions() != 0700 && stat.uid() != geteuid()) {
    return Error(
        llvm::formatv("Found incorrect permissions or UID on tmpdir '{0}'",
                      tmpdir_path.native())
            .str());
  }

  // Last but not least, the directory must also be empty (other than `.` and
  // `..`). Any files here would represent a security issue from injected
  // symlinks to trigger traversal out of our tmp directory and into another
  // directory we don't control.
  CARBON_ASSIGN_OR_RETURN(Dir::Reader reader, result_dir.Read());
  for (const auto& entry : reader) {
    llvm::StringRef name = entry.name();
    if (name != "." && name != "..") {
      // We found an existing directory *other* than the empty one we expected
      // to create. Likely this was created by something else racing with us and
      // we should not use it.
      return Error(
          llvm::formatv(
              "Found unexpected entry '{0}' in newly created tmpdir '{1}'",
              name, tmpdir_path.native())
              .str());
    }
  }

  return result_dir;
}

}  // namespace Carbon::Filesystem
