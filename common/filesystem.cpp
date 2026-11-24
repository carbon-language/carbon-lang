// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/filesystem.h"

#include <climits>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>

namespace Carbon::Filesystem {

auto Internal::FileRefBase::ReadFileToString()
    -> ErrorOr<std::string, FdError> {
  std::string result;

  // Read a buffer at a time until we reach the end. Use `PIPE_BUF` when
  // available (otherwise 4 KiB) as our chunk size, which is small enough for
  // stack allocation while matching typical OS pipe chunking.
  //
  // TODO: Replace this with a smaller buffer and using `resize_and_overwrite`
  // to read into the string in-place for larger strings. Unclear if that will
  // be any faster, but it will be much more friendly to callers with
  // constrained stack sizes and use less memory overall.
#ifdef PIPE_BUF
  constexpr size_t BufferSize = PIPE_BUF;
#else
  constexpr size_t BufferSize = 4096;
#endif
  std::byte buffer[BufferSize];
  CARBON_RETURN_IF_ERROR(SeekFromBeginning(0));
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

auto Internal::FileRefBase::WriteFileFromString(llvm::StringRef str)
    -> ErrorOr<Success, FdError> {
  CARBON_RETURN_IF_ERROR(SeekFromBeginning(0));
  CARBON_RETURN_IF_ERROR(WriteCompleteBuffer(llvm::ArrayRef<std::byte>(
      reinterpret_cast<const std::byte*>(str.data()), str.size())));
  CARBON_RETURN_IF_ERROR(Truncate(str.size()));
  return Success();
}

auto DirRef::AppendEntriesIf(
    llvm::SmallVectorImpl<std::filesystem::path>& entries,
    llvm::function_ref<auto(llvm::StringRef name)->bool> predicate)
    -> ErrorOr<Success, FdError> {
  CARBON_ASSIGN_OR_RETURN(Reader reader, Read());
  for (const Entry& entry : reader) {
    llvm::StringRef name = entry.name();
    if (name == "." || name == "..") {
      continue;
    }
    if (predicate && !predicate(name)) {
      continue;
    }
    entries.push_back(name.str());
  }
  return Success();
}

auto DirRef::AppendEntriesIf(
    llvm::SmallVectorImpl<std::filesystem::path>& dir_entries,
    llvm::SmallVectorImpl<std::filesystem::path>& non_dir_entries,
    llvm::function_ref<auto(llvm::StringRef name)->bool> predicate)
    -> ErrorOr<Success, FdError> {
  CARBON_ASSIGN_OR_RETURN(Reader reader, Read());
  for (const Entry& entry : reader) {
    llvm::StringRef name = entry.name();
    if (name == "." || name == "..") {
      continue;
    }
    if (predicate && !predicate(name)) {
      continue;
    }
    std::filesystem::path name_path = name.str();
    if (entry.is_known_dir()) {
      dir_entries.push_back(std::move(name_path));
      continue;
    }
    if (!entry.is_unknown_type()) {
      non_dir_entries.push_back(std::move(name_path));
      continue;
    }

    auto stat_result = Lstat(name_path);
    if (!stat_result.ok()) {
      return MakeFdError(stat_result.error(),
                         "Dir::AppendEntriesIf on '{0}' failed while stat-ing "
                         "entries to determine which are directories");
    }
    if (stat_result->is_dir()) {
      dir_entries.push_back(std::move(name_path));
    } else {
      non_dir_entries.push_back(std::move(name_path));
    }
  }
  return Success();
}

auto DirRef::ReadFileToString(const std::filesystem::path& path)
    -> ErrorOr<std::string, PathError> {
  CARBON_ASSIGN_OR_RETURN(ReadFile f, OpenReadOnly(path));
  auto result = f.ReadFileToString();
  if (result.ok()) {
    return *std::move(result);
  }
  return MakePathError(
      result.error(), "Dir::ReadFileToString on '{0}' relative to '{1}'", path);
}

auto DirRef::WriteFileFromString(const std::filesystem::path& path,
                                 llvm::StringRef content,
                                 CreationOptions creation_options)
    -> ErrorOr<Success, PathError> {
  CARBON_ASSIGN_OR_RETURN(WriteFile f, OpenWriteOnly(path, creation_options));
  auto write_result = f.WriteFileFromString(content);
  // Immediately close the file as even if there was a write error we don't want
  // to leave the file open.
  auto close_result = std::move(f).Close();

  // Now report the first error encountered or return success.
  if (!write_result.ok()) {
    return MakePathError(
        write_result.error(),
        "Write error in Dir::WriteFileFromString on '{0}' relative to '{1}'",
        path);
  }
  if (!close_result.ok()) {
    return MakePathError(
        close_result.error(),
        "Close error in Dir::WriteFileFromString on '{0}' relative to '{1}'",
        path);
  }
  return Success();
}

auto DirRef::CreateDirectories(const std::filesystem::path& path,
                               ModeType creation_mode)
    -> ErrorOr<Dir, PathError> {
  // Avoid having to handle an empty path by immediately rejecting it as
  // invalid.
  if (path.empty()) {
    return MakeInvalidArgPathError(
        "DirRef::CreateDirectories on '{0}' relative to '{1}'", path);
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
       !parent_path.empty();) {
    auto parent_open_result = OpenDir(parent_path, OpenExisting);
    if (parent_open_result.ok()) {
      work_dir = std::move(*parent_open_result);
      break;
    }
    std::filesystem::path next_parent = parent_path.parent_path();
    if (next_parent == parent_path) {
      return std::move(parent_open_result).error();
    }
    missing_components.push_back(parent_path.filename());
    parent_path = std::move(next_parent);
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
    std::ptrdiff_t dir_entry_start;
  };
  llvm::SmallVector<DirAndIterator> dir_stack;

  llvm::SmallVector<std::filesystem::path> dir_entries;
  llvm::SmallVector<std::filesystem::path> unknown_entries;

  dir_entries.push_back(path);
  for (;;) {
    // When we bottom out, we're removing the initial tree path and doing so
    // relative to `this` directory.
    DirRef current = dir_stack.empty() ? *this : dir_stack.back().dir;
    std::ptrdiff_t dir_entry_start =
        dir_stack.empty() ? 0 : dir_stack.back().dir_entry_start;

    // If we've finished all the child directories of the current entry in the
    // stack, pop it off and continue.
    if (dir_entry_start == static_cast<std::ptrdiff_t>(dir_entries.size())) {
      dir_stack.pop_back();
      continue;
    }
    CARBON_CHECK(dir_entry_start <
                 static_cast<std::ptrdiff_t>(dir_entries.size()));

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
      return current.MakePathError(
          read_result.error(),
          "Dir::Read on '{0}' relative to '{1}' during RmdirRecursively",
          entry_path);
    }
    dir_stack.push_back({*std::move(read_result),
                         static_cast<std::ptrdiff_t>(dir_entries.size())});

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

}  // namespace Carbon::Filesystem
