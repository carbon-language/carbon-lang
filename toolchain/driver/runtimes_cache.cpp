// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/runtimes_cache.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>

#include "common/filesystem.h"
#include "common/version.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"

namespace Carbon {

auto Runtimes::Make(std::filesystem::path path, llvm::raw_ostream* vlog_stream)
    -> ErrorOr<Runtimes> {
  if (!path.is_absolute()) {
    return Error("Runtimes require an absolute path");
  }

  CARBON_ASSIGN_OR_RETURN(
      Filesystem::Dir dir,
      Filesystem::Cwd().OpenDir(path, Filesystem::OpenExisting));
  return Runtimes(std::move(path), std::move(dir), {}, {}, vlog_stream);
}

auto Runtimes::Destroy() -> void {
  // Release the lock on the runtimes and close the lock file.
  flock_ = {};
  auto close_result = std::move(lock_file_).Close();
  if (!close_result.ok()) {
    // Log and continue on close errors.
    CARBON_VLOG("Error closing lock file for runtimes '{0}': {1}", base_path_,
                close_result.error());
  }
}

auto Runtimes::Get(Component component) -> ErrorOr<std::filesystem::path> {
  std::filesystem::path path = base_path_ / ComponentPath(component);
  auto open_result =
      base_dir_.OpenDir(ComponentPath(component), Filesystem::OpenExisting);
  if (open_result.ok()) {
    return path;
  }
  return open_result.error().ToError();
}

auto Runtimes::Build(Component component)
    -> ErrorOr<std::variant<std::filesystem::path, Builder>> {
  return BuildImpl(component, BuildLockDeadline, BuildLockPollInterval);
}

auto Runtimes::BuildImpl(Component component, Filesystem::Duration deadline,
                         Filesystem::Duration poll_interval)
    -> ErrorOr<std::variant<std::filesystem::path, Builder>> {
  // Try to get an existing resource directory first.
  auto existing_result = Get(component);
  if (existing_result.ok()) {
    return {*std::move(existing_result)};
  }

  // Otherwise, we will need to build the runtimes and commit them into this
  // directory once ready. Try and acquire an advisory lock to avoid redundant
  // computation.
  std::string_view component_path = ComponentPath(component);
  CARBON_ASSIGN_OR_RETURN(
      Filesystem::ReadWriteFile lock_file,
      base_dir_.OpenReadWrite(
          llvm::formatv(LockFileFormat, component_path).str(),
          Filesystem::OpenAlways, /*creation_mode=*/0700));
  CARBON_VLOG("PID {0} locking cache path: {1}\n", getpid(),
              base_path_ / component_path);
  Filesystem::FileLock flock;
  auto flock_result = lock_file.TryLock(Filesystem::FileLock::Exclusive,
                                        deadline, poll_interval);
  if (flock_result.ok()) {
    flock = *std::move(flock_result);
    CARBON_VLOG("Successfully locked cache path\n");
    // As a debugging aid, write our PID into the lock file when we
    // successfully acquire it. Ignore errors here though.
    (void)lock_file.WriteFileFromString(std::to_string(getpid()));
  } else if (!flock_result.error().would_block()) {
    // Some unexpected filesystem error, report that rather than trying to
    // continue.
    return std::move(flock_result).error();
  } else {
    CARBON_VLOG("Unable to lock cache path, held by: {1}\n",
                *lock_file.ReadFileToString());
    (void)std::move(lock_file).Close();
  }

  // See if another process has built the runtimes while we waited on the lock.
  // We do this even if we didn't successfully acquire the lock because we
  // ensure that a successful build atomically creates a viable directory.
  existing_result = Get(component);
  if (existing_result.ok()) {
    // Clear and close the lock file.
    (void)lock_file.WriteFileFromString("");
    flock = {};
    (void)std::move(lock_file).Close();
    return {*std::move(existing_result)};
  }

  // Whether we hold the lock file or not, we're going to now build these
  // runtimes. Create a temporary directory where we can do that safely
  // regardless of what else is happening.
  std::filesystem::path tmp_path =
      base_path_ / llvm::formatv(".{0}.tmp", component_path).str();
  CARBON_ASSIGN_OR_RETURN(Filesystem::RemovingDir tmp_dir,
                          Filesystem::MakeTmpDirWithPrefix(tmp_path));
  return {Builder(*this, std::move(lock_file), std::move(flock),
                  std::move(tmp_dir), component_path)};
}

auto Runtimes::Cache::FindXdgCachePath()
    -> std::optional<std::filesystem::path> {
  if (const char* xdg_cache_home = getenv("XDG_CACHE_HOME");
      xdg_cache_home != nullptr) {
    std::filesystem::path path = xdg_cache_home;
    if (path.is_absolute()) {
      CARBON_VLOG("Using '$XDG_CACHE_HOME' cache: {0}", path);
      return path;
    }
  }

  // Unable to use the standard environment variable. Try the designated
  // fallback of `$HOME/.cache`.
  const char* home = getenv("HOME");
  if (home == nullptr) {
    return std::nullopt;
  }

  std::filesystem::path path = home;
  if (!path.is_absolute()) {
    return std::nullopt;
  }
  path /= ".cache";
  CARBON_VLOG("Using '$HOME/.cache' cache: {0}", path);
  return path;
}

auto Runtimes::Cache::InitTmpSystemCache() -> ErrorOr<Success> {
  CARBON_ASSIGN_OR_RETURN(dir_owner_, Filesystem::MakeTmpDir());
  path_ = std::get<Filesystem::RemovingDir>(dir_owner_).abs_path();
  dir_ = std::get<Filesystem::RemovingDir>(dir_owner_);
  CARBON_VLOG("Using temporary cache: {0}", path_);
  return Success();
}

auto Runtimes::Cache::InitSystemCache(const InstallPaths& install)
    -> ErrorOr<Success> {
  constexpr llvm::StringLiteral CachePath = "carbon_runtimes";

  // If we have a digest to use as the cache key, save it and we can try to
  // use persistent caches.
  auto read_digest_result =
      Filesystem::Cwd().ReadFileToString(install.digest_path());
  if (!read_digest_result.ok()) {
    return InitTmpSystemCache();
  }
  cache_key_ = *std::move(read_digest_result);

  auto xdg_path_result = FindXdgCachePath();
  if (!xdg_path_result) {
    return InitTmpSystemCache();
  }

  // We have a candidate XDG-based cache path. Try to open that, and a
  // directory below it for Carbon's runtimes. Note that we don't error on a
  // missing directory, we fall through to using a temporary directory.
  auto open_result = Filesystem::Cwd().OpenDir(*xdg_path_result);
  if (!open_result.ok()) {
    if (!open_result.error().no_entity()) {
      // Some other unexpected error in the filesystem, propagate that.
      return std::move(open_result).error();
    }
    // Otherwise we fall back to a temporary system cache.
    return InitTmpSystemCache();
  }

  path_ = *std::move(xdg_path_result);
  // Now open a subdirectory of the cache for Carbon's usage. This will
  // create a subdirectory if one doesn't yet exist.
  path_ /= std::string_view(CachePath);
  CARBON_ASSIGN_OR_RETURN(
      dir_owner_, open_result->OpenDir(CachePath.str(), Filesystem::OpenAlways,
                                       /*creation_mode=*/0700));
  dir_ = std::get<Filesystem::Dir>(dir_owner_);

  // Ensure the directory has narrow permissions so runtimes can't be
  // overwritten.
  CARBON_ASSIGN_OR_RETURN(auto dir_stat, dir_.Stat());
  if (dir_stat.permissions() != 0700 || dir_stat.unix_uid() != geteuid()) {
    return Error(llvm::formatv(
        "Found runtimes cache path '{0}' with excessive permissions ({1}) "
        "or an invalid owning UID ({2})",
        path_, dir_stat.permissions(), dir_stat.unix_uid()));
  }

  return Success();
}

auto Runtimes::Cache::InitCachePath(const InstallPaths& install,
                                    std::filesystem::path cache_path)
    -> ErrorOr<Success> {
  auto read_digest_result =
      Filesystem::Cwd().ReadFileToString(install.digest_path());
  if (read_digest_result.ok()) {
    // If we have a digest to use as the cache key, save it and we can try to
    // use persistent caches.
    cache_key_ = *std::move(read_digest_result);
  } else {
    // Without a digest, use the path itself as the key.
    cache_key_ = cache_path.string();
  }

  CARBON_ASSIGN_OR_RETURN(dir_owner_, Filesystem::Cwd().OpenDir(cache_path));
  dir_ = std::get<Filesystem::Dir>(dir_owner_);
  path_ = std::move(cache_path);
  CARBON_VLOG("Using custom cache: {0}", path_);
  return Success();
}

auto Runtimes::Cache::Lookup(const Features& features) -> ErrorOr<Runtimes> {
  // Compute the hash of the features. We'll use this to build the subdirectory
  // within the cache.

  llvm::SHA256 entry_hasher;
  // First incorporate our cache key that comes from the installation's digest.
  // This ensures we don't share a cache entry with any other Carbon
  // installations using different inputs.
  entry_hasher.update(cache_key_);
  // Then incorporate the specific features that are enabled in this entry.
  entry_hasher.update(features.target);

  std::array<uint8_t, 32> entry_digest = entry_hasher.final();
  std::filesystem::path entry_path =
      llvm::formatv("runtimes-{0}-{1}", Version::String,
                    llvm::toHex(entry_digest, /*LowerCase=*/true))
          .str();

  Filesystem::Dir entry_dir;
  auto open_result = dir_.OpenDir(entry_path, Filesystem::OpenExisting);
  if (open_result.ok()) {
    entry_dir = *std::move(open_result);
  } else {
    if (!open_result.error().no_entity()) {
      return std::move(open_result).error();
    }

    // We're going to potentially create a new set of runtimes, prune the
    // existing runtimes first to provide a bound on the total size of runtimes.
    PruneStaleRuntimes(entry_path);

    // Now we can create or open, we don't care if a racing process created the
    // same runtime directory.
    CARBON_ASSIGN_OR_RETURN(entry_dir,
                            dir_.OpenDir(entry_path, Filesystem::OpenAlways));
  }

  CARBON_ASSIGN_OR_RETURN(
      auto lock_file, entry_dir.OpenWriteOnly(".lock", Filesystem::OpenAlways));
  CARBON_RETURN_IF_ERROR(lock_file.UpdateTimes());
  CARBON_ASSIGN_OR_RETURN(
      Filesystem::FileLock flock,
      lock_file.TryLock(Filesystem::FileLock::Shared, RuntimesLockDeadline,
                        RuntimesLockPollInterval));

  return Runtimes(path_ / entry_path, std::move(entry_dir),
                  std::move(lock_file), std::move(flock), vlog_stream_);
}

auto Runtimes::Cache::ComputeEntryAges(
    llvm::SmallVector<std::filesystem::path> entry_paths)
    -> llvm::SmallVector<Entry> {
  llvm::SmallVector<Entry> entries;

  Filesystem::TimePoint now = Filesystem::Clock::now();
  for (auto& path : entry_paths) {
    // We use the `mtime` from the lock file in the directory rather than the
    // directory itself to avoid any oddities with `mtime` on directories.
    //
    // Note that we also ignore errors here as if we can't read the stamp file
    // we will pick an arbitrary old time stamp, and we want pruning to be
    // maximally resilient to partially deleted or corrupted caches in order to
    // prune them back into a healthy state.
    auto stat_result = dir_.Lstat(path / ".lock");
    auto mtime = stat_result.ok()
                     ? stat_result->mtime()
                     : Filesystem::TimePoint(Filesystem::Duration(0));
    entries.push_back({.path = std::move(path), .age = now - mtime});
  }
  return entries;
}

auto Runtimes::Cache::PruneStaleRuntimes(
    const std::filesystem::path& new_entry_path) -> void {
  llvm::SmallVector<std::filesystem::path> dir_entries;
  llvm::SmallVector<std::filesystem::path> non_dir_entries;
  auto read_result = dir_.AppendEntriesIf(
      dir_entries, non_dir_entries,
      [](llvm::StringRef name) { return name.starts_with("runtimes-"); });
  if (!read_result.ok()) {
    CARBON_VLOG("Unable to read cache directory to prune stale entries: {0}",
                read_result.error());
    return;
  }

  // Directly attempt to remove non-directory and bad directory entries.
  for (const auto& name : non_dir_entries) {
    CARBON_VLOG("Unlinking non-directory entry '{0}'", name);
    auto result = dir_.Unlink(name);
    if (!result.ok()) {
      CARBON_VLOG("Error unlinking non-directory entry '{0}': {1}", name,
                  result.error());
    }
  }

  // If we only have a small number of entries, no need to prune.
  if (dir_entries.size() < MinNumEntries) {
    return;
  }

  llvm::SmallVector<Entry> entries = ComputeEntryAges(std::move(dir_entries));

  auto rm_entry = [&](const std::filesystem::path& entry_name) {
    // Note that we don't propagate errors here because we want to prune as much
    // as possible. We do log them.
    CARBON_VLOG("Removing cache entry '{0}'", entry_name);
    auto rm_result = dir_.Rmtree(entry_name);
    if (!rm_result.ok() && !rm_result.error().no_entity()) {
      CARBON_VLOG("Unable to remove old runtimes '{0}': {1}", entry_name,
                  rm_result.error());
      return false;
    }
    return true;
  };

  // Remove entries older than our max first. We don't need to check for locking
  // or other issues here given the age.
  llvm::erase_if(entries, [&](const Entry& entry) {
    return entry.age > MaxEntryAge && rm_entry(entry.path);
  });

  // Sort the entries so that the oldest is first.
  llvm::sort(entries, [](const Entry& lhs, const Entry& rhs) {
    return lhs.age > rhs.age;
  });

  // Now try to get the number of entries below our max target by removing the
  // least-recently used entries that are either more than our max locked age or
  // unlocked.
  auto rm_unlocked_entry = [&](const std::filesystem::path& name,
                               Filesystem::Duration age) {
    // Past a certain age, bypass the locking for efficiency and to avoid
    // retaining entries with stale locks.
    if (age > MaxLockedEntryAge) {
      return rm_entry(name);
    }

    CARBON_VLOG("Attempting to lock cache entry '{0}'", name);
    auto lock_file_open_result =
        dir_.OpenReadOnly(name / ".lock", Filesystem::OpenAlways);
    if (!lock_file_open_result.ok()) {
      if (lock_file_open_result.error().no_entity() ||
          lock_file_open_result.error().not_dir()) {
        // The only way these failures should be possible is if something
        // removed the cache directory between our read above and here. Assume
        // the entry is gone and continue.
        return true;
      }

      // For other errors, assume locked.
      CARBON_VLOG("Error opening lock file for cache entry '{0}': {1}", name,
                  lock_file_open_result.error());
      return false;
    }

    Filesystem::ReadFile lock_file = *std::move(lock_file_open_result);
    auto lock_result =
        lock_file.TryLock(Filesystem::FileLock::Exclusive, RuntimesLockDeadline,
                          RuntimesLockPollInterval);
    if (!lock_result.ok()) {
      // The normal case is when locking would block, log anything else.
      if (!lock_result.error().would_block()) {
        CARBON_VLOG("Error locking cache entry '{0}': {1}", name,
                    lock_result.error());
      }
      // However, don't try to remove it as we didn't acquire the lock.
      return false;
    }

    // The lock is held, remove the entry.
    return rm_entry(name);
  };

  int num_entries = entries.size();
  for (const auto& [name, age] : entries) {
    if (num_entries < MaxNumEntries) {
      break;
    }

    // Don't prune the currently being built entry. We should only reach here
    // when some other process created this entry in a race, and we don't want
    // to remove it or trigger rebuilds.
    if (name == new_entry_path) {
      continue;
    }

    if (rm_unlocked_entry(name, age)) {
      --num_entries;
    }
  }

  if (num_entries >= MaxNumEntries) {
    CARBON_VLOG(
        "Unable to prune cache to our target size due to held locks on recent "
        "cache entries or removal errors, leaving {0} entries in the cache",
        num_entries);
  }
}

auto Runtimes::Builder::Commit() && -> ErrorOr<std::filesystem::path> {
  std::filesystem::path dest_path = runtimes_->base_path() / dest_;

  // First, try to do the atomic commit of the built runtimes into the final
  // location.
  CARBON_CHECK(dir_.abs_path().parent_path() == runtimes_->base_path(),
               "Building a temporary directory '{0}' that is not in the "
               "runtimes tree '{1}'",
               dir_.abs_path(), runtimes_->base_path());
  auto rename_result = runtimes_->base_dir().Rename(
      dir_.abs_path().filename(), runtimes_->base_dir(), dest_);
  // If the rename was successful, then we don't need to remove anything so
  // release that state.
  if (rename_result.ok()) {
    std::move(dir_).Release();
  } else if (rename_result.error().not_empty()) {
    // Some other runtimes were successfully committed before ours, so we want
    // to discard ours. We report errors cleaning up here as we don't want to
    // pollute the filesystem excessively.
    //
    // TODO: Consider instead being more resilient to errors here and just log
    // them.
    CARBON_VLOG("PID {0} found racily built runtimes in cache path: {1}",
                getpid(), dest_path);
    CARBON_RETURN_IF_ERROR(std::move(dir_).Remove());
  } else {
    // An unexpected error occurred, propagate it and let the normal cleanup
    // occur.
    //
    // TODO: It's possible we need to handle `EBUSY` here, likely by ensuring it
    // is the *destination* that is busy and an existing, valid directory built
    // concurrently.
    return std::move(rename_result).error();
  }

  // Now that we've got a final path in place successfully, clear the flock if
  // it is currently held.
  ReleaseFileLock();

  // Finally, the build is committed so finish putting this into the moved-from
  // state by clearing the runtimes pointer.
  runtimes_ = nullptr;
  return dest_path;
}

auto Runtimes::Builder::ReleaseFileLock() -> void {
  CARBON_CHECK(runtimes_ != nullptr);

  if (flock_.is_locked()) {
    std::filesystem::path dest_path = runtimes_->base_path() / dest_;
    CARBON_VLOG("PID {0} releasing lock on cache path: {1}", getpid(),
                dest_path);
    (void)lock_file_.WriteFileFromString("");
    flock_ = {};
    (void)std::move(lock_file_).Close();
  } else {
    CARBON_CHECK(!lock_file_.is_valid());
  }
}

auto Runtimes::Builder::Destroy() -> void {
  // If the runtimes are null, no in-flight build is owned so nothing to do.
  if (runtimes_ == nullptr) {
    CARBON_CHECK(
        !lock_file_.is_valid() && !flock_.is_locked() && !dir_.is_valid(),
        "Builder left in a partially cleared state!");
    return;
  }

  // Otherwise we need to abandon an in-flight build. First release the lock.
  ReleaseFileLock();

  // The rest of the cleanup is handled by the `RemovingDir` destructor.
}

}  // namespace Carbon
