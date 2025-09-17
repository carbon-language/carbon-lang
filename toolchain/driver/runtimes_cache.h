// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_RUNTIMES_CACHE_H_
#define CARBON_TOOLCHAIN_DRIVER_RUNTIMES_CACHE_H_

#include <chrono>
#include <filesystem>
#include <utility>

#include "common/check.h"
#include "common/error.h"
#include "common/filesystem.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Manages a runtimes directory.
//
// Carbon (including Clang) relies on a set of runtime libraries and data files.
// These are organized into a directory modeled by a `Runtimes` object, with
// each individual runtime library built into a subdirectory. The `Runtimes`
// object in turn provides access to each of these subdirectories.
//
// Different runtime libraries are managed separately so that we don't force
// building all of them in a configuration where only one makes sense or is
// typically needed.
//
// Beyond providing access, a `Runtimes` object also supports orchestrating the
// _build_ of any of the runtime libraries into their designated subdirectories,
// including synchronizing between different threads or processes trying to
// build the same component of the runtimes.
//
// TODO: Add libc++ to the runtimes tree.
// TODO: Add the Core library to the runtimes tree.
class Runtimes {
 public:
  class Builder;
  class Cache;

  // Creates a `Runtimes` object for a specific existing directory.
  //
  // Requires that the `path` is an absolute path that is an existing directory.
  static auto Make(std::filesystem::path path,
                   llvm::raw_ostream* vlog_stream = nullptr)
      -> ErrorOr<Runtimes>;

  // Default construction produces an unformed runtimes that can only be
  // assigned to or destroyed.
  Runtimes() = default;
  // A specific runtimes object is move-only as it owns the relevant filesystem
  // resources.
  Runtimes(Runtimes&& arg) noexcept
      : base_path_(std::move(arg.base_path_)),
        base_dir_(std::exchange(arg.base_dir_, {})),
        lock_file_(std::exchange(arg.lock_file_, {})),
        flock_(std::exchange(arg.flock_, {})),
        vlog_stream_(arg.vlog_stream_) {}
  auto operator=(Runtimes&& arg) noexcept -> Runtimes& {
    Destroy();
    base_path_ = std::move(arg.base_path_);
    base_dir_ = std::exchange(arg.base_dir_, {});
    lock_file_ = std::exchange(arg.lock_file_, {});
    flock_ = std::exchange(arg.flock_, {});
    vlog_stream_ = arg.vlog_stream_;
    return *this;
  }
  ~Runtimes() { Destroy(); }

  // The base path for the runtimes.
  auto base_path() const -> const std::filesystem::path& { return base_path_; }

  // The base directory for the runtimes.
  auto base_dir() const -> Filesystem::DirRef { return base_dir_; }

  // Gets the path to an _existing_ Clang resource directory.
  //
  // Clang's resource directory contains all of the compiler-builtin runtime
  // libraries, headers, and data files.
  //
  // This will return the path to the Clang resource directory if it exists in
  // the runtimes tree. Otherwise, it will return an error.
  auto GetExistingClangResourceDir() -> ErrorOr<std::filesystem::path>;

  // Builds or returns a Clang resource directory.
  //
  // If there is an existing, built Clang resource directory, this will return
  // its path, the same as `GetExistingClangResourceDir` would. However, if
  // there is not yet a Clang resource directory in this runtimes tree, returns
  // a `Builder` object that can be used to build and commit a Clang resource
  // directory to this runtimes tree.
  auto BuildClangResourceDir()
      -> ErrorOr<std::variant<std::filesystem::path, Builder>>;

 private:
  friend Builder;
  friend Cache;
  friend class RuntimesTestPeer;

  // The deadline for acquiring a lock to build a new component of the runtimes.
  // This needs to be quite large as this is how long racing processes or
  // threads will wait to allow some other process to complete building the
  // runtimes. The result is that this should be significantly longer than the
  // expected slowest-to-build runtimes.
  //
  // Note, nothing goes _wrong_ if this deadline is exceeded, but multiple
  // copies of runtimes may end up being built and all but one thrown away.
  static constexpr Filesystem::Duration BuildLockDeadline =
      std::chrono::seconds(200);
  // The interval at which to poll for a build lock. This needs to be small
  // enough that we don't waste an excessive amount of time if a build of the
  // runtimes completes *just* after a poll. Typically, that means we want this
  // to be significant lower than the expected time it would take to build the
  // runtimes. Note that we don't poll if the runtimes have been completely
  // built prior to the query coming in, so this doesn't form the _minimum_ time
  // to find a component of the runtimes tree.
  static constexpr Filesystem::Duration BuildLockPollInterval =
      std::chrono::milliseconds(200);

  // The path to the clang resource directory within the runtimes.
  //
  // This uses `std::string_view` to simply using with paths.
  static constexpr std::string_view ClangResourceDirPath = "clang_resource_dir";

  // A format string used to form the lock file for a given directory in the
  // runtimes. This needs to be C-string, so directly expose the character
  // array.
  static constexpr char LockFileFormat[] = ".{0}.lock";

  explicit Runtimes(std::filesystem::path base_path, Filesystem::Dir base_dir,
                    Filesystem::WriteFile lock_file, Filesystem::FileLock flock,
                    llvm::raw_ostream* vlog_stream = nullptr)
      : base_path_(std::move(base_path)),
        base_dir_(std::move(base_dir)),
        lock_file_(std::move(lock_file)),
        flock_(std::move(flock)),
        vlog_stream_(vlog_stream) {
    CARBON_CHECK(base_path_.is_absolute(),
                 "The base path must be absolute: {0}", base_path_);
  }

  // Implementation of building the Clang resource directory. This exposes the
  // deadline and poll interval to allow testing with artificial values.
  auto BuildClangResourceDirImpl(Filesystem::Duration deadline,
                                 Filesystem::Duration poll_interval)
      -> ErrorOr<std::variant<std::filesystem::path, Builder>>;

  auto Destroy() -> void;

  std::filesystem::path base_path_;
  Filesystem::Dir base_dir_;
  Filesystem::WriteFile lock_file_;
  Filesystem::FileLock flock_;
  llvm::raw_ostream* vlog_stream_ = nullptr;
};

// A managed cache of `Runtimes` directories.
//
// This class manages and provides access to a cache of runtimes. Each entry in
// the cache is a collection of runtimes for a specific set of `Feature`s in a
// directory that can be modeled with an object of the `Runtimes` type. We call
// these entries "runtimes" typically. The cache keys and looks up these
// runtimes based on the set of `Feature`s and the input sources used to build
// them (including the compiler). Whenever looking up runtimes not already
// present in the cache, the cache will evict old runtimes before creating the
// new ones. The eviction strategy is to remove any runtimes more than a year
// old, as well as the least-recently used runtimes until there will only be a
// maximum of 50 runtimes cached. The goal is to allow multiple versions and
// build features to stay resident in the runtimes cache while providing a
// stable upper bound on the disk space used.
//
// The cache can be formed around a specific directory, or it can search for a
// system-default directory. The system default directory follows the guidance
// of the XDG Base Directory Specification:
// https://specifications.freedesktop.org/basedir-spec/latest/
//
// This tries to place the system cache in
// `$XDG_CACHE_HOME/carbon_runtimes_cache`, followed by
// `$HOME/.cache/carbon_runtimes_cache`. A fallback if neither works is to
// create a temporary directory for the cache. This temporary directory is owned
// by the `Cache` object and will be removed when it is destroyed.
//
// These system-wide paths are only used if the installation contains a digest
// file that can be used to ensure different builds and installs of Carbon don't
// incorrectly share cached runtimes. When missing, a temporary directory is
// used.
class Runtimes::Cache {
 public:
  // The features of a cached runtime.
  //
  // TODO: Add support for more build flags that we want to enable when building
  // runtimes such as sanitizers and CPU-specific optimizations.
  struct Features {
    std::string target;
  };

  Cache() = default;

  // The cache is move-only as it owns open resources for the cache directory.
  Cache(Cache&& arg) noexcept
      : vlog_stream_(arg.vlog_stream_),
        cache_key_(std::move(arg.cache_key_)),
        path_(std::move(arg.path_)),
        dir_owner_(std::exchange(arg.dir_owner_, {})),
        dir_(arg.dir_) {}
  auto operator=(Cache&& arg) noexcept -> Cache& {
    vlog_stream_ = arg.vlog_stream_;
    cache_key_ = std::move(arg.cache_key_);
    path_ = std::move(arg.path_);
    dir_owner_ = std::exchange(arg.dir_owner_, {});
    dir_ = arg.dir_;
    return *this;
  }

  // Creates a cache object for the current system.
  //
  // This will try to locate and use a persistent cache on the system if it can,
  // and otherwise fall back to creating a temporary cache. If either of these
  // hit unrecoverable errors, that error is returned instead. See the class
  // comment for more details about the overall strategy.
  static auto MakeSystem(const InstallPaths& install,
                         llvm::raw_ostream* vlog_stream = nullptr)
      -> ErrorOr<Cache> {
    Cache cache(vlog_stream);
    CARBON_RETURN_IF_ERROR(cache.InitSystemCache(install));
    return cache;
  }

  // Creates a cache object referencing an explicit cache path.
  //
  // The path must be an existing, writable directory.
  static auto MakeCustom(const InstallPaths& install,
                         std::filesystem::path cache_path,
                         llvm::raw_ostream* vlog_stream = nullptr)
      -> ErrorOr<Cache> {
    Cache cache(vlog_stream);
    CARBON_RETURN_IF_ERROR(cache.InitCachePath(install, cache_path));
    return cache;
  }

  // The path to the cache directory.
  auto path() const -> const std::filesystem::path& { return path_; }

  // Looks up a runtime in the cache.
  //
  // This will return a `Runtimes` object for the given features. If a runtime
  // for these features does not exist in the cache, any stale cache entries
  // will be pruned if needed, and then a new entry will be created and
  // returned.
  auto Lookup(const Features& features) -> ErrorOr<Runtimes>;

 private:
  friend class RuntimesTestPeer;

  static constexpr int MinNumEntries = 10;
  static constexpr int MaxNumEntries = 50;

  // The maximum age of a cache entry. Cache entries older than this will always
  // evicted if there are more than the minimum number of entries.
  static constexpr auto MaxEntryAge = std::chrono::years(1);
  // The maximum age of a locked cache entry. Cache entries older than this will
  // be evicted if needed without regard to any held lock from a process
  // currently using that entry.
  static constexpr auto MaxLockedEntryAge = std::chrono::days(10);

  // Runtimes are locked while in use to avoid them being removed concurrently,
  // but this is an even more advisory lock than usual. We use a relatively
  // short deadline and fast poll interval here as this is on the critical path
  // even for an existing, built runtimes entry.
  static constexpr auto RuntimesLockDeadline = std::chrono::milliseconds(100);
  static constexpr auto RuntimesLockPollInterval = std::chrono::milliseconds(1);

  explicit Cache(llvm::raw_ostream* vlog_stream) : vlog_stream_(vlog_stream) {}

  auto InitSystemCache(const InstallPaths& install) -> ErrorOr<Success>;
  auto InitCachePath(const InstallPaths& install,
                     std::filesystem::path cache_path) -> ErrorOr<Success>;

  auto PruneStaleRuntimes(const std::filesystem::path& new_entry_path) -> void;

  llvm::raw_ostream* vlog_stream_ = nullptr;
  std::string cache_key_;
  std::filesystem::path path_;
  std::variant<Filesystem::Dir, Filesystem::RemovingDir> dir_owner_;
  // A reference to whichever form of `dir_owner_` is in use.
  Filesystem::DirRef dir_;
};

// Builder for a new directory in a runtimes tree.
//
// This manages a staging directory for the build that will then be committed
// into the destination once fully built.
class Runtimes::Builder : public Printable<Builder> {
 public:
  Builder(Builder&& arg) noexcept
      : runtimes_(std::exchange(arg.runtimes_, nullptr)),
        lock_file_(std::exchange(arg.lock_file_, {})),
        flock_(std::exchange(arg.flock_, {})),
        dir_(std::exchange(arg.dir_, {})),
        dest_(arg.dest_) {}
  auto operator=(Builder&& arg) noexcept -> Builder& {
    Destroy();
    runtimes_ = std::exchange(arg.runtimes_, nullptr);
    lock_file_ = std::exchange(arg.lock_file_, {});
    flock_ = std::exchange(arg.flock_, {});
    dir_ = std::exchange(arg.dir_, {});
    dest_ = arg.dest_;
    return *this;
  }
  ~Builder() { Destroy(); }

  // The build's staging directory.
  auto dir() const -> Filesystem::DirRef { return dir_; }
  // The build's staging directory path.
  auto path() const -> const std::filesystem::path& { return dir_.abs_path(); }

  // Commits the new runtime to the cache.
  //
  // This will move the contents of the temporary directory to the final
  // destination in the cache.
  auto Commit() && -> ErrorOr<std::filesystem::path>;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "Runtimes::Builder{.path = '" << path() << "'}";
  }

 private:
  friend Runtimes;
  friend class RuntimesTestPeer;

  Builder() = default;
  explicit Builder(Runtimes& runtimes, Filesystem::ReadWriteFile lock_file,
                   Filesystem::FileLock flock, Filesystem::RemovingDir tmp_dir,
                   std::string_view dest)
      : runtimes_(&runtimes),
        vlog_stream_(runtimes.vlog_stream_),
        lock_file_(std::move(lock_file)),
        flock_(std::move(flock)),
        dir_(std::move(tmp_dir)),
        dest_(dest) {}

  auto ReleaseFileLock() -> void;
  auto Destroy() -> void;

  Runtimes* runtimes_ = nullptr;
  llvm::raw_ostream* vlog_stream_ = nullptr;
  Filesystem::ReadWriteFile lock_file_;
  Filesystem::FileLock flock_;
  Filesystem::RemovingDir dir_;
  std::string_view dest_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_RUNTIMES_CACHE_H_
