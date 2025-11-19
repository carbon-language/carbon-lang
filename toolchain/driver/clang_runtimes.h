// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_DRIVER_CLANG_RUNTIMES_H_
#define CARBON_TOOLCHAIN_DRIVER_CLANG_RUNTIMES_H_

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include "common/error.h"
#include "common/filesystem.h"
#include "common/latch.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/runtimes_cache.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

// Common APIs and utilities factored out of all of the Clang runtimes builders.
//
// The Clang runtimes builders are _asynchronous_ builders at their core, and so
// their primary API is to construct the builder (kicking off work across
// threads) and then `Wait` for it to finish.
class ClangRuntimesBuilderBase {
 public:
  // Waits until the runtimes are finished being built and then return either
  // any error encountered or the path of the built runtimes.
  //
  // If a path is returned, it will be the same as `Runtimes::Builder::Commit`
  // would return.
  auto Wait() && -> ErrorOr<std::filesystem::path> {
    tasks_.wait();
    return std::move(result_);
  }

 protected:
  class ArchiveBuilder;

  // Initializes the common state of a runtimes builder.
  //
  // Both the `clang` runner and the `threads` need to outlive this object.
  ClangRuntimesBuilderBase(ClangRunner* clang,
                           llvm::ThreadPoolInterface* threads,
                           llvm::Triple target_triple)
      : clang_(clang),
        vlog_stream_(clang_->vlog_stream_),
        tasks_(*threads),
        target_triple_(std::move(target_triple)),
        target_flag_(llvm::formatv("--target={0}", target_triple_.str())),
        result_(Error("Did not finish building the runtime!")) {}

  auto installation() -> const InstallPaths& { return *clang_->installation_; }

  // We use protected members as this base is just factoring out common
  // implementation details of other runners.
  //
  // NOLINTBEGIN(misc-non-private-member-variables-in-classes)

  ClangRunner* clang_;
  llvm::raw_ostream* vlog_stream_;

  // This task group will be used both to launch asynchronous work and to wait
  // for all of the work for a particular build to complete.
  llvm::ThreadPoolTaskGroup tasks_;

  llvm::Triple target_triple_;
  std::string target_flag_;

  ErrorOr<std::filesystem::path> result_;

  // If runtimes already exist, we may never need a builder for them. But if we
  // do need to build the runtimes, store the `Runtimes::Builder` here.
  std::optional<Runtimes::Builder> runtimes_builder_;

  // When building runtimes, this latch synchronizes all the steps required
  // to build the runtimes into the above builder's staging directory. Once
  // satisfied, it is typically used to schedule committing the runtimes as the
  // last task.
  Latch step_counter_;
  // NOLINTEND(misc-non-private-member-variables-in-classes)
};

// Helper class that factors out the logic to build an archive as part of a set
// of Clang runtimes.
//
// Runtimes can consist of one or more archives, and potentially other
// artifacts, but because archives are very common we factor out logic to build
// them here.
class ClangRuntimesBuilderBase::ArchiveBuilder {
 public:
  // Initialize an archive builder.
  //
  // - The `builder` must outlive this object.
  // - `archive_path` is the _relative_ path of the archive file within the
  // built
  //   runtimes directory.
  // - `srcs_path` is the _absolute_ root of source files used to build the
  // archive.
  // - `src_files` is a list of the file paths to build into the archive,
  //   relative to the `srcs_path`. These are `StringRef`s so they can reference
  //   constant `StringLiteral`s in common cases.
  // - `cflags` are the compile flags that should be used for all the compiles
  //   in this archive.
  ArchiveBuilder(ClangRuntimesBuilderBase* builder,
                 std::filesystem::path archive_path,
                 std::filesystem::path srcs_path,
                 llvm::ArrayRef<llvm::StringRef> src_files,
                 llvm::ArrayRef<llvm::StringRef> cflags)
      : builder_(builder),
        vlog_stream_(builder_->vlog_stream_),
        archive_path_(std::move(archive_path)),
        srcs_path_(std::move(srcs_path)),
        src_files_(src_files),
        cflags_(cflags) {}

  // Start building the archive, with a latch handle to signal its completion.
  //
  // This will launch asynchronous tasks on the `builder_->tasks` task group to
  // first compile all the members of the archive, and once compiled to put them
  // into the archive file. Only when this last step is complete will the
  // provided handle be destroyed, signaling this step of any concurrent build
  // is done.
  //
  // This must only be called once per instance.
  auto Setup(Latch::Handle latch_handle) -> void;

  // Accessor for the result of building the archive.
  //
  // This will return an error if accessed prior to `Start`-ing the archive's
  // build.
  //
  // Once `Start` has been called, this must not be called until the
  // `latch_handle` provided to `Start` is destroyed as doing so will race with
  // producing the result.
  auto result() -> ErrorOr<Success>& { return result_; }

 private:
  // Helper for finishing the build of the archive.
  //
  // Must only be called once all members have been compiled, and returns the
  // result of forming the archive file from those members.
  auto Finish() -> ErrorOr<Success>;

  // Given a specific `src_path` relative to our `srcs_path`, create any
  // necessary directories relative to the build's runtimes root to allow the
  // object file for this source file to be written there.
  //
  // Returns any errors encountered creating the necessary directories.
  //
  // Uses a cache to avoid redundant directory creations, and stores a list of
  // all the directories created so they can be removed at the end of the build.
  //
  // This method is thread-safe, and designed to be called concurrently from
  // each source file's compilation.
  auto CreateObjDir(const std::filesystem::path& src_path) -> ErrorOr<Success>;

  // Compiles the `src_file` and read the resulting object as a new archive
  // member.
  //
  // The argument should be one of the elements of `src_files_`. The compile is
  // performed using the `builder_->clang_` runner and the provided `cflags_`.
  //
  // Any errors encountered are returned.
  auto CompileMember(llvm::StringRef src_file)
      -> ErrorOr<llvm::NewArchiveMember>;

  ClangRuntimesBuilderBase* builder_;
  llvm::raw_ostream* vlog_stream_;

  std::filesystem::path archive_path_;

  std::filesystem::path srcs_path_;
  llvm::SmallVector<llvm::StringRef> src_files_;

  llvm::SmallVector<llvm::StringRef> cflags_;

  // A latch used to synchronize building the archive once all members have been
  // compiled.
  Latch compilation_latch_;
  // Storage for either the archive members or the error encountered while
  // compiling them.
  llvm::SmallVector<ErrorOr<llvm::NewArchiveMember>> objs_;

  // A mutex and vector used to maintain a thread-safe cache of directories
  // created to hold object files when compiling.
  std::mutex obj_dirs_mu_;
  llvm::SmallVector<std::filesystem::path> obj_dirs_;

  // Storage for the final result of building the requested archive.
  ErrorOr<Success> result_ = Error("No archive built!");
};

// A class template to build runtimes consisting of a single archive.
//
// The template argument comes from the `Runtimes::Component` enum, but is only
// intended for Clang-runtimes that consist of a single archive. We use a
// requires to enforce that the components used are exactly one of those
// supported so we can also move instantiation into the `.cpp` file.
template <Runtimes::Component Component>
  requires(Component == Runtimes::LibUnwind)
class ClangArchiveRuntimesBuilder : public ClangRuntimesBuilderBase {
 public:
  // Constructing this class will attempt to build the `Component` archive into
  // `runtimes`.
  //
  // If an existing build is found, it will immediately be available.
  // Otherwise, constructing this class will schedule asynchronous work on
  // `threads` to build the archive on-demand using `clang`.
  //
  // Once constructed, callers may call `Wait` (from the base class) to wait
  // until the asynchronous work is complete and the runtimes are available. If
  // they were already available, the call to `Wait` will not block.
  ClangArchiveRuntimesBuilder(ClangRunner* clang,
                              llvm::ThreadPoolInterface* threads,
                              llvm::Triple target_triple, Runtimes* runtimes);

 private:
  // Helpers to compute the list of source files and compile flags for a
  // particular archive. The implementations of these are expected to be
  // specialized for each different `Component`.
  auto CollectSrcFiles() -> llvm::SmallVector<llvm::StringRef>;
  auto CollectCflags() -> llvm::SmallVector<llvm::StringRef>;

  // Helper to encapsulate the initial, but still asynchronous setup work.
  auto Setup() -> void;

  // Helper to encapsulate the final asynchronous step in building the resource
  // directory.
  auto Finish() -> void;

  // The root path used for any of the source files.
  std::filesystem::path srcs_path_;

  // The (absolute) include path used during the compilation of the source
  // files.
  std::filesystem::path include_path_;

  // The relative archive path within the runtimes' build directory.
  std::filesystem::path archive_path_;

  // The archive builder if it is necessary to build the archive.
  std::optional<ArchiveBuilder> archive_;
};

extern template class ClangArchiveRuntimesBuilder<Runtimes::LibUnwind>;

// Builds the target-specific resource directory for Clang.
//
// There is a resource directory installed along side the Clang binary that
// contains all the target independent files such as headers. However, for
// target-specific files like the runtimes that are part of the resource
// directory, we build those on demand as runtimes.
//
class ClangResourceDirBuilder : public ClangRuntimesBuilderBase {
 public:
  // Constructing this class will attempt to build the Clang resource directory
  // into `runtimes`.
  //
  // If an existing build is found, it will immediately be available.
  // Otherwise, constructing this class will schedule asynchronous work on
  // `threads` to build them on-demand using `clang`.
  //
  // Once constructed, callers may call `Wait` (from the base class) to wait
  // until the asynchronous work is complete and the runtimes are available. If
  // they were already available, the call to `Wait` will not block.
  ClangResourceDirBuilder(ClangRunner* clang,
                          llvm::ThreadPoolInterface* threads,
                          llvm::Triple target_triple, Runtimes* runtimes);

 private:
  // Helper method to encapsulate the logic of configuring the list of source
  // files to use in the `builtins` archive within these runtimes on a
  // particular target.
  auto CollectBuiltinsSrcFiles() -> llvm::SmallVector<llvm::StringRef>;

  // Helper to encapsulate the initial, but still asynchronous setup work.
  auto Setup() -> void;

  // Helper to encapsulate the final asynchronous step in building the resource
  // directory.
  auto Finish() -> void;

  // Helper to compile a single file of the CRT runtimes.
  auto BuildCrtFile(llvm::StringRef src_file) -> ErrorOr<Success>;

  // The `lib` path and subdirectory of the being-built runtimes.
  std::filesystem::path lib_path_;
  Filesystem::Dir lib_dir_;

  // The results of compiling the CRT `begin` and `end` files.
  ErrorOr<Success> crt_begin_result_;
  ErrorOr<Success> crt_end_result_;

  // The archive builder for the builtins archive in the resource directory.
  std::optional<ArchiveBuilder> archive_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_DRIVER_CLANG_RUNTIMES_H_
