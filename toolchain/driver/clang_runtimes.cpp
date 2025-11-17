// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_runtimes.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

#include "common/check.h"
#include "common/error.h"
#include "common/filesystem.h"
#include "common/latch.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/runtime_sources.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/runtimes_cache.h"

namespace Carbon {

auto ClangRuntimesBuilderBase::ArchiveBuilder::Setup(Latch::Handle latch_handle)
    -> void {
  // `NewArchiveMember` isn't default constructable unfortunately, so we have to
  // manually populate the vector with errors that we'll replace with the actual
  // result in each thread.
  objs_.reserve(src_files_.size());
  for (auto _ : src_files_) {
    objs_.push_back(Error("Never constructed archive member!"));
  }

  // Finish building the archive when the last compile finishes.
  Latch::Handle comp_latch_handle =
      compilation_latch_.Init([this, latch_handle] { result_ = Finish(); });

  // Add all the compiles to the thread pool to run concurrently. The latch
  // handle ensures the last one triggers the finishing closure above.
  for (auto [src_file, obj] : llvm::zip_equal(src_files_, objs_)) {
    builder_->tasks_.async([this, comp_latch_handle, src_file, &obj]() mutable {
      obj = CompileMember(src_file);
    });
  }
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::Finish() -> ErrorOr<Success> {
  // We build this directly into the desired location as this is expected to be
  // a staging directory and cleaned up on errors. We do need to create any
  // intermediate directories.
  Filesystem::DirRef runtimes_dir = builder_->runtimes_builder_->dir();
  if (archive_path_.has_parent_path()) {
    CARBON_RETURN_IF_ERROR(
        runtimes_dir.CreateDirectories(archive_path_.parent_path()));
  }

  // Check if any compilations ended up producing an error. If so, return the
  // first error for the entire function. Otherwise, move the archive member
  // into a direct vector to match the required archive building API.
  llvm::SmallVector<llvm::NewArchiveMember> unwrapped_objs;
  unwrapped_objs.reserve(objs_.size());
  for (auto& obj : objs_) {
    if (!obj.ok()) {
      return std::move(obj).error();
    }
    unwrapped_objs.push_back(*std::move(obj));
  }
  objs_.clear();

  // Remove any directories created for the object files, the files should
  // already be removed. We walk the sorted list of these in reverse so we
  // remove child directories before parent directories.
  for (const auto& obj_dir : llvm::reverse(obj_dirs_)) {
    auto rmdir_result = runtimes_dir.Rmdir(obj_dir);
    // Don't return an error on failure here as this has no problematic
    // effect, just log that we couldn't clean up a directory.
    if (!rmdir_result.ok()) {
      CARBON_VLOG("Unable to remove object directory `{0}` in the runtime: {1}",
                  obj_dir.native(), rmdir_result.error());
    }
  }

  // Write the actual archive.
  CARBON_ASSIGN_OR_RETURN(
      Filesystem::WriteFile archive_file,
      runtimes_dir.OpenWriteOnly(archive_path_, Filesystem::CreateAlways));
  {
    llvm::raw_fd_ostream archive_os = archive_file.WriteStream();
    llvm::Error archive_err = llvm::writeArchiveToStream(
        archive_os, unwrapped_objs, llvm::SymtabWritingMode::NormalSymtab,
        builder_->target_triple_.isOSDarwin() ? llvm::object::Archive::K_DARWIN
                                              : llvm::object::Archive::K_GNU,
        /*Deterministic=*/true, /*Thin=*/false);
    // The presence of an error is `true`.
    if (archive_err) {
      (void)std::move(archive_file).Close();
      return Error(llvm::toString(std::move(archive_err)));
    }
  }
  // Close and return any errors, potentially from the writes above.
  CARBON_RETURN_IF_ERROR(std::move(archive_file).Close());
  return Success();
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::CreateObjDir(
    const std::filesystem::path& src_path) -> ErrorOr<Success> {
  auto obj_dir_path = src_path.parent_path();
  if (obj_dir_path.empty()) {
    return Success();
  }

  std::scoped_lock lock(obj_dirs_mu_);
  auto* it = std::lower_bound(obj_dirs_.begin(), obj_dirs_.end(), obj_dir_path);
  if (it != obj_dirs_.end() && *it == obj_dir_path) {
    return Success();
  }

  auto create_result =
      builder_->runtimes_builder_->dir().CreateDirectories(obj_dir_path);
  if (!create_result.ok()) {
    return Error(llvm::formatv(
        "Unable to create object directory mirroring source file `{0}`: {1}",
        src_path, create_result.error()));
  }

  it = obj_dirs_.insert(it, obj_dir_path);

  // Also insert any parent paths. These should always sort earlier.
  CARBON_DCHECK(!obj_dir_path.has_parent_path() ||
                obj_dir_path.parent_path() < obj_dir_path);
  obj_dir_path = obj_dir_path.parent_path();
  while (!obj_dir_path.empty()) {
    it = std::lower_bound(obj_dirs_.begin(), it, obj_dir_path);
    if (*it != obj_dir_path) {
      it = obj_dirs_.insert(it, obj_dir_path);
    }
    obj_dir_path = obj_dir_path.parent_path();
  }
  return Success();
}

auto ClangRuntimesBuilderBase::ArchiveBuilder::CompileMember(
    llvm::StringRef src_file) -> ErrorOr<llvm::NewArchiveMember> {
  // Create any obj subdirectories needed for this file.
  CARBON_RETURN_IF_ERROR(CreateObjDir(src_file.str()));

  std::filesystem::path obj_path =
      builder_->runtimes_builder_->path() / std::string_view(src_file);
  obj_path += ".o";
  std::filesystem::path src_path = srcs_path_ / std::string_view(src_file);
  CARBON_VLOG("Building `{0}' from `{1}`...\n", obj_path, src_path);

  llvm::SmallVector<llvm::StringRef> args(cflags_);

  // Add language-specific flags based on file extension.
  if (src_file.ends_with(".c")) {
    args.push_back("-std=c11");
  } else if (src_file.ends_with(".cpp")) {
    args.push_back("-std=c++20");
  }

  // Collect the additional required flags and dynamic flags for this builder.
  args.append({
      "-c",
      builder_->target_flag_,
      "-o",
      obj_path.native(),
      src_path.native(),
  });
  if (!builder_->clang_->RunWithNoRuntimes(args)) {
    return Error(
        llvm::formatv("Failed to compile runtime source file '{0}'", src_file));
  }

  auto obj_result = llvm::NewArchiveMember::getFile(obj_path.native(),
                                                    /*Deterministic=*/true);
  if (!obj_result) {
    return Error(llvm::formatv("Unable to read `{0}` object file: {1}",
                               src_file,
                               llvm::fmt_consume(obj_result.takeError())));
  }

  // Unlink the object file once we've read it. However, we log and ignore
  // any errors here as they aren't fatal.
  auto unlink_result = builder_->runtimes_builder_->dir().Unlink(obj_path);
  if (!unlink_result.ok()) {
    CARBON_VLOG("Unable to unlink object file `{0}`: {1}\n", obj_path,
                unlink_result.error());
  }

  return std::move(*obj_result);
}

template <Runtimes::Component Component>
  requires(Component == Runtimes::LibUnwind)
ClangArchiveRuntimesBuilder<Component>::ClangArchiveRuntimesBuilder(
    ClangRunner* clang, llvm::ThreadPoolInterface* threads,
    llvm::Triple target_triple, Runtimes* runtimes)
    : ClangRuntimesBuilderBase(clang, threads, std::move(target_triple)) {
  // Ensure we're on a platform where we _can_ build a working runtime.
  if (target_triple_.isOSWindows()) {
    result_ =
        Error("TODO: Windows runtimes are untested and not yet supported.");
    return;
  }

  auto build_dir_or_error = runtimes->Build(Component);
  if (!build_dir_or_error.ok()) {
    result_ = std::move(build_dir_or_error).error();
    return;
  }
  auto build_dir = *(std::move(build_dir_or_error));
  CARBON_KIND_SWITCH(std::move(build_dir)) {
    case CARBON_KIND(std::filesystem::path build_dir_path): {
      // Found cached build.
      result_ = std::move(build_dir_path);
      return;
    }
    case CARBON_KIND(Runtimes::Builder builder): {
      runtimes_builder_ = std::move(builder);
      // Building the runtimes is handled below.
      break;
    }
  }

  if constexpr (Component == Runtimes::LibUnwind) {
    srcs_path_ = installation().libunwind_path();
    include_path_ = installation().libunwind_path() / "include";
    archive_path_ = std::filesystem::path("lib") / "libunwind.a";
  } else {
    static_assert(false,
                  "Invalid runtimes component for an archive runtime builder.");
  }

  archive_.emplace(this, archive_path_, srcs_path_, CollectSrcFiles(),
                   CollectCflags());
  tasks_.async([this]() mutable { Setup(); });
}

template <Runtimes::Component Component>
  requires(Component == Runtimes::LibUnwind)
auto ClangArchiveRuntimesBuilder<Component>::CollectSrcFiles()
    -> llvm::SmallVector<llvm::StringRef> {
  if constexpr (Component == Runtimes::LibUnwind) {
    return llvm::SmallVector<llvm::StringRef>(llvm::make_filter_range(
        RuntimeSources::LibunwindSrcs, [](llvm::StringRef src) {
          return src.ends_with(".c") || src.ends_with(".cpp") ||
                 src.ends_with(".S");
        }));
  } else {
    static_assert(false,
                  "Invalid runtimes component for an archive runtime builder.");
  }
}

template <Runtimes::Component Component>
  requires(Component == Runtimes::LibUnwind)
auto ClangArchiveRuntimesBuilder<Component>::CollectCflags()
    -> llvm::SmallVector<llvm::StringRef> {
  if constexpr (Component == Runtimes::LibUnwind) {
    return {
        "-no-canonical-prefixes",
        "-O3",
        "-fPIC",
        "-funwind-tables",
        "-fno-exceptions",
        "-fno-rtti",
        "-nostdinc++",
        "-I",
        include_path_.native(),
        "-D_LIBUNWIND_IS_NATIVE_ONLY",
        "-w",
    };
  } else {
    static_assert(false,
                  "Invalid runtimes component for an archive runtime builder.");
  }
}

template <Runtimes::Component Component>
  requires(Component == Runtimes::LibUnwind)
auto ClangArchiveRuntimesBuilder<Component>::Setup() -> void {
  // Symlink the installation's `include` into the runtime.
  CARBON_CHECK(include_path_.is_absolute(),
               "Unexpected relative include path: {0}", include_path_);
  if (auto result = runtimes_builder_->dir().Symlink("include", include_path_);
      !result.ok()) {
    result_ = std::move(result).error();
    return;
  }

  // Finish building the runtime once the archive is built.
  Latch::Handle latch_handle = step_counter_.Init(
      [this]() mutable { tasks_.async([this]() mutable { Finish(); }); });

  // Start building the archive itself with a handle to detect when complete.
  archive_->Setup(std::move(latch_handle));
}

template <Runtimes::Component Component>
  requires(Component == Runtimes::LibUnwind)
auto ClangArchiveRuntimesBuilder<Component>::Finish() -> void {
  CARBON_VLOG("Finished building {0}...\n", archive_path_);
  if (!archive_->result().ok()) {
    result_ = std::move(archive_->result()).error();
    return;
  }

  result_ = (*std::move(runtimes_builder_)).Commit();
}

template class ClangArchiveRuntimesBuilder<Runtimes::LibUnwind>;

ClangResourceDirBuilder::ClangResourceDirBuilder(
    ClangRunner* clang, llvm::ThreadPoolInterface* threads,
    llvm::Triple target_triple, Runtimes* runtimes)
    : ClangRuntimesBuilderBase(clang, threads, std::move(target_triple)),
      crt_begin_result_(Error("Never built CRT begin file!")),
      crt_end_result_(Error("Never built CRT end file!")) {
  // Ensure we're on a platform where we _can_ build a working runtime.
  if (target_triple_.isOSWindows()) {
    result_ =
        Error("TODO: Windows runtimes are untested and not yet supported.");
    return;
  }

  auto build_dir_or_error = runtimes->Build(Runtimes::ClangResourceDir);
  if (!build_dir_or_error.ok()) {
    result_ = std::move(build_dir_or_error).error();
    return;
  }
  auto build_dir = *std::move(build_dir_or_error);
  if (std::holds_alternative<std::filesystem::path>(build_dir)) {
    // Found cached build.
    result_ = std::get<std::filesystem::path>(std::move(build_dir));
    return;
  }

  runtimes_builder_ = std::get<Runtimes::Builder>(std::move(build_dir));
  lib_path_ = std::filesystem::path("lib") / target_triple_.str();
  archive_.emplace(this, lib_path_ / "libclang_rt.builtins.a",
                   installation().llvm_runtime_srcs(),
                   CollectBuiltinsSrcFiles(), /*cflags=*/
                   llvm::SmallVector<llvm::StringRef>{
                       "-no-canonical-prefixes",
                       "-O3",
                       "-fPIC",
                       "-ffreestanding",
                       "-fno-builtin",
                       "-fomit-frame-pointer",
                       "-fvisibility=hidden",
                       "-w",
                   });
  tasks_.async([this]() { Setup(); });
}

auto ClangResourceDirBuilder::CollectBuiltinsSrcFiles()
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> src_files;
  auto append_src_files =
      [&](auto input_srcs,
          llvm::function_ref<bool(llvm::StringRef)> filter_out = {}) {
        for (llvm::StringRef input_src : input_srcs) {
          if (!input_src.ends_with(".c") && !input_src.ends_with(".S")) {
            // Not a compiled file.
            continue;
          }
          if (filter_out && filter_out(input_src)) {
            // Filtered out.
            continue;
          }

          src_files.push_back(input_src);
        }
      };
  append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsGenericSrcs));
  append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsBf16Srcs));
  if (target_triple_.isArch64Bit()) {
    append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsTfSrcs));
  }
  auto filter_out_chkstk = [&](llvm::StringRef src) {
    return !target_triple_.isOSWindows() || !src.ends_with("chkstk.S");
  };
  if (target_triple_.isAArch64()) {
    append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsAarch64Srcs),
                     filter_out_chkstk);
  } else if (target_triple_.isX86()) {
    append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsX86ArchSrcs));
    if (target_triple_.isArch64Bit()) {
      append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsX86_64Srcs),
                       filter_out_chkstk);
    } else {
      // TODO: This should be turned into a nice user-facing diagnostic about an
      // unsupported target.
      CARBON_CHECK(
          target_triple_.isArch32Bit(),
          "The Carbon toolchain doesn't currently support 16-bit x86.");
      append_src_files(llvm::ArrayRef(RuntimeSources::BuiltinsI386Srcs),
                       filter_out_chkstk);
    }
  } else {
    // TODO: This should be turned into a nice user-facing diagnostic about an
    // unsupported target.
    CARBON_FATAL("Target architecture is not supported: {0}",
                 target_triple_.str());
  }
  return src_files;
}

auto ClangResourceDirBuilder::Setup() -> void {
  // Symlink the installation's `include` and `share` directories.
  std::filesystem::path install_resource_path =
      installation().clang_resource_path();
  if (auto result = runtimes_builder_->dir().Symlink(
          "include", install_resource_path / "include");
      !result.ok()) {
    result_ = std::move(result).error();
    return;
  }
  if (auto result = runtimes_builder_->dir().Symlink(
          "share", install_resource_path / "share");
      !result.ok()) {
    result_ = std::move(result).error();
    return;
  }

  // Create the target's `lib` directory.
  auto lib_dir_result = runtimes_builder_->dir().CreateDirectories(lib_path_);
  if (!lib_dir_result.ok()) {
    result_ = std::move(lib_dir_result).error();
    return;
  }
  lib_dir_ = *std::move(lib_dir_result);

  Latch::Handle latch_handle =
      step_counter_.Init([this] { tasks_.async([this] { Finish(); }); });

  // For Linux targets, the system libc (typically glibc) doesn't necessarily
  // provide the CRT begin/end files, and so we need to build them.
  if (target_triple_.isOSLinux()) {
    tasks_.async([this, latch_handle] {
      crt_begin_result_ = BuildCrtFile(RuntimeSources::CrtBegin);
    });
    tasks_.async([this, latch_handle] {
      crt_end_result_ = BuildCrtFile(RuntimeSources::CrtEnd);
    });
  }

  archive_->Setup(std::move(latch_handle));
}

auto ClangResourceDirBuilder::Finish() -> void {
  CARBON_VLOG("Finished building resource dir...\n");
  if (!archive_->result().ok()) {
    result_ = std::move(archive_->result()).error();
    return;
  }
  if (target_triple_.isOSLinux()) {
    for (ErrorOr<Success>* result : {&crt_begin_result_, &crt_end_result_}) {
      if (!result->ok()) {
        result_ = std::move(*result).error();
        return;
      }
    }
  }

  result_ = (*std::move(runtimes_builder_)).Commit();
}

auto ClangResourceDirBuilder::BuildCrtFile(llvm::StringRef src_file)
    -> ErrorOr<Success> {
  CARBON_CHECK(src_file == RuntimeSources::CrtBegin ||
               src_file == RuntimeSources::CrtEnd);
  std::filesystem::path out_path =
      runtimes_builder_->path() / lib_path_ /
      (src_file == RuntimeSources::CrtBegin ? "clang_rt.crtbegin.o"
                                            : "clang_rt.crtend.o");
  std::filesystem::path src_path =
      installation().llvm_runtime_srcs() / std::string_view(src_file);
  CARBON_VLOG("Building `{0}' from `{1}`...\n", out_path, src_path);

  bool success = clang_->RunWithNoRuntimes({
      "-no-canonical-prefixes",
      "-DCRT_HAS_INITFINI_ARRAY",
      "-DEH_USE_FRAME_REGISTRY",
      "-O3",
      "-fPIC",
      "-ffreestanding",
      "-std=c11",
      "-w",
      "-c",
      target_flag_,
      "-o",
      out_path.native(),
      src_path.native(),
  });

  if (success) {
    return Success();
  }
  return Error(llvm::formatv("Failed to compile CRT file: {0}", src_file));
}

}  // namespace Carbon
