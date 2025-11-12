// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/carbon_runner.h"

#include <filesystem>
#include <utility>

#include "common/error.h"
#include "common/filesystem.h"
#include "common/vlog.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/build_runtimes_subcommand.h"
#include "toolchain/driver/compile_subcommand.h"
#include "toolchain/driver/driver.h"
#include "toolchain/driver/driver_env.h"
#include "toolchain/driver/driver_subcommand.h"
#include "toolchain/driver/link_subcommand.h"
#include "toolchain/driver/runtimes_cache.h"
#include "toolchain/driver/tool_runner_base.h"
#include "toolchain/install/install_paths.h"

namespace Carbon {

CarbonRunner::CarbonRunner(DriverEnv* driver_env)
    : ToolRunnerBase(driver_env->installation, driver_env->vlog_stream),
      driver_env_(driver_env) {}

auto CarbonRunner::BuildCoreLibraries(const Runtimes::Cache::Features& features,
                                      Runtimes& runtimes)
    -> ErrorOr<std::filesystem::path> {
  std::filesystem::path core_a_path = "carbon_core.a";

  // Build directory for core libs.
  CARBON_ASSIGN_OR_RETURN(auto build_dir,
                          runtimes.Build(Runtimes::CoreLibraries));
  if (std::holds_alternative<std::filesystem::path>(build_dir)) {
    // Found cached build.
    return std::get<std::filesystem::path>(std::move(build_dir)) / core_a_path;
  }

  std::string target = features.target;
  llvm::Triple target_triple(target);

  CARBON_VLOG("Building Carbon core libraries for target '{0}'", target);

  // Create temporary directory.
  CARBON_ASSIGN_OR_RETURN(auto tmp_path, Filesystem::MakeTmpDir());
  CARBON_ASSIGN_OR_RETURN(Filesystem::Dir tmp_dir,
                          Filesystem::Cwd().OpenDir(tmp_path.abs_path()));

  // Collect Core source files.
  std::filesystem::path core_src_dir = installation_->core_package();
  llvm::SmallVector<std::filesystem::path> core_src_files;

  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(core_src_dir)) {
    const auto& src_path = entry.path();
    if (std::filesystem::is_regular_file(src_path) &&
        src_path.extension() == ".carbon") {
      core_src_files.push_back(src_path);
    }
  }

  // Copy files into the temporary directory.
  llvm::SmallVector<std::filesystem::path> tmp_src_files;
  for (const auto& core_src_file : core_src_files) {
    // Add parent directories.
    std::filesystem::path relative_core_src_file =
        core_src_file.lexically_relative(core_src_dir);
    if (relative_core_src_file.has_parent_path()) {
      CARBON_RETURN_IF_ERROR(
          tmp_dir.CreateDirectories(relative_core_src_file.parent_path()));
    }

    std::filesystem::path tmp_src_file =
        tmp_path.abs_path() / relative_core_src_file;

    // Add to files for compiling.
    tmp_src_files.push_back(tmp_src_file);

    // Copy file.
    std::error_code ec = llvm::sys::fs::copy_file(
        std::filesystem::absolute(core_src_file).string(),
        tmp_src_file.string());
    if (ec) {
      return Error(ec.message());
    }
  }

  // Compile object files.
  DriverResult result =
      Compile(tmp_src_files, Lower::OptimizationLevel::Debug, target_triple,
              /*prelude_import=*/false);
  if (!result.success) {
    return Error("Failed to compile Core libraries.");
  }

  // Collect Core object files.
  llvm::SmallVector<llvm::NewArchiveMember> objs;

  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(tmp_path.abs_path())) {
    const auto& obj_path = entry.path();
    if (std::filesystem::is_regular_file(obj_path) &&
        obj_path.extension() == ".o") {
      auto obj_result = llvm::NewArchiveMember::getFile(obj_path.native(),
                                                        /*Deterministic=*/true);
      CARBON_CHECK(obj_result, "TODO: Diagnose this: {0}",
                   llvm::fmt_consume(obj_result.takeError()));
      objs.push_back(std::move(*obj_result));
    }
  }

  // Build archive.
  auto builder = std::get<Runtimes::Builder>(std::move(build_dir));

  CARBON_ASSIGN_OR_RETURN(
      Filesystem::WriteFile core_a_file,
      builder.dir().OpenWriteOnly(core_a_path, Filesystem::CreateAlways));

  {
    llvm::raw_fd_ostream core_a_os = core_a_file.WriteStream();
    llvm::Error archive_err = llvm::writeArchiveToStream(
        core_a_os, objs, llvm::SymtabWritingMode::NormalSymtab,
        target_triple.isOSDarwin() ? llvm::object::Archive::K_DARWIN
                                   : llvm::object::Archive::K_GNU,
        /*Deterministic=*/true, /*Thin=*/false);
    // The presence of an error is `true`.
    if (archive_err) {
      return Error(llvm::toString(std::move(archive_err)));
    }
  }
  CARBON_RETURN_IF_ERROR(std::move(core_a_file).Close());

  CARBON_ASSIGN_OR_RETURN(std::filesystem::path cached_core_path,
                          std::move(builder).Commit());

  return cached_core_path / core_a_path;
}

auto CarbonRunner::Compile(
    llvm::SmallVector<std::filesystem::path> input_filenames,
    Lower::OptimizationLevel opt_level, llvm::Triple target,
    bool prelude_import) -> DriverResult {
  // Convert to StringRef vector for options.
  llvm::SmallVector<llvm::StringRef> input_filename_refs;
  for (const auto& filename : input_filenames) {
    // Keep StringRef valid by using c_str() and length().
    input_filename_refs.push_back(
        llvm::StringRef(filename.c_str(), filename.string().length()));
  }

  CompileOptions options = {.opt_level = opt_level,
                            .codegen_options = {.target = target.str()},
                            .input_filenames = input_filename_refs,
                            .prelude_import = prelude_import};
  CompileSubcommand subcommand(std::move(options));
  return subcommand.Run(*driver_env_);
}

}  // namespace Carbon
