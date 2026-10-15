// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/carbon_runtimes.h"

#include "toolchain/base/install_paths.h"
#include "toolchain/base/runtimes_build_info.h"

namespace Carbon {

CarbonRuntimesBuilderBase::CarbonRuntimesBuilderBase(
    DriverEnv* driver_env, const Runtimes::Cache::Features& features)
    : target_(features.target),
      codegen_options_({.target = target_}),
      compile_options_(&codegen_options_),
      compile_driver_(&compile_options_),
      driver_env_(driver_env),
      install_root_(driver_env->installation->root()),
      result_(Error("Did not finish building the Carbon runtimes!")) {
  // Each prelude file explicitly imports the other parts of the prelude it
  // needs.
  compile_options_.prelude_import = false;
  // Don't also try and compile the rest of the core when compiling the prelude.
  compile_options_.include_carbon_core = false;
  if (!features.cpu.empty()) {
    clang_args_storage_.append(
        {"-Xclang", "-target-cpu", "-Xclang", features.cpu});
  }
  if (!features.tune_cpu.empty()) {
    clang_args_storage_.append(
        {"-Xclang", "-tune-cpu", "-Xclang", features.tune_cpu});
  }
  for (const std::string& feature : features.target_features) {
    clang_args_storage_.append(
        {"-Xclang", "-target-feature", "-Xclang", feature});
  }
  compile_options_.clang_args.reserve(clang_args_storage_.size());
  for (const std::string& arg : clang_args_storage_) {
    compile_options_.clang_args.push_back(arg);
  }
}

CarbonPreludeBuilder::CarbonPreludeBuilder(
    DriverEnv* driver_env, Runtimes* runtimes,
    const Runtimes::Cache::Features& features)
    : CarbonRuntimesBuilderBase(driver_env, features) {
  auto build_dir_or_error = runtimes->Build(Runtimes::CarbonCore);
  if (!build_dir_or_error.ok()) {
    result_ = std::move(build_dir_or_error).error();
    return;
  }
  auto build_dir = *std::move(build_dir_or_error);
  if (std::holds_alternative<std::filesystem::path>(build_dir)) {
    // Reuse cached build.
    result_ = std::get<std::filesystem::path>(std::move(build_dir));
    return;
  }

  runtimes_builder_ = std::get<Runtimes::Builder>(std::move(build_dir));
  lib_path_ = std::filesystem::path("lib/core") /
              compile_options_.codegen_options->target.str();
}

auto CarbonPreludeBuilder::Build() && -> ErrorOr<std::filesystem::path> {
  // If we didn't make a Builder in the constructor we either encountered an
  // error or an already cached build, return the result.
  if (!runtimes_builder_) {
    return std::move(result_);
  }

  // Create the output directory.
  auto lib_dir_result = runtimes_builder_->dir().CreateDirectories(lib_path_);
  if (!lib_dir_result.ok()) {
    return std::move(lib_dir_result).error();
  }
  auto lib_dir = *std::move(lib_dir_result);

  // Gather the absolute paths to the prelude source files into a string list.
  // This serves as a backing store for the `llvm::StringRef` inputs required to
  // the `CompileDriver` via `compile_options_`.
  llvm::SmallVector<std::string> prelude_source_files;
  llvm::for_each(RuntimesBuildInfo::CarbonCorePreludeSrcs,
                 [&](llvm::StringRef src) -> void {
                   prelude_source_files.emplace_back(
                       (install_root_ / std::filesystem::path(src.str())));
                   compile_options_.input_filenames.emplace_back(
                       llvm::StringRef(prelude_source_files.back()));
                 });

  auto install_root_length = install_root_.string().length();
  auto init_result = compile_driver_.Initialize(
      *driver_env_, [&](llvm::StringRef input_filename) -> std::string {
        // Make the input path relative to the input path root, and
        // replace the `.carbon` extension with a `.o`, so we can re-parent
        // this file and any subdirectories into the output path.
        auto relative_output_path =
            lib_path_ / std::filesystem::path(
                            input_filename.substr(install_root_length).str())
                            .replace_extension(".o");

        // The output path may contain subdirectories that haven't yet been
        // created, so create if necessary.
        // **Note** this call to `Dir::CreateDirectory` requires a path
        // _relative_to_ where the `Dir` object itself was opened, which the
        // runtimes builder opened at the path represented by
        // `runtimes_builder_->path()`. So we use the _relative_ path here.
        // The Builder exposes both `dir()` and `path()` methods because
        // `Dir` objects themselves don't know their own path, so the
        // Builder presents it separately.
        auto mkdir_result = runtimes_builder_->dir().CreateDirectories(
            relative_output_path.parent_path());

        auto absolute_output_path =
            runtimes_builder_->path() / relative_output_path;
        CARBON_CHECK(
            mkdir_result.ok(),
            "Failed to make output subdirectory {0} while building Carbon "
            "prelude.",
            absolute_output_path.parent_path());
        return absolute_output_path.string();
      });
  if (!init_result) {
    return Error("Failed to initialize compiler driver for Carbon prelude.");
  }

  auto compile_result = compile_driver_.Compile(*driver_env_);
  if (!compile_result.success) {
    return Error("Failed to compile Carbon prelude.");
  }

  result_ = (*std::move(runtimes_builder_)).Commit();
  return std::move(result_);
}

}  // namespace Carbon
