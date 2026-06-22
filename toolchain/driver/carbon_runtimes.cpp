// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/carbon_runtimes.h"

#include "toolchain/base/install_paths.h"
#include "toolchain/base/runtimes_build_info.h"

namespace Carbon {

CarbonRuntimesBuilderBase::CarbonRuntimesBuilderBase(
    DriverEnv* driver_env, std::shared_ptr<CodegenOptions> codegen_options)
    : compile_options_(),
      compile_driver_(&compile_options_),
      driver_env_(driver_env),
      install_root_(driver_env->installation->root()),
      result_(Error("Did not finish building the Carbon runtimes!")) {
  compile_options_.codegen_options = codegen_options;
}

CarbonPreludeBuilder::CarbonPreludeBuilder(
    DriverEnv* driver_env, std::shared_ptr<CodegenOptions> codegen_options,
    Runtimes* runtimes)
    : CarbonRuntimesBuilderBase(driver_env, codegen_options) {
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

  // Each prelude file explicitly imports the other parts of the prelude it
  // needs.
  compile_options_.prelude_import = false;
  auto install_root_length = install_root_.string().length();
  CARBON_CHECK(
      compile_driver_.Initialize(
          driver_env_,
          [&](llvm::StringRef input_filename) -> std::string {
            // Make the input path relative to the install_root_ again, and
            // replace the `.carbon` extension with a `.o`.
            auto object_path =
                std::filesystem::path(
                    input_filename.substr(install_root_length).str())
                    .replace_extension(".o");
            auto output_path =
                runtimes_builder_->path() / lib_path_ / object_path;
            // The output path may contain subdirectories that haven't yet been
            // created, so check for that and create if necessary.
            if (!std::filesystem::exists(output_path.parent_path())) {
              CARBON_CHECK(
                  runtimes_builder_->dir()
                      .CreateDirectories(lib_path_ / output_path.parent_path())
                      .ok(),
                  "Failed to make output subdirectory while building Carbon "
                  "prelude.");
            }
            llvm::errs() << "mapping: `" << input_filename << "` to: `"
                         << output_path << "`\n";
            return output_path.string();
          }),
      "Failed to initialize compiler driver for Carbon prelude.");
  CARBON_CHECK(compile_driver_.Compile(driver_env_).success,
               "Failed to compile Carbon prelude.");

  result_ = (*std::move(runtimes_builder_)).Commit();
  return std::move(result_);
}

}  // namespace Carbon
