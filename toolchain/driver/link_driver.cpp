// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/link_driver.h"

#include "common/filesystem.h"
#include "toolchain/base/runtimes_build_info.h"
#include "toolchain/driver/carbon_runtimes.h"

namespace Carbon {

LinkDriver::LinkDriver(LinkOptions* options) : options_(options) {}

namespace {

auto object_search(Filesystem::DirRef base_dir, std::filesystem::path base_path,
                   llvm::SmallVector<std::string>& prelude_paths) -> void {
  llvm::SmallVector<std::filesystem::path> work_paths({"."});
  while (!work_paths.empty()) {
    auto relative_path = work_paths.back();
    work_paths.pop_back();
    auto relative_dir = base_dir.OpenDir(relative_path);
    CARBON_CHECK(relative_dir.ok());
    llvm::SmallVector<std::filesystem::path> relative_sub_paths;
    llvm::SmallVector<std::filesystem::path> relative_file_paths;
    CARBON_CHECK(
        relative_dir
            ->AppendEntriesIf(relative_sub_paths, relative_file_paths,
                              [&relative_dir](llvm::StringRef name) -> bool {
                                auto path = std::filesystem::path(name.str());
                                auto stat = relative_dir->Stat(path);
                                CARBON_CHECK(stat.ok());
                                if (stat->is_dir()) {
                                  return true;
                                }
                                return path.extension() == ".o";
                              })
            .ok());
    llvm::for_each(relative_sub_paths, [relative_path, &work_paths](
                                           std::filesystem::path sub_path) {
      work_paths.push_back(relative_path / sub_path);
    });
    llvm::for_each(relative_file_paths, [relative_path, base_path,
                                         &prelude_paths](
                                            std::filesystem::path file_path) {
      prelude_paths.push_back((base_path / relative_path / file_path).string());
    });
  }
}

}  // namespace

auto LinkDriver::Link(DriverEnv& driver_env) -> DriverResult {
  // TODO: Currently we use the Clang driver to link. This works well on Unix
  // OSes but we likely need to directly build logic to invoke `link.exe` on
  // Windows where `cl.exe` doesn't typically cover that logic.

  // Use a reasonably large small vector here to minimize allocations. We expect
  // to link reasonably large numbers of object files.
  llvm::SmallVector<llvm::StringRef, 128> clang_args;

  // We link using a C++ mode of the driver.
  clang_args.push_back("--driver-mode=g++");

  // Pass the target down to Clang to pick up the correct defaults.
  std::string target_arg =
      llvm::formatv("--target={0}", options_->codegen_options->target).str();
  clang_args.push_back(target_arg);

  if (!options_->output_filename.empty()) {
    clang_args.push_back("-o");
    clang_args.push_back(options_->output_filename);
  } else if (options_->extra_clang_args.empty()) {
    CARBON_DIAGNOSTIC(LinkOutputOptionMissing, Error,
                      "no output specified to a link command and no extra "
                      "Clang options that can provide an output");
    driver_env.emitter.Emit(LinkOutputOptionMissing);
    return {.success = false};
  }

  if (options_->object_filenames.empty() &&
      options_->extra_clang_args.empty()) {
    CARBON_DIAGNOSTIC(LinkObjectFilesMissing, Error,
                      "no object files provided to link command and no extra "
                      "Clang options that could provide them");
    driver_env.emitter.Emit(LinkObjectFilesMissing);
    return {.success = false};
  }

  // Find or build the Carbon Core runtimes for linking the prelude into the
  // binary.
  bool include_prelude = false;
  std::filesystem::path core_path;
  Filesystem::DirRef runtimes_dir;
  std::filesystem::path runtimes_path;
  std::optional<Runtimes> runtimes_cache;
  if (options_->link_prelude_files) {
    if (driver_env.prebuilt_runtimes) {
      auto error_or_path =
          driver_env.prebuilt_runtimes->Get(Runtimes::CarbonCore);
      CARBON_CHECK(error_or_path.ok(),
                   "Prebuilt runtimes failed to fetch for Carbon prelude: {}",
                   error_or_path.error().message());
      core_path = std::move(*error_or_path);
      runtimes_dir = driver_env.prebuilt_runtimes->base_dir();
      runtimes_path = driver_env.prebuilt_runtimes->base_path();
      include_prelude = true;
    } else if (driver_env.build_runtimes_on_demand) {
      Runtimes::Cache::Features features = {
          .target = options_->codegen_options->target.str()};
      auto runtimes_or_error = driver_env.runtimes_cache.Lookup(features);
      CARBON_CHECK(runtimes_or_error.ok(), "Runtimes cache lookup failed: {}",
                   runtimes_or_error.error().message());
      auto runtimes = std::move(*runtimes_or_error);
      CarbonPreludeBuilder prelude_builder(
          &driver_env, options_->codegen_options, &runtimes);
      auto path_or_error = std::move(prelude_builder).Build();
      if (!path_or_error.ok()) {
        CARBON_DIAGNOSTIC(FailureBuildingRuntimes, Error,
                          "Failed to build Carbon prelude during linking: {0}",
                          std::string);
        driver_env.emitter.Emit(FailureBuildingRuntimes,
                                path_or_error.error().message());
        return {.success = false};
      }
      core_path = std::move(*path_or_error);
      runtimes_dir = runtimes.base_dir();
      runtimes_path = runtimes.base_path();
      // Keep the Runtimes object in scope so we can use it during linking.
      runtimes_cache = std::move(runtimes);
      include_prelude = true;
    }
  }

  // Note that we append any extra Clang args before our object filenames. This
  // allows us to propagate object filenames that collide with Clang flags using
  // `--` before the filenames. While in theory, this could create a problem in
  // the presence of mixtures of object files in the two lists and the order
  // being dependent, we don't expect that in practice.
  clang_args.append(options_->extra_clang_args.begin(),
                    options_->extra_clang_args.end());
  clang_args.push_back("--");

  // Append the Carbon prelude object files to the link.
  llvm::SmallVector<std::string> prelude_paths;
  if (include_prelude) {
    // Open subdirectory specifically for the object files relative to the
    // runtimes base path.
    auto relative_path = core_path.lexically_relative(runtimes_path);
    auto core_dir_or_error = runtimes_dir.OpenDir(relative_path);
    CARBON_CHECK(core_dir_or_error.ok(),
                 "Failed to open prelude binaries directory at {}, error: {}",
                 runtimes_path / relative_path, core_dir_or_error.error());
    auto core_dir = std::move(*core_dir_or_error);
    object_search(core_dir, core_path, prelude_paths);
    CARBON_CHECK(!prelude_paths.empty(), "Found no prelude files at {}",
                 runtimes_path / relative_path);
    llvm::for_each(prelude_paths,
                   [&clang_args](const std::string& path) -> void {
                     clang_args.push_back(llvm::StringRef(path));
                   });
  }

  clang_args.append(options_->object_filenames.begin(),
                    options_->object_filenames.end());

  ClangRunner runner(driver_env.installation, driver_env.fs,
                     driver_env.vlog_stream);
  ErrorOr<bool> run_result =
      driver_env.prebuilt_runtimes
          ? runner.RunWithPrebuiltRuntimes(clang_args,
                                           *driver_env.prebuilt_runtimes,
                                           driver_env.enable_leaking)
      : driver_env.build_runtimes_on_demand
          ? runner.Run(clang_args, driver_env.runtimes_cache,
                       *driver_env.thread_pool, driver_env.enable_leaking)
          : runner.RunWithNoRuntimes(clang_args, driver_env.enable_leaking);

  if (!run_result.ok()) {
    // This is not a Clang failure, but a failure to even run Clang, so we need
    // to diagnose it here.
    CARBON_DIAGNOSTIC(FailureRunningClangToLink, Error,
                      "failure running `clang` to perform linking: {0}",
                      std::string);
    driver_env.emitter.Emit(FailureRunningClangToLink,
                            run_result.error().message());
    return {.success = false};
  }
  // Successfully ran Clang to perform the link, return its result.
  return {.success = *run_result};
}

}  // namespace Carbon
