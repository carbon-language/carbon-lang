// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/build_runtimes_subcommand.h"

#include <variant>

#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/clang_runner.h"

namespace Carbon {

auto BuildRuntimesOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringOption(
      {
          .name = "output-directory",
          .value_name = "DIR",
          .help = R"""(
The directory to populate with runtime libraries suitable for the selected code
generation options.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&directory); });

  codegen_options.Build(b);
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "build-runtimes",
    .help = R"""(
Build Carbon's runtime libraries.

This subcommand builds Carbon's runtime libraries for a particular code
generation target, either in their default location or a specified one.

Running this command directly is not necessary as Carbon will build and cache
runtimes as needed when linking, but building them directly can aid in
debugging issues or allow them to be prebuilt, possibly with customized code
generation flags, and used explicitly when linking.
)""",
};

BuildRuntimesSubcommand::BuildRuntimesSubcommand()
    : DriverSubcommand(SubcommandInfo) {}

auto BuildRuntimesSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }

  // For diagnosing filesystem or other errors when building runtimes.
  CARBON_DIAGNOSTIC(FailureBuildingRuntimes, Error,
                    "failure building runtimes: {0}", std::string);

  auto run_result = RunInternal(driver_env);
  if (!run_result.ok()) {
    driver_env.emitter.Emit(FailureBuildingRuntimes,
                            run_result.error().message());
    return {.success = false};
  }

  llvm::outs() << "Built runtimes: " << *run_result << "\n";
  return {.success = true};
}

auto BuildRuntimesSubcommand::RunInternal(DriverEnv& driver_env)
    -> ErrorOr<std::filesystem::path> {
  ClangRunner runner(driver_env.installation, driver_env.fs,
                     driver_env.vlog_stream);

  Runtimes::Cache::Features features = {
      .target = options_.codegen_options.target.str()};

  bool is_cache = options_.directory.empty();
  std::filesystem::path explicit_output_path = options_.directory.str();
  if (!is_cache) {
    auto access_result = Filesystem::Cwd().Access(explicit_output_path);
    if (access_result.ok()) {
      return Error("output directory already exists");
    }
    if (!access_result.error().no_entity()) {
      return std::move(access_result).error();
    }
  }

  CARBON_ASSIGN_OR_RETURN(
      auto runtimes,
      is_cache ? driver_env.runtimes_cache.Lookup(features)
               : Runtimes::Make(explicit_output_path, driver_env.vlog_stream));
  CARBON_ASSIGN_OR_RETURN(auto tmp_dir, Filesystem::MakeTmpDir());

  return runner.BuildTargetResourceDir(features, runtimes, tmp_dir.abs_path(),
                                       *driver_env.thread_pool);
}

}  // namespace Carbon
