// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/build_runtimes_subcommand.h"

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
  ClangRunner runner(driver_env.installation, driver_env.fs,
                     driver_env.vlog_stream);

  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (!TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }

  // For diagnosing filesystem or other errors when building runtimes.
  CARBON_DIAGNOSTIC(FailureBuildingRuntimes, Error,
                    "failure building runtimes: {0}", std::string);

  auto tmp_result = Filesystem::MakeTmpDir();
  if (!tmp_result.ok()) {
    driver_env.emitter.Emit(FailureBuildingRuntimes,
                            tmp_result.error().message());
    return {.success = false};
  }
  Filesystem::RemovingDir tmp_dir = *std::move(tmp_result);

  // TODO: Currently, the default location is just a subdirectory of the
  // temporary directory used for the build. This allows the subcommand to be
  // used to test and debug runtime building, but not for the results to be
  // reused. Eventually, this should be connected to the same runtimes cache
  // used by link commands.
  std::filesystem::path output_path =
      options_.directory.empty()
          ? tmp_dir.abs_path() / "runtimes"
          : std::filesystem::path(options_.directory.str());

  // Hard code a subdirectory of the runtimes output for the Clang resource
  // directory runtimes.
  //
  // TODO: This should be replaced with an abstraction that manages the layout
  // of the generated runtimes rather than hardcoding it.
  std::filesystem::path resource_dir_path = output_path / "clang_resource_dir";

  auto build_result = runner.BuildTargetResourceDir(
      options_.codegen_options.target, resource_dir_path, tmp_dir.abs_path());
  if (!build_result.ok()) {
    driver_env.emitter.Emit(FailureBuildingRuntimes,
                            build_result.error().message());
  }

  return {.success = build_result.ok()};
}

}  // namespace Carbon
