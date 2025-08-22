// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/clang_subcommand.h"

#include <string>

#include "llvm/TargetParser/Host.h"
#include "toolchain/driver/clang_runner.h"

namespace Carbon {

auto ClangOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddFlag(
      {
          .name = "build-runtimes",
          .help = R"""(
Enables on-demand building of target-specific runtimes.

When enabled, any link actions using `clang` will build the necessary runtimes
on-demand. This build will use any customization it can from the link command
line flags to build the runtimes for the correct target and with any desired
features enabled.

Note: this only has an effect when `--prebuilt-runtimes` are not provided. If
there are no prebuilt runtimes and building runtimes is disabled, then it is
assumed the installed toolchain has had the necessary target runtimes added to
the installation tree in the default searched locations.
)""",
      },
      [&](auto& arg_b) {
        // TODO: Once runtimes are cached properly, the plan is to enable this
        // by default.
        arg_b.Default(false);
        arg_b.Set(&build_runtimes_on_demand);
      });
  b.AddStringPositionalArg(
      {
          .name = "ARG",
          .help = R"""(
Arguments passed to Clang.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&args); });
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "clang",
    .help = R"""(
Runs Clang on arguments.

This is equivalent to running the `clang` command line directly, and provides
the full command line interface.

Use `carbon clang -- ARGS` to pass flags to `clang`. Although there are
currently no flags for `carbon clang`, the `--` reserves the ability to add
flags in the future.

This is provided to help guarantee consistent compilation of C++ files, both
when Clang is invoked directly and when a Carbon file importing a C++ file
results in an indirect Clang invocation.
)""",
};

ClangSubcommand::ClangSubcommand() : DriverSubcommand(SubcommandInfo) {}

// TODO: This lacks a lot of features from the main driver code. We may need to
// add more.
// https://github.com/llvm/llvm-project/blob/main/clang/tools/driver/driver.cpp
auto ClangSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  ClangRunner runner(
      driver_env.installation, &driver_env.runtimes_cache, driver_env.fs,
      driver_env.vlog_stream,
      /*build_runtimes_on_demand=*/options_.build_runtimes_on_demand);

  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }

  // Only enable Clang's leaking of memory if the driver can support that.
  if (driver_env.enable_leaking) {
    runner.EnableLeakingMemory();
  }

  ErrorOr<bool> run_result = runner.Run(
      options_.args,
      driver_env.prebuilt_runtimes ? &*driver_env.prebuilt_runtimes : nullptr);
  if (!run_result.ok()) {
    // This is not a Clang failure, but a failure to even run Clang, so we need
    // to diagnose it here.
    CARBON_DIAGNOSTIC(FailureRunningClang, Error,
                      "failure running `clang` subcommand: {0}", std::string);
    driver_env.emitter.Emit(FailureRunningClang, run_result.error().message());
    return {.success = false};
  }

  // Successfully ran Clang, but return whether Clang itself succeeded.
  return {.success = *run_result};
}

}  // namespace Carbon
