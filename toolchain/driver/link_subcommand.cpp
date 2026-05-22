// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/link_subcommand.h"

#include "llvm/TargetParser/Triple.h"
#include "toolchain/driver/clang_runner.h"

namespace Carbon {

auto LinkOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringPositionalArg(
      {
          .name = "OBJECT_FILE",
          .help = R"""(
The input object files.

If empty, there must be extra Clang link arguments that provide object files
intermingled with linking flags.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&object_filenames); });

  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The linked file name. The output is always a linked binary.

If not provided, there must be extra Clang link arguments that include
specifying the output of the link. This allows supporting build systems that
intermingle the output flag with arbitrary other linker flags that need to use
legacy parsing logic.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });

  b.AddStringPositionalArg(
      {
          .name = "EXTRA_CLANG_LINK_ARGS",
          .help = R"""(
Extra arguments to pass to Clang when forming the link command. This is
primarily useful for expanding `LDFLAGS` or other baseline linking flags in a
build system.

These can also be used to pass object files to the link in the event your build
system mixes object files and linker flags.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&extra_clang_args); });

  codegen_options.Build(b);
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "link",
    .help = R"""(
Link Carbon executables.

This subcommand links Carbon executables by combining object files.

TODO: Support linking binary libraries, both archives and shared libraries.
TODO: Support linking against binary libraries.
)""",
};

LinkSubcommand::LinkSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto LinkSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
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
      llvm::formatv("--target={0}", options_.codegen_options.target).str();
  clang_args.push_back(target_arg);

  if (!options_.output_filename.empty()) {
    clang_args.push_back("-o");
    clang_args.push_back(options_.output_filename);
  } else if (options_.extra_clang_args.empty()) {
    CARBON_DIAGNOSTIC(LinkOutputOptionMissing, Error,
                      "no output specified to a link command and no extra "
                      "Clang options that can provide an output");
    driver_env.emitter.Emit(LinkOutputOptionMissing);
    return {.success = false};
  }

  if (options_.object_filenames.empty() && options_.extra_clang_args.empty()) {
    CARBON_DIAGNOSTIC(LinkObjectFilesMissing, Error,
                      "no object files provided to link command and no extra "
                      "Clang options that could provide them");
    driver_env.emitter.Emit(LinkObjectFilesMissing);
    return {.success = false};
  }

  // Note that we append any extra Clang args before our object filenames. This
  // allows us to propagate object filenames that collide with Clang flags using
  // `--` before the filenames. While in theory, this could create a problem in
  // the presence of mixtures of object files in the two lists and the order
  // being dependent, we don't expect that in practice.
  clang_args.append(options_.extra_clang_args.begin(),
                    options_.extra_clang_args.end());
  clang_args.push_back("--");
  clang_args.append(options_.object_filenames.begin(),
                    options_.object_filenames.end());

  ClangRunner runner(driver_env.installation, driver_env.fs,
                     driver_env.vlog_stream);
  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }
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
