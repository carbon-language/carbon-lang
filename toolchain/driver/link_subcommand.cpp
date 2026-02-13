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
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Append(&object_filenames);
      });

  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The linked file name. The output is always a linked binary.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Set(&output_filename);
      });

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

  clang_args.push_back("-o");
  clang_args.push_back(options_.output_filename);
  clang_args.append(options_.object_filenames.begin(),
                    options_.object_filenames.end());

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
