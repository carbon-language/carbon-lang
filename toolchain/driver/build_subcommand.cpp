// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/build_subcommand.h"

#include <filesystem>

#include "common/command_line.h"
#include "common/filesystem.h"
#include "common/hashing.h"
#include "common/pretty_stack_trace_function.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Target/TargetMachine.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/sorting_consumer.h"
#include "toolchain/driver/clang_runner.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/driver/driver_subcommand.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

auto BuildSubcommandOptions::Build(CommandLine::CommandBuilder& b) -> void {
  compile_options.BuildForBuildSubcommand(b);
  b.AddStringOption(
      {
          .name = "output",
          .short_name = "o",
          .value_name = "FILE",
          .help = R"""(
The file name for the output binary. If none is specified `build` will use the
name of the first provided input file.
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
      [&](auto& arg_b) { arg_b.Append(&extra_clang_link_args); });
  b.AddFlag(
      {
          .name = "use-temp-dir",
          .help = R"""(
Use a temporary directory for intermediate compilation artifacts.

When enabled (the default), carbon will compile all input files and necessary
dependencies into a temporary directory, before linking them into the final
output binary. If false, carbon will store the compilation artifacts as hashes
of the compiled input name in the current working directory.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&use_temp_dir);
      });
}

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "build",
    .help = R"""(
Compile and then link Carbon and C++ source code into a single executable.
)""",
};

BuildSubcommand::BuildSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto BuildSubcommand::BuildOptions(CommandLine::CommandBuilder& b) -> void {
  options_.Build(b);
}

auto BuildSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  if (driver_env.fuzzing && !options_.compile_options.clang_args.empty()) {
    // Parsing specific Clang arguments can reach deep into
    // external libraries that aren't fuzz clean.
    TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "build");
    return {.success = false};
  }

  std::optional<Filesystem::RemovingDir> temp_dir = std::nullopt;
  auto temp_dir_path = std::filesystem::path("");
  if (options_.use_temp_dir) {
    if (auto d = Filesystem::MakeTmpDir(); !d.ok()) {
      CARBON_DIAGNOSTIC(BuildTempDirectoryCreationError, Error, "{0}",
                        std::string);
      driver_env.emitter.Emit(BuildTempDirectoryCreationError,
                              PrintToString(d.error()));
      return {.success = false};
    } else {
      temp_dir = std::move(*d);
      temp_dir_path = temp_dir->path();
    }
  }

  auto on_exit = llvm::scope_exit([&]() {
    // Clean up the temporary directory created for compile results.
    if (temp_dir) {
      auto remove_result = std::move(*temp_dir).Remove();
      if (!remove_result.ok()) {
        CARBON_DIAGNOSTIC(BuildTempDirectoryDeletionError, Error, "{0}",
                          std::string);
        driver_env.emitter.Emit(BuildTempDirectoryDeletionError,
                                PrintToString(remove_result.error()));
      }
    }
  });

  auto compile_driver = CompileDriver(&options_.compile_options);
  if (!compile_driver.Initialize(
          driver_env, [&](llvm::StringRef input_filename) -> std::string {
            return (temp_dir_path /
                    llvm::formatv("{0:x16}.o", HashValue(input_filename)).str())
                .string();
          })) {
    return {.success = false};
  }

  auto compile_result = compile_driver.Compile(driver_env);
  if (!compile_result.success) {
    return compile_result;
  }

  // We've successfully compiled the inputs files, time to link them.
  llvm::SmallVector<llvm::StringRef> clang_link_args;

  // We link using a C++ mode of the driver.
  clang_link_args.push_back("--driver-mode=g++");

  // Pass the target down to Clang to pick up the correct defaults.
  std::string target_arg =
      llvm::formatv("--target={0}",
                    options_.compile_options.codegen_options.target)
          .str();
  clang_link_args.push_back(target_arg);

  llvm::SmallString<256> output_filename;
  if (!options_.output_filename.empty()) {
    clang_link_args.push_back("-o");
    clang_link_args.push_back(options_.output_filename);
  } else {
    output_filename = llvm::sys::path::filename(
        compile_driver.units()[compile_driver.first_input_index()]
            ->input_filename());
    llvm::sys::path::replace_extension(output_filename, "");
    clang_link_args.push_back("-o");
    clang_link_args.push_back(output_filename);
  }

  // Note that we append any extra Clang args before our object filenames. This
  // allows us to propagate object filenames that collide with Clang flags using
  // `--` before the filenames. While in theory, this could create a problem in
  // the presence of mixtures of object files in the two lists and the order
  // being dependent, we don't expect that in practice.
  clang_link_args.append(options_.extra_clang_link_args.begin(),
                         options_.extra_clang_link_args.end());
  clang_link_args.push_back("--");
  auto input_builder = [&](const std::unique_ptr<CompilationUnit>& unit) {
    return unit->output_filename();
  };
  append_range(clang_link_args,
               llvm::map_range(compile_driver.units(), input_builder));

  CARBON_VLOG_TO(driver_env.vlog_stream,
                 "*** Build Clang link call with these arguments:\n");
  for (auto a : clang_link_args) {
    CARBON_VLOG_TO(driver_env.vlog_stream, "    '{0}',\n", a);
  }

  ClangRunner runner(driver_env.installation, driver_env.fs,
                     driver_env.vlog_stream);
  // Don't run Clang when fuzzing, it is known to not be reliable under fuzzing
  // due to many unfixed issues.
  if (TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "clang")) {
    return {.success = false};
  }

  // Question: We're including some runtime stuff during compilation, is this
  // redundant?
  ErrorOr<bool> run_result =
      driver_env.prebuilt_runtimes
          ? runner.RunWithPrebuiltRuntimes(clang_link_args,
                                           *driver_env.prebuilt_runtimes,
                                           driver_env.enable_leaking)
      : driver_env.build_runtimes_on_demand
          ? runner.Run(clang_link_args, driver_env.runtimes_cache,
                       *driver_env.thread_pool, driver_env.enable_leaking)
          : runner.RunWithNoRuntimes(clang_link_args,
                                     driver_env.enable_leaking);

  if (!run_result.ok()) {
    // This is not a Clang failure, but a failure to even run Clang, so we need
    // to diagnose it here.
    CARBON_DIAGNOSTIC(BuildFailureRunningClangToLink, Error,
                      "failure running `clang` to perform linking: {0}",
                      std::string);
    driver_env.emitter.Emit(BuildFailureRunningClangToLink,
                            run_result.error().message());
  }

  // Successfully ran Clang to perform the link, return its result.
  return {.success = *run_result};
}

}  // namespace Carbon
