// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/build_subcommand.h"

#include <filesystem>

#include "common/filesystem.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/driver/link_driver.h"

namespace Carbon {

auto BuildSubcommandOptions::Build(CommandLine::CommandBuilder& b) -> void {
  compile_options.BuildForBuildSubcommand(b);
  link_options.BuildForBuildSubcommand(b);

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

  // Compute the needed LinkOptions for the LinkDriver from the output of the
  // compilation process.
  options_.link_options.codegen_options =
      options_.compile_options.codegen_options;

  llvm::SmallString<256> output_filename;
  if (options_.link_options.output_filename.empty()) {
    output_filename = llvm::sys::path::filename(
        compile_driver.units()[compile_driver.first_input_index()]
            ->input_filename());
    llvm::sys::path::replace_extension(output_filename, "");
    options_.link_options.output_filename = output_filename;
  }

  auto input_builder = [&](const std::unique_ptr<CompilationUnit>& unit) {
    return unit->output_filename();
  };
  append_range(options_.link_options.object_filenames,
               llvm::map_range(compile_driver.units(), input_builder));

  auto link_driver = LinkDriver(&options_.link_options);
  return link_driver.Link(driver_env);
}

}  // namespace Carbon
