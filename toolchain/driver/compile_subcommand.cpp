// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_subcommand.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Path.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/driver/compile_driver.h"

namespace Carbon {

static constexpr CommandLine::CommandInfo SubcommandInfo = {
    .name = "compile",
    .help = R"""(
Compile Carbon source code.

This subcommand runs the Carbon compiler over input source code, checking it for
errors and producing the requested output.

Error messages are written to the standard error stream.

Different phases of the compiler can be selected to run, and intermediate state
can be written to standard output as these phases progress.
)""",
};

CompileSubcommand::CompileSubcommand() : DriverSubcommand(SubcommandInfo) {}

auto CompileSubcommand::Run(DriverEnv& driver_env) -> DriverResult {
  options_.compile_options.FixupFlags();

  if (driver_env.fuzzing && !options_.compile_options.clang_args.empty()) {
    // Parsing specific Clang arguments can reach deep into
    // external libraries that aren't fuzz clean.
    TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "compile");
    return {.success = false};
  }

  // If we're lowering and have an output file name, we can only support a
  // single input filename.
  // TODO: Produce an error in this case rather than skipping all files but the
  // last, and remove the `--output-last-input-only` flag.
  if (options_.compile_options.phase >= CompileOptions::Phase::Lower &&
      options_.compile_options.input_filenames.size() > 1 &&
      !options_.output_last_input_only &&
      !options_.compile_options.output_filename.empty() &&
      options_.compile_options.output_filename != "-") {
    CARBON_DIAGNOSTIC(
        CompileMultipleInputsWithOutput, Warning,
        "only outputting {0} to {1}, skipping output of {2} input "
        "file{2:s}; pass `--output-last-input-only` to silence this "
        "warning",
        std::string, std::string, Diagnostics::IntAsSelect);
    driver_env.emitter.Emit(
        CompileMultipleInputsWithOutput,
        options_.compile_options.input_filenames.back().str(),
        options_.compile_options.output_filename.str(),
        options_.compile_options.input_filenames.size() - 1);
  }

  llvm::StringSet<> input_filenames(llvm::from_range,
                                    options_.compile_options.input_filenames);

  auto compile_driver = CompileDriver(&options_.compile_options);

  bool init_success = true;
  auto get_output_filename =
      [&](llvm::StringRef input_filename) -> std::string {
    // We only generate output for inputs specified on the command line,
    // not for inputs discovered through imports.
    if (!input_filenames.contains(input_filename)) {
      return "";
    }

    // If the output filename is "-", that's used for all inputs.
    if (options_.compile_options.output_filename == "-") {
      return "-";
    }

    // If single output filename was specified, it's used for the final
    // input filename only.
    if (!options_.compile_options.output_filename.empty()) {
      if (input_filename == options_.compile_options.input_filenames.back()) {
        return options_.compile_options.output_filename.str();
      }
      return "";
    }

    // Otherwise, generate an output filename for each explicitly-specified
    // input file.
    bool is_regular_file = true;
    if (input_filename == "-") {
      // TODO: If we would produce textual output, using "-" as the default
      // output filename here would be reasonable and useful.
      is_regular_file = false;
    } else if (auto status = driver_env.fs->status(input_filename);
               status && status->isOther()) {
      is_regular_file = false;
    }
    if (!is_regular_file) {
      CARBON_DIAGNOSTIC(CompileInputNotRegularFile, Error,
                        "output file name must be specified for input "
                        "`{0}` that is not a regular file",
                        std::string);
      driver_env.emitter.Emit(CompileInputNotRegularFile, input_filename.str());
      init_success = false;
      return "";
    }
    llvm::SmallString<256> output_filename = input_filename;
    llvm::sys::path::replace_extension(
        output_filename, options_.compile_options.asm_output ? ".s" : ".o");
    return output_filename.str().str();
  };

  if (!compile_driver.Initialize(driver_env, get_output_filename) ||
      !init_success) {
    return {.success = false};
  }

  return compile_driver.Compile(driver_env);
}

}  // namespace Carbon
