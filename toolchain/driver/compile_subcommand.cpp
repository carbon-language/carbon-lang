// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_subcommand.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "common/error.h"
#include "common/ostream.h"
#include "common/pretty_stack_trace_function.h"
#include "common/vlog.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "toolchain/base/clang_invocation.h"
#include "toolchain/base/timings.h"
#include "toolchain/check/check.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/emitter.h"
#include "toolchain/diagnostics/format_providers.h"
#include "toolchain/diagnostics/sorting_consumer.h"
#include "toolchain/driver/compile_driver.h"
#include "toolchain/lex/lex.h"
#include "toolchain/lower/lower.h"
#include "toolchain/lower/options.h"
#include "toolchain/parse/parse.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/source/source_buffer.h"

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
  if (driver_env.fuzzing && !options_.clang_args.empty()) {
    // Parsing specific Clang arguments can reach deep into
    // external libraries that aren't fuzz clean.
    TestAndDiagnoseIfFuzzingExternalLibraries(driver_env, "compile");
    return {.success = false};
  }

  auto compile_driver = CompileDriver(&options_);

  if (!compile_driver.Initialize(driver_env)) {
    return {.success = false};
  }

  return compile_driver.Compile(driver_env);
}

}  // namespace Carbon
