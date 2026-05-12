// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/shared_compile_options.h"

#include <optional>

#include "toolchain/base/clang_invocation.h"

namespace Carbon {

auto SharedCompileOptions::Build(CommandLine::CommandBuilder& b) -> void {
  b.AddStringPositionalArg(
      {
          .name = "FILE",
          .help = R"""(
The input Carbon source file to compile.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Append(&input_filenames);
      });

  b.AddStringOption(
      {
          .name = "clang-arg",
          .value_name = "CLANG-ARG",
          .help = R"""(
An argument to pass to the Clang compiler for use when compiling imported C++
code.

All flags that are accepted by the Clang driver are supported. However, you
cannot specify arguments that would result in additional compilations being
performed. Use `carbon clang` instead to compile additional source files.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&clang_args); });

  b.AddStringPositionalArg(
      {
          .name = "CLANG-ARG",
          .help = R"""(
Additional Clang arguments. See help for `--clang-arg` for details.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&clang_args); });

  b.AddOneOfOption(
      {
          .name = "optimize",
          .help = R"""(
Selects the amount of optimization to perform.
)""",
      },
      [&](auto& arg_b) {
        arg_b.SetOneOf(
            {
                // We intentionally don't expose O2 and Os. The difference
                // between these levels tends to reflect what achieves the
                // best speed for a specific application, as they all
                // largely optimize for speed as the primary factor.
                //
                // Instead of controlling this with more nuanced flags, we
                // plan to support profile and in-source hints to the
                // optimizer to adjust its strategy in the specific places
                // where the default doesn't have the desired results.
                arg_b.OneOfValue("none", Lower::OptimizationLevel::None),
                arg_b.OneOfValue("debug", Lower::OptimizationLevel::Debug),
                arg_b.OneOfValue("speed", Lower::OptimizationLevel::Speed),
                arg_b.OneOfValue("size", Lower::OptimizationLevel::Size),
            },
            &opt_level);
      });

  // Include the common code generation options at this point to render it
  // after the more common options above, but before the more unusual options
  // below.
  codegen_options.Build(b);

  b.AddFlag(
      {
          .name = "debug-info",
          .help = R"""(
Whether to emit DWARF debug information.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&include_debug_info);
      });
  b.AddFlag(
      {
          .name = "verify-llvm-ir",
          .help = R"""(
Whether to run the LLVM verifier on modules.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&run_llvm_verifier);
      });
}

auto SharedCompileOptions::ValidateTarget(Diagnostics::NoLocEmitter& emitter)
    -> ErrorOr<const llvm::Target*> {
  std::string target_error;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(codegen_options.target), target_error);
  if (!target) {
    CARBON_DIAGNOSTIC(CompileTargetInvalid, Error, "invalid target: {0}",
                      std::string);
    emitter.Emit(CompileTargetInvalid, target_error);
    return ErrorBuilder() << "Invalid LLVM target: " << target_error;
  }

  return target;
}

auto SharedCompileOptions::BuildClangInvocation(DriverEnv& driver_env)
    -> ErrorOr<std::shared_ptr<clang::CompilerInvocation>> {
  // TODO: Move this into `BuildClangInvocation` when it can accept an
  // optimization level.
  llvm::SmallVector<llvm::StringRef> all_clang_args = {
      // Propagate our optimization level to Clang as a default. This can be
      // overridden by Clang arguments, but doing so will only have an effect
      // if those arguments affect Clang's IR, not its pass pipeline.
      SharedCompileOptions::GetClangOptimizationFlag(opt_level),
  };
  all_clang_args.append(clang_args);
  auto clang_invocation = Carbon::BuildClangInvocation(
      driver_env.consumer, driver_env.fs, *driver_env.installation,
      codegen_options.target, all_clang_args);
  if (!clang_invocation) {
    return ErrorBuilder() << "Failed to build a valid clang invocation.";
  }
  // We will run our own pass pipeline over the IR in the `Optimize` phase, so
  // disable Clang's pipeline to avoid optimizing C++ code twice.
  clang_invocation->getCodeGenOpts().DisableLLVMPasses = true;
  return std::shared_ptr<clang::CompilerInvocation>(clang_invocation.release());
}

// Get the LLVM optimization level corresponding to a Carbon optimization level.
// static
auto SharedCompileOptions::GetLLVMOptimizationLevel(
    Lower::OptimizationLevel opt_level) -> llvm::OptimizationLevel {
  switch (opt_level) {
    case Lower::OptimizationLevel::None:
      return llvm::OptimizationLevel::O0;
    case Lower::OptimizationLevel::Debug:
      return llvm::OptimizationLevel::O1;
    case Lower::OptimizationLevel::Size:
      return llvm::OptimizationLevel::O2;
    case Lower::OptimizationLevel::Speed:
      return llvm::OptimizationLevel::O3;
  }
}

// Get the `-O` flag corresponding to an optimization level.
// static
auto SharedCompileOptions::GetClangOptimizationFlag(
    Lower::OptimizationLevel opt_level) -> llvm::StringLiteral {
  switch (opt_level) {
    case Lower::OptimizationLevel::None:
      return "-O0";
    case Lower::OptimizationLevel::Debug:
      return "-O1";
    case Lower::OptimizationLevel::Size:
      return "-O2";
    case Lower::OptimizationLevel::Speed:
      return "-O3";
  }
}

}  // namespace Carbon
