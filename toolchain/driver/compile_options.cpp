// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_options.h"

#include <optional>

#include "toolchain/base/clang_invocation.h"

namespace Carbon {

namespace {

// Provides command-line options common to both `compile` and `link`
// subcommands.
auto BuildSharedOptions(CommandLine::CommandBuilder& b, CompileOptions* options)
    -> void {
  b.AddStringPositionalArg(
      {
          .name = "FILE",
          .help = R"""(
The input Carbon source file to compile.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Required(true);
        arg_b.Append(&options->input_filenames);
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
      [&](auto& arg_b) { arg_b.Append(&options->clang_args); });

  b.AddStringPositionalArg(
      {
          .name = "CLANG-ARG",
          .help = R"""(
Additional Clang arguments. See help for `--clang-arg` for details.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&options->clang_args); });

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
            &options->opt_level);
      });

  // Include the common code generation options at this point to render it
  // after the more common options above, but before the more unusual options
  // below.
  options->codegen_options->Build(b);

  b.AddFlag(
      {
          .name = "debug-info",
          .help = R"""(
Whether to emit DWARF debug information.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&options->include_debug_info);
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
        arg_b.Set(&options->run_llvm_verifier);
      });
  b.AddFlag(
      {
          .name = "prelude-import",
          .help = R"""(
Whether to use the implicit prelude import. Enabled by default.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&options->prelude_import);
      });
}

}  // namespace

auto CompileOptions::BuildForCompileSubcommand(CommandLine::CommandBuilder& b)
    -> void {
  BuildSharedOptions(b, this);

  b.AddOneOfOption(
      {
          .name = "phase",
          .help = R"""(
Selects the compilation phase to run. These phases are always run in sequence,
so every phase before the one selected will also be run. The default is to
compile to machine code.
)""",
      },
      [&](auto& arg_b) {
        arg_b.SetOneOf(
            {
                arg_b.OneOfValue("lex", Phase::Lex),
                arg_b.OneOfValue("parse", Phase::Parse),
                arg_b.OneOfValue("check", Phase::Check),
                arg_b.OneOfValue("lower", Phase::Lower),
                arg_b.OneOfValue("optimize", Phase::Optimize),
                arg_b.OneOfValue("codegen", Phase::CodeGen).Default(true),
            },
            &phase);
      });

  // TODO: Rearrange the code setting this option and two related ones to
  // allow them to reference each other instead of hard-coding their names.
  b.AddStringOption(
      {
          .name = "output",
          .value_name = "FILE",
          .help = R"""(
The output filename for codegen.

When this is a file name, either textual assembly or a binary object will be
written to it based on the flag `--asm-output`. The default is to write a binary
object file.

Passing `--output=-` will write the output to stdout. In that case, the flag
`--asm-output` is ignored and the output defaults to textual assembly. Binary
object output can be forced by enabling `--force-obj-output`.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_filename); });
  b.AddFlag(
      {
          .name = "asm-output",
          .help = R"""(
Write textual assembly rather than a binary object file to the code generation
output.

This flag only applies when writing to a file. When writing to stdout, the
default is textual assembly and this flag is ignored.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&asm_output); });

  b.AddFlag(
      {
          .name = "force-obj-output",
          .help = R"""(
Force binary object output, even with `--output=-`.

When `--output=-` is set, the default is textual assembly; this forces printing
of a binary object file instead. Ignored for other `--output` values.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&force_obj_output); });

  b.AddFlag(
      {
          .name = "stream-errors",
          .help = R"""(
Stream error messages to stderr as they are generated rather than sorting them
and displaying them in source order.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&stream_errors); });

  b.AddFlag(
      {
          .name = "dump-shared-values",
          .help = R"""(
Dumps shared values. These aren't owned by any particular file or phase.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_shared_values); });
  b.AddFlag(
      {
          .name = "dump-tokens",
          .help = R"""(
Dump the tokens to stdout when lexed.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_tokens); });

  b.AddFlag(
      {
          .name = "omit-file-boundary-tokens",
          .help = R"""(
For `--dump-tokens`, omit file start and end boundary tokens.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&omit_file_boundary_tokens); });

  b.AddFlag(
      {
          .name = "dump-parse-tree",
          .help = R"""(
Dump the parse tree to stdout when parsed.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_parse_tree); });
  b.AddFlag(
      {
          .name = "preorder-parse-tree",
          .help = R"""(
When dumping the parse tree, reorder it so that it is in preorder rather than
postorder.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&preorder_parse_tree); });
  b.AddFlag(
      {
          .name = "dump-raw-sem-ir",
          .help = R"""(
Dump the raw JSON structure of SemIR to stdout when built.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_raw_sem_ir); });
  b.AddFlag(
      {
          .name = "dump-sem-ir",
          .help = R"""(
Dump the full SemIR to stdout when built.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_sem_ir); });
  b.AddFlag(
      {
          .name = "dump-cpp-ast",
          .help = R"""(
Dump the full C++ AST to stdout when built.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_cpp_ast); });

  b.AddOneOfOption(
      {
          .name = "dump-sem-ir-ranges",
          .help = R"""(
Selects handling of `//@dump-sem-ir-[begin|end]` markers when dumping SemIR.
By default, `if-present` prints ranges for files that have them, and full SemIR
for files that don't. `only` skips files with no ranges, and `ignore` always
prints full SemIR.
)""",
      },
      [&](auto& arg_b) {
        using DumpSemIRRanges = Check::CheckParseTreesOptions::DumpSemIRRanges;
        arg_b.SetOneOf(
            {
                arg_b.OneOfValue("if-present", DumpSemIRRanges::IfPresent)
                    .Default(true),
                arg_b.OneOfValue("only", DumpSemIRRanges::Only),
                arg_b.OneOfValue("ignore", DumpSemIRRanges::Ignore),
            },
            &dump_sem_ir_ranges);
      });

  b.AddFlag(
      {
          .name = "builtin-sem-ir",
          .help = R"""(
Include the SemIR for builtins when dumping it.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&builtin_sem_ir); });
  b.AddFlag(
      {
          .name = "dump-llvm-ir",
          .help = R"""(
Dump the LLVM IR to stdout after lowering.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_llvm_ir); });
  b.AddFlag(
      {
          .name = "dump-asm",
          .help = R"""(
Dump the generated assembly to stdout after codegen.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_asm); });
  b.AddFlag(
      {
          .name = "dump-mem-usage",
          .help = R"""(
Dumps the amount of memory used.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_mem_usage); });
  b.AddFlag(
      {
          .name = "dump-timings",
          .help = R"""(
Dumps the duration of each phase for each compilation unit.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_timings); });
  b.AddFlag(
      {
          .name = "custom-core",
          .value_name = "CUSTOM_CORE",
          .help = R"""(
Whether to use a custom Core package, the files for which must all be included
in the compile command line.

The prelude library in the Core package is imported automatically. By default,
the Core package shipped with the toolchain is used, and its files do not need
to be specified in the compile command line.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(false);
        arg_b.Set(&custom_core);
      });
  b.AddStringOption(
      {
          .name = "exclude-dump-file-prefix",
          .value_name = "PREFIX",
          .help = R"""(
Excludes files with the given prefix from dumps.
)""",
      },
      [&](auto& arg_b) { arg_b.Append(&exclude_dump_file_prefixes); });
  b.AddFlag(
      {
          .name = "output-last-input-only",
          .help = R"""(
Only write output for the last input file, ignoring all others.

TODO: This is a temporary workaround and should be removed once separate
compilation is better implemented.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&output_last_input_only); });
  b.AddStringOption(
      {
          .name = "sem-ir-crash-dump",
          .value_name = "PATH",
          .help = R"""(
Where to write a dump of the raw SemIR emitted so far, in the event of a crash
in the check phase. If empty, the dump is not written.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&sem_ir_crash_dump); });
  b.AddFlag(
      {
          .name = "mangle-string-fingerprint",
          .help = R"""(
Use the string form of the fingerprint from mangling instead of the hash form.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&mangle_string_fingerprint); });
}

auto CompileOptions::BuildForBuildSubcommand(CommandLine::CommandBuilder& b)
    -> void {
  BuildSharedOptions(b, this);
}

auto CompileOptions::ValidatePhase(Diagnostics::NoLocEmitter& emitter) const
    -> bool {
  CARBON_DIAGNOSTIC(
      CompilePhaseFlagConflict, Error,
      "requested dumping {0} but compile phase is limited to `{1}`",
      std::string, std::string);
  using Phase = CompileOptions::Phase;
  switch (phase) {
    case Phase::Lex:
      if (dump_parse_tree) {
        emitter.Emit(CompilePhaseFlagConflict, "parse tree",
                     CompileOptions::PhaseToString(phase));
        return false;
      }
      [[fallthrough]];
    case Phase::Parse:
      if (dump_sem_ir) {
        emitter.Emit(CompilePhaseFlagConflict, "SemIR",
                     CompileOptions::PhaseToString(phase));
        return false;
      }
      if (dump_cpp_ast) {
        emitter.Emit(CompilePhaseFlagConflict, "C++ AST",
                     CompileOptions::PhaseToString(phase));
        return false;
      }
      [[fallthrough]];
    case Phase::Check:
      if (dump_llvm_ir) {
        emitter.Emit(CompilePhaseFlagConflict, "LLVM IR",
                     CompileOptions::PhaseToString(phase));
        return false;
      }
      [[fallthrough]];
    case Phase::Lower:
    case Phase::Optimize:
    case Phase::CodeGen:
      // Everything can be dumped in these phases.
      break;
  }
  return true;
}

auto CompileOptions::ValidateTarget(Diagnostics::NoLocEmitter& emitter)
    -> ErrorOr<const llvm::Target*> {
  std::string target_error;
  const llvm::Target* target = llvm::TargetRegistry::lookupTarget(
      llvm::Triple(codegen_options->target), target_error);
  if (!target) {
    CARBON_DIAGNOSTIC(CompileTargetInvalid, Error, "invalid target: {0}",
                      std::string);
    emitter.Emit(CompileTargetInvalid, target_error);
    return ErrorBuilder() << "Invalid LLVM target: " << target_error;
  }

  return target;
}

auto CompileOptions::BuildClangInvocation(DriverEnv& driver_env)
    -> ErrorOr<std::shared_ptr<clang::CompilerInvocation>> {
  // TODO: Move this into `BuildClangInvocation` when it can accept an
  // optimization level.
  llvm::SmallVector<llvm::StringRef> all_clang_args = {
      // Propagate our optimization level to Clang as a default. This can be
      // overridden by Clang arguments, but doing so will only have an effect
      // if those arguments affect Clang's IR, not its pass pipeline.
      CompileOptions::GetClangOptimizationFlag(opt_level),
  };
  all_clang_args.append(clang_args);
  auto clang_invocation = Carbon::BuildClangInvocation(
      *driver_env.consumer, driver_env.fs, *driver_env.installation,
      codegen_options->target, all_clang_args);
  if (!clang_invocation) {
    return ErrorBuilder() << "Failed to build a valid clang invocation.";
  }
  // We will run our own pass pipeline over the IR in the `Optimize` phase, so
  // disable Clang's pipeline to avoid optimizing C++ code twice.
  clang_invocation->getCodeGenOpts().DisableLLVMPasses = true;
  return std::shared_ptr<clang::CompilerInvocation>(clang_invocation.release());
}

// static
auto CompileOptions::GetLLVMOptimizationLevel(
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

// static
auto CompileOptions::GetClangOptimizationFlag(
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

// static
auto CompileOptions::PhaseToString(CompileOptions::Phase phase) -> std::string {
  switch (phase) {
    case CompileOptions::Phase::Lex:
      return "lex";
    case CompileOptions::Phase::Parse:
      return "parse";
    case CompileOptions::Phase::Check:
      return "check";
    case CompileOptions::Phase::Lower:
      return "lower";
    case CompileOptions::Phase::Optimize:
      return "optimize";
    case CompileOptions::Phase::CodeGen:
      return "codegen";
  }
}

}  // namespace Carbon
