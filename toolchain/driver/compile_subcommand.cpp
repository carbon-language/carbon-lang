// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/compile_subcommand.h"

namespace Carbon {

auto operator<<(llvm::raw_ostream& out, CompileOptions::Phase phase)
    -> llvm::raw_ostream& {
  switch (phase) {
    case CompileOptions::Phase::Lex:
      out << "lex";
      break;
    case CompileOptions::Phase::Parse:
      out << "parse";
      break;
    case CompileOptions::Phase::Check:
      out << "check";
      break;
    case CompileOptions::Phase::Lower:
      out << "lower";
      break;
    case CompileOptions::Phase::CodeGen:
      out << "codegen";
      break;
  }
  return out;
}

constexpr CommandLine::CommandInfo CompileOptions::Info = {
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

auto CompileOptions::Build(CommandLine::CommandBuilder& b,
                           CodegenOptions& codegen_options) -> void {
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

  // Include the common code generation options at this point to render it
  // after the more common options above, but before the more unusual options
  // below.
  codegen_options.Build(b);

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
Dump the SemIR to stdout when built.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&dump_sem_ir); });
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
          .name = "prelude-import",
          .help = R"""(
Whether to use the implicit prelude import. Enabled by default.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&prelude_import);
      });
  b.AddStringOption(
      {
          .name = "exclude-dump-file-prefix",
          .value_name = "PREFIX",
          .help = R"""(
Excludes files with the given prefix from dumps.
)""",
      },
      [&](auto& arg_b) { arg_b.Set(&exclude_dump_file_prefix); });
  b.AddFlag(
      {
          .name = "debug-info",
          .help = R"""(
Emit DWARF debug information.
)""",
      },
      [&](auto& arg_b) {
        arg_b.Default(true);
        arg_b.Set(&include_debug_info);
      });
}

}  // namespace Carbon
