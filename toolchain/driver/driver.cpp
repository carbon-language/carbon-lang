// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include "common/command_line.h"
#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include "toolchain/codegen/codegen.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/lowering/lower_to_llvm.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/semantics/semantics_ir_formatter.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

struct Driver::CompileOptions {
  static constexpr CommandLine::CommandInfo Info = {
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

  enum class Phase {
    Lex,
    Parse,
    Check,
    Lower,
    CodeGen,
  };

  friend auto operator<<(llvm::raw_ostream& out, Phase phase)
      -> llvm::raw_ostream& {
    switch (phase) {
      case Phase::Lex:
        out << "lex";
        break;
      case Phase::Parse:
        out << "parse";
        break;
      case Phase::Check:
        out << "check";
        break;
      case Phase::Lower:
        out << "lower";
        break;
      case Phase::CodeGen:
        out << "codegen";
        break;
    }
    return out;
  }

  void Build(CommandLine::CommandBuilder& b) {
    b.AddStringPositionalArg(
        {
            .name = "FILE",
            .help = R"""(
The input Carbon source file to compile.
)""",
        },
        [&](auto& arg_b) {
          arg_b.Required(true);
          arg_b.Set(&input_file_name);
        });

    b.AddOneOfOption(
        {
            .name = "phase",
            .help = R"""(
Selects the compilation phase to run. These phases are always run in sequence,
so every phase before the one selected will also be run. The default is to
compile, lower, and generate machine code.
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

    b.AddStringOption(
        {
            .name = "output",
            .value_name = "FILE",
            .help = R"""(
The output filename for codegen.

When this is a file name, either textual assembly or a binary object will be
written to it based on the flag `--asm-output`. The default is to write a binary
object file.

Passing `--output=-` will write the output to stdout. In that
case, the flag `--asm-output` is ignored and the output defaults to textual
assembly. Binary object output can be forced by enabling `--force-obj-output`.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&output_file_name); });

    b.AddStringOption(
        {
            .name = "target",
            .help = R"""(
Select a target platform. Uses the LLVM target syntax, often called a "triple"
(despite not actually being a triple in many common cases).

This corresponds to the `target` flag to Clang and accepts the same strings
documented there:
https://clang.llvm.org/docs/CrossCompilation.html#target-triple
)""",
        },
        [&](auto& arg_b) {
          arg_b.Default(host);
          arg_b.Set(&target);
        });

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
Force writing a binary object file, even when writing to stdout.

This flag is only used when the code generation output file is set to stdout. In
that case, it will override the forced default of textual assembly and output a
binary output.
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
            .name = "dump-raw-semantics-ir",
            .help = R"""(
Dump the raw JSON structure of semantics IR to stdout when built.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_raw_semantics_ir); });
    b.AddFlag(
        {
            .name = "dump-semantics-ir",
            .help = R"""(
Dump the semantics IR to stdout when built.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&dump_semantics_ir); });
    b.AddFlag(
        {
            .name = "builtin-semantics-ir",
            .help = R"""(
Include the semantics IR for builtins when dumping it.
)""",
        },
        [&](auto& arg_b) { arg_b.Set(&builtin_semantics_ir); });
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
  }

  Phase phase;

  std::string host = llvm::sys::getDefaultTargetTriple();
  llvm::StringRef target;

  llvm::StringRef output_file_name;
  llvm::StringRef input_file_name;

  bool asm_output = false;
  bool force_obj_output = false;
  bool dump_tokens = false;
  bool dump_parse_tree = false;
  bool dump_raw_semantics_ir = false;
  bool dump_semantics_ir = false;
  bool dump_llvm_ir = false;
  bool dump_asm = false;
  bool stream_errors = false;
  bool preorder_parse_tree = false;
  bool builtin_semantics_ir = false;
};

struct Driver::Options {
  static constexpr CommandLine::CommandInfo Info = {
      .name = "carbon",
      // TODO: Setup more detailed version information and use that here.
      .version = R"""(
Carbon Language toolchain -- version 0.0.0
)""",
      .help = R"""(
This is the unified Carbon Language toolchain driver. It's subcommands provide
all of the core behavior of the toolchain, including compilation, linking, and
developer tools. Each of these has its own subcommand, and you can pass a
specific subcommand to the `help` subcommand to get details about is usage.
)""",
      .help_epilogue = R"""(
For questions, issues, or bug reports, please use our GitHub project:

  https://github.com/carbon-language/carbon-lang
)""",
  };

  enum class Subcommand {
    Compile,
  };

  void Build(CommandLine::CommandBuilder& b) {
    b.AddFlag(
        {
            .name = "verbose",
            .short_name = "v",
            .help = "Enable verbose logging to the stderr stream.",
        },
        [&](auto& arg_b) { arg_b.Set(&verbose); });

    b.AddSubcommand(CompileOptions::Info, [&](auto& sub_b) {
      compile_options.Build(sub_b);
      sub_b.Do([&] { subcommand = Subcommand::Compile; });
    });

    b.RequiresSubcommand();
  }

  bool verbose;
  Subcommand subcommand;

  CompileOptions compile_options;
};

auto Driver::ParseArgs(llvm::ArrayRef<llvm::StringRef> args, Options& options)
    -> CommandLine::ParseResult {
  return CommandLine::Parse(args, output_stream_, error_stream_, Options::Info,
                            [&](auto& b) { options.Build(b); });
}

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  Options options;
  CommandLine::ParseResult result = ParseArgs(args, options);
  if (result == CommandLine::ParseResult::Error) {
    return false;
  } else if (result == CommandLine::ParseResult::MetaSuccess) {
    return true;
  }

  if (options.verbose) {
    // Note this implies streamed output in order to interleave.
    vlog_stream_ = &error_stream_;
  }

  switch (options.subcommand) {
    case Options::Subcommand::Compile:
      return Compile(options.compile_options);
  }
  llvm_unreachable("All subcommands handled!");
}

auto Driver::Compile(const CompileOptions& options) -> bool {
  StreamDiagnosticConsumer stream_consumer(error_stream_);
  DiagnosticConsumer* consumer = &stream_consumer;
  std::unique_ptr<SortingDiagnosticConsumer> sorting_consumer;
  if (vlog_stream_ == nullptr && !options.stream_errors) {
    sorting_consumer = std::make_unique<SortingDiagnosticConsumer>(*consumer);
    consumer = sorting_consumer.get();
  }

  CARBON_VLOG() << "*** SourceBuffer::CreateFromFile on '"
                << options.input_file_name << "' ***\n";
  auto source = SourceBuffer::CreateFromFile(fs_, options.input_file_name);
  CARBON_VLOG() << "*** SourceBuffer::CreateFromFile done ***\n";
  // Require flushing the consumer before the source buffer is destroyed,
  // because diagnostics may reference the buffer.
  auto flush = llvm::make_scope_exit([&]() { consumer->Flush(); });
  if (!source.ok()) {
    error_stream_ << "ERROR: Unable to open input source file: "
                  << source.error();
    return false;
  }
  CARBON_VLOG() << "*** file:\n```\n" << source->text() << "\n```\n";

  bool has_errors = false;

  using Phase = CompileOptions::Phase;
  switch (options.phase) {
    case Phase::Lex:
      if (options.dump_parse_tree) {
        error_stream_ << "ERROR: Requested dumping the parse tree but compile "
                         "phase is limited to '"
                      << options.phase << "'\n";
        has_errors = true;
      }
      [[clang::fallthrough]];
    case Phase::Parse:
      if (options.dump_semantics_ir) {
        error_stream_ << "ERROR: Requested dumping the semantics IR but "
                         "compile phase is limited to '"
                      << options.phase << "'\n";
        has_errors = true;
      }
      [[clang::fallthrough]];
    case Phase::Check:
      if (options.dump_llvm_ir) {
        error_stream_ << "ERROR: Requested dumping the LLVM IR but compile "
                         "phase is limited to '"
                      << options.phase << "'\n";
        has_errors = true;
      }
      [[clang::fallthrough]];
    case Phase::Lower:
    case Phase::CodeGen:
      // Everything can be dumped in these phases.
      break;
  }

  CARBON_VLOG() << "*** TokenizedBuffer::Lex ***\n";
  auto tokenized_source = TokenizedBuffer::Lex(*source, *consumer);
  has_errors |= tokenized_source.has_errors();
  CARBON_VLOG() << "*** TokenizedBuffer::Lex done ***\n";
  if (options.dump_tokens) {
    CARBON_VLOG() << "Finishing output.";
    consumer->Flush();
    output_stream_ << tokenized_source;
  }
  CARBON_VLOG() << "tokenized_buffer: " << tokenized_source;
  if (options.phase == Phase::Lex) {
    return !has_errors;
  }

  CARBON_VLOG() << "*** ParseTree::Parse ***\n";
  auto parse_tree = ParseTree::Parse(tokenized_source, *consumer, vlog_stream_);
  has_errors |= parse_tree.has_errors();
  CARBON_VLOG() << "*** ParseTree::Parse done ***\n";
  if (options.dump_parse_tree) {
    consumer->Flush();
    parse_tree.Print(output_stream_, options.preorder_parse_tree);
  }
  CARBON_VLOG() << "parse_tree: " << parse_tree;
  if (options.phase == Phase::Parse) {
    return !has_errors;
  }

  const SemanticsIR builtin_ir = SemanticsIR::MakeBuiltinIR();
  CARBON_VLOG() << "*** SemanticsIR::MakeFromParseTree ***\n";
  const SemanticsIR semantics_ir = SemanticsIR::MakeFromParseTree(
      builtin_ir, tokenized_source, parse_tree, *consumer, vlog_stream_);
  has_errors |= semantics_ir.has_errors();
  CARBON_VLOG() << "*** SemanticsIR::MakeFromParseTree done ***\n";
  if (options.dump_raw_semantics_ir) {
    consumer->Flush();
    semantics_ir.Print(output_stream_, options.builtin_semantics_ir);
    if (options.dump_semantics_ir) {
      output_stream_ << "\n";
    }
  }
  if (options.dump_semantics_ir) {
    FormatSemanticsIR(tokenized_source, parse_tree, semantics_ir,
                      output_stream_);
  }
  CARBON_VLOG() << "semantics_ir: " << semantics_ir;
  if (options.phase == Phase::Check) {
    return !has_errors;
  }

  // Unlike previous steps, errors block further progress.
  if (has_errors) {
    CARBON_VLOG() << "*** Stopping before lowering due to syntax errors ***";
    return false;
  }
  consumer->Flush();

  CARBON_VLOG() << "*** LowerToLLVM ***\n";
  llvm::LLVMContext llvm_context;
  const std::unique_ptr<llvm::Module> module = LowerToLLVM(
      llvm_context, options.input_file_name, semantics_ir, vlog_stream_);
  CARBON_VLOG() << "*** LowerToLLVM done ***\n";
  if (options.dump_llvm_ir) {
    module->print(output_stream_, /*AAW=*/nullptr,
                  /*ShouldPreserveUseListOrder=*/true);
  }
  if (vlog_stream_) {
    CARBON_VLOG() << "module: ";
    module->print(*vlog_stream_, /*AAW=*/nullptr,
                  /*ShouldPreserveUseListOrder=*/false,
                  /*IsForDebug=*/true);
  }
  if (options.phase == Phase::Lower) {
    return true;
  }

  CARBON_VLOG() << "*** CodeGen ***\n";
  std::optional<CodeGen> codegen =
      CodeGen::Create(*module, options.target, error_stream_);
  if (!codegen) {
    return false;
  }
  if (vlog_stream_) {
    CARBON_VLOG() << "assembly:\n";
    codegen->EmitAssembly(*vlog_stream_);
  }

  if (options.output_file_name == "-") {
    if (options.force_obj_output) {
      if (!codegen->EmitObject(output_stream_)) {
        return false;
      }
    } else {
      if (!codegen->EmitAssembly(output_stream_)) {
        return false;
      }
    }
  } else {
    llvm::SmallString<256> output_file_name = options.output_file_name;
    if (output_file_name.empty()) {
      output_file_name = options.input_file_name;
      llvm::sys::path::replace_extension(output_file_name,
                                         options.asm_output ? ".s" : ".o");
    }
    CARBON_VLOG() << "Writing output to: " << output_file_name << "\n";

    std::error_code ec;
    llvm::raw_fd_ostream output_file(output_file_name, ec,
                                     llvm::sys::fs::OF_None);
    if (ec) {
      error_stream_ << "ERROR: Could not open output file '" << output_file_name
                    << "': " << ec.message() << "\n";
      return false;
    }
    if (options.asm_output) {
      if (!codegen->EmitAssembly(output_file)) {
        return false;
      }
    } else {
      if (!codegen->EmitObject(output_file)) {
        return false;
      }
    }
  }
  CARBON_VLOG() << "*** CodeGen done ***\n";
  return true;
}

}  // namespace Carbon
