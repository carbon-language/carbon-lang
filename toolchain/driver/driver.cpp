// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include "common/vlog.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/lowering/lower_to_llvm.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

constexpr Args::Flag VerboseFlag = {
    .name = "verbose",
    .short_name = "v",
};

namespace {
enum class CompilePhase {
  Tokenize,
  Parse,
  Syntax,
  LLVM,
  Object,
};
}  // namespace

static auto operator<<(llvm::raw_ostream& out, CompilePhase phase)
    -> llvm::raw_ostream& {
  switch (phase) {
    case CompilePhase::Tokenize:
      out << "tokenize";
      break;
    case CompilePhase::Parse:
      out << "parse";
      break;
    case CompilePhase::Syntax:
      out << "syntax";
      break;
    case CompilePhase::LLVM:
      out << "llvm";
      break;
    case CompilePhase::Object:
      out << "object";
      break;
  }
  return out;
}

constexpr std::tuple CompilePhaseOptionValues = {
  Args::EnumOptValue<CompilePhase::Tokenize>{.name = "tokenize"},
  Args::EnumOptValue<CompilePhase::Parse>{.name = "parse"},
  Args::EnumOptValue<CompilePhase::Syntax>{.name = "syntax"},
  Args::EnumOptValue<CompilePhase::LLVM>{.name = "llvm"},
  Args::EnumOptValue<CompilePhase::Object>{.name = "object"},
};

constexpr Args::EnumOpt<CompilePhaseOptionValues> CompilePhaseOption = {
    .name = "phase",
    .default_value = CompilePhase::Object,
};

constexpr Args::Flag DumpTokensFlag = {.name = "dump-tokens"};
constexpr Args::Flag DumpParseTreeFlag = {.name = "dump-parse-tree"};
constexpr Args::Flag DumpSemanticsIRFlag = {.name = "dump-semantics-ir"};
constexpr Args::Flag DumpLLVMIRFlag = {.name = "dump-llvm-ir"};

constexpr Args::Flag StreamErrorsFlag = {.name = "stream-errors"};

constexpr Args::Flag PreorderParseTreeFlag = {.name = "preorder-parse-tree"};

constexpr Args::Flag BuiltinSemanticsIRFlag = {.name = "builtin-semantics-ir"};

constexpr Args::Command<VerboseFlag> DriverCommand = {
    .name = "carbon",
    .info =
        {
            .description = "TODO",
            .usage = "TODO",
        },
};

namespace {
enum class Subcommands {
  Compile,
};
}

constexpr Args::Subcommand<
    Subcommands::Compile, CompilePhaseOption, DumpTokensFlag, DumpParseTreeFlag,
    DumpSemanticsIRFlag, DumpLLVMIRFlag, StreamErrorsFlag,
    PreorderParseTreeFlag, BuiltinSemanticsIRFlag>
    CompileSubcommand = {
        .name = "compile",
};

using ArgsParser = Args::Parser<DriverCommand, CompileSubcommand>;

namespace {
struct DriverSubcommands {
  llvm::raw_ostream& output_stream_;
  llvm::raw_ostream& error_stream_;
  llvm::raw_ostream* vlog_stream_ = nullptr;

  using ArgsT = ArgsParser::ArgsT;

  auto RunCompileSubcommand(const ArgsT& args) -> bool;
};
}  // namespace

auto Driver::RunCommand(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  auto parsed_args = ArgsParser::Parse(args, error_stream_);
  if (!parsed_args) {
    return false;
  }

  if (parsed_args.TestFlag<VerboseFlag>()) {
    vlog_stream_ = &error_stream_;
    CARBON_VLOG() << "*** Enabled verbose logging ***\n";
  }

  DriverSubcommands subcommands = {
    .output_stream_ = output_stream_,
    .error_stream_ = error_stream_,
    .vlog_stream_ = vlog_stream_,
  };
  switch (parsed_args.subcommand()) {
    case Subcommands::Compile: {
      CARBON_VLOG() << "*** Running compile subcommand ***\n";
      return subcommands.RunCompileSubcommand(parsed_args);
    }
  }
  CARBON_FATAL() << "Unhandled subcommand";
}

auto DriverSubcommands::RunCompileSubcommand(const ArgsT& args) -> bool {
  if (args.positional_args().empty()) {
    error_stream_ << "ERROR: No input file specified.\n";
    return false;
  }
  if (args.positional_args().size() != 1) {
    error_stream_
        << "ERROR: Unexpected additional inputs to the 'compile' subcommand:";
    for (auto arg : args.positional_args()) {
      error_stream_ << " " << arg;
    }
    error_stream_ << "\n";
    return false;
  }

  StreamDiagnosticConsumer stream_consumer(error_stream_);
  DiagnosticConsumer* consumer = &stream_consumer;
  std::unique_ptr<SortingDiagnosticConsumer> sorting_consumer;
  // Enable sorted diagnostics only if we're not using verbose logging and
  // streamed diagnostics haven't been requested.
  if (!args.TestFlag<VerboseFlag>() && !args.TestFlag<StreamErrorsFlag>()) {
    sorting_consumer = std::make_unique<SortingDiagnosticConsumer>(*consumer);
    consumer = sorting_consumer.get();
  }

  llvm::StringRef input_file_name = args.positional_args().front();

  CARBON_VLOG() << "*** SourceBuffer::CreateFromFile ***\n";
  auto source = SourceBuffer::CreateFromFile(input_file_name);
  CARBON_VLOG() << "*** SourceBuffer::CreateFromFile done ***\n";
  if (!source) {
    error_stream_ << "ERROR: Unable to open input source file: ";
    llvm::handleAllErrors(source.takeError(),
                          [&](const llvm::ErrorInfoBase& ei) {
                            ei.log(error_stream_);
                            error_stream_ << "\n";
                          });
    return false;
  }

  bool has_errors = false;

  CompilePhase phase = *args.GetOption<CompilePhaseOption>();
  switch (phase) {
    case CompilePhase::Tokenize:
      if (args.TestFlag<DumpParseTreeFlag>()) {
        error_stream_ << "ERROR: Requested dumping the parse tree but compile "
                         "phase is limited to '"
                      << phase << "'\n";
        has_errors = true;
      }
      [[clang::fallthrough]];
    case CompilePhase::Parse:
      if (args.TestFlag<DumpSemanticsIRFlag>()) {
        error_stream_ << "ERROR: Requested dumping the semantics IR but "
                         "compile phase is limited to '"
                      << phase << "'\n";
        has_errors = true;
      }
      [[clang::fallthrough]];
    case CompilePhase::Syntax:
      if (args.TestFlag<DumpLLVMIRFlag>()) {
        error_stream_ << "ERROR: Requested dumping the LLVM IR but compile "
                         "phase is limited to '"
                      << phase << "'\n";
        has_errors = true;
      }
      [[clang::fallthrough]];
    case CompilePhase::LLVM:
    case CompilePhase::Object:
      // Everything can be dumped in these phases.
      break;
  }

  CARBON_VLOG() << "*** TokenizedBuffer::Lex ***\n";
  auto tokenized_source = TokenizedBuffer::Lex(*source, *consumer);
  has_errors |= tokenized_source.has_errors();
  CARBON_VLOG() << "*** TokenizedBuffer::Lex done ***\n";
  if (args.TestFlag<DumpTokensFlag>()) {
    CARBON_VLOG() << "Finishing output.";
    consumer->Flush();
    output_stream_ << tokenized_source;
  }
  CARBON_VLOG() << "tokenized_buffer: " << tokenized_source;
  if (args.GetOption<CompilePhaseOption>() == CompilePhase::Tokenize) {
    return !has_errors;
  }

  CARBON_VLOG() << "*** ParseTree::Parse ***\n";
  auto parse_tree = ParseTree::Parse(tokenized_source, *consumer, vlog_stream_);
  has_errors |= parse_tree.has_errors();
  CARBON_VLOG() << "*** ParseTree::Parse done ***\n";
  if (args.TestFlag<DumpParseTreeFlag>()) {
    consumer->Flush();
    parse_tree.Print(output_stream_, args.TestFlag<PreorderParseTreeFlag>());
  }
  CARBON_VLOG() << "parse_tree: " << parse_tree;
  if (args.GetOption<CompilePhaseOption>() == CompilePhase::Parse) {
    return !has_errors;
  }

  const SemanticsIR builtin_ir = SemanticsIR::MakeBuiltinIR();
  CARBON_VLOG() << "*** SemanticsIR::MakeFromParseTree ***\n";
  const SemanticsIR semantics_ir = SemanticsIR::MakeFromParseTree(
      builtin_ir, tokenized_source, parse_tree, *consumer, vlog_stream_);
  has_errors |= semantics_ir.has_errors();
  CARBON_VLOG() << "*** SemanticsIR::MakeFromParseTree done ***\n";
  if (args.TestFlag<DumpSemanticsIRFlag>()) {
    consumer->Flush();
    semantics_ir.Print(output_stream_, args.TestFlag<BuiltinSemanticsIRFlag>());
  }
  CARBON_VLOG() << "semantics_ir: " << semantics_ir;
  if (args.GetOption<CompilePhaseOption>() == CompilePhase::Syntax) {
    return !has_errors;
  }

  // Unlike previous steps, errors block further progress.
  if (has_errors) {
    CARBON_VLOG()
        << "*** Stopping before lowering to LLVM IR due to syntax errors ***";
    return false;
  }

  CARBON_VLOG() << "*** LowerToLLVM ***\n";
  llvm::LLVMContext llvm_context;
  const std::unique_ptr<llvm::Module> module =
      LowerToLLVM(llvm_context, input_file_name, semantics_ir, vlog_stream_);
  CARBON_VLOG() << "*** LowerToLLVM done ***\n";
  if (args.TestFlag<DumpLLVMIRFlag>()) {
    consumer->Flush();
    module->print(output_stream_, /*AAW=*/nullptr,
                  /*ShouldPreserveUseListOrder=*/true);
  }
  if (vlog_stream_) {
    CARBON_VLOG() << "module: ";
    module->print(*vlog_stream_, /*AAW=*/nullptr,
                  /*ShouldPreserveUseListOrder=*/false,
                  /*IsForDebug=*/true);
  }
  if (args.GetOption<CompilePhaseOption>() == CompilePhase::LLVM) {
    return !has_errors;
  }

  CARBON_FATAL() << "ERROR: Object file emission not yet implemented.";
}

}  // namespace Carbon
