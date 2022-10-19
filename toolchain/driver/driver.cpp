// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/semantics_ir_factory.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

namespace {

enum class Subcommand {
#define CARBON_SUBCOMMAND(Name, ...) Name,
#include "toolchain/driver/flags.def"
  Unknown,
};

auto GetSubcommand(llvm::StringRef name) -> Subcommand {
  return llvm::StringSwitch<Subcommand>(name)
#define CARBON_SUBCOMMAND(Name, Spelling, ...) .Case(Spelling, Subcommand::Name)
#include "toolchain/driver/flags.def"
      .Default(Subcommand::Unknown);
}

}  // namespace

auto Driver::RunFullCommand(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  DiagnosticConsumer* consumer = &ConsoleDiagnosticConsumer();
  std::unique_ptr<SortingDiagnosticConsumer> sorting_consumer;
  // TODO: Figure out a command-line support library, this is temporary.
  if (!args.empty() && args[0] == "--print-errors=streamed") {
    args = args.drop_front();
  } else {
    sorting_consumer = std::make_unique<SortingDiagnosticConsumer>(*consumer);
    consumer = sorting_consumer.get();
  }

  if (args.empty()) {
    error_stream_ << "ERROR: No subcommand specified.\n";
    return false;
  }

  llvm::StringRef subcommand_text = args[0];
  args = args.drop_front();
  switch (GetSubcommand(subcommand_text)) {
    case Subcommand::Unknown:
      error_stream_ << "ERROR: Unknown subcommand '" << subcommand_text
                    << "'.\n";
      return false;

#define CARBON_SUBCOMMAND(Name, ...) \
  case Subcommand::Name:             \
    return Run##Name##Subcommand(*consumer, args);
#include "toolchain/driver/flags.def"
  }
  llvm_unreachable("All subcommands handled!");
}

auto Driver::RunHelpSubcommand(DiagnosticConsumer& /*consumer*/,
                               llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // TODO: We should support getting detailed help on a subcommand by looking
  // for it as a positional parameter here.
  if (!args.empty()) {
    ReportExtraArgs("help", args);
    return false;
  }

  output_stream_ << "List of subcommands:\n\n";

  constexpr llvm::StringLiteral SubcommandsAndHelp[][2] = {
#define CARBON_SUBCOMMAND(Name, Spelling, HelpText) {Spelling, HelpText},
#include "toolchain/driver/flags.def"
  };

  int max_subcommand_width = 0;
  for (auto subcommand_and_help : SubcommandsAndHelp) {
    max_subcommand_width = std::max(
        max_subcommand_width, static_cast<int>(subcommand_and_help[0].size()));
  }

  for (auto subcommand_and_help : SubcommandsAndHelp) {
    llvm::StringRef subcommand_text = subcommand_and_help[0];
    // TODO: We should wrap this to the number of columns left after the
    // subcommand on the terminal, and using a hanging indent.
    llvm::StringRef help_text = subcommand_and_help[1];
    output_stream_ << "  "
                   << llvm::left_justify(subcommand_text, max_subcommand_width)
                   << " - " << help_text << "\n";
  }

  output_stream_ << "\n";
  return true;
}

enum class DumpMode { TokenizedBuffer, ParseTree, SemanticsIR, Unknown };

auto Driver::RunDumpSubcommand(DiagnosticConsumer& consumer,
                               llvm::ArrayRef<llvm::StringRef> args) -> bool {
  if (args.empty()) {
    error_stream_ << "ERROR: No dump mode specified.\n";
    return false;
  }

  auto dump_mode = llvm::StringSwitch<DumpMode>(args.front())
                       .Case("tokens", DumpMode::TokenizedBuffer)
                       .Case("parse-tree", DumpMode::ParseTree)
                       .Case("semantics-ir", DumpMode::SemanticsIR)
                       .Default(DumpMode::Unknown);
  if (dump_mode == DumpMode::Unknown) {
    error_stream_ << "ERROR: Dump mode should be one of tokens, parse-tree, or "
                     "semantics-ir.\n";
    return false;
  }
  args = args.drop_front();

  if (args.empty()) {
    error_stream_ << "ERROR: No input file specified.\n";
    return false;
  }

  llvm::StringRef input_file_name = args.front();
  args = args.drop_front();
  if (!args.empty()) {
    ReportExtraArgs("dump", args);
    return false;
  }

  auto source = SourceBuffer::CreateFromFile(input_file_name);
  if (!source) {
    error_stream_ << "ERROR: Unable to open input source file: ";
    llvm::handleAllErrors(source.takeError(),
                          [&](const llvm::ErrorInfoBase& ei) {
                            ei.log(error_stream_);
                            error_stream_ << "\n";
                          });
    return false;
  }

  auto tokenized_source = TokenizedBuffer::Lex(*source, consumer);
  if (dump_mode == DumpMode::TokenizedBuffer) {
    consumer.Flush();
    tokenized_source.Print(output_stream_);
    return !tokenized_source.has_errors();
  }

  auto parse_tree = ParseTree::Parse(tokenized_source, consumer);
  if (dump_mode == DumpMode::ParseTree) {
    consumer.Flush();
    parse_tree.Print(output_stream_);
    return !tokenized_source.has_errors() && !parse_tree.has_errors();
  }

  auto semantics_ir = SemanticsIRFactory::Build(tokenized_source, parse_tree);
  if (dump_mode == DumpMode::SemanticsIR) {
    consumer.Flush();
    semantics_ir.Print(output_stream_);
    // TODO: Return false when SemanticsIR has errors (not supported right now).
    return !tokenized_source.has_errors() && !parse_tree.has_errors();
  }

  llvm_unreachable("should handle all dump modes");
}

auto Driver::ReportExtraArgs(llvm::StringRef subcommand_text,
                             llvm::ArrayRef<llvm::StringRef> args) -> void {
  error_stream_ << "ERROR: Unexpected additional arguments to the '"
                << subcommand_text << "' subcommand:";
  for (auto arg : args) {
    error_stream_ << " " << arg;
  }

  error_stream_ << "\n";
}

}  // namespace Carbon
