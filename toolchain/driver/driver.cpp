// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/driver/driver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "toolchain/diagnostics/sorting_diagnostic_consumer.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/source/source_buffer.h"

namespace Carbon {

namespace {

enum class Subcommand {
#define CARBON_SUBCOMMAND(Name, ...) Name,
#include "toolchain/driver/flags.def"
  Unknown,
};

// TODO: There should be better ways to version other than this.
#define CARBON_TOOLCHAIN_VERSION "Carbon version 0.0.0"
// TODO: rename/def/enum/header it something like Driver:RunVersionSubcommand.
auto PrintVersionMessage(llvm::raw_ostream& stream) -> bool {
  // TODO: Report Target, Thread model, InstalledDir.
  stream << CARBON_TOOLCHAIN_VERSION << '\n';
  return true;

}

std::string Desc = "Carbon Toolchain";
std::string ExtraDesc = "\n\n  Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n";

}  // namespace

auto Driver::RunFullCommand(int argc, char **argv) -> bool {
  // Discard general option and reset `GenericCategory` with generic options.
  llvm::cl::OptionCategory dummycat("","");
  llvm::cl::HideUnrelatedOptions(dummycat);
  llvm::cl::ResetCommandLineParser();

  // TODO: cases for piping(stdin, probably by default?), multiple files.
  llvm::cl::list<std::string> InputFilenames(
                                    llvm::cl::Positional, llvm::cl::ZeroOrMore,
                                    llvm::cl::desc("<input file(s)>"));
  // TODO: case for output to stdout(probably by default?).
  llvm::cl::opt<std::string> OutputFilename("o",
                                    llvm::cl::desc("Specify output filename"),
                                    llvm::cl::value_desc("output filename"));
  llvm::cl::opt<bool> Version("version",
                        llvm::cl::desc("Display the version of this program"));

// Positionals who correspond to subcommands.
#define CARBON_SUBCOMMAND(Name, Spelling, Description) \
        llvm::cl::opt<bool> Name(Spelling, llvm::cl::desc( Description ));
#include "toolchain/driver/flags.def"

  llvm::cl::ParseCommandLineOptions(argc, argv, Desc.append(ExtraDesc));

  // TODO: Refactor the same pattern in subcommands: CreateFromFile ...
  //       Organize with `args` that won't be entirely used
  // Somewhat stubbing arguments for RunSubcommands.
  llvm::SmallVector<llvm::StringRef, 16> subcommand_args(argv + 1, argv + argc);
  DiagnosticConsumer* consumer = &ConsoleDiagnosticConsumer();
  std::unique_ptr<SortingDiagnosticConsumer> sorting_consumer;
  // TODO: case of "--print-errors=streamed"
  sorting_consumer = std::make_unique<SortingDiagnosticConsumer>(*consumer);
  consumer = sorting_consumer.get();

  Carbon::Subcommand subcommand =
#define CARBON_SUBCOMMAND(Name, ...) \
 (Name ? Subcommand::Name :
#include "toolchain/driver/flags.def"
  Subcommand::Unknown
#define CARBON_SUBCOMMAND(Name, ...) \
  )
#include "toolchain/driver/flags.def"
  ;

  if (Version)
    return PrintVersionMessage(error_stream_);

  switch (subcommand) {
    // Note `llvm::cl` also handles cases of unknown subcommands
    case Subcommand::Unknown:
      error_stream_ << "ERROR: Unknown subcommand\n";
      return false;
#define CARBON_SUBCOMMAND(Name, ...)    \
  case Subcommand::Name:                \
  return Run##Name##Subcommand(*consumer, subcommand_args);
#include "toolchain/driver/flags.def"
  }
  llvm_unreachable("All subcommands handled!");
}

auto Driver::RunHelpSubcommand(DiagnosticConsumer& /*consumer*/,
                               llvm::ArrayRef<llvm::StringRef> args) -> bool {
  //if (!args.empty()) {
  //  ReportExtraArgs("help", args);
  //  return false;
  //}
  llvm::cl::PrintHelpMessage();
  return true;
}

auto Driver::RunDumpTokensSubcommand(DiagnosticConsumer& consumer,
                                     llvm::ArrayRef<llvm::StringRef> args)
    -> bool {
  if (args.empty()) {
    error_stream_ << "ERROR: No input file specified.\n";
    return false;
  }

  llvm::StringRef input_file_name = args.front();
  args = args.drop_front();
  if (!args.empty()) {
    ReportExtraArgs("dump-tokens", args);
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
  consumer.Flush();
  tokenized_source.Print(output_stream_);
  return !tokenized_source.has_errors();
}

auto Driver::RunDumpParseTreeSubcommand(DiagnosticConsumer& consumer,
                                        llvm::ArrayRef<llvm::StringRef> args)
    -> bool {
  if (args.empty()) {
    error_stream_ << "ERROR: No input file specified.\n";
    return false;
  }

  llvm::StringRef input_file_name = args.front();
  args = args.drop_front();
  if (!args.empty()) {
    ReportExtraArgs("dump-parse-tree", args);
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
  auto parse_tree = ParseTree::Parse(tokenized_source, consumer);
  consumer.Flush();
  parse_tree.Print(output_stream_);
  return !tokenized_source.has_errors() && !parse_tree.has_errors();
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
