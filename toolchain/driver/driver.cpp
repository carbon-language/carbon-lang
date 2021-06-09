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
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lexer/tokenized_buffer.h"
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
  if (args.empty()) {
    error_stream << "ERROR: No subcommand specified.\n";
    return false;
  }

  llvm::StringRef subcommand_text = args[0];
  llvm::SmallVector<llvm::StringRef, 16> subcommand_args(
      std::next(args.begin()), args.end());
  switch (GetSubcommand(subcommand_text)) {
    case Subcommand::Unknown:
      error_stream << "ERROR: Unknown subcommand '" << subcommand_text
                   << "'.\n";
      return false;

#define CARBON_SUBCOMMAND(Name, ...) \
  case Subcommand::Name:             \
    return Run##Name##Subcommand(subcommand_args);
#include "toolchain/driver/flags.def"
  }
  llvm_unreachable("All subcommands handled!");
}

auto Driver::RunHelpSubcommand(llvm::ArrayRef<llvm::StringRef> args) -> bool {
  // FIXME: We should support getting detailed help on a subcommand by looking
  // for it as a positional parameter here.
  if (!args.empty()) {
    ReportExtraArgs("help", args);
    return false;
  }

  output_stream << "List of subcommands:\n\n";

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
    // FIXME: We should wrap this to the number of columns left after the
    // subcommand on the terminal, and using a hanging indent.
    llvm::StringRef help_text = subcommand_and_help[1];
    output_stream << "  "
                  << llvm::left_justify(subcommand_text, max_subcommand_width)
                  << " - " << help_text << "\n";
  }

  output_stream << "\n";
  return true;
}

auto Driver::RunDumpTokensSubcommand(llvm::ArrayRef<llvm::StringRef> args)
    -> bool {
  if (args.empty()) {
    error_stream << "ERROR: No input file specified.\n";
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
    error_stream << "ERROR: Unable to open input source file: ";
    llvm::handleAllErrors(source.takeError(),
                          [&](const llvm::ErrorInfoBase& ei) {
                            ei.log(error_stream);
                            error_stream << "\n";
                          });
    return false;
  }
  auto tokenized_source =
      TokenizedBuffer::Lex(*source, ConsoleDiagnosticConsumer());
  if (tokenized_source.HasErrors()) {
    error_stream << "ERROR: Unable to tokenize source file '" << input_file_name
                 << "'!\n";
    return false;
  }
  tokenized_source.Print(output_stream);
  return true;
}

auto Driver::ReportExtraArgs(llvm::StringRef subcommand_text,
                             llvm::ArrayRef<llvm::StringRef> args) -> void {
  error_stream << "ERROR: Unexpected additional arguments to the '"
               << subcommand_text << "' subcommand:";
  for (auto arg : args) {
    error_stream << " " << arg;
  }

  error_stream << "\n";
}

}  // namespace Carbon
