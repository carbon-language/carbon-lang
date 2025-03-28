// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/command_line.h"

#include <memory>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/FormatVariadic.h"

// Recursion is used for subcommands. This should be okay since recursion is
// limited by command line architecture.
// NOLINTBEGIN(misc-no-recursion)

namespace Carbon::CommandLine {

auto operator<<(llvm::raw_ostream& output, ParseResult result)
    -> llvm::raw_ostream& {
  switch (result) {
    case ParseResult::MetaSuccess:
      return output << "MetaSuccess";
    case ParseResult::Success:
      return output << "Success";
  }
  CARBON_FATAL("Corrupt parse result!");
}

auto operator<<(llvm::raw_ostream& output, ArgKind kind) -> llvm::raw_ostream& {
  switch (kind) {
    case ArgKind::Flag:
      return output << "Boolean";
    case ArgKind::Integer:
      return output << "Integer";
    case ArgKind::String:
      return output << "String";
    case ArgKind::OneOf:
      return output << "OneOf";
    case ArgKind::MetaActionOnly:
      return output << "MetaActionOnly";
    case ArgKind::Invalid:
      return output << "Invalid";
  }
  CARBON_FATAL("Corrupt argument kind!");
}

auto operator<<(llvm::raw_ostream& output, CommandKind kind)
    -> llvm::raw_ostream& {
  switch (kind) {
    case CommandKind::Invalid:
      return output << "Invalid";
    case CommandKind::RequiresSubcommand:
      return output << "RequiresSubcommand";
    case CommandKind::Action:
      return output << "Action";
    case CommandKind::MetaAction:
      return output << "MetaAction";
  }
  CARBON_FATAL("Corrupt command kind!");
}

template <typename T, typename ToPrintable>
static auto PrintListOfAlternatives(llvm::raw_ostream& output,
                                    llvm::ArrayRef<T> alternatives,
                                    ToPrintable to_printable) -> void {
  for (const auto& alternative : alternatives.drop_back()) {
    output << "`" << to_printable(alternative)
           << (alternatives.size() > 2 ? "`, " : "` ");
  }
  if (alternatives.size() > 1) {
    output << "or ";
  }
  output << "`" << to_printable(alternatives.back()) << "`";
}

Arg::Arg(ArgInfo info) : info(info) {}

Arg::~Arg() {
  switch (kind) {
    case Kind::Flag:
    case Kind::Integer:
    case Kind::String:
    case Kind::MetaActionOnly:
    case Kind::Invalid:
      // Nothing to do!
      break;
    case Kind::OneOf:
      value_strings.~decltype(value_strings)();
      value_action.~ValueActionT();
      if (has_default) {
        default_action.~DefaultActionT();
      }
      break;
  }
}

Command::Command(CommandInfo info, Command* parent)
    : info(info), parent(parent) {}

class MetaPrinter {
 public:
  // `out` must not be null.
  explicit MetaPrinter(llvm::raw_ostream* out) : out_(out) {}

  // Registers this meta printer with a command through the provided builder.
  //
  // This adds meta subcommands or options to print both help and version
  // information for the command.
  auto RegisterWithCommand(const Command& command, CommandBuilder& builder)
      -> void;

  auto PrintHelp(const Command& command) const -> void;
  auto PrintHelpForSubcommandName(const Command& command,
                                  llvm::StringRef subcommand_name) const
      -> void;
  auto PrintVersion(const Command& command) const -> void;
  auto PrintSubcommands(const Command& command) const -> void;

 private:
  // The indent is calibrated to allow a short and long option after a two
  // character indent on the prior line to be visually recognized as separate
  // from the hanging indent.
  //
  // Visual guide:                               |  -x, --extract
  //                                             |          Hanging indented.
  static constexpr llvm::StringRef BlockIndent = "          ";

  // Width limit for parent command options in usage rendering.
  static constexpr int MaxParentOptionUsageWidth = 8;

  // Width limit for the leaf command options in usage rendering.
  static constexpr int MaxLeafOptionUsageWidth = 16;

  static constexpr CommandInfo HelpCommandInfo = {
      .name = "help",
      .help = R"""(
Prints help information for the command, including a description, command line
usage, and details of each subcommand and option that can be provided.
)""",
      .help_short = R"""(
Prints help information.
)""",
  };
  static constexpr ArgInfo HelpArgInfo = {
      .name = "help",
      .value_name = "(full|short)",
      .help = R"""(
Prints help information for the command, including a description, command line
usage, and details of each option that can be provided.
)""",
      .help_short = HelpCommandInfo.help_short,
  };

  // Provide a customized description for help on a subcommand to avoid
  // confusion with the top-level help.
  static constexpr CommandInfo SubHelpCommandInfo = {
      .name = "help",
      .help = R"""(
Prints help information for the subcommand, including a description, command
line usage, and details of each further subcommand and option that can be
provided.
)""",
      .help_short = R"""(
Prints subcommand help information.
)""",
  };
  static constexpr ArgInfo SubHelpArgInfo = {
      .name = "help",
      .value_name = "(full|short)",
      .help = R"""(
Prints help information for the subcommand, including a description, command
line usage, and details of each option that can be provided.
)""",
      .help_short = SubHelpCommandInfo.help_short,
  };

  static constexpr ArgInfo HelpSubcommandArgInfo = {
      .name = "subcommand",
      .help = R"""(
Which subcommand to print help information for.
)""",
  };

  static constexpr CommandInfo VersionCommandInfo = {
      .name = "version",
      .help = R"""(
Prints the version of this command.
)""",
  };
  static constexpr ArgInfo VersionArgInfo = {
      .name = "version",
      .help = VersionCommandInfo.help,
  };

  // A general helper for rendering a text block.
  auto PrintTextBlock(llvm::StringRef indent, llvm::StringRef text) const
      -> void;

  // Helpers for version and build information printing.
  auto PrintRawVersion(const Command& command, llvm::StringRef indent) const
      -> void;
  auto PrintRawBuildInfo(const Command& command, llvm::StringRef indent) const
      -> void;

  // Helpers for printing components of help and usage output for arguments,
  // including options and positional arguments.
  auto PrintArgValueUsage(const Arg& arg) const -> void;
  auto PrintOptionUsage(const Arg& option) const -> void;
  auto PrintOptionShortName(const Arg& arg) const -> void;
  auto PrintArgShortValues(const Arg& arg) const -> void;
  auto PrintArgLongValues(const Arg& arg, llvm::StringRef indent) const -> void;
  auto PrintArgHelp(const Arg& arg, llvm::StringRef indent) const -> void;

  // Helpers for printing command usage summaries.
  auto PrintRawUsageCommandAndOptions(
      const Command& command,
      int max_option_width = MaxLeafOptionUsageWidth) const -> void;
  auto PrintRawUsage(const Command& command, llvm::StringRef indent) const
      -> void;
  auto PrintUsage(const Command& command) const -> void;

  // Helpers to print various sections of `PrintHelp` that only occur within
  // that output.
  auto PrintHelpSubcommands(const Command& command) const -> void;
  auto PrintHelpPositionalArgs(const Command& command) const -> void;
  auto PrintHelpOptions(const Command& command) const -> void;

  llvm::raw_ostream* out_;

  // A flag that may be configured during command line parsing to select between
  // long and short form help output.
  bool short_help_ = false;

  // The requested subcommand to print help information for.
  llvm::StringRef help_subcommand_;
};

auto MetaPrinter::RegisterWithCommand(const Command& command,
                                      CommandBuilder& builder) -> void {
  bool is_subcommand = command.parent;
  bool has_subcommands = !command.subcommands.empty();

  // If this command has subcommands, we prefer that model for access meta
  // actions, but still silently support using the flags. But we never want to
  // *add* subcommands if they aren't already being used.
  if (has_subcommands) {
    builder.AddSubcommand(
        is_subcommand ? SubHelpCommandInfo : HelpCommandInfo,
        [&](CommandBuilder& sub_b) {
          sub_b.AddStringPositionalArg(HelpSubcommandArgInfo, [&](auto& arg_b) {
            arg_b.Set(&help_subcommand_);
          });
          sub_b.Meta([this, &command]() {
            if (help_subcommand_.empty()) {
              PrintHelp(command);
            } else {
              PrintHelpForSubcommandName(command, help_subcommand_);
            }
          });
        });

    // Only add version printing support if there is a version string
    // configured for this command.
    if (!command.info.version.empty()) {
      builder.AddSubcommand(VersionCommandInfo, [&](CommandBuilder& sub_b) {
        sub_b.Meta([this, &command]() { PrintVersion(command); });
      });
    }
  }
  builder.AddOneOfOption(
      is_subcommand ? SubHelpArgInfo : HelpArgInfo, [&](auto& arg_b) {
        arg_b.HelpHidden(has_subcommands);
        arg_b.SetOneOf(
            {
                arg_b.OneOfValue("full", false).Default(true),
                arg_b.OneOfValue("short", true),
            },
            &short_help_);
        arg_b.MetaAction([this, &command]() { PrintHelp(command); });
      });

  // Only add version printing support if there is a version string configured
  // for this command.
  if (!command.info.version.empty()) {
    builder.AddMetaActionOption(VersionArgInfo, [&](auto& arg_b) {
      arg_b.HelpHidden(has_subcommands);
      arg_b.MetaAction([this, &command]() { PrintVersion(command); });
    });
  }
}

auto MetaPrinter::PrintHelp(const Command& command) const -> void {
  // TODO: begin using the short setting to customize the output.
  (void)short_help_;

  const CommandInfo& info = command.info;
  if (!info.version.empty()) {
    // We use the version string as a header for the command help when present.
    PrintRawVersion(command, /*indent=*/"");
    *out_ << "\n";
  }
  if (!command.info.help.empty()) {
    PrintTextBlock("", info.help);
    *out_ << "\n";
  }
  if (!info.build_info.empty()) {
    *out_ << "Build info:\n";
    PrintRawBuildInfo(command, /*indent=*/"  ");
    *out_ << "\n";
  }

  PrintUsage(command);
  PrintHelpSubcommands(command);
  PrintHelpPositionalArgs(command);
  PrintHelpOptions(command);

  if (!info.help_epilogue.empty()) {
    *out_ << "\n";
    PrintTextBlock("", info.help_epilogue);
  }

  // End with a blank line for the long help to make it easier to separate from
  // anything that follows in the shell.
  *out_ << "\n";
}

auto MetaPrinter::PrintHelpForSubcommandName(
    const Command& command, llvm::StringRef subcommand_name) const -> void {
  for (const auto& subcommand : command.subcommands) {
    if (subcommand->info.name == subcommand_name) {
      PrintHelp(*subcommand);
      return;
    }
  }

  // TODO: This should really be connected up so that parsing can return an
  // Error instead of ParseResult::MetaSuccess in this case.
  *out_ << "error: could not find a subcommand named '" << subcommand_name
        << "'\n";
}

auto MetaPrinter::PrintVersion(const Command& command) const -> void {
  CARBON_CHECK(
      !command.info.version.empty(),
      "Printing should not be enabled without a version string configured.");
  PrintRawVersion(command, /*indent=*/"");
  if (!command.info.build_info.empty()) {
    *out_ << "\n";
    // If there is build info to print, we also render that without any indent.
    PrintRawBuildInfo(command, /*indent=*/"");
  }
}

auto MetaPrinter::PrintSubcommands(const Command& command) const -> void {
  PrintListOfAlternatives(*out_, llvm::ArrayRef(command.subcommands),
                          [](const std::unique_ptr<Command>& subcommand) {
                            return subcommand->info.name;
                          });
}

auto MetaPrinter::PrintRawVersion(const Command& command,
                                  llvm::StringRef indent) const -> void {
  // Newlines are trimmed from the version string an a closing newline added but
  // no other formatting is performed.
  *out_ << indent << command.info.version.trim('\n') << "\n";
}
auto MetaPrinter::PrintRawBuildInfo(const Command& command,
                                    llvm::StringRef indent) const -> void {
  // Print the build info line-by-line without any wrapping in case it
  // contains line-oriented formatted text, but drop leading and trailing blank
  // lines.
  llvm::SmallVector<llvm::StringRef, 128> lines;
  command.info.build_info.trim('\n').split(lines, "\n");
  for (auto line : lines) {
    *out_ << indent << line << "\n";
  }
}

auto MetaPrinter::PrintTextBlock(llvm::StringRef indent,
                                 llvm::StringRef text) const -> void {
  // Strip leading and trailing newlines to make it easy to use multiline raw
  // string literals that will naturally have those.
  text = text.trim('\n');
  // For empty text, print nothing at all. The caller formatting will work to
  // handle this gracefully.
  if (text.empty()) {
    return;
  }

  // Remove line breaks from the text that would typically be removed when
  // rendering it as Markdown. The goal is to preserve:
  //
  // - Blank lines as paragraph separators.
  // - Line breaks after list items or other structural components in Markdown.
  // - Fenced regions exactly as they appear.
  //
  // And within paragraphs (including those nested in lists), reflow the
  // paragraph intelligently to the column width. There are TODOs below about
  // both lists and reflowing.
  llvm::SmallVector<llvm::StringRef, 128> input_lines;
  text.split(input_lines, "\n");

  for (int i = 0, size = input_lines.size(); i < size;) {
    if (input_lines[i].empty()) {
      // Blank lines are preserved.
      *out_ << "\n";
      ++i;
      continue;
    }

    if (input_lines[i].starts_with("```")) {
      // Fenced regions are preserved verbatim.
      llvm::StringRef fence =
          input_lines[i].slice(0, input_lines[i].find_first_not_of("`"));
      do {
        *out_ << indent << input_lines[i] << "\n";
        ++i;
      } while (i < size && !input_lines[i].starts_with(fence));
      if (i >= size) {
        // Don't error on malformed text blocks, just print what we've got.
        break;
      }
      // Including the close of the fence.
      *out_ << indent << input_lines[i] << "\n";
      ++i;
      continue;
    }

    if (input_lines[i].starts_with("    ")) {
      // Indented code blocks ar preserved verbatim, but we don't support tabs
      // in the indent for simplicity.
      do {
        *out_ << indent << input_lines[i] << "\n";
        ++i;
      } while (i < size && input_lines[i].starts_with("    "));
      continue;
    }

    // TODO: Detect other Markdown structures, especially lists and tables.

    // Otherwise, collect all of the lines until the end or the next blank line
    // as a block of text.
    //
    // TODO: This is where we should re-flow.
    llvm::StringRef space = indent;
    do {
      *out_ << space << input_lines[i].trim();
      space = " ";
      ++i;
    } while (i < size && !input_lines[i].empty());
    *out_ << "\n";
  }
}

auto MetaPrinter::PrintArgValueUsage(const Arg& arg) const -> void {
  if (!arg.info.value_name.empty()) {
    *out_ << arg.info.value_name;
    return;
  }
  if (arg.kind == Arg::Kind::OneOf) {
    *out_ << "(";
    llvm::ListSeparator sep("|");
    for (llvm::StringRef value_string : arg.value_strings) {
      *out_ << sep << value_string;
    }
    *out_ << ")";
    return;
  }
  *out_ << "...";
}

auto MetaPrinter::PrintOptionUsage(const Arg& option) const -> void {
  if (option.kind == Arg::Kind::Flag) {
    *out_ << "--" << (option.default_flag ? "no-" : "") << option.info.name;
    return;
  }
  *out_ << "--" << option.info.name;
  if (option.kind != Arg::Kind::MetaActionOnly) {
    *out_ << (option.has_default ? "[" : "") << "=";
    PrintArgValueUsage(option);
    if (option.has_default) {
      *out_ << "]";
    }
  }
}

auto MetaPrinter::PrintOptionShortName(const Arg& arg) const -> void {
  CARBON_CHECK(!arg.info.short_name.empty(), "No short name to use.");
  *out_ << "-" << arg.info.short_name;
}

auto MetaPrinter::PrintArgShortValues(const Arg& arg) const -> void {
  CARBON_CHECK(
      arg.kind == Arg::Kind::OneOf,
      "Only one-of arguments have interesting value snippets to print.");
  llvm::ListSeparator sep;
  for (llvm::StringRef value_string : arg.value_strings) {
    *out_ << sep << value_string;
  }
}
auto MetaPrinter::PrintArgLongValues(const Arg& arg,
                                     llvm::StringRef indent) const -> void {
  *out_ << indent << "Possible values:\n";
  // TODO: It would be good to add help text for each value and then print it
  // here.
  for (auto [i, value_string] : llvm::enumerate(arg.value_strings)) {
    *out_ << indent << "- " << value_string;
    if (arg.has_default && static_cast<int>(i) == arg.default_value_index) {
      *out_ << " (default)";
    }
    *out_ << "\n";
  }
}

auto MetaPrinter::PrintArgHelp(const Arg& arg, llvm::StringRef indent) const
    -> void {
  // Print out the main help text.
  PrintTextBlock(indent, arg.info.help);

  // Then print out any help based on the values.
  switch (arg.kind) {
    case Arg::Kind::Integer:
      if (arg.has_default) {
        *out_ << "\n";
        *out_ << indent << "Default value: " << arg.default_integer << "\n";
      }
      break;
    case Arg::Kind::String:
      if (arg.has_default) {
        *out_ << "\n";
        *out_ << indent << "Default value: " << arg.default_string << "\n";
      }
      break;
    case Arg::Kind::OneOf:
      *out_ << "\n";
      PrintArgLongValues(arg, indent);
      break;
    case Arg::Kind::Flag:
    case Arg::Kind::MetaActionOnly:
      // No value help.
      break;
    case Arg::Kind::Invalid:
      CARBON_FATAL("Argument configured without any action or kind!");
  }
}

auto MetaPrinter::PrintRawUsageCommandAndOptions(const Command& command,
                                                 int max_option_width) const
    -> void {
  // Recursively print parent usage first with a compressed width.
  if (command.parent) {
    PrintRawUsageCommandAndOptions(*command.parent, MaxParentOptionUsageWidth);
    *out_ << " ";
  }

  *out_ << command.info.name;

  // Buffer the options rendering so we can limit its length.
  RawStringOstream buffer_out;
  MetaPrinter buffer_printer(&buffer_out);
  bool have_short_flags = false;
  for (const auto& arg : command.options) {
    if (static_cast<int>(buffer_out.size()) > max_option_width) {
      break;
    }
    // We can summarize positive boolean flags with a short name using a
    // sequence of short names in a single rendered argument.
    if (arg->kind == Arg::Kind::Flag && !arg->default_flag &&
        !arg->info.short_name.empty()) {
      if (!have_short_flags) {
        have_short_flags = true;
        buffer_out << "-";
      }
      buffer_out << arg->info.short_name;
    }
  }
  llvm::StringRef space = have_short_flags ? " " : "";
  for (const auto& option : command.options) {
    if (static_cast<int>(buffer_out.size()) > max_option_width) {
      break;
    }
    if (option->is_help_hidden || option->meta_action) {
      // Skip hidden and options with meta actions attached.
      continue;
    }
    if (option->kind == Arg::Kind::Flag && !option->default_flag &&
        !option->info.short_name.empty()) {
      // Handled with short names above.
      continue;
    }
    buffer_out << space;
    buffer_printer.PrintOptionUsage(*option);
    space = " ";
  }
  if (!buffer_out.empty()) {
    if (static_cast<int>(buffer_out.size()) <= max_option_width) {
      *out_ << " [" << buffer_out.TakeStr() << "]";
    } else {
      buffer_out.clear();
      *out_ << " [OPTIONS]";
    }
  }
}

auto MetaPrinter::PrintRawUsage(const Command& command,
                                llvm::StringRef indent) const -> void {
  if (!command.info.usage.empty()) {
    PrintTextBlock(indent, command.info.usage);
    return;
  }

  if (command.kind != Command::Kind::RequiresSubcommand) {
    // We're a valid leaf command, so synthesize a full usage line.
    *out_ << indent;
    PrintRawUsageCommandAndOptions(command);

    if (!command.positional_args.empty()) {
      bool open_optional = false;
      for (auto [i, arg] : llvm::enumerate(command.positional_args)) {
        *out_ << " ";
        if (i != 0 && command.positional_args[i - 1]->is_append) {
          *out_ << "-- ";
        }
        if (!arg->is_required && !open_optional) {
          *out_ << "[";
          open_optional = true;
        }
        *out_ << "<" << arg->info.name << ">";
        if (arg->is_append) {
          *out_ << "...";
        }
      }
      if (open_optional) {
        *out_ << "]";
      }
    }
    *out_ << "\n";
  }

  // If we have subcommands, also recurse into them so each one can print their
  // usage lines.
  for (const auto& subcommand : command.subcommands) {
    if (subcommand->is_help_hidden ||
        subcommand->kind == Command::Kind::MetaAction) {
      continue;
    }
    PrintRawUsage(*subcommand, indent);
  }
}

auto MetaPrinter::PrintUsage(const Command& command) const -> void {
  if (!command.parent) {
    *out_ << "Usage:\n";
  } else {
    *out_ << "Subcommand `" << command.info.name << "` usage:\n";
  }
  PrintRawUsage(command, "  ");
}

auto MetaPrinter::PrintHelpSubcommands(const Command& command) const -> void {
  bool first_subcommand = true;
  for (const auto& subcommand : command.subcommands) {
    if (subcommand->is_help_hidden) {
      continue;
    }
    if (first_subcommand) {
      first_subcommand = false;
      if (!command.parent) {
        *out_ << "\nSubcommands:";
      } else {
        *out_ << "\nSubcommand `" << command.info.name << "` subcommands:";
      }
    }
    *out_ << "\n";
    *out_ << "  " << subcommand->info.name << "\n";
    PrintTextBlock(BlockIndent, subcommand->info.help);
  }
}

auto MetaPrinter::PrintHelpPositionalArgs(const Command& command) const
    -> void {
  bool first_positional_arg = true;
  for (const auto& positional_arg : command.positional_args) {
    if (positional_arg->is_help_hidden) {
      continue;
    }
    if (first_positional_arg) {
      first_positional_arg = false;
      if (!command.parent) {
        *out_ << "\nPositional arguments:";
      } else {
        *out_ << "\nSubcommand `" << command.info.name
              << "` positional arguments:";
      }
    }
    *out_ << "\n";
    *out_ << "  " << positional_arg->info.name << "\n";
    PrintArgHelp(*positional_arg, BlockIndent);
  }
}

auto MetaPrinter::PrintHelpOptions(const Command& command) const -> void {
  bool first_option = true;
  for (const auto& option : command.options) {
    if (option->is_help_hidden) {
      continue;
    }
    if (first_option) {
      first_option = false;
      if (!command.parent && command.subcommands.empty()) {
        // Only one command level.
        *out_ << "\nOptions:";
      } else if (!command.parent) {
        *out_ << "\nCommand options:";
      } else {
        *out_ << "\nSubcommand `" << command.info.name << "` options:";
      }
    }
    *out_ << "\n";
    *out_ << "  ";
    if (!option->info.short_name.empty()) {
      PrintOptionShortName(*option);
      *out_ << ", ";
    } else {
      *out_ << "    ";
    }
    PrintOptionUsage(*option);
    *out_ << "\n";
    PrintArgHelp(*option, BlockIndent);
  }
}

class Parser {
 public:
  // `out` must not be null.
  explicit Parser(llvm::raw_ostream* out, CommandInfo command_info,
                  llvm::function_ref<auto(CommandBuilder&)->void> build);

  auto Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args)
      -> ErrorOr<ParseResult>;

 private:
  friend CommandBuilder;

  // For the option and subcommand maps, we use somewhat large small size
  // buffers (16) as there is no real size pressure on these and its nice to
  // avoid heap allocation in the small cases.
  using OptionMapT =
      llvm::SmallDenseMap<llvm::StringRef, llvm::PointerIntPair<Arg*, 1, bool>,
                          16>;
  using SubcommandMapT = llvm::SmallDenseMap<llvm::StringRef, Command*, 16>;

  // This table is sized to be 128 so that it can hold ASCII characters. We
  // don't need any more than this and using a direct table indexed by the
  // character's numeric value makes for a convenient map.
  using ShortOptionTableT = std::array<OptionMapT::mapped_type*, 128>;

  auto PopulateMaps(const Command& command) -> void;

  auto SetOptionDefault(const Arg& option) -> void;

  auto ParseNegatedFlag(const Arg& flag, std::optional<llvm::StringRef> value)
      -> ErrorOr<Success>;
  auto ParseFlag(const Arg& flag, std::optional<llvm::StringRef> value)
      -> ErrorOr<Success>;
  auto ParseIntegerArgValue(const Arg& arg, llvm::StringRef value)
      -> ErrorOr<Success>;
  auto ParseStringArgValue(const Arg& arg, llvm::StringRef value)
      -> ErrorOr<Success>;
  auto ParseOneOfArgValue(const Arg& arg, llvm::StringRef value)
      -> ErrorOr<Success>;
  auto ParseArg(const Arg& arg, bool short_spelling,
                std::optional<llvm::StringRef> value, bool negated_name = false)
      -> ErrorOr<Success>;

  auto SplitValue(llvm::StringRef& unparsed_arg)
      -> std::optional<llvm::StringRef>;
  auto ParseLongOption(llvm::StringRef unparsed_arg) -> ErrorOr<Success>;
  auto ParseShortOptionSeq(llvm::StringRef unparsed_arg) -> ErrorOr<Success>;
  auto FinalizeParsedOptions() -> ErrorOr<Success>;

  auto ParsePositionalArg(llvm::StringRef unparsed_arg) -> ErrorOr<Success>;
  auto ParseSubcommand(llvm::StringRef unparsed_arg) -> ErrorOr<Success>;

  auto ParsePositionalSuffix(llvm::ArrayRef<llvm::StringRef> unparsed_args)
      -> ErrorOr<Success>;

  auto FinalizeParse() -> ErrorOr<ParseResult>;

  // When building a command, it registers arguments and potentially subcommands
  // that are meta actions to print things to standard out, so we build a meta
  // printer for that here.
  MetaPrinter meta_printer_;

  Command root_command_;

  const Command* command_;

  OptionMapT option_map_;
  ShortOptionTableT short_option_table_;
  SubcommandMapT subcommand_map_;

  int positional_arg_index_ = 0;
  bool appending_to_positional_arg_ = false;

  ActionT arg_meta_action_;
};

auto Parser::PopulateMaps(const Command& command) -> void {
  option_map_.clear();
  for (const auto& option : command.options) {
    option_map_.insert({option->info.name, {option.get(), false}});
  }
  short_option_table_.fill(nullptr);
  for (auto& map_entry : option_map_) {
    const Arg* option = map_entry.second.getPointer();
    if (option->info.short_name.empty()) {
      continue;
    }
    CARBON_CHECK(option->info.short_name.size() == 1,
                 "Short option names must have exactly one character.");
    unsigned char short_char = option->info.short_name[0];
    CARBON_CHECK(short_char < short_option_table_.size(),
                 "Short option name outside of the expected range.");
    short_option_table_[short_char] = &map_entry.second;
  }
  subcommand_map_.clear();
  for (const auto& subcommand : command.subcommands) {
    subcommand_map_.insert({subcommand->info.name, subcommand.get()});
  }
}

auto Parser::SetOptionDefault(const Arg& option) -> void {
  CARBON_CHECK(option.has_default, "No default value available!");
  switch (option.kind) {
    case Arg::Kind::Flag:
      *option.flag_storage = option.default_flag;
      break;
    case Arg::Kind::Integer:
      *option.integer_storage = option.default_integer;
      break;
    case Arg::Kind::String:
      *option.string_storage = option.default_string;
      break;
    case Arg::Kind::OneOf:
      option.default_action(option);
      break;
    case Arg::Kind::MetaActionOnly:
      CARBON_FATAL("Can't set a default value for a meta action!");
    case Arg::Kind::Invalid:
      CARBON_FATAL("Option configured without any action or kind!");
  }
}

auto Parser::ParseNegatedFlag(const Arg& flag,
                              std::optional<llvm::StringRef> value)
    -> ErrorOr<Success> {
  if (flag.kind != Arg::Kind::Flag) {
    return Error(
        "cannot use a negated flag name by prefixing it with `no-` when it "
        "isn't a boolean flag argument");
  }
  if (value) {
    return Error(
        "cannot specify a value when using a flag name prefixed with `no-` -- "
        "that prefix implies a value of `false`");
  }
  *flag.flag_storage = false;
  return Success();
}

auto Parser::ParseFlag(const Arg& flag, std::optional<llvm::StringRef> value)
    -> ErrorOr<Success> {
  CARBON_CHECK(flag.kind == Arg::Kind::Flag, "Incorrect kind: {0}", flag.kind);
  if (!value || *value == "true") {
    *flag.flag_storage = true;
  } else if (*value == "false") {
    *flag.flag_storage = false;
  } else {
    return Error(llvm::formatv(
        "invalid value specified for the boolean flag `--{0}`: {1}",
        flag.info.name, *value));
  }
  return Success();
}

auto Parser::ParseIntegerArgValue(const Arg& arg, llvm::StringRef value)
    -> ErrorOr<Success> {
  CARBON_CHECK(arg.kind == Arg::Kind::Integer, "Incorrect kind: {0}", arg.kind);
  int integer_value;
  // Note that this method returns *true* on error!
  if (value.getAsInteger(/*Radix=*/0, integer_value)) {
    return Error(llvm::formatv(
        "cannot parse value for option `--{0}` as an integer: {1}",
        arg.info.name, value));
  }
  if (!arg.is_append) {
    *arg.integer_storage = integer_value;
  } else {
    arg.integer_sequence->push_back(integer_value);
  }
  return Success();
}

auto Parser::ParseStringArgValue(const Arg& arg, llvm::StringRef value)
    -> ErrorOr<Success> {
  CARBON_CHECK(arg.kind == Arg::Kind::String, "Incorrect kind: {0}", arg.kind);
  if (!arg.is_append) {
    *arg.string_storage = value;
  } else {
    arg.string_sequence->push_back(value);
  }
  return Success();
}

auto Parser::ParseOneOfArgValue(const Arg& arg, llvm::StringRef value)
    -> ErrorOr<Success> {
  CARBON_CHECK(arg.kind == Arg::Kind::OneOf, "Incorrect kind: {0}", arg.kind);
  if (!arg.value_action(arg, value)) {
    RawStringOstream error;
    error << "option `--" << arg.info.name << "=";
    llvm::printEscapedString(value, error);
    error << "` has an invalid value `";
    llvm::printEscapedString(value, error);
    error << "`; valid values are: ";
    PrintListOfAlternatives(error, arg.value_strings,
                            [](llvm::StringRef x) { return x; });
    return Error(error.TakeStr());
  }
  return Success();
}

auto Parser::ParseArg(const Arg& arg, bool short_spelling,
                      std::optional<llvm::StringRef> value, bool negated_name)
    -> ErrorOr<Success> {
  // If this argument has a meta action, replace the current meta action with
  // it.
  if (arg.meta_action) {
    arg_meta_action_ = arg.meta_action;
  }

  // Boolean flags have special parsing logic.
  if (negated_name) {
    return ParseNegatedFlag(arg, value);
  }
  if (arg.kind == Arg::Kind::Flag) {
    return ParseFlag(arg, value);
  }

  std::string name;
  if (short_spelling) {
    name = llvm::formatv("`-{0}` (short for `--{1}`)", arg.info.short_name,
                         arg.info.name);
  } else {
    name = llvm::formatv("`--{0}`", arg.info.name);
  }

  if (!value) {
    // We can't have a positional argument without a value, so we know this is
    // an option and handle it as such.
    if (arg.kind == Arg::Kind::MetaActionOnly) {
      // Nothing further to do here, this is only a meta-action.
      return Success();
    }
    if (!arg.has_default) {
      return Error(llvm::formatv(
          "option {0} requires a value to be provided and none was", name));
    }
    SetOptionDefault(arg);
    return Success();
  }

  // There is a value to parse as part of the argument.
  switch (arg.kind) {
    case Arg::Kind::Integer:
      return ParseIntegerArgValue(arg, *value);
    case Arg::Kind::String:
      return ParseStringArgValue(arg, *value);
    case Arg::Kind::OneOf:
      return ParseOneOfArgValue(arg, *value);
    case Arg::Kind::MetaActionOnly:
      // TODO: Improve message.
      return Error(llvm::formatv(
          "option {0} cannot be used with a value, and '{1}' was provided",
          name, value));
    case Arg::Kind::Flag:
    case Arg::Kind::Invalid:
      CARBON_FATAL("Invalid kind!");
  }
}

auto Parser::SplitValue(llvm::StringRef& unparsed_arg)
    -> std::optional<llvm::StringRef> {
  // Split out a value if present.
  std::optional<llvm::StringRef> value;
  auto index = unparsed_arg.find('=');
  if (index != llvm::StringRef::npos) {
    value = unparsed_arg.substr(index + 1);
    unparsed_arg = unparsed_arg.substr(0, index);
  }
  return value;
}

auto Parser::ParseLongOption(llvm::StringRef unparsed_arg) -> ErrorOr<Success> {
  CARBON_CHECK(unparsed_arg.starts_with("--") && unparsed_arg.size() > 2,
               "Must only be called on a potential long option.");

  // Walk past the double dash.
  unparsed_arg = unparsed_arg.drop_front(2);
  bool negated_name = unparsed_arg.consume_front("no-");
  std::optional<llvm::StringRef> value = SplitValue(unparsed_arg);

  auto option_it = option_map_.find(unparsed_arg);
  if (option_it == option_map_.end()) {
    // TODO: Improve error.
    return Error(llvm::formatv("unknown option `--{0}{1}`",
                               negated_name ? "no-" : "", unparsed_arg));
  }

  // Mark this option as parsed.
  option_it->second.setInt(true);

  // Parse this specific option and any value.
  const Arg& option = *option_it->second.getPointer();
  return ParseArg(option, /*short_spelling=*/false, value, negated_name);
}

auto Parser::ParseShortOptionSeq(llvm::StringRef unparsed_arg)
    -> ErrorOr<Success> {
  CARBON_CHECK(unparsed_arg.starts_with("-") && unparsed_arg.size() > 1,
               "Must only be called on a potential short option sequence.");

  unparsed_arg = unparsed_arg.drop_front();
  std::optional<llvm::StringRef> value = SplitValue(unparsed_arg);
  if (value && unparsed_arg.size() != 1) {
    return Error(llvm::formatv(
        "cannot provide a value to the group of multiple short options "
        "`-{0}=...`; values must be provided to a single option, using "
        "either the short or long spelling",
        unparsed_arg));
  }

  for (unsigned char c : unparsed_arg) {
    auto* arg_entry =
        (c < short_option_table_.size()) ? short_option_table_[c] : nullptr;
    if (!arg_entry) {
      return Error(
          llvm::formatv("unknown short option `-{0}`", static_cast<char>(c)));
    }
    // Mark this argument as parsed.
    arg_entry->setInt(true);

    // Parse the argument, including the value if this is the last.
    const Arg& arg = *arg_entry->getPointer();
    CARBON_RETURN_IF_ERROR(ParseArg(arg, /*short_spelling=*/true, value));
  }
  return Success();
}

auto Parser::FinalizeParsedOptions() -> ErrorOr<Success> {
  llvm::SmallVector<const Arg*> missing_options;
  for (const auto& option_entry : option_map_) {
    const Arg* option = option_entry.second.getPointer();
    if (!option_entry.second.getInt()) {
      // If the argument has a default value and isn't a meta-action, we need to
      // act on that when it isn't passed.
      if (option->has_default && !option->meta_action) {
        SetOptionDefault(*option);
      }
      // Remember any missing required arguments, we'll diagnose those.
      if (option->is_required) {
        missing_options.push_back(option);
      }
    }
  }
  if (missing_options.empty()) {
    return Success();
  }

  // Sort the missing arguments by name to provide a stable and deterministic
  // error message. We know there can't be duplicate names because these came
  // from a may keyed on the name, so this provides a total ordering.
  llvm::sort(missing_options, [](const Arg* lhs, const Arg* rhs) {
    return lhs->info.name < rhs->info.name;
  });

  RawStringOstream error;
  error << "required options not provided: ";
  llvm::ListSeparator sep;
  for (const Arg* option : missing_options) {
    error << sep << "--" << option->info.name;
  }

  return Error(error.TakeStr());
}

auto Parser::ParsePositionalArg(llvm::StringRef unparsed_arg)
    -> ErrorOr<Success> {
  if (static_cast<size_t>(positional_arg_index_) >=
      command_->positional_args.size()) {
    return Error(llvm::formatv(
        "completed parsing all {0} configured positional arguments, and found "
        "an additional positional argument: `{1}`",
        command_->positional_args.size(), unparsed_arg));
  }

  const Arg& arg = *command_->positional_args[positional_arg_index_];

  // Mark that we'll keep appending here until a `--` marker. When already
  // appending this is redundant but harmless.
  appending_to_positional_arg_ = arg.is_append;
  if (!appending_to_positional_arg_) {
    // If we're not continuing to append to a current positional arg,
    // increment the positional arg index to find the next argument we
    // should use here.
    ++positional_arg_index_;
  }

  return ParseArg(arg, /*short_spelling=*/false, unparsed_arg);
}

auto Parser::ParseSubcommand(llvm::StringRef unparsed_arg) -> ErrorOr<Success> {
  auto subcommand_it = subcommand_map_.find(unparsed_arg);
  if (subcommand_it == subcommand_map_.end()) {
    RawStringOstream error;
    error << "invalid subcommand `" << unparsed_arg
          << "`; available subcommands: ";
    MetaPrinter(&error).PrintSubcommands(*command_);
    return Error(error.TakeStr());
  }

  // Before we recurse into the subcommand, verify that all the required
  // arguments for this command were in fact parsed.
  CARBON_RETURN_IF_ERROR(FinalizeParsedOptions());

  // Recurse into the subcommand, tracking the active command.
  command_ = subcommand_it->second;
  PopulateMaps(*command_);
  return Success();
}

auto Parser::FinalizeParse() -> ErrorOr<ParseResult> {
  // If an argument action is provided, we run that and consider the parse
  // meta-successful rather than verifying required arguments were provided and
  // the (sub)command action.
  if (arg_meta_action_) {
    arg_meta_action_();
    return ParseResult::MetaSuccess;
  }

  // Verify we're not missing any arguments.
  CARBON_RETURN_IF_ERROR(FinalizeParsedOptions());

  // If we were appending to a positional argument, mark that as complete.
  llvm::ArrayRef positional_args = command_->positional_args;
  if (appending_to_positional_arg_) {
    CARBON_CHECK(
        static_cast<size_t>(positional_arg_index_) < positional_args.size(),
        "Appending to a positional argument with an invalid index: {0}",
        positional_arg_index_);
    ++positional_arg_index_;
  }

  // See if any positional args are required and unparsed.
  auto unparsed_positional_args = positional_args.slice(positional_arg_index_);
  if (!unparsed_positional_args.empty()) {
    // There are un-parsed positional arguments, make sure they aren't required.
    const Arg& missing_arg = *unparsed_positional_args.front();
    if (missing_arg.is_required) {
      return Error(
          llvm::formatv("not all required positional arguments were provided; "
                        "first missing and required positional argument: `{0}`",
                        missing_arg.info.name));
    }
    for (const auto& arg_ptr : unparsed_positional_args) {
      CARBON_CHECK(
          !arg_ptr->is_required,
          "Cannot have required positional parameters after an optional one.");
    }
  }

  switch (command_->kind) {
    case Command::Kind::Invalid:
      CARBON_FATAL("Should never have a parser with an invalid command!");
    case Command::Kind::RequiresSubcommand: {
      RawStringOstream error;
      error << "no subcommand specified; available subcommands: ";
      MetaPrinter(&error).PrintSubcommands(*command_);
      return Error(error.TakeStr());
    }
    case Command::Kind::Action:
      // All arguments have been successfully parsed, run any action for the
      // most specific selected command. Only the leaf command's action is run.
      command_->action();
      return ParseResult::Success;
    case Command::Kind::MetaAction:
      command_->action();
      return ParseResult::MetaSuccess;
  }
}

auto Parser::ParsePositionalSuffix(
    llvm::ArrayRef<llvm::StringRef> unparsed_args) -> ErrorOr<Success> {
  CARBON_CHECK(
      !command_->positional_args.empty(),
      "Cannot do positional suffix parsing without positional arguments!");
  CARBON_CHECK(
      !unparsed_args.empty() && unparsed_args.front() == "--",
      "Must be called with a suffix of arguments starting with a `--` that "
      "switches to positional suffix parsing.");
  // Once we're in the positional suffix, we can track empty positional
  // arguments.
  bool empty_positional = false;
  while (!unparsed_args.empty()) {
    llvm::StringRef unparsed_arg = unparsed_args.front();
    unparsed_args = unparsed_args.drop_front();

    if (unparsed_arg != "--") {
      CARBON_RETURN_IF_ERROR(ParsePositionalArg(unparsed_arg));
      empty_positional = false;
      continue;
    }

    if (appending_to_positional_arg_ || empty_positional) {
      ++positional_arg_index_;
      if (static_cast<size_t>(positional_arg_index_) >=
          command_->positional_args.size()) {
        return Error(
            llvm::formatv("completed parsing all {0} configured positional "
                          "arguments, but found a subsequent `--` and have no "
                          "further positional arguments to parse beyond it",
                          command_->positional_args.size()));
      }
    }
    appending_to_positional_arg_ = false;
    empty_positional = true;
  }

  return Success();
}

Parser::Parser(llvm::raw_ostream* out, CommandInfo command_info,
               llvm::function_ref<auto(CommandBuilder&)->void> build)
    : meta_printer_(out), root_command_(command_info) {
  // Run the command building lambda on a builder for the root command.
  CommandBuilder builder(&root_command_, &meta_printer_);
  build(builder);
  builder.Finalize();
  command_ = &root_command_;
}

auto Parser::Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args)
    -> ErrorOr<ParseResult> {
  PopulateMaps(*command_);

  while (!unparsed_args.empty()) {
    llvm::StringRef unparsed_arg = unparsed_args.front();

    // Peak at the front for an exact `--` argument that switches to a
    // positional suffix parsing without dropping this argument.
    if (unparsed_arg == "--") {
      if (command_->positional_args.empty()) {
        return Error(
            "cannot meaningfully end option and subcommand arguments with a "
            "`--` argument when there are no positional arguments to parse");
      }
      if (static_cast<size_t>(positional_arg_index_) >=
          command_->positional_args.size()) {
        return Error(
            "switched to purely positional arguments with a `--` argument "
            "despite already having parsed all positional arguments for this "
            "command");
      }
      CARBON_RETURN_IF_ERROR(ParsePositionalSuffix(unparsed_args));
      // No more unparsed arguments to handle.
      break;
    }

    // Now that we're not switching parse modes, drop the current unparsed
    // argument and parse it.
    unparsed_args = unparsed_args.drop_front();

    if (unparsed_arg.starts_with("--")) {
      // Note that the exact argument "--" has been handled above already.
      CARBON_RETURN_IF_ERROR(ParseLongOption(unparsed_arg));
      continue;
    }

    if (unparsed_arg.starts_with("-") && unparsed_arg.size() > 1) {
      CARBON_RETURN_IF_ERROR(ParseShortOptionSeq(unparsed_arg));
      continue;
    }

    CARBON_CHECK(
        command_->positional_args.empty() || command_->subcommands.empty(),
        "Cannot have both positional arguments and subcommands!");
    if (command_->positional_args.empty() && command_->subcommands.empty()) {
      return Error(llvm::formatv(
          "found unexpected positional argument or subcommand: `{0}`",
          unparsed_arg));
    }

    if (!command_->positional_args.empty()) {
      CARBON_RETURN_IF_ERROR(ParsePositionalArg(unparsed_arg));
      continue;
    }
    CARBON_RETURN_IF_ERROR(ParseSubcommand(unparsed_arg));
  }

  return FinalizeParse();
}

auto ArgBuilder::Required(bool is_required) -> void {
  arg_->is_required = is_required;
}

auto ArgBuilder::HelpHidden(bool is_help_hidden) -> void {
  arg_->is_help_hidden = is_help_hidden;
}

ArgBuilder::ArgBuilder(Arg* arg) : arg_(arg) {}

auto FlagBuilder::Default(bool flag_value) -> void {
  arg()->has_default = true;
  arg()->default_flag = flag_value;
}

auto FlagBuilder::Set(bool* flag) -> void { arg()->flag_storage = flag; }

auto IntegerArgBuilder::Default(int integer_value) -> void {
  arg()->has_default = true;
  arg()->default_integer = integer_value;
}

auto IntegerArgBuilder::Set(int* integer) -> void {
  arg()->is_append = false;
  arg()->integer_storage = integer;
}

auto IntegerArgBuilder::Append(llvm::SmallVectorImpl<int>* sequence) -> void {
  arg()->is_append = true;
  arg()->integer_sequence = sequence;
}

auto StringArgBuilder::Default(llvm::StringRef string_value) -> void {
  arg()->has_default = true;
  arg()->default_string = string_value;
}

auto StringArgBuilder::Set(llvm::StringRef* string) -> void {
  arg()->is_append = false;
  arg()->string_storage = string;
}

auto StringArgBuilder::Append(llvm::SmallVectorImpl<llvm::StringRef>* sequence)
    -> void {
  arg()->is_append = true;
  arg()->string_sequence = sequence;
}

static auto IsValidName(llvm::StringRef name) -> bool {
  if (name.size() <= 1) {
    return false;
  }
  if (!llvm::isAlnum(name.front())) {
    return false;
  }
  if (!llvm::isAlnum(name.back())) {
    return false;
  }
  for (char c : name.drop_front().drop_back()) {
    if (c != '-' && c != '_' && !llvm::isAlnum(c)) {
      return false;
    }
  }
  // We disallow names starting with "no-" as we will parse those for boolean
  // flags.
  return !name.starts_with("no-");
}

auto CommandBuilder::AddFlag(const ArgInfo& info,
                             llvm::function_ref<auto(FlagBuilder&)->void> build)
    -> void {
  FlagBuilder builder(AddArgImpl(info, Arg::Kind::Flag));
  // All boolean flags have an implicit default of `false`, although it can be
  // overridden in the build callback.
  builder.Default(false);
  build(builder);
}

auto CommandBuilder::AddIntegerOption(
    const ArgInfo& info,
    llvm::function_ref<auto(IntegerArgBuilder&)->void> build) -> void {
  IntegerArgBuilder builder(AddArgImpl(info, Arg::Kind::Integer));
  build(builder);
}

auto CommandBuilder::AddStringOption(
    const ArgInfo& info,
    llvm::function_ref<auto(StringArgBuilder&)->void> build) -> void {
  StringArgBuilder builder(AddArgImpl(info, Arg::Kind::String));
  build(builder);
}

auto CommandBuilder::AddOneOfOption(
    const ArgInfo& info, llvm::function_ref<auto(OneOfArgBuilder&)->void> build)
    -> void {
  OneOfArgBuilder builder(AddArgImpl(info, Arg::Kind::OneOf));
  build(builder);
}

auto CommandBuilder::AddMetaActionOption(
    const ArgInfo& info, llvm::function_ref<auto(ArgBuilder&)->void> build)
    -> void {
  ArgBuilder builder(AddArgImpl(info, Arg::Kind::MetaActionOnly));
  build(builder);
}

auto CommandBuilder::AddIntegerPositionalArg(
    const ArgInfo& info,
    llvm::function_ref<auto(IntegerArgBuilder&)->void> build) -> void {
  AddPositionalArgImpl(info, Arg::Kind::Integer, [build](Arg& arg) {
    IntegerArgBuilder builder(&arg);
    build(builder);
  });
}

auto CommandBuilder::AddStringPositionalArg(
    const ArgInfo& info,
    llvm::function_ref<auto(StringArgBuilder&)->void> build) -> void {
  AddPositionalArgImpl(info, Arg::Kind::String, [build](Arg& arg) {
    StringArgBuilder builder(&arg);
    build(builder);
  });
}

auto CommandBuilder::AddOneOfPositionalArg(
    const ArgInfo& info, llvm::function_ref<auto(OneOfArgBuilder&)->void> build)
    -> void {
  AddPositionalArgImpl(info, Arg::Kind::OneOf, [build](Arg& arg) {
    OneOfArgBuilder builder(&arg);
    build(builder);
  });
}

auto CommandBuilder::AddSubcommand(
    const CommandInfo& info,
    llvm::function_ref<auto(CommandBuilder&)->void> build) -> void {
  CARBON_CHECK(IsValidName(info.name), "Invalid subcommand name: {0}",
               info.name);
  CARBON_CHECK(subcommand_names_.insert(info.name).second,
               "Added a duplicate subcommand: {0}", info.name);
  CARBON_CHECK(
      command_->positional_args.empty(),
      "Cannot add subcommands to a command with a positional argument.");

  command_->subcommands.emplace_back(new Command(info, command_));
  CommandBuilder builder(command_->subcommands.back().get(), meta_printer_);
  build(builder);
  builder.Finalize();
}

auto CommandBuilder::HelpHidden(bool is_help_hidden) -> void {
  command_->is_help_hidden = is_help_hidden;
}

auto CommandBuilder::RequiresSubcommand() -> void {
  CARBON_CHECK(!command_->subcommands.empty(),
               "Cannot require subcommands unless there are subcommands.");
  CARBON_CHECK(command_->positional_args.empty(),
               "Cannot require subcommands and have a positional argument.");
  CARBON_CHECK(command_->kind == Kind::Invalid,
               "Already established the kind of this command as: {0}",
               command_->kind);
  command_->kind = Kind::RequiresSubcommand;
}

auto CommandBuilder::Do(ActionT action) -> void {
  CARBON_CHECK(command_->kind == Kind::Invalid,
               "Already established the kind of this command as: {0}",
               command_->kind);
  command_->kind = Kind::Action;
  command_->action = std::move(action);
}

auto CommandBuilder::Meta(ActionT action) -> void {
  CARBON_CHECK(command_->kind == Kind::Invalid,
               "Already established the kind of this command as: {0}",
               command_->kind);
  command_->kind = Kind::MetaAction;
  command_->action = std::move(action);
}

CommandBuilder::CommandBuilder(Command* command, MetaPrinter* meta_printer)
    : command_(command), meta_printer_(meta_printer) {}

auto CommandBuilder::AddArgImpl(const ArgInfo& info, Arg::Kind kind) -> Arg* {
  CARBON_CHECK(IsValidName(info.name), "Invalid argument name: {0}", info.name);
  CARBON_CHECK(arg_names_.insert(info.name).second,
               "Added a duplicate argument name: {0}", info.name);

  command_->options.emplace_back(new Arg(info));
  Arg* arg = command_->options.back().get();
  arg->kind = kind;
  return arg;
}

auto CommandBuilder::AddPositionalArgImpl(
    const ArgInfo& info, Arg::Kind kind,
    llvm::function_ref<auto(Arg&)->void> build) -> void {
  CARBON_CHECK(IsValidName(info.name), "Invalid argument name: {0}", info.name);
  CARBON_CHECK(
      command_->subcommands.empty(),
      "Cannot add a positional argument to a command with subcommands.");

  command_->positional_args.emplace_back(new Arg(info));
  Arg& arg = *command_->positional_args.back();
  arg.kind = kind;
  build(arg);

  CARBON_CHECK(!arg.is_help_hidden,
               "Cannot have a help-hidden positional argument.");

  if (arg.is_required && command_->positional_args.size() > 1) {
    CARBON_CHECK((*std::prev(command_->positional_args.end(), 2))->is_required,
                 "A required positional argument cannot be added after an "
                 "optional one.");
  }
}

auto CommandBuilder::Finalize() -> void {
  meta_printer_->RegisterWithCommand(*command_, *this);
}

auto Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args,
           llvm::raw_ostream& out, CommandInfo command_info,
           llvm::function_ref<auto(CommandBuilder&)->void> build)
    -> ErrorOr<ParseResult> {
  // Build a parser, which includes building the command description provided by
  // the user.
  Parser parser(&out, command_info, build);

  // Now parse the arguments provided using that parser.
  return parser.Parse(unparsed_args);
}

}  // namespace Carbon::CommandLine

// NOLINTEND(misc-no-recursion)
