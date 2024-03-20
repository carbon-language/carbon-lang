// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/command_line.h"

#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon::CommandLine {

auto operator<<(llvm::raw_ostream& output, ParseResult result)
    -> llvm::raw_ostream& {
  switch (result) {
    case ParseResult::Error:
      return output << "Error";
    case ParseResult::MetaSuccess:
      return output << "MetaSuccess";
    case ParseResult::Success:
      return output << "Success";
  }
  CARBON_FATAL() << "Corrupt parse result!";
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
  CARBON_FATAL() << "Corrupt argument kind!";
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
  CARBON_FATAL() << "Corrupt command kind!";
}
Arg::Arg(const ArgInfo& info) : info(info) {}

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

Command::Command(const CommandInfo& info, Command* parent)
    : info(info), parent(parent) {}

class MetaPrinter {
 public:
  explicit MetaPrinter(llvm::raw_ostream& out) : out_(out) {}

  // Registers this meta printer with a command through the provided builder.
  //
  // This adds meta subcommands or options to print both help and version
  // information for the command.
  void RegisterWithCommand(const Command& command, CommandBuilder& builder);

  void PrintHelp(const Command& command) const;
  void PrintHelpForSubcommandName(const Command& command,
                                  llvm::StringRef subcommand_name) const;
  void PrintVersion(const Command& command) const;
  void PrintSubcommands(const Command& command) const;

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
  void PrintTextBlock(llvm::StringRef indent, llvm::StringRef text) const;

  // Helpers for version and build information printing.
  void PrintRawVersion(const Command& command, llvm::StringRef indent) const;
  void PrintRawBuildInfo(const Command& command, llvm::StringRef indent) const;

  // Helpers for printing components of help and usage output for arguments,
  // including options and positional arguments.
  void PrintArgValueUsage(const Arg& arg) const;
  void PrintOptionUsage(const Arg& option) const;
  void PrintOptionShortName(const Arg& arg) const;
  void PrintArgShortValues(const Arg& arg) const;
  void PrintArgLongValues(const Arg& arg, llvm::StringRef indent) const;
  void PrintArgHelp(const Arg& arg, llvm::StringRef indent) const;

  // Helpers for printing command usage summaries.
  void PrintRawUsageCommandAndOptions(
      const Command& command,
      int max_option_width = MaxLeafOptionUsageWidth) const;
  void PrintRawUsage(const Command& command, llvm::StringRef indent) const;
  void PrintUsage(const Command& command) const;

  // Helpers to print various sections of `PrintHelp` that only occur within
  // that output.
  void PrintHelpSubcommands(const Command& command) const;
  void PrintHelpPositionalArgs(const Command& command) const;
  void PrintHelpOptions(const Command& command) const;

  llvm::raw_ostream& out_;

  // A flag that may be configured during command line parsing to select between
  // long and short form help output.
  bool short_help_ = false;

  // The requested subcommand to print help information for.
  llvm::StringRef help_subcommand_;
};

void MetaPrinter::RegisterWithCommand(const Command& command,
                                      CommandBuilder& builder) {
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

void MetaPrinter::PrintHelp(const Command& command) const {
  // TODO: begin using the short setting to customize the output.
  (void)short_help_;

  const CommandInfo& info = command.info;
  if (!info.version.empty()) {
    // We use the version string as a header for the command help when present.
    PrintRawVersion(command, /*indent=*/"");
    out_ << "\n";
  }
  if (!command.info.help.empty()) {
    PrintTextBlock("", info.help);
    out_ << "\n";
  }
  if (!info.build_info.empty()) {
    out_ << "Build info:\n";
    PrintRawBuildInfo(command, /*indent=*/"  ");
    out_ << "\n";
  }

  PrintUsage(command);
  PrintHelpSubcommands(command);
  PrintHelpPositionalArgs(command);
  PrintHelpOptions(command);

  if (!info.help_epilogue.empty()) {
    out_ << "\n";
    PrintTextBlock("", info.help_epilogue);
  }

  // End with a blank line for the long help to make it easier to separate from
  // anything that follows in the shell.
  out_ << "\n";
}

void MetaPrinter::PrintHelpForSubcommandName(
    const Command& command, llvm::StringRef subcommand_name) const {
  for (const auto& subcommand : command.subcommands) {
    if (subcommand->info.name == subcommand_name) {
      PrintHelp(*subcommand);
      return;
    }
  }

  out_ << "ERROR: Could not find a subcommand named '" << subcommand_name
       << "'.\n";
}

void MetaPrinter::PrintVersion(const Command& command) const {
  CARBON_CHECK(!command.info.version.empty())
      << "Printing should not be enabled without a version string configured.";
  PrintRawVersion(command, /*indent=*/"");
  if (!command.info.build_info.empty()) {
    out_ << "\n";
    // If there is build info to print, we also render that without any indent.
    PrintRawBuildInfo(command, /*indent=*/"");
  }
}

void MetaPrinter::PrintSubcommands(const Command& command) const {
  for (const auto& subcommand :
       llvm::ArrayRef(command.subcommands).drop_back()) {
    out_ << "'" << subcommand->info.name << "', ";
  }
  if (command.subcommands.size() > 1) {
    out_ << "or ";
  }
  out_ << "'" << command.subcommands.back()->info.name << "'";
}

void MetaPrinter::PrintRawVersion(const Command& command,
                                  llvm::StringRef indent) const {
  // Newlines are trimmed from the version string an a closing newline added but
  // no other formatting is performed.
  out_ << indent << command.info.version.trim('\n') << "\n";
}
void MetaPrinter::PrintRawBuildInfo(const Command& command,
                                    llvm::StringRef indent) const {
  // Print the build info line-by-line without any wrapping in case it
  // contains line-oriented formatted text, but drop leading and trailing blank
  // lines.
  llvm::SmallVector<llvm::StringRef, 128> lines;
  command.info.build_info.trim('\n').split(lines, "\n");
  for (auto line : lines) {
    out_ << indent << line << "\n";
  }
}

void MetaPrinter::PrintTextBlock(llvm::StringRef indent,
                                 llvm::StringRef text) const {
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
      out_ << "\n";
      ++i;
      continue;
    }

    if (input_lines[i].starts_with("```")) {
      // Fenced regions are preserved verbatim.
      llvm::StringRef fence =
          input_lines[i].slice(0, input_lines[i].find_first_not_of("`"));
      do {
        out_ << indent << input_lines[i] << "\n";
        ++i;
      } while (i < size && !input_lines[i].starts_with(fence));
      if (i >= size) {
        // Don't error on malformed text blocks, just print what we've got.
        break;
      }
      // Including the close of the fence.
      out_ << indent << input_lines[i] << "\n";
      ++i;
      continue;
    }

    if (input_lines[i].starts_with("    ")) {
      // Indented code blocks ar preserved verbatim, but we don't support tabs
      // in the indent for simplicity.
      do {
        out_ << indent << input_lines[i] << "\n";
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
      out_ << space << input_lines[i].trim();
      space = " ";
      ++i;
    } while (i < size && !input_lines[i].empty());
    out_ << "\n";
  }
}

void MetaPrinter::PrintArgValueUsage(const Arg& arg) const {
  if (!arg.info.value_name.empty()) {
    out_ << arg.info.value_name;
    return;
  }
  if (arg.kind == Arg::Kind::OneOf) {
    out_ << "(";
    llvm::ListSeparator sep("|");
    for (llvm::StringRef value_string : arg.value_strings) {
      out_ << sep << value_string;
    }
    out_ << ")";
    return;
  }
  out_ << "...";
}

void MetaPrinter::PrintOptionUsage(const Arg& option) const {
  if (option.kind == Arg::Kind::Flag) {
    out_ << "--" << (option.default_flag ? "no-" : "") << option.info.name;
    return;
  }
  out_ << "--" << option.info.name;
  if (option.kind != Arg::Kind::MetaActionOnly) {
    out_ << (option.has_default ? "[" : "") << "=";
    PrintArgValueUsage(option);
    if (option.has_default) {
      out_ << "]";
    }
  }
}

void MetaPrinter::PrintOptionShortName(const Arg& arg) const {
  CARBON_CHECK(!arg.info.short_name.empty()) << "No short name to use.";
  out_ << "-" << arg.info.short_name;
}

void MetaPrinter::PrintArgShortValues(const Arg& arg) const {
  CARBON_CHECK(arg.kind == Arg::Kind::OneOf)
      << "Only one-of arguments have interesting value snippets to print.";
  llvm::ListSeparator sep;
  for (llvm::StringRef value_string : arg.value_strings) {
    out_ << sep << value_string;
  }
}
void MetaPrinter::PrintArgLongValues(const Arg& arg,
                                     llvm::StringRef indent) const {
  out_ << indent << "Possible values:\n";
  // TODO: It would be good to add help text for each value and then print it
  // here.
  for (int i : llvm::seq<int>(0, arg.value_strings.size())) {
    llvm::StringRef value_string = arg.value_strings[i];
    out_ << indent << "- " << value_string;
    if (arg.has_default && i == arg.default_value_index) {
      out_ << " (default)";
    }
    out_ << "\n";
  }
}

void MetaPrinter::PrintArgHelp(const Arg& arg, llvm::StringRef indent) const {
  // Print out the main help text.
  PrintTextBlock(indent, arg.info.help);

  // Then print out any help based on the values.
  switch (arg.kind) {
    case Arg::Kind::Integer:
      if (arg.has_default) {
        out_ << "\n";
        out_ << indent << "Default value: " << arg.default_integer << "\n";
      }
      break;
    case Arg::Kind::String:
      if (arg.has_default) {
        out_ << "\n";
        out_ << indent << "Default value: " << arg.default_string << "\n";
      }
      break;
    case Arg::Kind::OneOf:
      out_ << "\n";
      PrintArgLongValues(arg, indent);
      break;
    case Arg::Kind::Flag:
    case Arg::Kind::MetaActionOnly:
      // No value help.
      break;
    case Arg::Kind::Invalid:
      CARBON_FATAL() << "Argument configured without any action or kind!";
  }
}

void MetaPrinter::PrintRawUsageCommandAndOptions(const Command& command,
                                                 int max_option_width) const {
  // Recursively print parent usage first with a compressed width.
  if (command.parent) {
    PrintRawUsageCommandAndOptions(*command.parent, MaxParentOptionUsageWidth);
    out_ << " ";
  }

  out_ << command.info.name;

  // Buffer the options rendering so we can limit its length.
  std::string buffer_str;
  llvm::raw_string_ostream buffer_out(buffer_str);
  MetaPrinter buffer_printer(buffer_out);
  bool have_short_flags = false;
  for (const auto& arg : command.options) {
    if (static_cast<int>(buffer_str.size()) > max_option_width) {
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
    if (static_cast<int>(buffer_str.size()) > max_option_width) {
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
  if (!buffer_str.empty()) {
    if (static_cast<int>(buffer_str.size()) <= max_option_width) {
      out_ << " [" << buffer_str << "]";
    } else {
      out_ << " [OPTIONS]";
    }
  }
}

void MetaPrinter::PrintRawUsage(const Command& command,
                                llvm::StringRef indent) const {
  if (!command.info.usage.empty()) {
    PrintTextBlock(indent, command.info.usage);
    return;
  }

  if (command.kind != Command::Kind::RequiresSubcommand) {
    // We're a valid leaf command, so synthesize a full usage line.
    out_ << indent;
    PrintRawUsageCommandAndOptions(command);

    if (!command.positional_args.empty()) {
      bool open_optional = false;
      for (int i : llvm::seq<int>(0, command.positional_args.size())) {
        out_ << " ";
        if (i != 0 && command.positional_args[i - 1]->is_append) {
          out_ << "-- ";
        }
        const auto& arg = command.positional_args[i];
        if (!arg->is_required && !open_optional) {
          out_ << "[";
          open_optional = true;
        }
        out_ << "<" << arg->info.name << ">";
        if (arg->is_append) {
          out_ << "...";
        }
      }
      if (open_optional) {
        out_ << "]";
      }
    }
    out_ << "\n";
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

void MetaPrinter::PrintUsage(const Command& command) const {
  if (!command.parent) {
    out_ << "Usage:\n";
  } else {
    out_ << "Subcommand '" << command.info.name << "' usage:\n";
  }
  PrintRawUsage(command, "  ");
}

void MetaPrinter::PrintHelpSubcommands(const Command& command) const {
  bool first_subcommand = true;
  for (const auto& subcommand : command.subcommands) {
    if (subcommand->is_help_hidden) {
      continue;
    }
    if (first_subcommand) {
      first_subcommand = false;
      if (!command.parent) {
        out_ << "\nSubcommands:";
      } else {
        out_ << "\nSubcommand '" << command.info.name << "' subcommands:";
      }
    }
    out_ << "\n";
    out_ << "  " << subcommand->info.name << "\n";
    PrintTextBlock(BlockIndent, subcommand->info.help);
  }
}

void MetaPrinter::PrintHelpPositionalArgs(const Command& command) const {
  bool first_positional_arg = true;
  for (const auto& positional_arg : command.positional_args) {
    if (positional_arg->is_help_hidden) {
      continue;
    }
    if (first_positional_arg) {
      first_positional_arg = false;
      if (!command.parent) {
        out_ << "\nPositional arguments:";
      } else {
        out_ << "\nSubcommand '" << command.info.name
             << "' positional arguments:";
      }
    }
    out_ << "\n";
    out_ << "  " << positional_arg->info.name << "\n";
    PrintArgHelp(*positional_arg, BlockIndent);
  }
}

void MetaPrinter::PrintHelpOptions(const Command& command) const {
  bool first_option = true;
  for (const auto& option : command.options) {
    if (option->is_help_hidden) {
      continue;
    }
    if (first_option) {
      first_option = false;
      if (!command.parent && command.subcommands.empty()) {
        // Only one command level.
        out_ << "\nOptions:";
      } else if (!command.parent) {
        out_ << "\nCommand options:";
      } else {
        out_ << "\nSubcommand '" << command.info.name << "' options:";
      }
    }
    out_ << "\n";
    out_ << "  ";
    if (!option->info.short_name.empty()) {
      PrintOptionShortName(*option);
      out_ << ", ";
    } else {
      out_ << "    ";
    }
    PrintOptionUsage(*option);
    out_ << "\n";
    PrintArgHelp(*option, BlockIndent);
  }
}

class Parser {
 public:
  explicit Parser(llvm::raw_ostream& out, llvm::raw_ostream& errors,
                  const CommandInfo& command_info,
                  llvm::function_ref<void(CommandBuilder&)> build);

  auto Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args) -> ParseResult;

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

  void PopulateMaps(const Command& command);

  void SetOptionDefault(const Arg& option);

  auto ParseNegatedFlag(const Arg& flag, std::optional<llvm::StringRef> value)
      -> bool;
  auto ParseFlag(const Arg& flag, std::optional<llvm::StringRef> value) -> bool;
  auto ParseIntegerArgValue(const Arg& arg, llvm::StringRef value) -> bool;
  auto ParseStringArgValue(const Arg& arg, llvm::StringRef value) -> bool;
  auto ParseOneOfArgValue(const Arg& arg, llvm::StringRef value) -> bool;
  auto ParseArg(const Arg& arg, bool short_spelling,
                std::optional<llvm::StringRef> value, bool negated_name = false)
      -> bool;

  auto SplitValue(llvm::StringRef& unparsed_arg)
      -> std::optional<llvm::StringRef>;
  auto ParseLongOption(llvm::StringRef unparsed_arg) -> bool;
  auto ParseShortOptionSeq(llvm::StringRef unparsed_arg) -> bool;
  auto FinalizeParsedOptions() -> bool;

  auto ParsePositionalArg(llvm::StringRef unparsed_arg) -> bool;
  auto ParseSubcommand(llvm::StringRef unparsed_arg) -> bool;

  auto ParsePositionalSuffix(llvm::ArrayRef<llvm::StringRef> unparsed_args)
      -> bool;

  auto FinalizeParse() -> ParseResult;

  // When building a command, it registers arguments and potentially subcommands
  // that are meta actions to print things to standard out, so we build a meta
  // printer for that here.
  MetaPrinter meta_printer_;

  // Most parsing output goes to an error stream, and we also provide an
  // error-oriented meta printer for when that is useful during parsing.
  llvm::raw_ostream& errors_;
  MetaPrinter error_meta_printer_;

  Command root_command_;

  const Command* command_;

  OptionMapT option_map_;
  ShortOptionTableT short_option_table_;
  SubcommandMapT subcommand_map_;

  int positional_arg_index_ = 0;
  bool appending_to_positional_arg_ = false;

  ActionT arg_meta_action_;
};

void Parser::PopulateMaps(const Command& command) {
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
    CARBON_CHECK(option->info.short_name.size() == 1)
        << "Short option names must have exactly one character.";
    unsigned char short_char = option->info.short_name[0];
    CARBON_CHECK(short_char < short_option_table_.size())
        << "Short option name outside of the expected range.";
    short_option_table_[short_char] = &map_entry.second;
  }
  subcommand_map_.clear();
  for (const auto& subcommand : command.subcommands) {
    subcommand_map_.insert({subcommand->info.name, subcommand.get()});
  }
}

void Parser::SetOptionDefault(const Arg& option) {
  CARBON_CHECK(option.has_default) << "No default value available!";
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
      CARBON_FATAL() << "Can't set a default value for a meta action!";
    case Arg::Kind::Invalid:
      CARBON_FATAL() << "Option configured without any action or kind!";
  }
}

auto Parser::ParseNegatedFlag(const Arg& flag,
                              std::optional<llvm::StringRef> value) -> bool {
  if (flag.kind != Arg::Kind::Flag) {
    errors_ << "ERROR: Cannot use a negated flag name by prefixing it with "
               "'no-' when it isn't a boolean flag argument.\n";
    return false;
  }
  if (value) {
    errors_ << "ERROR: Cannot specify a value when using a flag name prefixed "
               "with 'no-' -- that prefix implies a value of 'false'.\n";
    return false;
  }
  *flag.flag_storage = false;
  return true;
}

auto Parser::ParseFlag(const Arg& flag, std::optional<llvm::StringRef> value)
    -> bool {
  CARBON_CHECK(flag.kind == Arg::Kind::Flag) << "Incorrect kind: " << flag.kind;
  if (!value || *value == "true") {
    *flag.flag_storage = true;
  } else if (*value == "false") {
    *flag.flag_storage = false;
  } else {
    errors_ << "ERROR: Invalid value specified for the boolean flag '--"
            << flag.info.name << "': " << *value << "\n";

    return false;
  }
  return true;
}

auto Parser::ParseIntegerArgValue(const Arg& arg, llvm::StringRef value)
    -> bool {
  CARBON_CHECK(arg.kind == Arg::Kind::Integer)
      << "Incorrect kind: " << arg.kind;
  int integer_value;
  // Note that this method returns *true* on error!
  if (value.getAsInteger(/*Radix=*/0, integer_value)) {
    errors_ << "ERROR: Cannot parse value for option '--" << arg.info.name
            << "' as an integer: " << value << "\n";
    return false;
  }
  if (!arg.is_append) {
    *arg.integer_storage = integer_value;
  } else {
    arg.integer_sequence->push_back(integer_value);
  }
  return true;
}

auto Parser::ParseStringArgValue(const Arg& arg, llvm::StringRef value)
    -> bool {
  CARBON_CHECK(arg.kind == Arg::Kind::String) << "Incorrect kind: " << arg.kind;
  if (!arg.is_append) {
    *arg.string_storage = value;
  } else {
    arg.string_sequence->push_back(value);
  }
  return true;
}

auto Parser::ParseOneOfArgValue(const Arg& arg, llvm::StringRef value) -> bool {
  CARBON_CHECK(arg.kind == Arg::Kind::OneOf) << "Incorrect kind: " << arg.kind;
  if (!arg.value_action(arg, value)) {
    errors_ << "ERROR: Option '--" << arg.info.name << "=";
    llvm::printEscapedString(value, errors_);
    errors_ << "' has an invalid value '";
    llvm::printEscapedString(value, errors_);
    errors_ << "'; valid values are: ";
    for (auto value_string : arg.value_strings.drop_back()) {
      errors_ << "'" << value_string << "', ";
    }
    if (arg.value_strings.size() > 1) {
      errors_ << "or ";
    }
    errors_ << "'" << arg.value_strings.back() << "'\n";
    return false;
  }
  return true;
}

auto Parser::ParseArg(const Arg& arg, bool short_spelling,
                      std::optional<llvm::StringRef> value, bool negated_name)
    -> bool {
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

  auto name =
      llvm::formatv(short_spelling ? "'-{0}' (short for '--{1}')" : "'--{1}'",
                    arg.info.short_name, arg.info.name);

  if (!value) {
    // We can't have a positional argument without a value, so we know this is
    // an option and handle it as such.
    if (arg.kind == Arg::Kind::MetaActionOnly) {
      // Nothing further to do here, this is only a meta-action.
      return true;
    }
    if (!arg.has_default) {
      errors_ << "ERROR: Option " << name
              << " requires a value to be provided and none was.\n";
      return false;
    }
    SetOptionDefault(arg);
    return true;
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
      errors_ << "ERROR: Option " << name
              << " cannot be used with a value, and '" << *value
              << "' was provided.\n";
      // TODO: improve message
      return false;
    case Arg::Kind::Flag:
    case Arg::Kind::Invalid:
      CARBON_FATAL() << "Invalid kind!";
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

auto Parser::ParseLongOption(llvm::StringRef unparsed_arg) -> bool {
  CARBON_CHECK(unparsed_arg.starts_with("--") && unparsed_arg.size() > 2)
      << "Must only be called on a potential long option.";

  // Walk past the double dash.
  unparsed_arg = unparsed_arg.drop_front(2);
  bool negated_name = unparsed_arg.consume_front("no-");
  std::optional<llvm::StringRef> value = SplitValue(unparsed_arg);

  auto option_it = option_map_.find(unparsed_arg);
  if (option_it == option_map_.end()) {
    errors_ << "ERROR: Unknown option '--" << (negated_name ? "no-" : "")
            << unparsed_arg << "'\n";
    // TODO: improve error
    return false;
  }

  // Mark this option as parsed.
  option_it->second.setInt(true);

  // Parse this specific option and any value.
  const Arg& option = *option_it->second.getPointer();
  return ParseArg(option, /*short_spelling=*/false, value, negated_name);
}

auto Parser::ParseShortOptionSeq(llvm::StringRef unparsed_arg) -> bool {
  CARBON_CHECK(unparsed_arg.starts_with("-") && unparsed_arg.size() > 1)
      << "Must only be called on a potential short option sequence.";

  unparsed_arg = unparsed_arg.drop_front();
  std::optional<llvm::StringRef> value = SplitValue(unparsed_arg);
  if (value && unparsed_arg.size() != 1) {
    errors_ << "ERROR: Cannot provide a value to the group of multiple short "
               "options '-"
            << unparsed_arg
            << "=...'; values must be provided to a single option, using "
               "either the short or long spelling.\n";
    return false;
  }

  for (unsigned char c : unparsed_arg) {
    auto* arg_entry =
        (c < short_option_table_.size()) ? short_option_table_[c] : nullptr;
    if (!arg_entry) {
      errors_ << "ERROR: Unknown short option '" << c << "'\n";
      return false;
    }
    // Mark this argument as parsed.
    arg_entry->setInt(true);

    // Parse the argument, including the value if this is the last.
    const Arg& arg = *arg_entry->getPointer();
    if (!ParseArg(arg, /*short_spelling=*/true, value)) {
      return false;
    }
  }
  return true;
}

auto Parser::FinalizeParsedOptions() -> bool {
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
    return true;
  }

  // Sort the missing arguments by name to provide a stable and deterministic
  // error message. We know there can't be duplicate names because these came
  // from a may keyed on the name, so this provides a total ordering.
  std::sort(missing_options.begin(), missing_options.end(),
            [](const Arg* lhs, const Arg* rhs) {
              return lhs->info.name < rhs->info.name;
            });

  for (const Arg* option : missing_options) {
    errors_ << "ERROR: Required option '--" << option->info.name
            << "' not provided.\n";
  }

  return false;
}

auto Parser::ParsePositionalArg(llvm::StringRef unparsed_arg) -> bool {
  if (static_cast<size_t>(positional_arg_index_) >=
      command_->positional_args.size()) {
    errors_ << "ERROR: Completed parsing all "
            << command_->positional_args.size()
            << " configured positional arguments, and found an additional "
               "positional argument: '"
            << unparsed_arg << "'\n";
    return false;
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

auto Parser::ParseSubcommand(llvm::StringRef unparsed_arg) -> bool {
  auto subcommand_it = subcommand_map_.find(unparsed_arg);
  if (subcommand_it == subcommand_map_.end()) {
    errors_ << "ERROR: Invalid subcommand '" << unparsed_arg
            << "'. Available subcommands: ";
    error_meta_printer_.PrintSubcommands(*command_);
    errors_ << "\n";
    return false;
  }

  // Before we recurse into the subcommand, verify that all the required
  // arguments for this command were in fact parsed.
  if (!FinalizeParsedOptions()) {
    return false;
  }

  // Recurse into the subcommand, tracking the active command.
  command_ = subcommand_it->second;
  PopulateMaps(*command_);
  return true;
}

auto Parser::FinalizeParse() -> ParseResult {
  // If an argument action is provided, we run that and consider the parse
  // meta-successful rather than verifying required arguments were provided and
  // the (sub)command action.
  if (arg_meta_action_) {
    arg_meta_action_();
    return ParseResult::MetaSuccess;
  }

  // Verify we're not missing any arguments.
  if (!FinalizeParsedOptions()) {
    return ParseResult::Error;
  }

  // If we were appending to a positional argument, mark that as complete.
  llvm::ArrayRef positional_args = command_->positional_args;
  if (appending_to_positional_arg_) {
    CARBON_CHECK(static_cast<size_t>(positional_arg_index_) <
                 positional_args.size())
        << "Appending to a positional argument with an invalid index: "
        << positional_arg_index_;
    ++positional_arg_index_;
  }

  // See if any positional args are required and unparsed.
  auto unparsed_positional_args = positional_args.slice(positional_arg_index_);
  if (!unparsed_positional_args.empty()) {
    // There are un-parsed positional arguments, make sure they aren't required.
    const Arg& missing_arg = *unparsed_positional_args.front();
    if (missing_arg.is_required) {
      errors_ << "ERROR: Not all required positional arguments were provided. "
                 "First missing and required positional argument: '"
              << missing_arg.info.name << "'\n";
      return ParseResult::Error;
    }
    for (const auto& arg_ptr : unparsed_positional_args) {
      CARBON_CHECK(!arg_ptr->is_required)
          << "Cannot have required positional parameters after an optional "
             "one.";
    }
  }

  switch (command_->kind) {
    case Command::Kind::Invalid:
      CARBON_FATAL() << "Should never have a parser with an invalid command!";
    case Command::Kind::RequiresSubcommand:
      errors_ << "ERROR: No subcommand specified. Available subcommands: ";
      error_meta_printer_.PrintSubcommands(*command_);
      errors_ << "\n";
      return ParseResult::Error;
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
    llvm::ArrayRef<llvm::StringRef> unparsed_args) -> bool {
  CARBON_CHECK(!command_->positional_args.empty())
      << "Cannot do positional suffix parsing without positional arguments!";
  CARBON_CHECK(!unparsed_args.empty() && unparsed_args.front() == "--")
      << "Must be called with a suffix of arguments starting with a `--` that "
         "switches to positional suffix parsing.";
  // Once we're in the positional suffix, we can track empty positional
  // arguments.
  bool empty_positional = false;
  while (!unparsed_args.empty()) {
    llvm::StringRef unparsed_arg = unparsed_args.front();
    unparsed_args = unparsed_args.drop_front();

    if (unparsed_arg != "--") {
      if (!ParsePositionalArg(unparsed_arg)) {
        return false;
      }
      empty_positional = false;
      continue;
    }

    if (appending_to_positional_arg_ || empty_positional) {
      ++positional_arg_index_;
      if (static_cast<size_t>(positional_arg_index_) >=
          command_->positional_args.size()) {
        errors_
            << "ERROR: Completed parsing all "
            << command_->positional_args.size()
            << " configured positional arguments, but found a subsequent `--` "
               "and have no further positional arguments to parse beyond it.\n";
        return false;
      }
    }
    appending_to_positional_arg_ = false;
    empty_positional = true;
  }

  return true;
}

Parser::Parser(llvm::raw_ostream& out, llvm::raw_ostream& errors,
               const CommandInfo& command_info,
               llvm::function_ref<void(CommandBuilder&)> build)
    : meta_printer_(out),
      errors_(errors),
      error_meta_printer_(errors),
      root_command_(command_info) {
  // Run the command building lambda on a builder for the root command.
  CommandBuilder builder(root_command_, meta_printer_);
  build(builder);
  builder.Finalize();
  command_ = &root_command_;
}

auto Parser::Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args)
    -> ParseResult {
  PopulateMaps(*command_);

  while (!unparsed_args.empty()) {
    llvm::StringRef unparsed_arg = unparsed_args.front();

    // Peak at the front for an exact `--` argument that switches to a
    // positional suffix parsing without dropping this argument.
    if (unparsed_arg == "--") {
      if (command_->positional_args.empty()) {
        errors_ << "ERROR: Cannot meaningfully end option and subcommand "
                   "arguments with a `--` argument when there are no "
                   "positional arguments to parse.\n";
        return ParseResult::Error;
      }
      if (static_cast<size_t>(positional_arg_index_) >=
          command_->positional_args.size()) {
        errors_ << "ERROR: Switched to purely positional arguments with a `--` "
                   "argument despite already having parsed all positional "
                   "arguments for this command.\n";
        return ParseResult::Error;
      }
      if (!ParsePositionalSuffix(unparsed_args)) {
        return ParseResult::Error;
      }
      // No more unparsed arguments to handle.
      break;
    }

    // Now that we're not switching parse modes, drop the current unparsed
    // argument and parse it.
    unparsed_args = unparsed_args.drop_front();

    if (unparsed_arg.starts_with("--")) {
      // Note that the exact argument "--" has been handled above already.
      if (!ParseLongOption(unparsed_arg)) {
        return ParseResult::Error;
      }
      continue;
    }

    if (unparsed_arg.starts_with("-") && unparsed_arg.size() > 1) {
      if (!ParseShortOptionSeq(unparsed_arg)) {
        return ParseResult::Error;
      }
      continue;
    }

    CARBON_CHECK(command_->positional_args.empty() ||
                 command_->subcommands.empty())
        << "Cannot have both positional arguments and subcommands!";
    if (command_->positional_args.empty() && command_->subcommands.empty()) {
      errors_ << "ERROR: Found unexpected positional argument or subcommand: '"
              << unparsed_arg << "'\n";
      return ParseResult::Error;
    }

    if (!command_->positional_args.empty()) {
      if (!ParsePositionalArg(unparsed_arg)) {
        return ParseResult::Error;
      }
      continue;
    }
    if (!ParseSubcommand(unparsed_arg)) {
      return ParseResult::Error;
    }
  }

  return FinalizeParse();
}

void ArgBuilder::Required(bool is_required) { arg_.is_required = is_required; }

void ArgBuilder::HelpHidden(bool is_help_hidden) {
  arg_.is_help_hidden = is_help_hidden;
}

ArgBuilder::ArgBuilder(Arg& arg) : arg_(arg) {}

void FlagBuilder::Default(bool flag_value) {
  arg_.has_default = true;
  arg_.default_flag = flag_value;
}

void FlagBuilder::Set(bool* flag) { arg_.flag_storage = flag; }

void IntegerArgBuilder::Default(int integer_value) {
  arg_.has_default = true;
  arg_.default_integer = integer_value;
}

void IntegerArgBuilder::Set(int* integer) {
  arg_.is_append = false;
  arg_.integer_storage = integer;
}

void IntegerArgBuilder::Append(llvm::SmallVectorImpl<int>* sequence) {
  arg_.is_append = true;
  arg_.integer_sequence = sequence;
}

void StringArgBuilder::Default(llvm::StringRef string_value) {
  arg_.has_default = true;
  arg_.default_string = string_value;
}

void StringArgBuilder::Set(llvm::StringRef* string) {
  arg_.is_append = false;
  arg_.string_storage = string;
}

void StringArgBuilder::Append(
    llvm::SmallVectorImpl<llvm::StringRef>* sequence) {
  arg_.is_append = true;
  arg_.string_sequence = sequence;
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

void CommandBuilder::AddFlag(const ArgInfo& info,
                             llvm::function_ref<void(FlagBuilder&)> build) {
  FlagBuilder builder(AddArgImpl(info, Arg::Kind::Flag));
  // All boolean flags have an implicit default of `false`, although it can be
  // overridden in the build callback.
  builder.Default(false);
  build(builder);
}

void CommandBuilder::AddIntegerOption(
    const ArgInfo& info, llvm::function_ref<void(IntegerArgBuilder&)> build) {
  IntegerArgBuilder builder(AddArgImpl(info, Arg::Kind::Integer));
  build(builder);
}

void CommandBuilder::AddStringOption(
    const ArgInfo& info, llvm::function_ref<void(StringArgBuilder&)> build) {
  StringArgBuilder builder(AddArgImpl(info, Arg::Kind::String));
  build(builder);
}

void CommandBuilder::AddOneOfOption(
    const ArgInfo& info, llvm::function_ref<void(OneOfArgBuilder&)> build) {
  OneOfArgBuilder builder(AddArgImpl(info, Arg::Kind::OneOf));
  build(builder);
}

void CommandBuilder::AddMetaActionOption(
    const ArgInfo& info, llvm::function_ref<void(ArgBuilder&)> build) {
  ArgBuilder builder(AddArgImpl(info, Arg::Kind::MetaActionOnly));
  build(builder);
}

void CommandBuilder::AddIntegerPositionalArg(
    const ArgInfo& info, llvm::function_ref<void(IntegerArgBuilder&)> build) {
  AddPositionalArgImpl(info, Arg::Kind::Integer, [build](Arg& arg) {
    IntegerArgBuilder builder(arg);
    build(builder);
  });
}

void CommandBuilder::AddStringPositionalArg(
    const ArgInfo& info, llvm::function_ref<void(StringArgBuilder&)> build) {
  AddPositionalArgImpl(info, Arg::Kind::String, [build](Arg& arg) {
    StringArgBuilder builder(arg);
    build(builder);
  });
}

void CommandBuilder::AddOneOfPositionalArg(
    const ArgInfo& info, llvm::function_ref<void(OneOfArgBuilder&)> build) {
  AddPositionalArgImpl(info, Arg::Kind::OneOf, [build](Arg& arg) {
    OneOfArgBuilder builder(arg);
    build(builder);
  });
}

void CommandBuilder::AddSubcommand(
    const CommandInfo& info, llvm::function_ref<void(CommandBuilder&)> build) {
  CARBON_CHECK(IsValidName(info.name))
      << "Invalid subcommand name: " << info.name;
  CARBON_CHECK(subcommand_names_.insert(info.name).second)
      << "Added a duplicate subcommand: " << info.name;
  CARBON_CHECK(command_.positional_args.empty())
      << "Cannot add subcommands to a command with a positional argument.";

  command_.subcommands.emplace_back(new Command(info, &command_));
  CommandBuilder builder(*command_.subcommands.back(), meta_printer_);
  build(builder);
  builder.Finalize();
}

void CommandBuilder::HelpHidden(bool is_help_hidden) {
  command_.is_help_hidden = is_help_hidden;
}

void CommandBuilder::RequiresSubcommand() {
  CARBON_CHECK(!command_.subcommands.empty())
      << "Cannot require subcommands unless there are subcommands.";
  CARBON_CHECK(command_.positional_args.empty())
      << "Cannot require subcommands and have a positional argument.";
  CARBON_CHECK(command_.kind == Kind::Invalid)
      << "Already established the kind of this command as: " << command_.kind;
  command_.kind = Kind::RequiresSubcommand;
}

void CommandBuilder::Do(ActionT action) {
  CARBON_CHECK(command_.kind == Kind::Invalid)
      << "Already established the kind of this command as: " << command_.kind;
  command_.kind = Kind::Action;
  command_.action = std::move(action);
}

void CommandBuilder::Meta(ActionT action) {
  CARBON_CHECK(command_.kind == Kind::Invalid)
      << "Already established the kind of this command as: " << command_.kind;
  command_.kind = Kind::MetaAction;
  command_.action = std::move(action);
}

CommandBuilder::CommandBuilder(Command& command, MetaPrinter& meta_printer)
    : command_(command), meta_printer_(meta_printer) {}

auto CommandBuilder::AddArgImpl(const ArgInfo& info, Arg::Kind kind) -> Arg& {
  CARBON_CHECK(IsValidName(info.name))
      << "Invalid argument name: " << info.name;
  CARBON_CHECK(arg_names_.insert(info.name).second)
      << "Added a duplicate argument name: " << info.name;

  command_.options.emplace_back(new Arg(info));
  Arg& arg = *command_.options.back();
  arg.kind = kind;
  return arg;
}

void CommandBuilder::AddPositionalArgImpl(
    const ArgInfo& info, Arg::Kind kind, llvm::function_ref<void(Arg&)> build) {
  CARBON_CHECK(IsValidName(info.name))
      << "Invalid argument name: " << info.name;
  CARBON_CHECK(command_.subcommands.empty())
      << "Cannot add a positional argument to a command with subcommands.";

  command_.positional_args.emplace_back(new Arg(info));
  Arg& arg = *command_.positional_args.back();
  arg.kind = kind;
  build(arg);

  CARBON_CHECK(!arg.is_help_hidden)
      << "Cannot have a help-hidden positional argument.";

  if (arg.is_required && command_.positional_args.size() > 1) {
    CARBON_CHECK((*std::prev(command_.positional_args.end(), 2))->is_required)
        << "A required positional argument cannot be added after an optional "
           "one.";
  }
}

void CommandBuilder::Finalize() {
  meta_printer_.RegisterWithCommand(command_, *this);
}

auto Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args,
           llvm::raw_ostream& out, llvm::raw_ostream& errors,
           const CommandInfo& command_info,
           llvm::function_ref<void(CommandBuilder&)> build) -> ParseResult {
  // Build a parser, which includes building the command description provided by
  // the user.
  Parser parser(out, errors, command_info, build);

  // Now parse the arguments provided using that parser.
  return parser.Parse(unparsed_args);
}

}  // namespace Carbon::CommandLine
