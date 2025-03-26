// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_COMMAND_LINE_H_
#define CARBON_COMMON_COMMAND_LINE_H_

#include <memory>
#include <utility>

#include "common/check.h"
#include "common/error.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

// # Command-line argument parsing library.
//
// This is a collection of tools to describe both simple and reasonably complex
// command line interfaces, and parse arguments based on those descriptions. It
// optimizes for command line tools that will parse arguments exactly once per
// execution, and specifically tools that make heavy use of subcommand-style
// command line interfaces.
//
// ## Terminology used by this library
//
// _Argument_ or _arg_: One of the parsed components of the command line.
//
// _Option_: A _named_ argument starting with `--` to configure some aspect of
// the tool. These often have both long spellings that start with `--` and
// short, single character spellings that start with `-` and can be bundled with
// other single character options.
//
// _Flag_: A boolean or binary option. These can only be enabled or disabled.
//
// _Positional argument_: An argument that is not named but is identified based
// on the order in which it is encountered in the command line, with options
// removed. Only a leaf command can contain positional arguments.
//
// _Value_: The value parameter provided to an argument. For options, this is
// provided after an `=` and the option name. For positional arguments, the
// value is the only thing provided. The string of the value is parsed according
// to the kind of argument with fairly simple rules. See the argument builders
// for more details.
//
// _Command_: The container of options, subcommands, and positional arguments to
// parse and an action to take when successful.
//
// _Leaf command_: A command that doesn't contain subcommands. This is the only
// kind of command that can contain positional arguments.
//
// _Subcommand_: A command nested within another command and identified by a
// specific name that ends the parsing of arguments based on the parent and
// switches to parse based on the specific subcommand's options and positional
// arguments. A command with subcommands cannot parse positional arguments.
//
// _Action_: An open-ended callback, typically reflecting a specific subcommand
// being parsed. Can either directly perform the operation or simply mark which
// operation was selected.
//
// _Meta action_: An action fully handled by argument parsing such as displaying
// help, version information, or completion.
//
// Example command to illustrate the different components:
//
//     git --no-pager clone --shared --filter=blob:none my-repo my-directory
//
// This command breaks down as:
// - `git`: The top-level command.
// - `--no-pager`: A negated flag on the top-level command (`git`).
// - `clone`: A subcommand.
// - `--shared`: A positive flag for the subcommand (`clone`).
// - `--filter=blob:none`: An option named `filter` with a value `blob:none`.
// - `my-repo`: The first positional argument to the subcommand.
// - `my-directory`: the second positional argument to the subcommand.
//
// **Note:** while the example uses a `git` command to be relatively familiar
// and well documented, this library does not support the same flag syntaxes as
// `git` or use anything similar for its subcommand dispatch. This example is
// just to help clarify the terminology used, and carefully chosen to only use a
// syntax that overlaps with this library's parsed syntax.
//
// ## Options and flags
//
// The option syntax and behavior supported by this library is designed to be
// strict and relatively simple while still supporting a wide range of expected
// use cases:
//
// - All options must have a unique long name that is accessed with a `--`
//   prefix. The name must consist of characters in the set [-a-zA-Z0-9], and it
//   must not start with a `-` or `no-`.
//
// - Values are always attached using an `=` after the name. Only a few simple
//   value formats are supported:
//   - Arbitrary strings
//   - Integers as parsed by `llvm::StringRef` and whose value fits in an
//     `int`.
//   - One of a fixed set of strings
//
// - Options may be parsed multiple times, and the behavior can be configured:
//   - Each time, they can set a new value, overwriting any previous.
//   - They can append the value to a container.
//   - TODO: They can increment a count.
//
// - Options may have a default value that will be synthesized even if they do
//   not occur in the parsed command line.
//
// - Flags (boolean options) have some special rules.
//   - They may be spelled normally, and default to setting that flag to `true`.
//   - They may also accept a value of either exactly `true` or `false` and that
//     is the value.
//   - They may be spelled with a `no-` prefix, such as `--no-verbose`, and that
//     is exactly equivalent to `--verbose=false`.
//   - For flags with a default `true` value, they are rendered in help using
//     the `no-` prefix.
//
// - Options may additionally have a short name of a single character [a-zA-Z].
//   - There is no distinction between the behavior of long and short names.
//   - The short name can only specify the positive or `true` value for flags.
//     There is no negative form of short names.
//   - Short names are parsed after a single `-`, such as `-v`.
//   - Any number of short names for boolean flags or options with default
//     values may be grouped after `-`, such as `-xyz`: this is three options,
//     `x`, `y`, and `z`.
//   - Short options may include a value after an `=`, but not when grouped with
//     other short options.
//
// - Options are parsed from any argument until either a subcommand switches to
//   that subcommand's options or a `--` argument ends all option parsing to
//   allow arguments that would be ambiguous with the option syntax.
//
// ## Subcommands
//
// Subcommands can be nested to any depth, and each subcommand gets its own
// option space.
//
// ## Positional arguments
//
// Leaf commands (and subcommands) can accept positional arguments. These work
// similar to options but consist *only* of the value. They have names, but the
// name is only used in documentation and error messages. Except for the special
// case of the exact argument `-`, positional argument values cannot start with
// a `-` character until after option parsing is ended with a `--` argument.
//
// Singular positional arguments store the single value in that position.
// Appending positional arguments append all of the values in sequence.
//
// Multiple positional arguments to a single command are parsed in sequence. For
// appending positional arguments, the sequence of values appended is ended with
// a `--` argument. For example, a command that takes two sequences of
// positional arguments might look like:
//
//     my-diff-tool a.txt b.txt c.txt -- new-a.txt new-b.txt new-c.txt
//
// The `--` used in this way *both* ends option parsing and the positional
// argument sequence. Note that if there are no positional arguments prior to
// the first `--`, then it will just end option parsing. To pass an empty
// sequence of positional arguments two `--` arguments would be required. Once
// option parsing is ended, even a single positional argument can be skipped
// using a `--` argument without a positional argument.
//
// ## Help text blocks and rendering
//
// At many points in this library, a block of text is specified as a string.
// These should be written using multi-line raw string literals that start on
// their own line such as:
//
// ```cpp
//     ...
//     .help = R"""(
// Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
// tempor incididunt ut labore et dolore magna aliqua.
// )""",
// ```
//
// The code using these blocks will remove leading and trailing newlines. TODO:
// It should also reflow all of the text in a way that tries to be roughly
// similar to how Markdown would be reflowed when rendered:
//
// - Blank lines will be preserved.
// - (TODO) A best effort to detect lists and other common Markdown constructs
//   and preserve the newlines between those.
// - (TODO) Fenced regions will be preserved exactly.
//
// The remaining blocks will have all newlines removed when there is no column
// width information on the output stream, or will be re-flowed to the output
// stream's column width when available.
//
// ## The `Args` class
//
// The `Args` class is not a meaningful type, it serves two purposes: to give a
// structured name prefix to the inner types, and to allow marking many
// components `private` and using `friend` to grant access to implementation
// details.
//
// ## Roadmap / TODOs / planned future work
//
// - Detect column width when the stream is a terminal and re-flow text.
// - Implement short help display mode (and enable that by default).
// - Add help text to one-of values and render it when present.
// - Add formatting when the stream supports it (colors, bold, etc).
// - Improve error messages when parsing fails.
//   - Reliably display the most appropriate usage string.
//   - Suggest typo corrections?
//   - Print snippet or otherwise render the failure better?
// - Add support for responding to shell autocomplete queries.
// - Finish adding support for setting and printing version information.
// - Add short option counting support (`-vvv` -> `--verbose=3`).
//
namespace Carbon::CommandLine {

// Forward declare some implementation detail classes and classes that are
// friended.
struct Arg;
struct Command;
class MetaPrinter;
class Parser;
class CommandBuilder;

// The result of parsing arguments.
enum class ParseResult : int8_t {
  // Signifies that program arguments were successfully parsed and can be
  // used.
  Success,

  // Signifies a successful meta operation such as displaying help text
  // was performed. No parsed arguments are available, and the side-effects
  // have been directly provided via the streams provided to the parser.
  MetaSuccess,
};
auto operator<<(llvm::raw_ostream& output, ParseResult result)
    -> llvm::raw_ostream&;

// Actions are stored in data structures so we use an owning closure to model
// them.
using ActionT = std::function<auto()->void>;

// The core argument info used to render help and other descriptive
// information. This is used for both options and positional arguments.
// Commands use an extended version below.
struct ArgInfo {
  // The name of the argument. For options, this is the long name parsed after
  // `--`. Option names must also be unique. Conventionally, these are spelled
  // in lower case with a single dash separating words as that tends to be
  // especially easy to type and reasonably easy to read.
  //
  // For positional arguments this is only used for help text, and is often
  // spelled with `ALL-CAPS` to make it clear that it is a placeholder name
  // and does not appear in the actual command line.
  llvm::StringRef name;

  // Optional short name for options. This must consist of a single character
  // in the set [a-zA-Z].
  llvm::StringRef short_name = "";

  // For options, it is sometimes useful to render a distinct value
  // placeholder name to help clarify what the value should contain. These are
  // often in `ALL-CAPS` to make it clear they are placeholders. For example,
  // an `output` option might set this to `FILE` so it renders as
  // `--output=FILE`.
  llvm::StringRef value_name = "";

  // A long-form help string that will be rendered for this argument in
  // command help strings. It supports multiple lines and is rendered as a
  // text block described in the `Args` top-level comment.
  llvm::StringRef help = "";

  // A short help string for the argument. This should be as short as
  // possible, and always fit easily on a single line. Ideally, it can fit in
  // a narrow column of a single line and is in the range of 40 columns wide.
  //
  // If not provided, the first paragraph (up to a blank line) from `help`
  // will be used.
  llvm::StringRef help_short = "";
};

// The kinds of arguments that can be parsed.
enum class ArgKind : int8_t {
  Invalid,
  Flag,
  Integer,
  String,
  OneOf,
  MetaActionOnly,
};
auto operator<<(llvm::raw_ostream& output, ArgKind kind) -> llvm::raw_ostream&;

// A builder used to configure an argument for parsing.
class ArgBuilder {
 public:
  // When marked as required, if an argument is not provided explicitly in the
  // command line the parse will produce an error.
  auto Required(bool is_required) -> void;

  // An argument can be hidden from the help output.
  auto HelpHidden(bool is_help_hidden) -> void;

  // Sets a meta-action to run when this argument is parsed. This is used to
  // set up arguments like `--help` or `--version` that can be entirely
  // handled during parsing and rather than produce parsed information about
  // the command line, override that for some custom behavior.
  template <typename T>
  auto MetaAction(T action) -> void;

 protected:
  friend class CommandBuilder;
  // `arg` must not be null.
  explicit ArgBuilder(Arg* arg);

  auto arg() -> Arg* { return arg_; }

 private:
  Arg* arg_;
};

// Customized argument builder when the value is a boolean flag.
class FlagBuilder : public ArgBuilder {
 public:
  // Flags can be defaulted to true. However, flags always have *some*
  // default, this merely customizes which value is default. If uncustomized,
  // the default of a flag is false.
  auto Default(bool flag_value) -> void;

  // Configures the argument to store a parsed value in the provided storage.
  //
  // This must be called on the builder.
  auto Set(bool* flag) -> void;

 private:
  using ArgBuilder::ArgBuilder;
};

// Customized argument builder when the value is an integer.
class IntegerArgBuilder : public ArgBuilder {
 public:
  // Sets a default for the argument with the provided value. There is no
  // default for integer arguments unless this is called.
  //
  // For arguments that are `Set` below, this value will be stored even if the
  // argument is never explicitly provided. For arguments that are `Append`ed
  // below, this value will be used whenever the argument occurs without an
  // explicit value, but unless the argument is parsed nothing will be
  // appended.
  auto Default(int integer_value) -> void;

  // Configures the argument to store a parsed value in the provided storage.
  // Each time the argument is parsed, it will write a new value to this
  // storage.
  //
  // Exactly one of this method or `Append` below must be configured for the
  // argument.
  auto Set(int* integer) -> void;

  // Configures the argument to append a parsed value to the provided
  // container. Each time the argument is parsed, a new value will be
  // appended.
  //
  // Exactly one of this method or `Set` above must be configured for the
  // argument.
  auto Append(llvm::SmallVectorImpl<int>* sequence) -> void;

 private:
  using ArgBuilder::ArgBuilder;
};

// Customized argument builder when the value is a string.
class StringArgBuilder : public ArgBuilder {
 public:
  // Sets a default for the argument with the provided value. There is no
  // default for string arguments unless this is called.
  //
  // For arguments that are `Set` below, this value will be stored even if the
  // argument is never explicitly provided. For arguments that are `Append`ed
  // below, this value will be used whenever the argument occurs without an
  // explicit value, but unless the argument is parsed nothing will be
  // appended.
  auto Default(llvm::StringRef string_value) -> void;

  // Configures the argument to store a parsed value in the provided storage.
  // Each time the argument is parsed, it will write a new value to this
  // storage.
  //
  // Exactly one of this method or `Append` below must be configured for the
  // argument.
  auto Set(llvm::StringRef* string) -> void;

  // Configures the argument to append a parsed value to the provided
  // container. Each time the argument is parsed, a new value will be
  // appended.
  //
  // Exactly one of this method or `Set` above must be configured for the
  // argument.
  auto Append(llvm::SmallVectorImpl<llvm::StringRef>* sequence) -> void;

 private:
  using ArgBuilder::ArgBuilder;
};

// Customized argument builder when the value is required to be one of a fixed
// set of possible strings, and each one maps to a specific value of some
// type, often an enumerator.
class OneOfArgBuilder : public ArgBuilder {
 public:
  // A tiny helper / builder type for describing one of the possible values
  // that a particular argument accepts.
  //
  // Beyond carrying the string, type, and value for this candidate, it also
  // allows marking the candidate as the default.
  template <typename T>
  class OneOfValueT {
   public:
    // Configure whether a candidate is the default. If not called, it is not
    // the default. This can only be used when setting, not when appending.
    auto Default(bool is_default) && -> OneOfValueT;

   private:
    friend OneOfArgBuilder;

    explicit OneOfValueT(llvm::StringRef str, T value);

    llvm::StringRef str;
    T value;
    bool is_default = false;
  };

  // A helper function to create an object that models one of the candidate
  // values for this argument. It takes the string used to select this value
  // on the command line, deduces the type of the value, and accepts the value
  // itself that should be used when this candidate is parsed.
  template <typename T>
  static auto OneOfValue(llvm::StringRef string, T value) -> OneOfValueT<T>;

  // Configures the argument to store a parsed value in the provided storage.
  // Each time the argument is parsed, it will write a new value to this
  // storage.
  //
  // Exactly one of this method or `AppendOneOf` below must be configured for
  // the argument.
  //
  // For one-of arguments, this method also provides an array of possible
  // values that may be used:
  //
  // ```cpp
  //   arg_b.SetOneOf(
  //       {
  //           arg_b.OneOfValue("x", 1),
  //           arg_b.OneOfValue("y", 2),
  //           arg_b.OneOfValue("z", 3).Default(true),
  //       },
  //       &value);
  // ```
  //
  // The only value strings that will be parsed are those described in the
  // array, and the value stored in the storage for a particular parsed value
  // string is the one associated with it. There must be a single homogeneous
  // type for the all of the candidate values and that type must be storable
  // into the result storage pointee type. At most one of the array can also
  // be marked as a default value, which will make specifying a value
  // optional, and also will cause that value to be stored into the result
  // even if the argument is not parsed explicitly.
  template <typename T, typename U, size_t N>
  auto SetOneOf(const OneOfValueT<U> (&values)[N], T* result) -> void;

  // Configures the argument to append a parsed value to the provided
  // container. Each time the argument is parsed, a new value will be
  // appended.
  //
  // Exactly one of this method or `SetOneOf` above must be configured for the
  // argument.
  //
  // Similar to `SetOneOf`, this must also describe the candidate value
  // strings that can be parsed and the consequent values that are used for
  // those strings:
  //
  // ```cpp
  //   arg_b.AppendOneOf(
  //       {
  //           arg_b.OneOfValue("x", 1),
  //           arg_b.OneOfValue("y", 2),
  //           arg_b.OneOfValue("z", 3),
  //       },
  //       &values);
  // ```
  //
  // Instead of storing, these values are appended.
  //
  // However, appending one-of arguments cannot use a default. The values must
  // always be explicitly parsed.
  template <typename T, typename U, size_t N>
  auto AppendOneOf(const OneOfValueT<U> (&values)[N],
                   llvm::SmallVectorImpl<T>* sequence) -> void;

 private:
  using ArgBuilder::ArgBuilder;

  template <typename U, size_t N, typename MatchT, size_t... Indices>
  auto OneOfImpl(const OneOfValueT<U> (&input_values)[N], MatchT match,
                 std::index_sequence<Indices...> /*indices*/) -> void;
};

// The extended info for a command, including for a subcommand.
//
// This provides the primary descriptive information for commands used by help
// and other diagnostic messages. For subcommands, it also provides the name
// used to access the subcommand.
struct CommandInfo {
  // The name of the command. For subcommands, this is also the argument
  // spelling that accesses this subcommand. It must consist of characters in
  // the set [-a-zA-Z0-9], and must not start with a `-`.
  llvm::StringRef name;

  // An optional version string that will be rendered as part of version meta
  // action by the library. While this library does not impose a machine
  // parsable structure, users will expect this to be extractable and parsable
  // in practice.
  //
  // Subcommands with an empty version will inherit the first non-empty parent
  // version.
  //
  // Whether a (possibly inherited) version string is non-empty determines
  // whether this library provides the version-printing meta actions via a
  // `--version` flag or, if there are subcommands, a `version` subcommand.
  llvm::StringRef version = "";

  // Optional build information to include when printing the version.
  llvm::StringRef build_info = "";

  // An optional long-form help string for the command.
  //
  // When accessing a command's dedicated help output, this will form the main
  // prose output at the beginning of the help message. When listing
  // subcommands, the subcommand's `help` string will be used to describe it
  // in the list.
  //
  // The help meta actions are available regardless of whether this is
  // provided in order to mechanically describe other aspects of the command.
  //
  // This field supports multiple lines and uses the text block handling
  // described in the top-level `Args` comment.
  llvm::StringRef help = "";

  // An optional short help string for the command.
  //
  // This should be very short, at most one line and ideally fitting within a
  // narrow column of a line. If left blank, the first paragraph of text (up
  // to the first blank line) in the `help` string will be used.
  llvm::StringRef help_short = "";

  // An optional custom block of text to describe the usage of the command.
  //
  // The usage should just be a series of lines with possible invocations of
  // the command summarized as briefly as possible. Ideally, each variation on
  // a single line. The goal is to show the *structure* of different command
  // invocations, not to be comprehensive.
  //
  // When omitted, the library will generate these based on the parsing logic.
  // The generated usage is expected to be suitable for the vast majority of
  // users, but can be customized in cases where necessary.
  llvm::StringRef usage = "";

  // An optional epilogue multi-line block of text appended to the help display
  // for this command. It is only used for this command's dedicated help, but
  // can contain extra, custom guidance that is especially useful to have at
  // the very end of the output.
  llvm::StringRef help_epilogue = "";
};

// Commands are classified based on the action they result in when run.
//
// Commands that require a subcommand have no action and are just a means to
// reach the subcommand actions.
//
// Commands with _meta_ actions are also a separate kind from those with
// normal actions.
enum class CommandKind : int8_t {
  Invalid,
  RequiresSubcommand,
  Action,
  MetaAction,
};
auto operator<<(llvm::raw_ostream& output, CommandKind kind)
    -> llvm::raw_ostream&;

// Builder used to configure a command to parse.
//
// An instance of this is used to configure the top-level command, and a fresh
// instance is provided for configuring each subcommand as well.
class CommandBuilder {
 public:
  using Kind = CommandKind;

  auto AddFlag(const ArgInfo& info,
               llvm::function_ref<auto(FlagBuilder&)->void> build) -> void;
  auto AddIntegerOption(
      const ArgInfo& info,
      llvm::function_ref<auto(IntegerArgBuilder&)->void> build) -> void;
  auto AddStringOption(const ArgInfo& info,
                       llvm::function_ref<auto(StringArgBuilder&)->void> build)
      -> void;
  auto AddOneOfOption(const ArgInfo& info,
                      llvm::function_ref<auto(OneOfArgBuilder&)->void> build)
      -> void;
  auto AddMetaActionOption(const ArgInfo& info,
                           llvm::function_ref<auto(ArgBuilder&)->void> build)
      -> void;

  auto AddIntegerPositionalArg(
      const ArgInfo& info,
      llvm::function_ref<auto(IntegerArgBuilder&)->void> build) -> void;
  auto AddStringPositionalArg(
      const ArgInfo& info,
      llvm::function_ref<auto(StringArgBuilder&)->void> build) -> void;
  auto AddOneOfPositionalArg(
      const ArgInfo& info,
      llvm::function_ref<auto(OneOfArgBuilder&)->void> build) -> void;

  auto AddSubcommand(const CommandInfo& info,
                     llvm::function_ref<auto(CommandBuilder&)->void> build)
      -> void;

  // Subcommands can be hidden from the help listing of their parents with
  // this setting. Hiding a subcommand doesn't disable its own help, it just
  // removes it from the listing.
  auto HelpHidden(bool is_help_hidden) -> void;

  // Exactly one of these three should be called to select and configure the
  // kind of the built command.
  auto RequiresSubcommand() -> void;
  auto Do(ActionT action) -> void;
  auto Meta(ActionT meta_action) -> void;

 private:
  friend Parser;

  // `command` and `meta_printer` must not be null.
  explicit CommandBuilder(Command* command, MetaPrinter* meta_printer);

  auto AddArgImpl(const ArgInfo& info, ArgKind kind) -> Arg*;
  auto AddPositionalArgImpl(const ArgInfo& info, ArgKind kind,
                            llvm::function_ref<auto(Arg&)->void> build) -> void;
  auto Finalize() -> void;

  Command* command_;
  MetaPrinter* meta_printer_;

  llvm::SmallDenseSet<llvm::StringRef> arg_names_;
  llvm::SmallDenseSet<llvm::StringRef> subcommand_names_;
};

// Builds a description of a command and then parses the provided arguments
// for that command.
//
// This is the main entry point to both build up the description of the
// command whose arguments are being parsed and to do the parsing. Everything
// is done in a single invocation as the common case is to build a command
// description, parse the arguments once, and then run with that
// configuration.
//
// The `out` stream is treated like `stdout` would be for a Unix-style command
// line tool, and `errors` like `stderr`: any errors or diagnostic information
// are printed to `errors`, but meta-actions like printing a command's help go
// to `out`.
auto Parse(llvm::ArrayRef<llvm::StringRef> unparsed_args,
           llvm::raw_ostream& out, CommandInfo command_info,
           llvm::function_ref<auto(CommandBuilder&)->void> build)
    -> ErrorOr<ParseResult>;

// Implementation details only below.

// The internal representation of a parsable argument description.
struct Arg {
  using Kind = ArgKind;
  using ValueActionT =
      std::function<auto(const Arg& arg, llvm::StringRef value_string)->bool>;
  using DefaultActionT = std::function<auto(const Arg& arg)->void>;

  explicit Arg(ArgInfo info);
  ~Arg();

  ArgInfo info;
  Kind kind = Kind::Invalid;
  bool has_default = false;
  bool is_required = false;
  bool is_append = false;
  bool is_help_hidden = false;

  // Meta action storage, only populated if this argument causes a meta action.
  ActionT meta_action;

  // Storage options depending on the kind.
  union {
    // Singular argument storage pointers.
    bool* flag_storage;
    int* integer_storage;
    llvm::StringRef* string_storage;

    // Appending argument storage pointers.
    llvm::SmallVectorImpl<int>* integer_sequence;
    llvm::SmallVectorImpl<llvm::StringRef>* string_sequence;

    // One-of information.
    struct {
      llvm::OwningArrayRef<llvm::StringRef> value_strings;
      ValueActionT value_action;
    };
  };

  // Default values depending on the kind.
  union {
    bool default_flag;
    int default_integer;
    llvm::StringRef default_string;
    struct {
      DefaultActionT default_action;
      int default_value_index;
    };
  };
};

// The internal representation of a parsable command description, including its
// options, positional arguments, and subcommands.
struct Command {
  using Kind = CommandBuilder::Kind;

  explicit Command(CommandInfo info, Command* parent = nullptr);

  CommandInfo info;
  Command* parent;
  ActionT action;
  Kind kind = Kind::Invalid;

  bool is_help_hidden = false;

  llvm::SmallVector<std::unique_ptr<Arg>> options;
  llvm::SmallVector<std::unique_ptr<Arg>> positional_args;
  llvm::SmallVector<std::unique_ptr<Command>> subcommands;
};

template <typename T>
auto ArgBuilder::MetaAction(T action) -> void {
  CARBON_CHECK(!arg_->meta_action, "Cannot set a meta action twice!");
  arg_->meta_action = std::move(action);
}

template <typename T>
auto OneOfArgBuilder::OneOfValueT<T>::Default(
    bool is_default) && -> OneOfValueT {
  OneOfValueT result = std::move(*this);
  result.is_default = is_default;
  return result;
}

template <typename T>
OneOfArgBuilder::OneOfValueT<T>::OneOfValueT(llvm::StringRef str, T value)
    : str(str), value(std::move(value)) {}

template <typename T>
auto OneOfArgBuilder::OneOfValue(llvm::StringRef str, T value)
    -> OneOfValueT<T> {
  return OneOfValueT<T>(str, value);
}

template <typename T, typename U, size_t N>
auto OneOfArgBuilder::SetOneOf(const OneOfValueT<U> (&values)[N], T* result)
    -> void {
  static_assert(N > 0, "Must include at least one value.");
  arg()->is_append = false;
  OneOfImpl(
      values, [result](T value) { *result = value; },
      std::make_index_sequence<N>{});
}

template <typename T, typename U, size_t N>
auto OneOfArgBuilder::AppendOneOf(const OneOfValueT<U> (&values)[N],
                                  llvm::SmallVectorImpl<T>* sequence) -> void {
  static_assert(N > 0, "Must include at least one value.");
  arg()->is_append = true;
  OneOfImpl(
      values, [sequence](T value) { sequence->push_back(value); },
      std::make_index_sequence<N>{});
}

// An implementation tool for the one-of value candidate handling. Delegating to
// this allows us to deduce a pack of indices from the array of candidates, and
// then use that variadic pack to operate on the array in the variadic space.
// This includes packaging the components up separately into our storage
// representation, as well as processing the array to find and register any
// default.
//
// The representation is especially tricky because we want all of the actual
// values and even the *type* of values to be erased. We do that by building
// lambdas that do the type-aware operations and storing those into type-erased
// function objects.
template <typename U, size_t N, typename MatchT, size_t... Indices>
auto OneOfArgBuilder::OneOfImpl(const OneOfValueT<U> (&input_values)[N],
                                MatchT match,
                                std::index_sequence<Indices...> /*indices*/)
    -> void {
  std::array<llvm::StringRef, N> value_strings = {input_values[Indices].str...};
  std::array<U, N> values = {input_values[Indices].value...};

  // Directly copy the value strings into a heap-allocated array in the
  // argument.
  new (&arg()->value_strings)
      llvm::OwningArrayRef<llvm::StringRef>(value_strings);

  // And build a type-erased action that maps a specific value string to a value
  // by index.
  new (&arg()->value_action) Arg::ValueActionT(
      [values, match](const Arg& arg, llvm::StringRef value_string) -> bool {
        for (auto [value, arg_value_string] :
             llvm::zip(values, arg.value_strings)) {
          if (value_string == arg_value_string) {
            match(value);
            return true;
          }
        }
        return false;
      });

  // Fold over all the input values to see if there is a default.
  if ((input_values[Indices].is_default || ...)) {
    CARBON_CHECK(!arg()->is_append, "Can't append default.");
    CARBON_CHECK((input_values[Indices].is_default + ... + 0) == 1,
                 "Cannot default more than one value.");

    arg()->has_default = true;

    // First build a lambda that configures the default using an index. We'll
    // call this below, this lambda isn't the one that is stored.
    auto set_default = [&](size_t index, const auto& default_value) {
      // Now that we have the desired default index, build a lambda and store it
      // as the default action. This lambda is stored and so it captures the
      // necessary information explicitly and by value.
      new (&arg()->default_action)
          Arg::DefaultActionT([value = default_value.value,
                               match](const Arg& /*arg*/) { match(value); });

      // Also store the index itself for use when printing help.
      arg()->default_value_index = index;
    };

    // Now we fold across the inputs and in the one case that is the default, we
    // call the lambda. This is just a somewhat awkward way to write a loop with
    // a condition in it over a pack.
    ((input_values[Indices].is_default
          ? set_default(Indices, input_values[Indices])
          : static_cast<void>(0)),
     ...);
  }
}

}  // namespace Carbon::CommandLine

#endif  // CARBON_COMMON_COMMAND_LINE_H_
