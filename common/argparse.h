// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ARG_PARSE_H_
#define CARBON_COMMON_ARG_PARSE_H_

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "common/ostream.h"

namespace Carbon {

class ProgramArgs {
 public:
  enum class FlagKind {
    Boolean,
    String,
  };

  struct BooleanFlag {
    bool default_value;
  };

  struct StringFlag {
    llvm::StringLiteral default_value;
  };

  struct Flag {
    llvm::StringLiteral name;
    llvm::StringLiteral short_name = "";

    std::optional<BooleanFlag> boolean = {};
    std::optional<StringFlag> string = {};
  };

  struct Command {
    llvm::StringLiteral name;

    llvm::ArrayRef<Flag> flags = {};

    llvm::StringLiteral description = "";
    llvm::StringLiteral usage = "";
    llvm::StringLiteral epilog = "";
  };

  struct Options {
    Command command;
    llvm::ArrayRef<Command> subcommands = {};
  };

  static auto Parse(const Options& options, llvm::ArrayRef<llvm::StringRef> args,
             llvm::raw_ostream& output_stream, llvm::raw_ostream& error_stream)
      -> ProgramArgs;

  // Query whether there are useful parsed arguments to continue executing the
  // program. Only returns true when the parse result is successful and not
  // handled with a meta operation like displaying help text.
  explicit operator bool() const {
    return parse_result_ == ParseResult::Success;
  }

  // An exit code based on whether the parse encountered an error. Typically
  // used to return from `main`.
  auto main_exit_code() const -> int {
    return parse_result_ == ParseResult::Error ? EXIT_FAILURE : EXIT_SUCCESS;
  }

  // Test whether a boolean flag value is true, either by being set explicitly
  // or having a true default. An error to call with a non-boolean flag.
  auto TestFlag(llvm::StringRef name) const -> bool;

  // Get's a string flag's value if available, whether via a default or
  // explicitly set value. If unavailable, returns an empty optional. An error
  // to call with a non-string flag.
  auto GetStringFlag(llvm::StringRef name) const -> std::optional<llvm::StringRef>;

  auto has_subcommand() const -> bool { return subcommand_name_.has_value(); }
  auto subcommand_name() const -> std::optional<llvm::StringRef> {
    return subcommand_name_;
  }

  // Test whether a boolean subcommand flag value is true, either by being set
  // explicitly or having a true default. An error to call without a present
  // subcommand or with a non-boolean flag.
  auto TestSubcommandFlag(llvm::StringRef name) const -> bool;

  // Get's a subcommand string flag's value if available, whether via a default
  // or explicitly set value. If unavailable, returns an empty optional. An
  // error to call without a present subcommand or with a non-string flag.
  auto GetSubcommandStringFlag(llvm::StringRef name) const
      -> std::optional<llvm::StringRef>;

  auto positional_args() const -> llvm::ArrayRef<llvm::StringRef> {
    return positional_args_;
  }

 private:
  enum class ParseResult {
    // Signifies an error parsing arguments. It will have been diagnosed using
    // the streams provided to the parser, and no useful parsed arguments are
    // available.
    Error,

    // Signifies that program arguments were successfully parsed and can be
    // used.
    Success,

    // Signifies a successful meta operation such as displaying help text
    // was performed. No parsed arguments are available, and the side-effects
    // have been directly provided via the streams provided to the parser.
    MetaSuccess,
  };

  ParseResult parse_result_;

  llvm::SmallBitVector boolean_flag_values_;
  llvm::SmallVector<llvm::StringRef, 4> string_flag_values_;

  struct FlagKindAndValueIndex {
    FlagKind kind;
    int value_index;
  };

  llvm::SmallDenseMap<llvm::StringRef, FlagKindAndValueIndex, 4> flags_;

  std::optional<llvm::StringRef> subcommand_name_;

  llvm::SmallDenseMap<llvm::StringRef, FlagKindAndValueIndex, 4> subcommand_flags_;

  llvm::SmallVector<llvm::StringRef, 12> positional_args_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_ARG_PARSE_H_
