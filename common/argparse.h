// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ARG_PARSE_H_
#define CARBON_COMMON_ARG_PARSE_H_

#include <string>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "common/ostream.h"
#include "common/check.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

namespace ArgParser {

enum class FlagKind {
  Boolean,
  String,
  Enum,

  StringList,
};

struct Flag {
  llvm::StringLiteral name;
  // llvm::StringLiteral short_name = "";

  // The address of a flag is used as the identity after parsing.
  Flag(const Flag&) = delete;
};

struct BooleanFlag : Flag {
  bool default_value = false;
};

struct BooleanDefault {
  bool default_value = false;
};
constexpr inline auto MakeBooleanFlag(llvm::StringLiteral name,
                                      BooleanDefault defaults = {})
    -> BooleanFlag {
  return {{.name = name}, defaults.default_value};
}

struct StringFlag : Flag {
  std::optional<llvm::StringLiteral> default_value = {};
};

struct StringDefault {
  llvm::StringLiteral default_value;
};
constexpr inline auto MakeStringFlag(llvm::StringLiteral name) -> StringFlag {
  return {{.name = name}};
}
constexpr inline auto MakeStringFlag(llvm::StringLiteral name,
                                     StringDefault defaults)
    -> StringFlag {
  return {{.name = name}, {defaults.default_value}};
}

template <typename EnumT>
struct EnumValue {
  llvm::StringLiteral name;
  EnumT value;
};

template <typename EnumT, ssize_t N>
struct EnumFlag : Flag {
  EnumValue<EnumT> values[N];

  std::optional<EnumT> default_value = {};
};

namespace Detail {
template <typename EnumT, ssize_t N, size_t... Indices>
constexpr inline auto MakeEnumFlagHelper(
    llvm::StringLiteral name, const EnumValue<EnumT> (&args)[N],
    std::index_sequence<Indices...> /*indices*/) -> EnumFlag<EnumT, N> {
  return {{.name = name}, {args[Indices]...}};
}
}  // namespace Detail

template <typename EnumT, ssize_t N>
constexpr inline auto MakeEnumFlag(llvm::StringLiteral name,
                                   const EnumValue<EnumT> (&args)[N])
    -> EnumFlag<EnumT, N> {
  return Detail::MakeEnumFlagHelper(name, args, std::make_index_sequence<N>{});
}

struct StringListFlag : Flag {
  llvm::ArrayRef<llvm::StringLiteral> default_values = {};
};


struct StringListDefault {
  llvm::ArrayRef<llvm::StringLiteral> default_values = {};
};
constexpr inline auto MakeStringListFlag(
    llvm::StringLiteral name, StringListDefault defaults = {})
    -> StringListFlag {
  return {{.name = name}, defaults.default_values};
}

struct CommandInfo {
  llvm::StringLiteral description = "";
  llvm::StringLiteral usage = "";
  llvm::StringLiteral epilog = "";
};
template <typename... Ts>
struct Command {
  llvm::StringLiteral name;

  std::tuple<const Ts*...> flags = {};

  CommandInfo info;
};

template <typename... Ts>
constexpr inline auto MakeCommand(llvm::StringLiteral name, const Ts*... flags)
    -> Command<Ts...> {
  return {.name = name, .flags = std::tuple{flags...}, .info = {}};
}

template <typename... Ts>
constexpr inline auto MakeCommand(llvm::StringLiteral name, CommandInfo info,
                                  const Ts*... flags) -> Command<Ts...> {
  return {.name = name, .flags = std::tuple{flags...}, .info = info};
}

template <typename EnumT, typename... FlagTs>
struct Subcommand : Command<FlagTs...> {
  static_assert(std::is_enum_v<EnumT>,
                "Must provide an enum type to enumerate subcommands.");
  using Enum = EnumT;
  EnumT enumerator;
};

template <typename EnumT, typename... FlagTs>
constexpr inline auto MakeSubcommand(llvm::StringLiteral name, EnumT enumerator,
                                     const FlagTs*... flags)
    -> Subcommand<EnumT, FlagTs...> {
  return {{.name = name, .flags = std::tuple{flags...}, .info = {}},
          enumerator};
}

template <typename EnumT, typename... FlagTs>
constexpr inline auto MakeSubcommand(llvm::StringLiteral name, EnumT enumerator,
                                     CommandInfo info, const FlagTs*... flags)
    -> Subcommand<EnumT, FlagTs...> {
  return {{.name = name, .flags = std::tuple{flags...}, .info = info},
          enumerator};
}

template <typename CommandT, typename... SubcommandTs>
auto Parse(llvm::ArrayRef<llvm::StringRef> raw_args, llvm::raw_ostream& errors,
           const CommandT& command, const SubcommandTs&... subcommands);

}  // namespace ArgParser

class Args {
 public:
  using FlagKind = ArgParser::FlagKind;
  using Flag = ArgParser::Flag;
  using BooleanFlag = ArgParser::BooleanFlag;
  using StringFlag = ArgParser::StringFlag;
  template <typename EnumT, ssize_t N>
  using EnumFlag = ArgParser::EnumFlag<EnumT, N>;
  using StringListFlag = ArgParser::StringListFlag;
  template <typename... FlagTs>
  using Command = ArgParser::Command<FlagTs...>;

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
  // or having a true default.
  auto TestFlag(const BooleanFlag *flag) const -> bool {
    return TestFlagImpl(flags_, flag);
  }

  // Gets a string flag's value if available, whether via a default or
  // explicitly set value. If unavailable, returns an empty optional.
  auto GetStringFlag(const StringFlag* flag) const
      -> std::optional<llvm::StringRef> {
    return GetStringFlagImpl(flags_, flag);
  }

  // Gets an enum flag's value if available, whether via a default or explicitly
  // set value. If unavailable, returns an empty optional.
  template <typename EnumT, ssize_t N>
  auto GetEnumFlag(const EnumFlag<EnumT, N>* flag) const
      -> std::optional<EnumT> {
    return GetEnumFlagImpl(flags_, flag);
  }

  auto GetStringListFlag(const StringListFlag* flag) const
      -> llvm::ArrayRef<llvm::StringRef> {
    return GetStringListFlagImpl(flags_, flag);
  }

  auto positional_args() const -> llvm::ArrayRef<llvm::StringRef> {
    return positional_args_;
  }

 protected:
  template <typename CommandT, typename... SubcommandTs>
  friend auto ArgParser::Parse(llvm::ArrayRef<llvm::StringRef> raw_args,
                               llvm::raw_ostream& errors,
                               const CommandT& command,
                               const SubcommandTs&... subcommands);

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

  struct FlagKindAndValue {
    FlagKind kind;

    // For values that can be stored in-line, we do so. Otherwise, we store an
    // index into a side array. Which member is active, and if an index which
    // array is indexed, is determined by the kind.
    union {
      bool boolean_value;
      int value_index;
      int enum_value;
    };
  };
  
  using FlagMap = llvm::SmallDenseMap<const Flag*, FlagKindAndValue, 4>;

  ParseResult parse_result_;

  llvm::SmallVector<llvm::StringRef, 4> string_flag_values_;

  llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 1>, 4>
      string_list_flag_values_;

  FlagMap flags_;

  llvm::SmallVector<llvm::StringRef, 12> positional_args_;

  auto TestFlagImpl(const FlagMap& flags, const BooleanFlag* flag) const -> bool;
  auto GetStringFlagImpl(const FlagMap& flags, const StringFlag* flag) const
      -> std::optional<llvm::StringRef>;
  template <typename EnumT, ssize_t N>
  auto GetEnumFlagImpl(const FlagMap& flags,
                       const EnumFlag<EnumT, N>* flag) const
      -> std::optional<EnumT>;
  auto GetStringListFlagImpl(const FlagMap& flags, const StringListFlag* flag) const
      -> llvm::ArrayRef<llvm::StringRef>;

  void AddFlagDefault(Args::FlagMap& flags, const BooleanFlag* flag);
  void AddFlagDefault(Args::FlagMap& flags, const StringFlag* flag);
  template <typename EnumT, ssize_t N>
  void AddFlagDefault(Args::FlagMap& flags, const EnumFlag<EnumT, N>* flag);
  void AddFlagDefault(Args::FlagMap& flags, const StringListFlag* flag);

  auto AddParsedFlagToMap(FlagMap& flags, const Flag* flag, FlagKind kind)
      -> std::pair<bool, FlagKindAndValue&>;
  auto AddParsedFlag(FlagMap& flags, const BooleanFlag* flag,
                     std::optional<llvm::StringRef> value,
                     llvm::raw_ostream& errors) -> bool;
  auto AddParsedFlag(FlagMap& flags, const StringFlag* flag,
                     std::optional<llvm::StringRef> value,
                     llvm::raw_ostream& errors) -> bool;
  template <typename EnumT, ssize_t N>
  auto AddParsedFlag(FlagMap& flags, const EnumFlag<EnumT, N>* flag,
                     std::optional<llvm::StringRef> value,
                     llvm::raw_ostream& errors) -> bool;
  auto AddParsedFlag(FlagMap& flags, const StringListFlag* flag,
                     std::optional<llvm::StringRef> value,
                     llvm::raw_ostream& errors) -> bool;
};

template <typename SubcommandEnumT>
class SubcommandArgs : public Args {
 public:
  static_assert(std::is_enum_v<SubcommandEnumT>,
                "Must provide an enum type to enumerate subcommands.");
  using SubcommandEnum = SubcommandEnumT;

  using FlagKind = Args::FlagKind;
  using Flag = Args::Flag;
  using BooleanFlag = Args::BooleanFlag;
  using StringFlag = Args::StringFlag;
  template <typename... FlagTs>
  using Command = Args::Command<FlagTs...>;

  template <typename... FlagTs>
  using Subcommand = ArgParser::Subcommand<SubcommandEnum, FlagTs...>;

  auto subcommand() const -> SubcommandEnum {
    return subcommand_;
  }

  // Test whether a boolean subcommand flag value is true, either by being set
  // explicitly or having a true default.
  auto TestSubcommandFlag(const BooleanFlag *flag) const -> bool {
    return TestFlagImpl(subcommand_flags_, flag);
  }

  // Get's a subcommand string flag's value if available, whether via a default
  // or explicitly set value. If unavailable, returns an empty optional.
  auto GetSubcommandStringFlag(const StringFlag *flag) const
      -> std::optional<llvm::StringRef> {
    return GetStringFlagImpl(subcommand_flags_, flag);
  }

  // Gets a subcommand enum flag's value if available, whether via a default or
  // explicitly set value. If unavailable, returns an empty optional.
  template <typename EnumT, ssize_t N>
  auto GetSubcommandEnumFlag(const EnumFlag<EnumT, N>* flag) const
      -> std::optional<EnumT> {
    return GetEnumFlagImpl(subcommand_flags_, flag);
  }

  auto GetSubcommandStringListFlag(const StringListFlag* flag) const
      -> llvm::ArrayRef<llvm::StringRef> {
    return GetStringListFlagImpl(subcommand_flags_, flag);
  }

 private:
  template <typename CommandT, typename... SubcommandTs>
  friend auto ArgParser::Parse(llvm::ArrayRef<llvm::StringRef> raw_args,
                               llvm::raw_ostream& errors,
                               const CommandT& command,
                               const SubcommandTs&... subcommands);

  using Args::ParseResult;
  using Args::FlagMap;

  SubcommandEnum subcommand_;
  FlagMap subcommand_flags_;
};

namespace ArgParser {

namespace Detail {

template <typename... SubcommandTs> struct SubcommandEnum;

template <typename SubcommandT, typename... SubcommandTs>
struct SubcommandEnum<SubcommandT, SubcommandTs...> {
  using Type = typename SubcommandT::Enum;
};

enum NoSubcommands {};
template <> struct SubcommandEnum<> { using Type = NoSubcommands; };

}  // namespace Detail

template <typename CommandT, typename... SubcommandTs>
auto Parse(llvm::ArrayRef<llvm::StringRef> raw_args, llvm::raw_ostream& errors,
           const CommandT& command, const SubcommandTs&... subcommands) {
  // Extract the enum type from the subcommand types, and ensure it is a single type.
  using SubcommandEnum = typename Detail::SubcommandEnum<SubcommandTs...>::Type;
  constexpr bool HasSubcommands = sizeof...(SubcommandTs) > 0;
  if constexpr (HasSubcommands) {
    static_assert(
        (std::is_same_v<SubcommandEnum, typename SubcommandTs::Enum> && ...),
        "Must have the same enum type for all subcommands.");
  }

  using ArgsType = std::conditional_t<HasSubcommands, SubcommandArgs<SubcommandEnum>, Args>;
  ArgsType args;

  // Start in the error state to allow early returns whenever a parse error is
  // found.
  args.parse_result_ = ArgsType::ParseResult::Error;

  auto* flags = &args.flags_;
  llvm::SmallDenseMap<
      llvm::StringRef,
      std::function<bool(std::optional<llvm::StringRef> arg_value)>, 16>
      flag_map;
  auto build_flag_map = [&](const auto*... command_flags) {
    // Process the input flags into a lookup table for parsing, also setting up
    // any default values.
    flag_map.clear();

    auto add_flag = [&](const auto* flag) {
      bool inserted =
          flag_map
              .insert({flag->name,
                       [flag, &args, &flags,
                        &errors](std::optional<llvm::StringRef> arg_value) {
                         return args.AddParsedFlag(*flags, flag, arg_value,
                                                   errors);
                       }})
              .second;
      CARBON_CHECK(inserted) << "Duplicate flags named: " << flag->name;
      args.AddFlagDefault(*flags, flag);
    };
    // Fold over the flags, calling `add_flag` for each one.
    (add_flag(command_flags), ...);
  };
  std::apply(build_flag_map, command.flags);

  // Process the input subcommands into a lookup table. We just handle the
  // subcommand name here to be lazy. We'll process the subcommand itself only
  // if it is needed.
  llvm::SmallDenseMap<llvm::StringRef, std::function<void()>, 16> subcommand_map;
  auto parsed_subcommand = [&](const auto* subcommand) {
    if constexpr (HasSubcommands) {
      args.subcommand_ = subcommand->enumerator;
      flags = &args.subcommand_flags_;
      // Rebuild the flag map for this subcommand.
      std::apply(build_flag_map, subcommand->flags);
    }
  };
  if constexpr (HasSubcommands) {
    auto add_subcommand = [&](const auto* subcommand) {
      bool inserted = subcommand_map
                          .insert({subcommand->name,
                                   [subcommand, &parsed_subcommand] {
                                     parsed_subcommand(subcommand);
                                   }})
                          .second;
      CARBON_CHECK(inserted)
          << "Duplicate subcommands named: " << subcommand->name;
    };
    (add_subcommand(&subcommands), ...);
  }

  // Now walk the input args, and build up the program args from them. Part-way
  // through, if we discover a subcommand, we'll re-set the mappings and switch
  // to parsing the subcommand.
  bool is_subcommand_parsed = false;
  for (int i = 0, size = raw_args.size(); i < size; ++i) {
    llvm::StringRef arg = raw_args[i];
    if (arg[0] != '-') {
      if (!HasSubcommands || is_subcommand_parsed) {
        args.positional_args_.push_back(arg);
        continue;
      }
      if constexpr (HasSubcommands) {
        is_subcommand_parsed = true;
        // This should be a subcommand, parse it as such.
        auto subcommand_it = subcommand_map.find(arg);
        if (subcommand_it == subcommand_map.end()) {
          errors << "ERROR: Invalid subcommand: " << arg << "\n";
          // TODO: show usage
          return args;
        }

        // Switch to subcommand parsing and continue.
        subcommand_it->second();
        continue;
      }
    }
    if (arg[1] != '-') {
      CARBON_FATAL() << "TODO: handle short flags";
    }
    if (arg.size() == 2) {
      // A parameter of `--` disables all flag processing making the remaining args always positional.
      args.positional_args_.append(raw_args.begin() + i + 1, raw_args.end());
      break;
    }
    // Walk past the double dash.
    arg = arg.drop_front(2);

    // Split out a value if present.
    std::optional<llvm::StringRef> value = {};
    auto index = arg.find('=');
    if (index != llvm::StringRef::npos) {
      value = arg.substr(index + 1);
      arg = arg.substr(0, index);
    }

    auto flag_it = flag_map.find(arg);
    if (flag_it == flag_map.end()) {
      errors << "ERROR: Flag '--" << arg << "' does not exist.\n";
      // TODO: show usage
      return args;
    }
    if (!flag_it->second(value)) {
      // TODO: show usage
      return args;
    }
  }

  // We successfully parsed all the arguments.
  args.parse_result_ = ArgsType::ParseResult::Success;
  return args;
}

inline auto operator<<(llvm::raw_ostream& out, FlagKind kind)
    -> llvm::raw_ostream& {
  switch (kind) {
    case FlagKind::Boolean:
      out << "Boolean";
      break;
    case FlagKind::String:
      out << "String";
      break;
    case FlagKind::Enum:
      out << "Enum";
      break;
    case FlagKind::StringList:
      out << "StringList";
      break;
  }
  return out;
}

}  // namespace ArgParser

template <typename EnumT, ssize_t N>
auto Args::GetEnumFlagImpl(const FlagMap& flags,
                           const EnumFlag<EnumT, N>* flag) const
    -> std::optional<EnumT> {
  auto flag_iterator = flags.find(flag);
  if (flag_iterator == flags.end()) {
    // No value for this flag.
    return {};
  }
  FlagKind kind = flag_iterator->second.kind;
  CARBON_CHECK(kind == FlagKind::Enum)
      << "Flag '" << flag->name << "' has inconsistent kinds";
  return static_cast<EnumT>(flag_iterator->second.enum_value);
}

template <typename EnumT, ssize_t N>
void Args::AddFlagDefault(Args::FlagMap& flags, const EnumFlag<EnumT, N>* flag) {
  if (!flag->default_value.has_value()) {
    return;
  }
  auto [flag_it, inserted] = flags.insert({flag, {.kind = FlagKind::String}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of flags!";

  // Make sure any value we store will round-trip through our type erased
  // storage of `int` correctly.
  EnumT enum_value = *flag->default_value;
  int storage_value = static_cast<int>(enum_value);
  CARBON_CHECK(enum_value == static_cast<EnumT>(storage_value))
      << "Default for enum flag '--" << flag->name << "' has a storage value '"
      << storage_value << "' which won't round-trip!";

  flag_it->second.enum_value = storage_value;
}

template <typename EnumT, ssize_t N>
auto Args::AddParsedFlag(FlagMap& flags, const EnumFlag<EnumT, N>* flag,
                         std::optional<llvm::StringRef> arg_value,
                         llvm::raw_ostream& errors) -> bool {
  auto [inserted, value] = AddParsedFlagToMap(flags, flag, FlagKind::Enum);
  if (!arg_value && !flag->default_value) {
    errors << "ERROR: Invalid missing value for the enum flag '--"
           << flag->name << "' which does not have a default value\n";
    return false;
  }
  EnumT enum_value;
  if (arg_value) {
    bool matched_value = false;
    for (ssize_t i = 0; i < N; ++i) {
      if (*arg_value == flag->values[i].name) {
        enum_value = flag->values[i].value;
        matched_value = true;
        break;
      }
    }
    if (!matched_value) {
      errors << "ERROR: Invalid value '" << *arg_value
             << "' for the enum flag '--" << flag->name
             << "', must be one of the following: ";
      for (ssize_t i = 0; i < N; ++i) {
        if (i != 0) {
          errors << ", ";
        }
        errors << flag->values[i].name;
      }
      errors << "\n";
      return false;
    }
  } else {
    enum_value = *flag->default_value;
  }

  // Make sure any value we store will round-trip through our type erased
  // storage of `int` correctly.
  int storage_value = static_cast<int>(enum_value);
  CARBON_CHECK(enum_value == static_cast<EnumT>(storage_value))
      << "Parsed value for enum flag '--" << flag->name
      << "' has a storage value '" << storage_value
      << "' which won't round-trip!";

  value.enum_value = storage_value;
  return true;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_ARG_PARSE_H_
