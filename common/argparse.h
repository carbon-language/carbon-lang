// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ARGPARSE_H_
#define CARBON_COMMON_ARGPARSE_H_

#include <array>
#include <forward_list>
#include <string>
#include <type_traits>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

class Args {
 public:
  struct Flag;
  struct StringOpt;
  struct IntOpt;
  template <typename EnumT, ssize_t N>
  struct EnumOpt;
  struct StringListOpt;
  template <typename... Ts>
  struct Command;
  template <typename EnumT, typename... OptTs>
  struct Subcommand;

  constexpr static auto MakeFlag(llvm::StringRef name,
                                 llvm::StringRef short_name = "",
                                 bool default_value = false) -> Flag;

  constexpr static auto MakeStringOpt(
      llvm::StringRef name, llvm::StringRef short_name = "",
      std::optional<llvm::StringRef> default_value = {}) -> StringOpt;

  constexpr static auto MakeIntOpt(llvm::StringRef name,
                                   llvm::StringRef short_name = "",
                                   std::optional<ssize_t> default_value = {})
      -> IntOpt;

  template <typename EnumT>
  struct EnumValue;
  template <typename EnumT, ssize_t N>
  constexpr static auto MakeEnumOpt(llvm::StringRef name,
                                    const EnumValue<EnumT> (&args)[N],
                                    llvm::StringRef short_name = "",
                                    std::optional<EnumT> default_value = {})
      -> EnumOpt<EnumT, N>;

  constexpr static auto MakeStringListOpt(
      llvm::StringRef name, llvm::StringRef short_name = "",
      std::optional<llvm::ArrayRef<llvm::StringRef>> default_values = {})
      -> StringListOpt;

  struct CommandInfo {
    llvm::StringRef description = "";
    llvm::StringRef usage = "";
    llvm::StringRef epilog = "";
  };
  template <typename... Ts>
  constexpr static auto MakeCommand(llvm::StringRef name, const Ts*... opts)
      -> Command<Ts...>;
  template <typename... Ts>
  constexpr static auto MakeCommand(llvm::StringRef name, CommandInfo info,
                                    const Ts*... opts) -> Command<Ts...>;
  template <typename EnumT, typename... OptTs>
  constexpr static auto MakeSubcommand(llvm::StringRef name, EnumT enumerator,
                                       const OptTs*... opts)
      -> Subcommand<EnumT, OptTs...>;
  template <typename EnumT, typename... OptTs>
  constexpr static auto MakeSubcommand(llvm::StringRef name, EnumT enumerator,
                                       CommandInfo info, const OptTs*... opts)
      -> Subcommand<EnumT, OptTs...>;

  template <typename CommandT, typename... SubcommandTs>
  static auto Parse(llvm::ArrayRef<llvm::StringRef> raw_args,
                    llvm::raw_ostream& errors, const CommandT& command,
                    const SubcommandTs&... subcommands);

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

  // Test whether a flag value is true, either by being set explicitly or having
  // a true default.
  auto TestFlag(const Flag* opt) const -> bool {
    return TestFlagImpl(opts_, opt);
  }

  // Gets a string opt's value if available, whether via a default or
  // explicitly set value. If unavailable, returns an empty optional.
  auto GetStringOpt(const StringOpt* opt) const
      -> std::optional<llvm::StringRef> {
    return GetStringOptImpl(opts_, opt);
  }

  // Gets an int opt's value if available, whether via a default or
  // explicitly set value. If unavailable, returns an empty optional.
  auto GetIntOpt(const IntOpt* opt) const -> std::optional<ssize_t> {
    return GetIntOptImpl(opts_, opt);
  }

  // Gets an enum opt's value if available, whether via a default or explicitly
  // set value. If unavailable, returns an empty optional.
  template <typename EnumT, ssize_t N>
  auto GetEnumOpt(const EnumOpt<EnumT, N>* opt) const -> std::optional<EnumT> {
    return GetEnumOptImpl(opts_, opt);
  }

  auto GetStringListOpt(const StringListOpt* opt) const
      -> llvm::ArrayRef<llvm::StringRef> {
    return GetStringListOptImpl(opts_, opt);
  }

  auto positional_args() const -> llvm::ArrayRef<llvm::StringRef> {
    return positional_args_;
  }

 protected:
  enum class OptKind {
    Flag,
    String,
    Int,
    Enum,

    StringList,
  };

  struct Opt;

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

  struct OptKindAndValue {
    OptKind kind;

    // For values that can be stored in-line, we do so. Otherwise, we store an
    // index into a side array. Which member is active, and if an index which
    // array is indexed, is determined by the kind.
    union {
      bool flag_value;
      int enum_value;

      int value_index;
    };
  };

  using OptMap = llvm::SmallDenseMap<const Opt*, OptKindAndValue, 4>;

  ParseResult parse_result_;

  llvm::SmallVector<llvm::StringRef, 4> string_opt_values_;
  llvm::SmallVector<ssize_t, 4> int_opt_values_;
  llvm::SmallVector<llvm::SmallVector<llvm::StringRef, 1>, 4>
      string_list_opt_values_;

  OptMap opts_;

  llvm::SmallVector<llvm::StringRef, 12> positional_args_;

  template <typename EnumT, ssize_t N, size_t... Indices>
  constexpr static auto MakeEnumOptHelper(
      llvm::StringRef name, const EnumValue<EnumT> (&args)[N],
      llvm::StringRef short_name, std::optional<EnumT> default_value,
      std::index_sequence<Indices...> /*indices*/) -> EnumOpt<EnumT, N>;

  auto TestFlagImpl(const OptMap& opts, const Flag* opt) const -> bool;
  auto GetStringOptImpl(const OptMap& opts, const StringOpt* opt) const
      -> std::optional<llvm::StringRef>;
  auto GetIntOptImpl(const OptMap& opts, const IntOpt* opt) const
      -> std::optional<ssize_t>;
  template <typename EnumT, ssize_t N>
  auto GetEnumOptImpl(const OptMap& opts, const EnumOpt<EnumT, N>* opt) const
      -> std::optional<EnumT>;
  auto GetStringListOptImpl(const OptMap& opts, const StringListOpt* opt) const
      -> llvm::ArrayRef<llvm::StringRef>;

  void AddOptDefault(Args::OptMap& opts, const Flag* opt);
  void AddOptDefault(Args::OptMap& opts, const StringOpt* opt);
  void AddOptDefault(Args::OptMap& opts, const IntOpt* opt);
  template <typename EnumT, ssize_t N>
  void AddOptDefault(Args::OptMap& opts, const EnumOpt<EnumT, N>* opt);
  void AddOptDefault(Args::OptMap& opts, const StringListOpt* opt);

  auto AddParsedOptToMap(OptMap& opts, const Opt* opt, OptKind kind)
      -> std::pair<bool, OptKindAndValue&>;
  auto AddParsedOpt(OptMap& opts, const Flag* opt,
                    std::optional<llvm::StringRef> value,
                    llvm::raw_ostream& errors) -> bool;
  auto AddParsedOpt(OptMap& opts, const StringOpt* opt,
                    std::optional<llvm::StringRef> value,
                    llvm::raw_ostream& errors) -> bool;
  auto AddParsedOpt(OptMap& opts, const IntOpt* opt,
                    std::optional<llvm::StringRef> value,
                    llvm::raw_ostream& errors) -> bool;
  template <typename EnumT, ssize_t N>
  auto AddParsedOpt(OptMap& opts, const EnumOpt<EnumT, N>* opt,
                    std::optional<llvm::StringRef> value,
                    llvm::raw_ostream& errors) -> bool;
  auto AddParsedOpt(OptMap& opts, const StringListOpt* opt,
                    std::optional<llvm::StringRef> value,
                    llvm::raw_ostream& errors) -> bool;

  friend auto operator<<(llvm::raw_ostream& out, OptKind kind)
      -> llvm::raw_ostream& {
    switch (kind) {
      case OptKind::Flag:
        out << "Flag";
        break;
      case OptKind::String:
        out << "String";
        break;
      case OptKind::Int:
        out << "Int";
        break;
      case OptKind::Enum:
        out << "Enum";
        break;
      case OptKind::StringList:
        out << "StringList";
        break;
    }
    return out;
  }
};

template <typename SubcommandEnumT>
class SubcommandArgs : public Args {
 public:
  static_assert(std::is_enum_v<SubcommandEnumT>,
                "Must provide an enum type to enumerate subcommands.");
  using SubcommandEnum = SubcommandEnumT;

  auto subcommand() const -> SubcommandEnum { return subcommand_; }

  // Test whether a subcommand flag value is true, either by being set
  // explicitly or having a true default.
  auto TestSubcommandFlag(const Flag* opt) const -> bool {
    return TestFlagImpl(subcommand_opts_, opt);
  }

  // Gets a subcommand string opt's value if available, whether via a default
  // or explicitly set value. If unavailable, returns an empty optional.
  auto GetSubcommandStringOpt(const StringOpt* opt) const
      -> std::optional<llvm::StringRef> {
    return GetStringOptImpl(subcommand_opts_, opt);
  }

  // Gets a subcommand int opt's value if available, whether via a default
  // or explicitly set value. If unavailable, returns an empty optional.
  auto GetSubcommandIntOpt(const IntOpt* opt) const -> std::optional<ssize_t> {
    return GetIntOptImpl(subcommand_opts_, opt);
  }

  // Gets a subcommand enum opt's value if available, whether via a default or
  // explicitly set value. If unavailable, returns an empty optional.
  template <typename EnumT, ssize_t N>
  auto GetSubcommandEnumOpt(const EnumOpt<EnumT, N>* opt) const
      -> std::optional<EnumT> {
    return GetEnumOptImpl(subcommand_opts_, opt);
  }

  auto GetSubcommandStringListOpt(const StringListOpt* opt) const
      -> llvm::ArrayRef<llvm::StringRef> {
    return GetStringListOptImpl(subcommand_opts_, opt);
  }

 private:
  friend class Args;

  SubcommandEnum subcommand_;
  OptMap subcommand_opts_;
};

struct Args::Opt {
  llvm::StringRef name;
  llvm::StringRef short_name = "";

  // The address of a opt is used as the identity after parsing.
  Opt(const Opt&) = delete;
};

struct Args::Flag : Opt {
  bool default_value = false;
};

constexpr inline auto Args::MakeFlag(llvm::StringRef name,
                                     llvm::StringRef short_name,
                                     bool default_value) -> Flag {
  return {{.name = name, .short_name = short_name}, default_value};
}

struct Args::StringOpt : Opt {
  std::optional<llvm::StringRef> default_value = {};
};

constexpr inline auto Args::MakeStringOpt(
    llvm::StringRef name, llvm::StringRef short_name,
    std::optional<llvm::StringRef> default_value) -> StringOpt {
  return {{.name = name, .short_name = short_name}, default_value};
}

struct Args::IntOpt : Opt {
  std::optional<ssize_t> default_value = {};
};

constexpr inline auto Args::MakeIntOpt(llvm::StringRef name,
                                       llvm::StringRef short_name,
                                       std::optional<ssize_t> default_value)
    -> IntOpt {
  return {{.name = name, .short_name = short_name}, default_value};
}

template <typename EnumT>
struct Args::EnumValue {
  llvm::StringRef name;
  EnumT value;
};

template <typename EnumT, ssize_t N>
struct Args::EnumOpt : Opt {
  std::array<EnumValue<EnumT>, N> values;

  std::optional<EnumT> default_value = {};
};

template <typename EnumT, ssize_t N, size_t... Indices>
constexpr inline auto Args::MakeEnumOptHelper(
    llvm::StringRef name, const EnumValue<EnumT> (&args)[N],
    llvm::StringRef short_name, std::optional<EnumT> default_value,
    std::index_sequence<Indices...> /*indices*/) -> EnumOpt<EnumT, N> {
  return {{.name = name, .short_name = short_name},
          {args[Indices]...},
          default_value};
}

template <typename EnumT, ssize_t N>
constexpr inline auto Args::MakeEnumOpt(llvm::StringRef name,
                                        const EnumValue<EnumT> (&args)[N],
                                        llvm::StringRef short_name,
                                        std::optional<EnumT> default_value)
    -> EnumOpt<EnumT, N> {
  return MakeEnumOptHelper(name, args, short_name, default_value,
                           std::make_index_sequence<N>{});
}

struct Args::StringListOpt : Opt {
  std::optional<llvm::ArrayRef<llvm::StringRef>> default_value = {};
};

constexpr inline auto Args::MakeStringListOpt(
    llvm::StringRef name, llvm::StringRef short_name,
    std::optional<llvm::ArrayRef<llvm::StringRef>> default_values)
    -> StringListOpt {
  return {{.name = name, .short_name = short_name}, default_values};
}

template <typename... Ts>
struct Args::Command {
  llvm::StringRef name;

  std::tuple<const Ts*...> opts = {};

  CommandInfo info;
};

template <typename... Ts>
constexpr inline auto Args::MakeCommand(llvm::StringRef name, const Ts*... opts)
    -> Command<Ts...> {
  return {.name = name, .opts = std::tuple{opts...}, .info = {}};
}

template <typename... Ts>
constexpr inline auto Args::MakeCommand(llvm::StringRef name, CommandInfo info,
                                        const Ts*... opts) -> Command<Ts...> {
  return {.name = name, .opts = std::tuple{opts...}, .info = info};
}

template <typename EnumT, typename... OptTs>
struct Args::Subcommand : Command<OptTs...> {
  static_assert(std::is_enum_v<EnumT>,
                "Must provide an enum type to enumerate subcommands.");
  using Enum = EnumT;
  EnumT enumerator;
};

template <typename EnumT, typename... OptTs>
constexpr inline auto Args::MakeSubcommand(llvm::StringRef name,
                                           EnumT enumerator,
                                           const OptTs*... opts)
    -> Subcommand<EnumT, OptTs...> {
  return {{.name = name, .opts = std::tuple{opts...}, .info = {}}, enumerator};
}

template <typename EnumT, typename... OptTs>
constexpr inline auto Args::MakeSubcommand(llvm::StringRef name,
                                           EnumT enumerator, CommandInfo info,
                                           const OptTs*... opts)
    -> Subcommand<EnumT, OptTs...> {
  return {{.name = name, .opts = std::tuple{opts...}, .info = info},
          enumerator};
}

namespace Detail {

template <typename... SubcommandTs>
struct SubcommandEnum;

template <typename SubcommandT, typename... SubcommandTs>
struct SubcommandEnum<SubcommandT, SubcommandTs...> {
  using Type = typename SubcommandT::Enum;
};

enum NoSubcommands {};
template <>
struct SubcommandEnum<> {
  using Type = NoSubcommands;
};

}  // namespace Detail

template <typename CommandT, typename... SubcommandTs>
auto Args::Parse(llvm::ArrayRef<llvm::StringRef> raw_args,
                 llvm::raw_ostream& errors, const CommandT& command,
                 const SubcommandTs&... subcommands) {
  // Extract the enum type from the subcommand types, and ensure it is a single
  // type.
  using SubcommandEnum = typename Detail::SubcommandEnum<SubcommandTs...>::Type;
  constexpr bool HasSubcommands = sizeof...(SubcommandTs) > 0;
  if constexpr (HasSubcommands) {
    static_assert(
        (std::is_same_v<SubcommandEnum, typename SubcommandTs::Enum> && ...),
        "Must have the same enum type for all subcommands.");
  }

  using ArgsType =
      std::conditional_t<HasSubcommands, SubcommandArgs<SubcommandEnum>, Args>;
  ArgsType args;

  // Start in the error state to allow early returns whenever a parse error is
  // found.
  args.parse_result_ = ArgsType::ParseResult::Error;

  using OptParserFunctionT =
      std::function<bool(std::optional<llvm::StringRef> arg_value)>;

  auto* opts = &args.opts_;
  std::forward_list<OptParserFunctionT> parsers;
  llvm::SmallDenseMap<llvm::StringRef, OptParserFunctionT*, 16> opt_parse_map;
  OptParserFunctionT* opt_parse_char_map[128] = {};
  auto add_opt = [&](const auto* opt) {
    auto& parser = parsers.emplace_front(
        [opt, &args, &opts, &errors](std::optional<llvm::StringRef> arg_value) {
          return args.AddParsedOpt(*opts, opt, arg_value, errors);
        });
    bool inserted = opt_parse_map.insert({opt->name, &parser}).second;
    CARBON_CHECK(inserted) << "Duplicate opts named: " << opt->name;
    if (!opt->short_name.empty()) {
      // TODO: extract to a method on `Opt`.
      CARBON_CHECK(opt->short_name.size() == 1)
          << "Option with a short name longer than a single character: "
          << opt->name;
      CARBON_CHECK(llvm::isAlpha(opt->short_name[0]))
          << "Option with a short name that isn't a valid letter in the 'C' "
             "locale: "
          << opt->name;
      int short_index = static_cast<int>(opt->short_name[0]);
      CARBON_CHECK(!opt_parse_char_map[short_index])
          << "Duplicate option short name '" << opt->short_name
          << "' for option: " << opt->name;

      opt_parse_char_map[short_index] = &parser;
    }
    args.AddOptDefault(*opts, opt);
  };
  auto build_opt_parse_map = [&](const auto*... command_flags) {
    // Process the input opts into a lookup table for parsing, also setting up
    // any default values.
    opt_parse_map.clear();

    // Fold over the opts, calling `add_flag` for each one.
    (add_opt(command_flags), ...);
  };
  std::apply(build_opt_parse_map, command.opts);

  // Process the input subcommands into a lookup table. We just handle the
  // subcommand name here to be lazy. We'll process the subcommand itself only
  // if it is needed.
  llvm::SmallDenseMap<llvm::StringRef, std::function<void()>, 16>
      subcommand_parse_map;
  auto parsed_subcommand = [&](const auto* subcommand) {
    if constexpr (HasSubcommands) {
      args.subcommand_ = subcommand->enumerator;
      opts = &args.subcommand_opts_;
      // Rebuild the opt map for this subcommand.
      std::apply(build_opt_parse_map, subcommand->opts);
    }
  };
  if constexpr (HasSubcommands) {
    auto add_subcommand = [&](const auto* subcommand) {
      bool inserted = subcommand_parse_map
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
        auto subcommand_it = subcommand_parse_map.find(arg);
        if (subcommand_it == subcommand_parse_map.end()) {
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
      auto short_args = arg.drop_front();
      std::optional<llvm::StringRef> value = {};
      auto index = short_args.find('=');
      if (index != llvm::StringRef::npos) {
        value = short_args.substr(index + 1);
        short_args = short_args.substr(0, index);
      }
      for (unsigned char c : short_args) {
        if (!llvm::isAlpha(c)) {
          errors << "ERROR: Invalid short option string: '-";
          llvm::printEscapedString(short_args, errors);
          errors << "'\n";
          // TODO: show usage
          return args;
        }
      }
      // All but the last short character are parsed without a value.
      for (unsigned char c : short_args.drop_back()) {
        (*opt_parse_char_map[c])(std::nullopt);
      }
      // The last character gets the value if present.
      (*opt_parse_char_map[static_cast<unsigned char>(short_args.back())])(
          value);
      continue;
    }
    if (arg.size() == 2) {
      // A parameter of `--` disables all opt processing making the remaining
      // args always positional.
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

    auto opt_it = opt_parse_map.find(arg);
    if (opt_it == opt_parse_map.end()) {
      errors << "ERROR: Opt '--" << arg << "' does not exist.\n";
      // TODO: show usage
      return args;
    }
    if (!(*opt_it->second)(value)) {
      // TODO: show usage
      return args;
    }
  }

  // We successfully parsed all the arguments.
  args.parse_result_ = ArgsType::ParseResult::Success;
  return args;
}

template <typename EnumT, ssize_t N>
auto Args::GetEnumOptImpl(const OptMap& opts,
                          const EnumOpt<EnumT, N>* opt) const
    -> std::optional<EnumT> {
  auto opt_iterator = opts.find(opt);
  if (opt_iterator == opts.end()) {
    // No value for this opt.
    return {};
  }
  OptKind kind = opt_iterator->second.kind;
  CARBON_CHECK(kind == OptKind::Enum)
      << "Opt '" << opt->name << "' has inconsistent kinds";
  return static_cast<EnumT>(opt_iterator->second.enum_value);
}

template <typename EnumT, ssize_t N>
void Args::AddOptDefault(Args::OptMap& opts, const EnumOpt<EnumT, N>* opt) {
  if (!opt->default_value.has_value()) {
    return;
  }
  auto [opt_it, inserted] = opts.insert({opt, {.kind = OptKind::String}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of opts!";

  // Make sure any value we store will round-trip through our type erased
  // storage of `int` correctly.
  EnumT enum_value = *opt->default_value;
  int storage_value = static_cast<int>(enum_value);
  CARBON_CHECK(enum_value == static_cast<EnumT>(storage_value))
      << "Default for enum opt '--" << opt->name << "' has a storage value '"
      << storage_value << "' which won't round-trip!";

  opt_it->second.enum_value = storage_value;
}

template <typename EnumT, ssize_t N>
auto Args::AddParsedOpt(OptMap& opts, const EnumOpt<EnumT, N>* opt,
                        std::optional<llvm::StringRef> arg_value,
                        llvm::raw_ostream& errors) -> bool {
  auto [inserted, value] = AddParsedOptToMap(opts, opt, OptKind::Enum);
  if (!arg_value && !opt->default_value) {
    errors << "ERROR: Invalid missing value for the enum opt '--" << opt->name
           << "' which does not have a default value\n";
    return false;
  }
  EnumT enum_value;
  if (arg_value) {
    bool matched_value = false;
    for (ssize_t i = 0; i < N; ++i) {
      if (*arg_value == opt->values[i].name) {
        enum_value = opt->values[i].value;
        matched_value = true;
        break;
      }
    }
    if (!matched_value) {
      errors << "ERROR: Invalid value '" << *arg_value
             << "' for the enum opt '--" << opt->name
             << "', must be one of the following: ";
      for (ssize_t i = 0; i < N; ++i) {
        if (i != 0) {
          errors << ", ";
        }
        errors << opt->values[i].name;
      }
      errors << "\n";
      return false;
    }
  } else {
    enum_value = *opt->default_value;
  }

  // Make sure any value we store will round-trip through our type erased
  // storage of `int` correctly.
  int storage_value = static_cast<int>(enum_value);
  CARBON_CHECK(enum_value == static_cast<EnumT>(storage_value))
      << "Parsed value for enum opt '--" << opt->name
      << "' has a storage value '" << storage_value
      << "' which won't round-trip!";

  value.enum_value = storage_value;
  return true;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_ARGPARSE_H_
