// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_ARGPARSE_H_
#define CARBON_COMMON_ARGPARSE_H_

#include <tuple>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon {

namespace Detail {

template <const auto& Value>
using TypeOfValue = llvm::remove_cvref_t<decltype(Value)>;

template <const auto& Param>
struct ValueHolderT {
  constexpr static auto& Value = Param;
};

template <typename EnumOptValuesT>
struct EnumOptTypeImpl;

template <typename EnumOptValueT, typename... EnumOptValueTs>
struct EnumOptTypeImpl<std::tuple<EnumOptValueT, EnumOptValueTs...>> {
  using Type = decltype(EnumOptValueT::Enumerator);

  static_assert(std::is_enum_v<Type>,
                "Must use an enum value for the enumerator.");
  static_assert((std::is_same_v<Type, decltype(EnumOptValueTs::Enumerator)> &&
                 ...),
                "Must have the same enum type for all enum option values!");
};

template <const auto& EnumOptValues>
using EnumOptType = std::remove_const_t<
    typename EnumOptTypeImpl<TypeOfValue<EnumOptValues>>::Type>;

}  // namespace Detail

class Args {
 public:
  struct Flag {
    llvm::StringLiteral name;
    llvm::StringLiteral short_name = "";

    bool default_value = false;
  };

  struct StringOpt {
    llvm::StringLiteral name;
    llvm::StringLiteral short_name = "";

    std::optional<llvm::StringRef> default_value = {};
  };

  struct IntOpt {
    llvm::StringLiteral name;
    llvm::StringLiteral short_name = "";

    std::optional<ssize_t> default_value = {};
  };

  template <auto EnumeratorV>
  struct EnumOptValue {
    constexpr static auto Enumerator = EnumeratorV;
    static_assert(std::is_enum_v<decltype(Enumerator)>,
                  "Must use an enum value for the enumerator.");

    llvm::StringLiteral name;
  };

  template <const auto& EnumOptValues>
  struct EnumOpt {
    constexpr static auto& Values = EnumOptValues;
    using EnumT = Detail::EnumOptType<Values>;

    llvm::StringLiteral name;
    llvm::StringLiteral short_name = "";

    std::optional<EnumT> default_value = {};
  };

  template <const auto& EnumOptValues>
  using EnumOptEnumT = typename EnumOpt<EnumOptValues>::EnumT;

  struct StringListOpt {
    llvm::StringLiteral name;
    llvm::StringLiteral short_name = "";

    llvm::ArrayRef<llvm::StringLiteral> default_values = {};
  };

  struct CommandInfo {
    llvm::StringLiteral description = "";
    llvm::StringLiteral usage = "";
    llvm::StringLiteral epilog = "";
  };
  template <const auto&... Options>
  struct Command {
    llvm::StringLiteral name;

    CommandInfo info;

    // Implementation detail to hold the template parameters. Never needs to be
    // initialized by users.
    std::tuple<Detail::ValueHolderT<Options>...> options = {};
  };

  template <auto EnumeratorV, const auto&... Options>
  struct Subcommand {
    // Capture the template parameter into a usable name.
    constexpr static auto Enumerator = EnumeratorV;
    static_assert(std::is_enum_v<decltype(Enumerator)>,
                  "Must use an enum to enumerate subcommands.");

    llvm::StringLiteral name;

    CommandInfo info;

    // Implementation detail to hold the template parameters. Never needs to be
    // initialized by users.
    std::tuple<Detail::ValueHolderT<Options>...> options = {};
  };

  template <auto EnumeratorV, const auto&... Options>
  struct Subcommand;

  template <const auto& CommandT, const auto&... Subcommands>
  class Parser;

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

  auto positional_args() const -> llvm::ArrayRef<llvm::StringRef> {
    return positional_args_;
  }

 protected:
  template <const auto& CommandT, const auto&... Subcommands>
  friend class Parser;

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

  ParseResult parse_result_;

  llvm::SmallVector<llvm::StringRef, 12> positional_args_;

  void SetOptionDefaultImpl(const Flag& flag, bool& value);
  void SetOptionDefaultImpl(const StringOpt& option,
                            std::optional<llvm::StringRef>& value);
  void SetOptionDefaultImpl(const IntOpt& option,
                            std::optional<ssize_t>& value);
  template <const auto& EnumOptValues>
  void SetOptionDefaultImpl(const EnumOpt<EnumOptValues>& option,
                            std::optional<EnumOptEnumT<EnumOptValues>>& value);
  void SetOptionDefaultImpl(const StringListOpt& option,
                            llvm::SmallVectorImpl<llvm::StringRef>& value);

  auto ParseOptionImpl(const Flag& flag, std::optional<llvm::StringRef> arg,
                       llvm::raw_ostream& errors, bool& value) -> bool;
  auto ParseOptionImpl(const StringOpt& opt, std::optional<llvm::StringRef> arg,
                       llvm::raw_ostream& errors,
                       std::optional<llvm::StringRef>& value) -> bool;
  auto ParseOptionImpl(const IntOpt& opt, std::optional<llvm::StringRef> arg,
                       llvm::raw_ostream& errors, std::optional<ssize_t>& value)
      -> bool;
  template <const auto& EnumOptValues, const EnumOpt<EnumOptValues>& Option>
  auto ParseOptionImpl(std::optional<llvm::StringRef> arg_value,
                       llvm::raw_ostream& errors,
                       std::optional<EnumOptEnumT<EnumOptValues>>& value)
      -> bool;
  auto ParseOptionImpl(const StringListOpt& opt,
                       std::optional<llvm::StringRef> arg,
                       llvm::raw_ostream& errors,
                       llvm::SmallVectorImpl<llvm::StringRef>& value) -> bool;

  auto ParseOneArg(
      llvm::StringRef arg, llvm::ArrayRef<llvm::StringRef>& remaining_args,
      llvm::raw_ostream& errors,
      llvm::function_ref<bool(llvm::StringRef)> parse_subcommand,
      llvm::function_ref<bool(llvm::StringRef name,
                              std::optional<llvm::StringRef> value)>
          parse_option,
      llvm::function_ref<bool(unsigned char c,
                              std::optional<llvm::StringRef> value)>
          parse_short_option) -> bool;

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

namespace Detail {

template <typename SubcommandEnumT>
class SubcommandImpl {
 public:
  static_assert(std::is_enum_v<SubcommandEnumT>,
                "Must provide an enum type to enumerate subcommands.");
  using SubcommandEnum = SubcommandEnumT;

  auto subcommand() const -> SubcommandEnum { return subcommand_; }

 private:
  friend Args;

  SubcommandEnum subcommand_ = {};
};

enum NoSubcommands {};

template <>
class SubcommandImpl<NoSubcommands> {
  friend class Args;
};

template <typename OptionT>
struct OptionValue;

template <>
struct OptionValue<Args::Flag> {
  bool value = false;
};

template <>
struct OptionValue<Args::StringOpt> {
  std::optional<llvm::StringRef> value = {};
};

template <>
struct OptionValue<Args::IntOpt> {
  std::optional<ssize_t> value = {};
};

template <const auto& EnumOptValues>
struct OptionValue<Args::EnumOpt<EnumOptValues>> {
  std::optional<Args::EnumOptEnumT<EnumOptValues>> value = {};
};

template <>
struct OptionValue<Args::StringListOpt> {
  llvm::SmallVector<llvm::StringRef, 4> value = {};
};

template <typename LHSOptionT, typename RHSOptionT, const LHSOptionT& LHSOption,
          const RHSOptionT& RHSOption>
struct IsSameOptionImpl {
  constexpr static bool Value = false;
};

template <typename OptionT, const OptionT& LHSOption, const OptionT& RHSOption>
struct IsSameOptionImpl<OptionT, OptionT, LHSOption, RHSOption> {
  constexpr static bool Value = &LHSOption == &RHSOption;
};

template <const auto& LHSOption, const auto& RHSOption>
using IsSameOption = IsSameOptionImpl<llvm::remove_cvref_t<decltype(LHSOption)>,
                                      llvm::remove_cvref_t<decltype(RHSOption)>,
                                      LHSOption, RHSOption>;

template <const auto& Option, const auto&... Options, size_t... Is>
constexpr inline auto FindIndexForOption(std::index_sequence<Is...> /*indices*/)
    -> size_t {
  static_assert(
      (static_cast<size_t>(IsSameOption<Option, Options>::Value) + ...) > 0,
      "Queried an option that was not parsed into the program arguments!");
  static_assert(
      (static_cast<size_t>(IsSameOption<Option, Options>::Value) + ...) == 1,
      "Queried an option that was provided multiple times to the "
      "program arguments!");
  return ((static_cast<size_t>(IsSameOption<Option, Options>::Value) * Is) +
          ...);
}

template <const auto& Option>
struct ParseOptionImpl {
  template <typename ArgsT, typename ValueT>
  static auto Parse(ArgsT& args, std::optional<llvm::StringRef> arg_value,
                    llvm::raw_ostream& errors, ValueT& value) -> bool {
    return args.ParseOptionImpl(Option, arg_value, errors, value);
  }
};

template <const auto& EnumOptValues, const Args::EnumOpt<EnumOptValues>& Option>
struct ParseOptionImpl<Option> {
  template <typename ArgsT>
  static auto Parse(ArgsT& args, std::optional<llvm::StringRef> arg_value,
                    llvm::raw_ostream& errors,
                    std::optional<Args::EnumOptEnumT<EnumOptValues>>& value)
      -> bool {
    return args.ParseOptionImpl<EnumOptValues, Option>(arg_value, errors,
                                                       value);
  }
};

}  // namespace Detail

template <typename SubcommandEnumT, const auto&... Options>
class ArgsImpl : public Args, public Detail::SubcommandImpl<SubcommandEnumT> {
 public:
  // Test whether a flag value is true, either by being set explicitly or having
  // a true default.
  template <const Flag& F>
  auto TestFlag() const -> bool {
    constexpr size_t N = GetOptionIndex<F>();
    return std::get<N>(option_values_).value;
  }

  // Gets an option value if available, whether via a default or explicitly set
  // value. If unavailable, returns an empty optional. Cannot be used with
  // flags, only with options that store an optional value.
  template <const auto& Option>
  auto GetOption() const {
    static_assert(!std::is_same_v<Detail::TypeOfValue<Option>, Flag>,
                  "Use `TestFlag` with flag options which avoids the "
                  "`std::optional` wrapping.");
    constexpr size_t N = GetOptionIndex<Option>();
    return std::get<N>(option_values_).value;
  }

 private:
  friend Args;
  template <const auto& Option>
  friend struct Detail::ParseOptionImpl;

  std::tuple<Detail::OptionValue<llvm::remove_cvref_t<decltype(Options)>>...>
      option_values_;

  template <const auto& Option>
  constexpr static auto GetOptionIndex() -> size_t {
    return Detail::FindIndexForOption<Option, Options...>(
        std::make_index_sequence<sizeof...(Options)>{});
  }

  template <const auto& Option>
  auto SetOptionDefault() -> void {
    constexpr size_t N = GetOptionIndex<Option>();
    return SetOptionDefaultImpl(Option, std::get<N>(option_values_).value);
  }

  template <const auto& Option>
  auto ParseOption(std::optional<llvm::StringRef> arg_value,
                   llvm::raw_ostream& errors) -> bool {
    constexpr size_t N = GetOptionIndex<Option>();
    return Detail::ParseOptionImpl<Option>::Parse(
        *this, arg_value, errors, std::get<N>(option_values_).value);
  }

  template <typename OptionHoldersT>
  auto ParseOptions(llvm::StringRef name, std::optional<llvm::StringRef> value,
                    llvm::raw_ostream& errors, OptionHoldersT options) -> bool {
    auto parse_option_impl = [&](auto holder) {
      constexpr auto& Option = decltype(holder)::Value;
      if (name != Option.name) {
        return false;
      }
      return ParseOption<Option>(value, errors);
    };
    auto parse_options_impl = [&](auto... holders) {
      return (parse_option_impl(holders) || ...);
    };
    if (std::apply(parse_options_impl, options)) {
      return true;
    }

    // Otherwise, no option name matched so diagnose this.
    errors << "ERROR: Invalid option '--" << name << "'\n";
    return false;
  }

  template <typename OptionHoldersT>
  auto ParseShortOptions(unsigned char c, std::optional<llvm::StringRef> value,
                         llvm::raw_ostream& errors, OptionHoldersT options)
      -> bool {
    auto parse_short_option_impl = [&](auto holder) {
      constexpr auto& Option = decltype(holder)::Value;
      constexpr llvm::StringLiteral ShortName = Option.short_name;
      if constexpr (!ShortName.empty()) {
        if (c == static_cast<unsigned char>(ShortName[0])) {
          return ParseOption<Option>(value, errors);
        }
      }
      return false;
    };
    auto parse_short_options_impl = [&](auto... holders) {
      return (parse_short_option_impl(holders) || ...);
    };
    if (std::apply(parse_short_options_impl, options)) {
      return true;
    }

    // Otherwise, no option name matched so diagnose this.
    errors << "ERROR: Invalid short option character '" << c << "'\n";
    return false;
  }
};

namespace Detail {

template <const auto&... Subcommands>
struct SubcommandEnum;

template <const auto& Subcommand, const auto&... Subcommands>
struct SubcommandEnum<Subcommand, Subcommands...> {
  using Type = TypeOfValue<Subcommand.Enumerator>;
};

template <>
struct SubcommandEnum<> {
  using Type = NoSubcommands;
};

template <typename SubcommandEnumT, typename T>
struct ArgsImplFromHolderTuple;

template <typename SubcommandEnumT, const auto&... Options>
struct ArgsImplFromHolderTuple<SubcommandEnumT,
                               std::tuple<ValueHolderT<Options>...>> {
  using ArgsImplT = ArgsImpl<SubcommandEnumT, Options...>;
};

// Build some utilities to unique the options in the options tuple as
// different subcommands can share a flag.
// FIXME: This seems likely to be ... very expensive in terms of compile time,
// but more efficient approaches seem very complex.
struct MergeOptionHolders {
  template <typename... OptionHolderTs>
  auto operator()(OptionHolderTs... initial_holders) {
    if constexpr (sizeof...(initial_holders) == 0) {
      return std::tuple<>{};
    } else {
      auto impl = [](auto self, auto holder, auto... holders) {
        if constexpr (sizeof...(holders) == 0) {
          return std::tuple<decltype(holder)>{};
        } else {
          if constexpr ((std::is_same_v<decltype(holder), decltype(holders)> ||
                         ...)) {
            return self(self, holders...);
          } else {
            return std::tuple_cat(std::tuple<decltype(holder)>{},
                                  self(self, holders...));
          }
        }
      };
      return impl(impl, initial_holders...);
    }
  }
};

// Now use both the enum and the merge tools to collect all the options across
// subcommands and build the specific implementation type used for our parsed
// arguments.
template <typename SubcommandEnum, const auto&... Commands>
using ArgsImplT = typename Detail::ArgsImplFromHolderTuple<
    SubcommandEnum,
    decltype(std::apply(MergeOptionHolders{},
                        std::tuple_cat(Commands.options...)))>::ArgsImplT;

}  // namespace Detail

template <const auto& ThisCommand, const auto&... Subcommands>
class Args::Parser {
 public:
  constexpr static bool HasSubcommands = sizeof...(Subcommands) > 0;

  // Extract the enum type from the subcommand types, and ensure it is a single
  // type.
  using SubcommandEnum = typename Detail::SubcommandEnum<Subcommands...>::Type;
  static_assert((std::is_same_v<SubcommandEnum,
                                Detail::TypeOfValue<Subcommands.Enumerator>> &&
                 ...),
                "Must have the same enum type for all subcommands.");

  // Now use both the enum and the merge tools to collect all the options across
  // subcommands and build the specific implementation type used for our parsed
  // arguments.
  using ArgsT = Detail::ArgsImplT<SubcommandEnum, ThisCommand, Subcommands...>;

  static auto Parse(llvm::ArrayRef<llvm::StringRef> raw_args,
                    llvm::raw_ostream& errors) -> ArgsT;

 private:
  // Compile time checking for duplicate options within a command.
  template <typename OptionHoldersT>
  constexpr static auto TestForDuplicateOptions(OptionHoldersT option_holders)
      -> bool {
    if constexpr ((std::tuple_size_v<OptionHoldersT>) > 0) {
      auto impl = [](auto self, auto holder, auto... holders) {
        if constexpr (sizeof...(holders) > 0) {
          return (std::is_same_v<decltype(holder), decltype(holders)> || ...) ||
                 self(self, holders...);
        } else {
          return false;
        }
      };
      return std::apply(
          [impl](auto... holders) { return impl(impl, holders...); },
          option_holders);
    } else {
      return false;
    }
  };
  static_assert(!TestForDuplicateOptions(ThisCommand.options),
                "Found a duplicate option within a single command!");
  static_assert(!(TestForDuplicateOptions(Subcommands.options) || ...),
                "Found a duplicate option within a single subcommand!");
};

template <const auto& ThisCommand, const auto&... Subcommands>
auto Args::Parser<ThisCommand, Subcommands...>::Parse(
    llvm::ArrayRef<llvm::StringRef> raw_args, llvm::raw_ostream& errors)
    -> ArgsT {
  ArgsT args;

  // Start in the error state to allow early returns whenever a parse error is
  // found.
  args.parse_result_ = ArgsT::ParseResult::Error;

  auto set_option_defaults = [&args](auto... command_options) {
    (args.template SetOptionDefault<decltype(command_options)::Value>(), ...);
  };
  std::apply(set_option_defaults, ThisCommand.options);

  auto parse_args = [&](const auto& parse_subcommand, const auto& parse_option,
                        const auto& parse_short_option) {
    while (!raw_args.empty()) {
      llvm::StringRef arg = raw_args.front();
      raw_args = raw_args.drop_front();
      if (!args.ParseOneArg(arg, raw_args, errors, parse_subcommand,
                            parse_option, parse_short_option)) {
        return false;
      }
    }
    return true;
  };

  auto parse_options = [&](llvm::StringRef name,
                           std::optional<llvm::StringRef> value) -> bool {
    return args.ParseOptions(name, value, errors, ThisCommand.options);
  };
  auto parse_short_options = [&](unsigned char c,
                                 std::optional<llvm::StringRef> value) -> bool {
    return args.ParseShortOptions(c, value, errors, ThisCommand.options);
  };

  if constexpr (HasSubcommands) {
    auto parse_subcommand_args = [&](auto holder) {
      constexpr auto& Subcommand = decltype(holder)::Value;

      args.subcommand_ = Subcommand.Enumerator;
      std::apply(set_option_defaults, Subcommand.options);

      // Finish parsing args, but with the subcommand's options.
      auto parse_subcommand_options =
          [&](llvm::StringRef name,
              std::optional<llvm::StringRef> value) -> bool {
        return args.ParseOptions(name, value, errors, Subcommand.options);
      };
      auto parse_subcommand_short_options =
          [&](unsigned char c, std::optional<llvm::StringRef> value) -> bool {
        return args.ParseShortOptions(c, value, errors, Subcommand.options);
      };

      return parse_args(nullptr, parse_subcommand_options,
                        parse_subcommand_short_options);
    };
    auto parse_subcommands = [&](llvm::StringRef arg) -> bool {
      auto parse_subcommand_impl = [&](auto holder) {
        constexpr auto& Subcommand = decltype(holder)::Value;
        if (arg != Subcommand.name) {
          return false;
        }

        return parse_subcommand_args(holder);
      };
      if ((parse_subcommand_impl(Detail::ValueHolderT<Subcommands>{}) || ...)) {
        return true;
      }

      // Otherwise, did not match any of the subcommand names.
      errors << "ERROR: Invalid subcommand '" << arg << "'\n";
      return false;
    };
    if (!parse_args(parse_subcommands, parse_options, parse_short_options)) {
      // TODO: usage
      return args;
    }
  } else {
    if (!parse_args(nullptr, parse_options, parse_short_options)) {
      // TODO: usage
      return args;
    }
  }

  // We successfully parsed all the arguments.
  args.parse_result_ = ArgsT::ParseResult::Success;
  return args;
}

template <const auto& EnumOptValues>
void Args::SetOptionDefaultImpl(
    const EnumOpt<EnumOptValues>& option,
    std::optional<EnumOptEnumT<EnumOptValues>>& value) {
  if (!option.default_value) {
    return;
  }
  value = *option.default_value;
}

namespace Detail {

template <typename Indices>
struct IndexSeqTypesHelper;

template <size_t... Indices>
struct IndexSeqTypesHelper<std::index_sequence<Indices...>> {
  using Type = std::tuple<std::integral_constant<size_t, Indices>...>;
};

template <size_t N>
constexpr inline typename IndexSeqTypesHelper<std::make_index_sequence<N>>::Type
    IndexConstantSeq = {};

}  // namespace Detail

template <const auto& EnumOptValues, const Args::EnumOpt<EnumOptValues>& Option>
auto Args::ParseOptionImpl(std::optional<llvm::StringRef> arg_value,
                           llvm::raw_ostream& errors,
                           std::optional<EnumOptEnumT<EnumOptValues>>& value)
    -> bool {
  if (!arg_value && !Option.default_value) {
    errors << "ERROR: Invalid missing value for the enum option '--"
           << Option.name << "' which does not have a default value\n";
    return false;
  }
  if (!arg_value) {
    value = *Option.default_value;
    return true;
  }

  using EnumT = EnumOptEnumT<EnumOptValues>;
  constexpr size_t NumValues =
      std::tuple_size_v<Detail::TypeOfValue<EnumOptValues>>;
  static_assert(NumValues > 0, "Enum options must provide values to parse.");

  auto parse_value = [&](auto i) {
    constexpr auto& EnumValue = std::get<decltype(i)::value>(EnumOptValues);
    constexpr EnumT Enumerator = EnumValue.Enumerator;
    constexpr llvm::StringLiteral Name = EnumValue.name;
    if (Name != arg_value) {
      return false;
    }

    value = Enumerator;
    return true;
  };
  auto parse_values = [&](auto... indices) {
    return (parse_value(indices) || ...);
  };
  if (std::apply(parse_values, Detail::IndexConstantSeq<NumValues>)) {
    return true;
  }

  errors << "ERROR: Invalid value '" << *arg_value << "' for the enum opt '--"
         << Option.name << "', must be one of the following: ";
  auto print_enum_value_name = [&](auto i) {
    constexpr auto& EnumValue = std::get<decltype(i)::value>(EnumOptValues);
    constexpr llvm::StringLiteral Name = EnumValue.name;
    errors << Name;
  };
  auto print_enum_value_names = [&](auto zero, auto... indices) {
    print_enum_value_name(zero);
    ((errors << ", ", print_enum_value_name(indices)), ...);
  };
  std::apply(print_enum_value_names, Detail::IndexConstantSeq<NumValues>);
  errors << "\n";
  return false;
}

}  // namespace Carbon

#endif  // CARBON_COMMON_ARGPARSE_H_
