// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/argparse.h"

#include "common/check.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

void Args::SetOptionDefaultImpl(const Flag& flag, bool& value) {
  value = flag.default_value;
}

template <typename OptionT, typename ValueT>
auto SetOptionDefaultImplGeneric(const OptionT& option,
                                 std::optional<ValueT>& value) {
  if (!option.default_value) {
    return;
  }
  value = *option.default_value;
}

void Args::SetOptionDefaultImpl(const StringOpt& option,
                                std::optional<llvm::StringRef>& value) {
  SetOptionDefaultImplGeneric(option, value);
}

void Args::SetOptionDefaultImpl(const IntOpt& option,
                                std::optional<ssize_t>& value) {
  SetOptionDefaultImplGeneric(option, value);
}

void Args::SetOptionDefaultImpl(const StringListOpt& option,
                                llvm::SmallVectorImpl<llvm::StringRef>& value) {
  if (option.default_values.empty()) {
    return;
  }
  value.assign(option.default_values.begin(), option.default_values.end());
}

auto Args::ParseOptionImpl(const Flag& flag, std::optional<llvm::StringRef> arg,
                           llvm::raw_ostream& errors, bool& value) -> bool {
  if (!arg || *arg == "true") {
    value = true;
    return true;
  }
  if (*arg == "false") {
    value = false;
    return true;
  }
  errors << "ERROR: Invalid value '" << *arg << "' provided for the flag '--"
         << flag.name << "'\n";
  return false;
}

auto Args::ParseOptionImpl(const StringOpt& opt,
                           std::optional<llvm::StringRef> arg,
                           llvm::raw_ostream& errors,
                           std::optional<llvm::StringRef>& value) -> bool {
  if (!arg && !opt.default_value) {
    errors << "ERROR: Invalid missing value for the string opt '--" << opt.name
           << "' which does not have a default value\n";
    return false;
  }
  value = arg ? *arg : static_cast<llvm::StringRef>(*opt.default_value);
  return true;
}

auto Args::ParseOptionImpl(const IntOpt& opt,
                           std::optional<llvm::StringRef> arg_value,
                           llvm::raw_ostream& errors,
                           std::optional<ssize_t>& value) -> bool {
  if (!arg_value && !opt.default_value) {
    errors << "ERROR: Invalid missing value for the integer option '--"
           << opt.name << "' which does not have a default value\n";
    return false;
  }
  ssize_t value_int;
  if (arg_value) {
    // Note that LLVM's function for parsing as an integer confusingly returns
    // true *on an error* in parsing.
    if (arg_value->getAsInteger(/*Radix=*/0, value_int)) {
      errors << "ERROR: Unable to parse integer option '--" << opt.name
             << "' value '" << *arg_value << "' as an integer\n";
      return false;
    }
  } else {
    value_int = *opt.default_value;
  }
  value = value_int;
  return true;
}

auto Args::ParseOptionImpl(const StringListOpt& opt,
                           std::optional<llvm::StringRef> arg_value,
                           llvm::raw_ostream& errors,
                           llvm::SmallVectorImpl<llvm::StringRef>& value)
    -> bool {
  if (!arg_value) {
    errors << "ERROR: Invalid missing value for the string list option '--"
           << opt.name << "'\n";
    return false;
  }
  value.push_back(*arg_value);
  return true;
}

auto Args::ParseOneArg(
    llvm::StringRef arg, llvm::ArrayRef<llvm::StringRef>& remaining_args,
    llvm::raw_ostream& errors,
    llvm::function_ref<bool(llvm::StringRef)> parse_subcommand,
    llvm::function_ref<bool(llvm::StringRef name,
                            std::optional<llvm::StringRef> value)>
        parse_option,
    llvm::function_ref<bool(unsigned char c,
                            std::optional<llvm::StringRef> value)>
        parse_short_option) -> bool {
  if (arg[0] != '-' || arg.size() <= 1) {
    if (parse_subcommand) {
      return parse_subcommand(arg);
    }

    positional_args_.push_back(arg);
    return true;
  }

  if (arg[1] == '-') {
    if (arg.size() == 2) {
      // A parameter of `--` consumes all remaining args as positional ones.
      positional_args_.append(remaining_args.begin(), remaining_args.end());
      remaining_args = {};
      return true;
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

    return parse_option(arg, value);
  }

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
      return false;
    }
  }
  // All but the last short character are parsed without a value.
  for (unsigned char c : short_args.drop_back()) {
    if (!parse_short_option(c, std::nullopt)) {
      return false;
    }
  }
  // The last character gets the value if present.
  return parse_short_option(static_cast<unsigned char>(short_args.back()),
                            value);
}

}  // namespace Carbon
