// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/argparse.h"

#include "common/check.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

auto Args::TestFlagImpl(const FlagMap& flags, const BooleanFlag* flag) const -> bool {
  auto flag_iterator = flags.find(flag);
  CARBON_CHECK(flag_iterator != flags.end()) << "Invalid flag: " << flag->name;
  FlagKind kind = flag_iterator->second.kind;
  CARBON_CHECK(kind == FlagKind::Boolean)
      << "Flag '" << flag->name << "' has inconsistent kinds";
  return flag_iterator->second.boolean_value;
}

template <typename ValueT, typename FlagMapT, typename FlagT,
          typename FlagKindT, typename ValuesT>
static auto GetFlagGenericImpl(const FlagMapT& flags, const FlagT* flag,
                               FlagKindT kind, const ValuesT& values)
    -> std::optional<ValueT> {
  auto flag_iterator = flags.find(flag);
  if (flag_iterator == flags.end()) {
    // No value for this flag.
    return {};
  }
  FlagKindT stored_kind = flag_iterator->second.kind;
  CARBON_CHECK(stored_kind == kind)
      << "Flag '" << flag->name << "' has inconsistent kinds: expected '"
      << kind << "' but found '" << stored_kind << "'";
  return values[flag_iterator->second.value_index];
}

auto Args::GetStringFlagImpl(const FlagMap& flags, const StringFlag* flag) const
    -> std::optional<llvm::StringRef> {
  return GetFlagGenericImpl<llvm::StringRef>(flags, flag, FlagKind::String,
                                             string_flag_values_);
}

auto Args::GetIntFlagImpl(const FlagMap& flags, const IntFlag* flag) const
    -> std::optional<ssize_t> {
  return GetFlagGenericImpl<ssize_t>(flags, flag, FlagKind::Int,
                                     int_flag_values_);
}

auto Args::GetStringListFlagImpl(const FlagMap& flags,
                                 const StringListFlag* flag) const
    -> llvm::ArrayRef<llvm::StringRef> {
  if (auto flag_values = GetFlagGenericImpl<llvm::ArrayRef<llvm::StringRef>>(
          flags, flag, FlagKind::StringList, string_list_flag_values_)) {
    return *flag_values;
  }
  return {};
}

void Args::AddFlagDefault(Args::FlagMap& flags, const BooleanFlag* flag) {
  auto [flag_it, inserted] = flags.insert({flag, {.kind = FlagKind::Boolean}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of flags!";
  auto& value = flag_it->second;
  value.boolean_value = flag->default_value;
}

template <typename FlagMapT, typename FlagT, typename FlagKindT,
          typename ValuesT>
auto AddFlagDefaultGeneric(FlagMapT& flags, const FlagT* flag, FlagKindT kind,
                           ValuesT& values) {
  if (!flag->default_value) {
    return;
  }
  auto [flag_it, inserted] = flags.insert({flag, {.kind = kind}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of flags!";
  auto& value = flag_it->second;
  value.value_index = values.size();
  values.emplace_back(*flag->default_value);
}

void Args::AddFlagDefault(Args::FlagMap& flags, const StringFlag* flag) {
  AddFlagDefaultGeneric(flags, flag, FlagKind::String, string_flag_values_);
}

void Args::AddFlagDefault(Args::FlagMap& flags, const IntFlag* flag) {
  AddFlagDefaultGeneric(flags, flag, FlagKind::Int, int_flag_values_);
}

void Args::AddFlagDefault(Args::FlagMap& flags, const StringListFlag* flag) {
  AddFlagDefaultGeneric(flags, flag, FlagKind::StringList, string_list_flag_values_);
}

auto Args::AddParsedFlagToMap(FlagMap& flags, const Flag* flag, FlagKind kind)
    -> std::pair<bool, FlagKindAndValue&> {
  auto [flag_it, inserted] = flags.insert({flag, {.kind = kind}});
  auto& value = flag_it->second;
  if (!inserted) {
    CARBON_CHECK(value.kind == kind)
        << "Inconsistent flag kind for repeated flag '--" << flag->name
        << "': originally '" << value.kind << "' and now '" << kind << "'";
  }
  return {inserted, value};
}

auto Args::AddParsedFlag(FlagMap& flags, const BooleanFlag* flag,
                         std::optional<llvm::StringRef> arg_value, llvm::raw_ostream& errors)
    -> bool {
  auto [_, value] = AddParsedFlagToMap(flags, flag, FlagKind::Boolean);
  if (!arg_value || *arg_value == "true") {
    value.boolean_value = true;
    return true;
  }
  if (*arg_value == "false") {
    value.boolean_value = false;
    return true;
  }
  errors << "ERROR: Invalid value '" << *arg_value
         << "' provided for the boolean flag '--" << flag->name << "'\n";
  return false;
}

auto Args::AddParsedFlag(FlagMap& flags, const StringFlag* flag,
                         std::optional<llvm::StringRef> arg_value, llvm::raw_ostream& errors)
    -> bool {
  auto [inserted, value] = AddParsedFlagToMap(flags, flag, FlagKind::String);
  if (!arg_value && !flag->default_value) {
    errors << "ERROR: Invalid missing value for the string flag '--"
           << flag->name << "' which does not have a default value\n";
    return false;
  }
  llvm::StringRef value_str =
      arg_value ? *arg_value
                : static_cast<llvm::StringRef>(*flag->default_value);
  if (inserted) {
    value.value_index = string_flag_values_.size();
    string_flag_values_.push_back(value_str);
  } else {
    string_flag_values_[value.value_index] = value_str;
  }
  return true;
}

auto Args::AddParsedFlag(FlagMap& flags, const IntFlag* flag,
                         std::optional<llvm::StringRef> arg_value, llvm::raw_ostream& errors)
    -> bool {
  auto [inserted, value] = AddParsedFlagToMap(flags, flag, FlagKind::Int);
  if (!arg_value && !flag->default_value) {
    errors << "ERROR: Invalid missing value for the int flag '--"
           << flag->name << "' which does not have a default value\n";
    return false;
  }
  ssize_t value_int;
  if (arg_value) {
    // Note that LLVM's function for parsing as an integer confusingly returns
    // true *on an error* in parsing.
    if (arg_value->getAsInteger(/*Radix=*/0, value_int)) {
      errors << "ERROR: Unable to parse int flag '--" << flag->name << "' value '"
             << *arg_value << "' as an integer\n";
      return false;
    }
  } else {
    value_int = *flag->default_value;
  }
  if (inserted) {
    value.value_index = int_flag_values_.size();
    int_flag_values_.push_back(value_int);
  } else {
    int_flag_values_[value.value_index] = value_int;
  }
  return true;
}

auto Args::AddParsedFlag(FlagMap& flags, const StringListFlag* flag,
                         std::optional<llvm::StringRef> arg_value, llvm::raw_ostream& errors)
    -> bool {
  auto [inserted, value] = AddParsedFlagToMap(flags, flag, FlagKind::StringList);
  if (!arg_value) {
    errors << "ERROR: Must specify a value for the string list flag '--"
           << flag->name << "'\n";
    return false;
  }
  if (inserted) {
    value.value_index = string_list_flag_values_.size();
    string_list_flag_values_.push_back({*arg_value});
  } else {
    string_list_flag_values_[value.value_index].push_back(*arg_value);
  }
  return true;
}

}  // namespace Carbon
