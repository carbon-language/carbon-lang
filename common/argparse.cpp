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

auto Args::GetStringFlagImpl(const FlagMap& flags, const StringFlag* flag) const
    -> std::optional<llvm::StringRef> {
  auto flag_iterator = flags.find(flag);
  if (flag_iterator == flags.end()) {
    // No value for this flag.
    return {};
  }
  FlagKind kind = flag_iterator->second.kind;
  CARBON_CHECK(kind == FlagKind::String)
      << "Flag '" << flag->name << "' has inconsistent kinds";
  return string_flag_values_[flag_iterator->second.value_index];
}

auto Args::GetStringListFlagImpl(const FlagMap& flags,
                                 const StringListFlag* flag) const
    -> llvm::ArrayRef<llvm::StringRef> {
  auto flag_iterator = flags.find(flag);
  if (flag_iterator == flags.end()) {
    // An empty sequence is always used when list flags are absent.
    return {};
  }
  FlagKind kind = flag_iterator->second.kind;
  CARBON_CHECK(kind == FlagKind::StringList)
      << "Flag '" << flag->name << "' has inconsistent kinds";
  return string_list_flag_values_[flag_iterator->second.value_index];
}

void Args::AddFlagDefault(Args::FlagMap& flags, const BooleanFlag* flag) {
  auto [flag_it, inserted] = flags.insert({flag, {.kind = FlagKind::Boolean}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of flags!";
  auto& value = flag_it->second;
  value.boolean_value = flag->default_value;
}

void Args::AddFlagDefault(Args::FlagMap& flags, const StringFlag* flag) {
  if (!flag->default_value.has_value()) {
    return;
  }
  auto [flag_it, inserted] = flags.insert({flag, {.kind = FlagKind::String}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of flags!";
  auto& value = flag_it->second;
  value.value_index = string_flag_values_.size();
  string_flag_values_.push_back(*flag->default_value);
}

void Args::AddFlagDefault(Args::FlagMap& flags, const StringListFlag* flag) {
  if (flag->default_values.empty()) {
    return;
  }
  auto [flag_it, inserted] = flags.insert({flag, {.kind = FlagKind::StringList}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of flags!";
  auto& value = flag_it->second;
  value.value_index = string_list_flag_values_.size();
  string_list_flag_values_.emplace_back(flag->default_values.begin(),
                                        flag->default_values.end());
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
