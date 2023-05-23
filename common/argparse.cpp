// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/argparse.h"

#include "common/check.h"
#include "llvm/ADT/PointerUnion.h"

namespace Carbon {

auto Args::TestFlagImpl(const OptMap& opts, const Flag* opt) const
    -> bool {
  auto opt_iterator = opts.find(opt);
  CARBON_CHECK(opt_iterator != opts.end()) << "Invalid opt: " << opt->name;
  OptKind kind = opt_iterator->second.kind;
  CARBON_CHECK(kind == OptKind::Flag)
      << "Opt '" << opt->name << "' has inconsistent kinds";
  return opt_iterator->second.flag_value;
}

template <typename ValueT, typename OptMapT, typename OptT,
          typename OptKindT, typename ValuesT>
static auto GetFlagGenericImpl(const OptMapT& opts, const OptT* opt,
                               OptKindT kind, const ValuesT& values)
    -> std::optional<ValueT> {
  auto opt_iterator = opts.find(opt);
  if (opt_iterator == opts.end()) {
    // No value for this opt.
    return {};
  }
  OptKindT stored_kind = opt_iterator->second.kind;
  CARBON_CHECK(stored_kind == kind)
      << "Opt '" << opt->name << "' has inconsistent kinds: expected '"
      << kind << "' but found '" << stored_kind << "'";
  return values[opt_iterator->second.value_index];
}

auto Args::GetStringOptImpl(const OptMap& opts, const StringOpt* opt) const
    -> std::optional<llvm::StringRef> {
  return GetFlagGenericImpl<llvm::StringRef>(opts, opt, OptKind::String,
                                             string_opt_values_);
}

auto Args::GetIntOptImpl(const OptMap& opts, const IntOpt* opt) const
    -> std::optional<ssize_t> {
  return GetFlagGenericImpl<ssize_t>(opts, opt, OptKind::Int,
                                     int_opt_values_);
}

auto Args::GetStringListOptImpl(const OptMap& opts,
                                 const StringListOpt* opt) const
    -> llvm::ArrayRef<llvm::StringRef> {
  if (auto opt_values = GetFlagGenericImpl<llvm::ArrayRef<llvm::StringRef>>(
          opts, opt, OptKind::StringList, string_list_opt_values_)) {
    return *opt_values;
  }
  return {};
}

void Args::AddOptDefault(Args::OptMap& opts, const Flag* opt) {
  auto [opt_it, inserted] = opts.insert({opt, {.kind = OptKind::Flag}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of opts!";
  auto& value = opt_it->second;
  value.flag_value = opt->default_value;
}

template <typename OptMapT, typename OptT, typename OptKindT,
          typename ValuesT>
auto AddOptDefaultGeneric(OptMapT& opts, const OptT* opt, OptKindT kind,
                           ValuesT& values) {
  if (!opt->default_value) {
    return;
  }
  auto [opt_it, inserted] = opts.insert({opt, {.kind = kind}});
  CARBON_CHECK(inserted) << "Defaults must be added to an empty set of opts!";
  auto& value = opt_it->second;
  value.value_index = values.size();
  values.emplace_back(*opt->default_value);
}

void Args::AddOptDefault(Args::OptMap& opts, const StringOpt* opt) {
  AddOptDefaultGeneric(opts, opt, OptKind::String, string_opt_values_);
}

void Args::AddOptDefault(Args::OptMap& opts, const IntOpt* opt) {
  AddOptDefaultGeneric(opts, opt, OptKind::Int, int_opt_values_);
}

void Args::AddOptDefault(Args::OptMap& opts, const StringListOpt* opt) {
  AddOptDefaultGeneric(opts, opt, OptKind::StringList,
                        string_list_opt_values_);
}

auto Args::AddParsedOptToMap(OptMap& opts, const Opt* opt, OptKind kind)
    -> std::pair<bool, OptKindAndValue&> {
  auto [opt_it, inserted] = opts.insert({opt, {.kind = kind}});
  auto& value = opt_it->second;
  if (!inserted) {
    CARBON_CHECK(value.kind == kind)
        << "Inconsistent opt kind for repeated opt '--" << opt->name
        << "': originally '" << value.kind << "' and now '" << kind << "'";
  }
  return {inserted, value};
}

auto Args::AddParsedOpt(OptMap& opts, const Flag* opt,
                         std::optional<llvm::StringRef> arg_value,
                         llvm::raw_ostream& errors) -> bool {
  auto [_, value] = AddParsedOptToMap(opts, opt, OptKind::Flag);
  if (!arg_value || *arg_value == "true") {
    value.flag_value = true;
    return true;
  }
  if (*arg_value == "false") {
    value.flag_value = false;
    return true;
  }
  errors << "ERROR: Invalid value '" << *arg_value
         << "' provided for the flag '--" << opt->name << "'\n";
  return false;
}

auto Args::AddParsedOpt(OptMap& opts, const StringOpt* opt,
                         std::optional<llvm::StringRef> arg_value,
                         llvm::raw_ostream& errors) -> bool {
  auto [inserted, value] = AddParsedOptToMap(opts, opt, OptKind::String);
  if (!arg_value && !opt->default_value) {
    errors << "ERROR: Invalid missing value for the string opt '--"
           << opt->name << "' which does not have a default value\n";
    return false;
  }
  llvm::StringRef value_str =
      arg_value ? *arg_value
                : static_cast<llvm::StringRef>(*opt->default_value);
  if (inserted) {
    value.value_index = string_opt_values_.size();
    string_opt_values_.push_back(value_str);
  } else {
    string_opt_values_[value.value_index] = value_str;
  }
  return true;
}

auto Args::AddParsedOpt(OptMap& opts, const IntOpt* opt,
                         std::optional<llvm::StringRef> arg_value,
                         llvm::raw_ostream& errors) -> bool {
  auto [inserted, value] = AddParsedOptToMap(opts, opt, OptKind::Int);
  if (!arg_value && !opt->default_value) {
    errors << "ERROR: Invalid missing value for the int opt '--" << opt->name
           << "' which does not have a default value\n";
    return false;
  }
  ssize_t value_int;
  if (arg_value) {
    // Note that LLVM's function for parsing as an integer confusingly returns
    // true *on an error* in parsing.
    if (arg_value->getAsInteger(/*Radix=*/0, value_int)) {
      errors << "ERROR: Unable to parse int opt '--" << opt->name
             << "' value '" << *arg_value << "' as an integer\n";
      return false;
    }
  } else {
    value_int = *opt->default_value;
  }
  if (inserted) {
    value.value_index = int_opt_values_.size();
    int_opt_values_.push_back(value_int);
  } else {
    int_opt_values_[value.value_index] = value_int;
  }
  return true;
}

auto Args::AddParsedOpt(OptMap& opts, const StringListOpt* opt,
                         std::optional<llvm::StringRef> arg_value,
                         llvm::raw_ostream& errors) -> bool {
  auto [inserted, value] =
      AddParsedOptToMap(opts, opt, OptKind::StringList);
  if (!arg_value) {
    errors << "ERROR: Must specify a value for the string list opt '--"
           << opt->name << "'\n";
    return false;
  }
  if (inserted) {
    value.value_index = string_list_opt_values_.size();
    string_list_opt_values_.push_back({*arg_value});
  } else {
    string_list_opt_values_[value.value_index].push_back(*arg_value);
  }
  return true;
}

}  // namespace Carbon
