// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/name_scope.h"

namespace Carbon::SemIR {

auto NameScope::Print(llvm::raw_ostream& out) const -> void {
  out << "{inst: " << inst_id_ << ", parent_scope: " << parent_scope_id_
      << ", has_error: " << (has_error_ ? "true" : "false");

  out << ", extended_scopes: [";
  llvm::ListSeparator scope_sep;
  for (auto id : extended_scopes_) {
    out << scope_sep << id;
  }
  out << "]";

  out << ", names: {";
  llvm::ListSeparator sep;
  for (auto entry : names_) {
    out << sep << entry.name_id << ": " << entry.inst_id;
  }
  out << "}";

  out << "}";
}

auto NameScope::AddRequired(Entry name_entry) -> void {
  auto add_name = [&] {
    EntryId index(names_.size());
    names_.push_back(name_entry);
    return index;
  };
  auto result = name_map_.Insert(name_entry.name_id, add_name);
  CARBON_CHECK(result.is_inserted(), "Failed to add required name: {0}",
               name_entry.name_id);
}

auto NameScope::LookupOrAdd(SemIR::NameId name_id, InstId inst_id,
                            AccessKind access_kind)
    -> std::pair<bool, EntryId> {
  auto insert_result = name_map_.Insert(name_id, EntryId(names_.size()));
  if (!insert_result.is_inserted()) {
    return {false, EntryId(insert_result.value())};
  }

  names_.push_back(
      {.name_id = name_id, .inst_id = inst_id, .access_kind = access_kind});
  return {true, EntryId(names_.size() - 1)};
}

}  // namespace Carbon::SemIR
