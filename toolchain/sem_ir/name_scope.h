// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_

#include "common/map.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// Access control for an entity.
enum class AccessKind : int8_t {
  Public,
  Protected,
  Private,
};

struct NameScope : Printable<NameScope> {
  struct Entry {
    NameId name_id;
    InstId inst_id;
    AccessKind access_kind;
  };

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{inst: " << inst_id << ", parent_scope: " << parent_scope_id
        << ", has_error: " << (has_error ? "true" : "false");

    out << ", extended_scopes: [";
    llvm::ListSeparator scope_sep;
    for (auto id : extended_scopes) {
      out << scope_sep << id;
    }
    out << "]";

    out << ", names: {";
    llvm::ListSeparator sep;
    for (auto entry : names) {
      out << sep << entry.name_id << ": " << entry.inst_id;
    }
    out << "}";

    out << "}";
  }

  // Adds a name to the scope that must not already exist.
  auto AddRequired(Entry name_entry) -> void {
    auto add_name = [&] {
      int index = names.size();
      names.push_back(name_entry);
      return index;
    };
    auto result = name_map.Insert(name_entry.name_id, add_name);
    CARBON_CHECK(result.is_inserted(), "Failed to add required name: {0}",
                 name_entry.name_id);
  }

  // Returns true if this name scope describes an imported package.
  auto is_imported_package() const -> bool {
    return is_closed_import && parent_scope_id == NameScopeId::Package;
  }

  // Names in the scope. We store both an insertion-ordered vector for iterating
  // and a map from `NameId` to the index of that vector for name lookup.
  //
  // Optimization notes: this is somewhat memory inefficient. If this ends up
  // either hot or a significant source of memory allocation, we should consider
  // switching to a SOA model where the `AccessKind` is stored in a separate
  // vector so that these can pack densely. If this ends up both cold and memory
  // intensive, we can also switch the lookup to a set of indices into the
  // vector rather than a map from `NameId` to index.
  llvm::SmallVector<Entry> names;
  Map<NameId, int> name_map;

  // Instructions returning values that are extended by this scope.
  //
  // Small vector size is set to 1: we expect that there will rarely be more
  // than a single extended scope.
  // TODO: Revisit this once we have more kinds of extended scope and data.
  // TODO: Consider using something like `TinyPtrVector` for this.
  llvm::SmallVector<InstId, 1> extended_scopes;

  // The instruction which owns the scope.
  InstId inst_id;

  // When the scope is a namespace, the name. Otherwise, invalid.
  NameId name_id;

  // The parent scope.
  NameScopeId parent_scope_id;

  // Whether we have diagnosed an error in a construct that would have added
  // names to this scope. For example, this can happen if an `import` failed or
  // an `extend` declaration was ill-formed. If true, the `names` map is assumed
  // to be missing names as a result of the error, and no further errors are
  // produced for lookup failures in this scope.
  bool has_error = false;

  // True if this is a closed namespace created by importing a package.
  bool is_closed_import = false;

  // Imported IR scopes that compose this namespace. This will be empty for
  // scopes that correspond to the current package.
  llvm::SmallVector<std::pair<SemIR::ImportIRId, SemIR::NameScopeId>, 0>
      import_ir_scopes;
};

// Provides a ValueStore wrapper for an API specific to name scopes.
class NameScopeStore {
 public:
  explicit NameScopeStore(InstStore* insts) : insts_(insts) {}

  // Adds a name scope, returning an ID to reference it.
  auto Add(InstId inst_id, NameId name_id, NameScopeId parent_scope_id)
      -> NameScopeId {
    return values_.Add({.inst_id = inst_id,
                        .name_id = name_id,
                        .parent_scope_id = parent_scope_id});
  }

  // Adds a name that is required to exist in a name scope, such as `Self`.
  // These must never conflict.
  auto AddRequiredName(NameScopeId scope_id, NameId name_id, InstId inst_id)
      -> void {
    Get(scope_id).AddRequired({.name_id = name_id,
                               .inst_id = inst_id,
                               .access_kind = AccessKind::Public});
  }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) -> NameScope& { return values_.Get(scope_id); }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) const -> const NameScope& {
    return values_.Get(scope_id);
  }

  // Returns the instruction owning the requested name scope, or Invalid with
  // nullopt if the scope is either invalid or has no associated instruction.
  auto GetInstIfValid(NameScopeId scope_id) const
      -> std::pair<InstId, std::optional<Inst>> {
    if (!scope_id.is_valid()) {
      return {InstId::Invalid, std::nullopt};
    }
    auto inst_id = Get(scope_id).inst_id;
    if (!inst_id.is_valid()) {
      return {InstId::Invalid, std::nullopt};
    }
    return {inst_id, insts_->Get(inst_id)};
  }

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(std::string(label), values_);
  }

 private:
  InstStore* insts_;
  ValueStore<NameScopeId> values_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_
