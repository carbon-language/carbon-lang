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

class NameScope : public Printable<NameScope> {
 public:
  struct Entry {
    NameId name_id;
    InstId inst_id;
    AccessKind access_kind;
    bool is_poisoned = false;
  };
  static_assert(sizeof(Entry) == 12);

  struct EntryId : public IdBase<EntryId> {
    static constexpr llvm::StringLiteral Label = "name_scope_entry";
    using IdBase::IdBase;
  };

  explicit NameScope(InstId inst_id, NameId name_id,
                     NameScopeId parent_scope_id)
      : inst_id_(inst_id),
        name_id_(name_id),
        parent_scope_id_(parent_scope_id) {}

  auto Print(llvm::raw_ostream& out) const -> void;

  auto entries() const -> llvm::ArrayRef<Entry> { return names_; }

  // Get a specific Name entry based on an EntryId that return from a lookup.
  //
  // The Entry could become invalidated if the scope object is invalidated or if
  // a name is added.
  auto GetEntry(EntryId entry_id) const -> const Entry& {
    return names_[entry_id.index];
  }
  auto GetEntry(EntryId entry_id) -> Entry& { return names_[entry_id.index]; }

  // Searches for the given name and returns an EntryId if found or nullopt if
  // not. The returned entry may be poisoned.
  auto Lookup(NameId name_id) const -> std::optional<EntryId> {
    auto lookup = name_map_.Lookup(name_id);
    if (!lookup) {
      return std::nullopt;
    }
    return lookup.value();
  }

  // Adds a new name that is known to not exist. The new entry is not allowed to
  // be poisoned. An existing poisoned entry can be overwritten.
  auto AddRequired(Entry name_entry) -> void;

  // Searches for the given name. If found, including if a poisoned entry is
  // found, returns true with the existing EntryId. Otherwise, adds the name
  // using inst_id and access_kind and returns false with the new EntryId.
  //
  // This cannot be used to add poisoned entries; use LookupOrPoison instead.
  auto LookupOrAdd(SemIR::NameId name_id, InstId inst_id,
                   AccessKind access_kind) -> std::pair<bool, EntryId>;

  // Searches for the given name. If found, including if a poisoned entry is
  // found, returns the corresponding EntryId. Otherwise, returns nullopt and
  // poisons the name so it can't be declared later.
  auto LookupOrPoison(NameId name_id) -> std::optional<EntryId>;

  auto extended_scopes() const -> llvm::ArrayRef<InstId> {
    return extended_scopes_;
  }

  auto AddExtendedScope(SemIR::InstId extended_scope) -> void {
    extended_scopes_.push_back(extended_scope);
  }

  auto inst_id() const -> InstId { return inst_id_; }

  auto name_id() const -> NameId { return name_id_; }

  auto parent_scope_id() const -> NameScopeId { return parent_scope_id_; }

  auto has_error() const -> bool { return has_error_; }

  // Mark that we have diagnosed an error in a construct that would have added
  // names to this scope.
  auto set_has_error() -> void { has_error_ = true; }

  auto is_closed_import() const -> bool { return is_closed_import_; }

  auto set_is_closed_import(bool is_closed_import) -> void {
    is_closed_import_ = is_closed_import;
  }

  // Returns true if this name scope describes an imported package.
  auto is_imported_package() const -> bool {
    return is_closed_import() && parent_scope_id() == NameScopeId::Package;
  }

  auto import_ir_scopes() const
      -> llvm::ArrayRef<std::pair<SemIR::ImportIRId, SemIR::NameScopeId>> {
    return import_ir_scopes_;
  }

  auto AddImportIRScope(
      const std::pair<SemIR::ImportIRId, SemIR::NameScopeId>& import_ir_scope)
      -> void {
    import_ir_scopes_.push_back(import_ir_scope);
  }

 private:
  // Names in the scope, including poisoned names.
  //
  // Entries could become invalidated if the scope object is invalidated or if a
  // name is added.
  //
  // We store both an insertion-ordered vector for iterating
  // and a map from `NameId` to the index of that vector for name lookup.
  //
  // Optimization notes: this is somewhat memory inefficient. If this ends up
  // either hot or a significant source of memory allocation, we should consider
  // switching to a SOA model where the `AccessKind` is stored in a separate
  // vector so that these can pack densely. If this ends up both cold and memory
  // intensive, we can also switch the lookup to a set of indices into the
  // vector rather than a map from `NameId` to index.
  llvm::SmallVector<Entry> names_;
  Map<NameId, EntryId> name_map_;

  // Instructions returning values that are extended by this scope.
  //
  // Small vector size is set to 1: we expect that there will rarely be more
  // than a single extended scope.
  // TODO: Revisit this once we have more kinds of extended scope and data.
  // TODO: Consider using something like `TinyPtrVector` for this.
  llvm::SmallVector<InstId, 1> extended_scopes_;

  // The instruction which owns the scope.
  InstId inst_id_;

  // When the scope is a namespace, the name. Otherwise, invalid.
  NameId name_id_;

  // The parent scope.
  NameScopeId parent_scope_id_;

  // Whether we have diagnosed an error in a construct that would have added
  // names to this scope. For example, this can happen if an `import` failed or
  // an `extend` declaration was ill-formed. If true, names are assumed to be
  // missing as a result of the error, and no further errors are produced for
  // lookup failures in this scope.
  bool has_error_ = false;

  // True if this is a closed namespace created by importing a package.
  bool is_closed_import_ = false;

  // Imported IR scopes that compose this namespace. This will be empty for
  // scopes that correspond to the current package.
  llvm::SmallVector<std::pair<SemIR::ImportIRId, SemIR::NameScopeId>, 0>
      import_ir_scopes_;
};

// Provides a ValueStore wrapper for an API specific to name scopes.
class NameScopeStore {
 public:
  explicit NameScopeStore(const File* file) : file_(file) {}

  // Adds a name scope, returning an ID to reference it.
  auto Add(InstId inst_id, NameId name_id, NameScopeId parent_scope_id)
      -> NameScopeId {
    return values_.Add(NameScope(inst_id, name_id, parent_scope_id));
  }

  // Adds a name that is required to exist in a name scope, such as `Self`.
  // The name must never conflict. inst_id must not be poisoned.
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
      -> std::pair<InstId, std::optional<Inst>>;

  // Returns whether the provided scope ID is for a package scope.
  auto IsPackage(NameScopeId scope_id) const -> bool {
    if (!scope_id.is_valid()) {
      return false;
    }
    // A package is either the current package or an imported package.
    return scope_id == SemIR::NameScopeId::Package ||
           Get(scope_id).is_imported_package();
  }

  // Returns whether the provided scope ID is for the Core package.
  auto IsCorePackage(NameScopeId scope_id) const -> bool;

  auto OutputYaml() const -> Yaml::OutputMapping {
    return values_.OutputYaml();
  }

  // Collects memory usage of members.
  auto CollectMemUsage(MemUsage& mem_usage, llvm::StringRef label) const
      -> void {
    mem_usage.Collect(std::string(label), values_);
  }

 private:
  const File* file_;
  ValueStore<NameScopeId> values_;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_
