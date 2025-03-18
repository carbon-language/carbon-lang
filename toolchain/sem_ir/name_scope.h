// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_NAME_SCOPE_H_

#include "clang/AST/DeclBase.h"
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

// Represents the result of a name lookup.
//
// Lookup results are constructed through the `Make()` factory functions. Each
// result takes one of a few forms, depending on the function used:
// - Found when the lookup was successful returning an existing `InstId`. Can be
//   constructed using `MakeFound()` or `MakeWrappedLookupResult()` with an
//   existing `inst_id`.
// - Not found when the name wasn't declared or nor poisoned. Can be constructed
//   using `MakeNotFound()` or using `MakeWrappedLookupResult()` with a `None`
//   `inst_id`.
// - Poisoned when the name wasn't declared but was poisoned and so also
//   considered to not be found in that scope. Can be constructed using
//   `MakePoisoned()`.
// - Represent that an error has occurred during lookup. This is still
//   considered found and the error `InstId` is considered existing. Can be
//   constructed using `MakeError()` or using `MakeWrappedLookupResult()` with
//   `ErrorInst::SingletonInstId`.
class ScopeLookupResult {
 public:
  static auto MakeFound(InstId target_inst_id, AccessKind access_kind)
      -> ScopeLookupResult {
    CARBON_CHECK(target_inst_id.has_value());
    return MakeWrappedLookupResult(target_inst_id, access_kind);
  }

  static auto MakeNotFound() -> ScopeLookupResult {
    return MakeWrappedLookupResult(InstId::None, AccessKind::Public);
  }

  static auto MakePoisoned(LocId poisoning_loc_id) -> ScopeLookupResult {
    return ScopeLookupResult(poisoning_loc_id);
  }

  static auto MakeError() -> ScopeLookupResult {
    return MakeFound(ErrorInst::SingletonInstId, AccessKind::Public);
  }

  static auto MakeWrappedLookupResult(InstId target_inst_id,
                                      AccessKind access_kind)
      -> ScopeLookupResult {
    return ScopeLookupResult(target_inst_id, access_kind);
  }

  // True iff CreatePoisoned() was used.
  auto is_poisoned() const -> bool { return is_poisoned_; }

  // True when lookup was successful or resulted with an error. False for
  // poisoned or not found.
  auto is_found() const -> bool {
    return !is_poisoned() && target_inst_id_.has_value();
  }

  // The `InstId` of the result of the lookup. Must only be called when lookup
  // was successful; in other words, when `is_found()` returns true. Always
  // returns an existing `InstId`.
  auto target_inst_id() const -> InstId {
    CARBON_CHECK(is_found());
    return target_inst_id_;
  }

  // The `LocId` where the name poisoning was triggered. Must only be called
  // when lookup returned a poisoned name; in other words, when `is_poisoned()`
  // returns true. Always returns an existing `InstId`.
  auto poisoning_loc_id() const -> LocId {
    CARBON_CHECK(is_poisoned());
    return poisoning_loc_id_;
  }

  auto access_kind() const -> AccessKind { return access_kind_; }

  // Equality means either:
  // - Both are not poisoned and have the same `InstId` and `AccessKind`.
  // - Both are poisoned and have the same `LocId`.
  friend auto operator==(const ScopeLookupResult& lhs,
                         const ScopeLookupResult& rhs) -> bool {
    return lhs.is_poisoned_ == rhs.is_poisoned_ &&
           lhs.access_kind_ == rhs.access_kind_ &&
           (lhs.is_poisoned_ ? lhs.poisoning_loc_id_ == rhs.poisoning_loc_id_
                             : lhs.target_inst_id_ == rhs.target_inst_id_);
  }

 private:
  explicit ScopeLookupResult(InstId target_inst_id, AccessKind access_kind)
      : target_inst_id_(target_inst_id),
        access_kind_(access_kind),
        is_poisoned_(false) {}

  explicit ScopeLookupResult(LocId loc_id)
      : poisoning_loc_id_(loc_id),
        access_kind_(AccessKind::Public),
        is_poisoned_(true) {}

  union {
    InstId target_inst_id_;
    LocId poisoning_loc_id_;
  };
  AccessKind access_kind_;
  bool is_poisoned_;
};
static_assert(sizeof(ScopeLookupResult) == 8);

class NameScope : public Printable<NameScope> {
 public:
  struct Entry {
    NameId name_id;
    ScopeLookupResult result;

    // Equality means they have the same `name_id` and equal `result`.
    friend auto operator==(const Entry&, const Entry&) -> bool = default;
  };
  static_assert(sizeof(Entry) == 12);

  struct EntryId : public IdBase<EntryId> {
    static constexpr llvm::StringLiteral Label = "name_scope_entry";
    using IdBase::IdBase;
  };

  // Disallow copy, allow move.
  NameScope(NameScope&& other) = default;
  auto operator=(NameScope&& other) -> NameScope& = default;

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
  auto LookupOrAdd(NameId name_id, InstId inst_id, AccessKind access_kind)
      -> std::pair<bool, EntryId>;

  // Searches for the given name. If found, including if a poisoned entry is
  // found, returns the corresponding EntryId. Otherwise, returns nullopt and
  // poisons the name so it can't be declared later. Names that are not
  // identifiers will not be poisoned.
  auto LookupOrPoison(LocId loc_id, NameId name_id) -> std::optional<EntryId>;

  auto extended_scopes() const -> llvm::ArrayRef<InstId> {
    return extended_scopes_;
  }

  auto AddExtendedScope(InstId extended_scope) -> void {
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

  auto is_cpp_scope() const -> bool { return cpp_decl_context(); }

  auto cpp_decl_context() const -> const clang::DeclContext* {
    return cpp_decl_context_;
  }
  auto cpp_decl_context() -> clang::DeclContext* { return cpp_decl_context_; }

  auto set_cpp_decl_context(clang::DeclContext* cpp_decl_context) -> void {
    cpp_decl_context_ = cpp_decl_context;
  }

  // Returns true if this name scope describes an imported package.
  auto is_imported_package() const -> bool {
    return is_closed_import() && parent_scope_id() == NameScopeId::Package;
  }

  auto import_ir_scopes() const
      -> llvm::ArrayRef<std::pair<ImportIRId, NameScopeId>> {
    return import_ir_scopes_;
  }

  auto AddImportIRScope(
      const std::pair<ImportIRId, NameScopeId>& import_ir_scope) -> void {
    import_ir_scopes_.push_back(import_ir_scope);
  }

  auto is_interface_definition() const -> bool {
    return is_interface_definition_;
  }

  // TODO: Figure out a better way of setting this and is_cpp_scope() than
  // calling a function immediately after construction.
  auto set_is_interface_definition() -> void {
    is_interface_definition_ = true;
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

  // When the scope is a namespace, the name. Otherwise, `None`.
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

  // Set if this is the `Cpp` scope or a scope inside `Cpp`. Points to the
  // matching Clang declaration context to look for names. This is mutable since
  // `clang::Sema::LookupQualifiedName()` requires a mutable `DeclContext`.
  // TODO: Ensure we can easily serialize/deserialize this. Consider decl ID to
  // point into the AST. This is related to:
  // https://github.com/carbon-language/carbon-lang/issues/4666.
  clang::DeclContext* cpp_decl_context_ = nullptr;

  // True if this is the scope of an interface definition, where associated
  // entities will be bound to the interface's `Self` symbolic type.
  bool is_interface_definition_ = false;

  // Imported IR scopes that compose this namespace. This will be empty for
  // scopes that correspond to the current package.
  llvm::SmallVector<std::pair<ImportIRId, NameScopeId>, 0> import_ir_scopes_;
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
  // The name must never conflict.
  auto AddRequiredName(NameScopeId scope_id, NameId name_id, InstId inst_id)
      -> void {
    Get(scope_id).AddRequired(
        {.name_id = name_id,
         .result = ScopeLookupResult::MakeFound(inst_id, AccessKind::Public)});
  }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) -> NameScope& { return values_.Get(scope_id); }

  // Returns the requested name scope.
  auto Get(NameScopeId scope_id) const -> const NameScope& {
    return values_.Get(scope_id);
  }

  // Returns the instruction owning the requested name scope, or `None` with
  // nullopt if the scope is either `None` or has no associated instruction.
  auto GetInstIfValid(NameScopeId scope_id) const
      -> std::pair<InstId, std::optional<Inst>>;

  // Returns whether the provided scope ID is for a package scope.
  auto IsPackage(NameScopeId scope_id) const -> bool {
    if (!scope_id.has_value()) {
      return false;
    }
    // A package is either the current package or an imported package.
    return scope_id == NameScopeId::Package ||
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
