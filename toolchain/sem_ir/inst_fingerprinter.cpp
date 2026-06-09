// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_fingerprinter.h"

#include <array>
#include <optional>
#include <utility>
#include <variant>

#include "common/concepts.h"
#include "common/ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/sem_ir/cpp_overload_set.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// A fingerprint store that computes a hash via a Merkle tree.
class HashFingerprintStore {
 public:
  using ResultType = uint64_t;

  explicit HashFingerprintStore(int total_ir_count)
      : fingerprints_(FilesFingerprintStores::MakeWithExplicitSizeFrom(
            total_ir_count, [] {
              return FingerprintStore::MakeForOverwriteWithExplicitSize(
                  0, CheckIRId::None);
            })) {}

  auto GetFingerprint(const File* file, InstId inst_id)
      -> std::optional<ResultType> {
    auto& store = fingerprints_.Get(file->check_ir_id());
    if (store.size() == 0) {
      return std::nullopt;
    }
    if (inst_id == InstId::InitTombstone ||
        inst_id == InstId::ImplWitnessTablePlaceholder) {
      return inst_id.index;
    }
    auto fingerprint = store.Get(inst_id);
    if (fingerprint == 0) {
      return std::nullopt;
    }
    return fingerprint;
  }

  auto SetFingerprint(const File* file, InstId inst_id, ResultType fingerprint)
      -> void {
    auto& store = fingerprints_.Get(file->check_ir_id());
    if (store.size() == 0) {
      store = FingerprintStore::MakeWithExplicitSize(
          file->insts().size(), file->insts().GetIdTag(), 0);
    }
    store.Set(inst_id, fingerprint ? fingerprint : 1);
  }

  auto Prepare() -> void { contents_.clear(); }

  auto Finish() -> ResultType { return llvm::stable_hash_combine(contents_); }

  auto AddFingerprint(ResultType fingerprint) -> void {
    contents_.push_back(fingerprint);
  }

  auto AddInvalid() -> void { contents_.push_back(-1); }

  auto AddString(llvm::StringRef string) -> void {
    contents_.push_back(llvm::stable_hash_name(string));
  }

  auto AddInteger(uint64_t value) -> void { contents_.push_back(value); }

  auto AddAPInt(const llvm::APInt& value) -> void {
    contents_.push_back(value.getBitWidth());
    contents_.append(value.getRawData(),
                     value.getRawData() + value.getNumWords());
  }

 private:
  using FingerprintStore =
      FixedSizeValueStore<InstId, uint64_t, Tag<CheckIRId>>;
  using FilesFingerprintStores =
      FixedSizeValueStore<CheckIRId, FingerprintStore>;
  FilesFingerprintStores fingerprints_;
  llvm::SmallVector<llvm::stable_hash> contents_;
};

// A fingerprint store that produces a string representation of the entity being
// fingerprinted.
class StringFingerprintStore {
 public:
  using ResultType = llvm::StringRef;

  explicit StringFingerprintStore(int total_ir_count)
      : fingerprints_(FilesFingerprintStores::MakeWithExplicitSizeFrom(
            total_ir_count, [] {
              return FingerprintStore::MakeForOverwriteWithExplicitSize(
                  0, CheckIRId::None);
            })) {}

  auto GetFingerprint(const File* file, InstId inst_id)
      -> std::optional<ResultType> {
    auto& store = fingerprints_.Get(file->check_ir_id());
    if (store.size() == 0) {
      return std::nullopt;
    }
    if (inst_id == InstId::InitTombstone) {
      return llvm::StringRef("!tombstone");
    }
    if (inst_id == InstId::ImplWitnessTablePlaceholder) {
      return llvm::StringRef("!placeholder");
    }
    auto fingerprint = store.Get(inst_id);
    if (fingerprint.empty()) {
      return std::nullopt;
    }
    return fingerprint;
  }

  auto SetFingerprint(const File* file, InstId inst_id, ResultType fingerprint)
      -> void {
    auto& store = fingerprints_.Get(file->check_ir_id());
    if (store.size() == 0) {
      store = FingerprintStore::MakeWithExplicitSize(
          file->insts().size(), file->insts().GetIdTag(), llvm::StringRef());
    }
    store.Set(inst_id, fingerprint);
  }

  auto Prepare() -> void { contents_.clear(); }

  auto Finish() -> ResultType {
    std::string result = "{";
    bool first = true;
    for (const auto& item : contents_) {
      if (!first) {
        result += ",";
      }
      first = false;
      result += item;
    }
    result += "}";
    return SaveString(std::move(result));
  }

  auto AddFingerprint(ResultType fingerprint) -> void {
    contents_.push_back(fingerprint);
  }

  auto AddInvalid() -> void { contents_.push_back("!invalid"); }

  auto AddString(llvm::StringRef string) -> void {
    constexpr llvm::StringRef SpecialChars = "{},!\\";
    auto num_special_chars = llvm::count_if(
        string, [&](char c) { return SpecialChars.contains(c); });
    if (num_special_chars) {
      std::string escaped;
      escaped.reserve(string.size() + num_special_chars);
      for (char c : string) {
        if (SpecialChars.contains(c)) {
          escaped.push_back('\\');
        }
        escaped.push_back(c);
      }
      contents_.push_back(SaveString(std::move(escaped)));
    } else {
      contents_.push_back(string);
    }
  }

  auto AddInteger(uint64_t value) -> void {
    contents_.push_back(SaveString(std::to_string(value)));
  }

  auto AddAPInt(const llvm::APInt& value) -> void {
    contents_.push_back(
        SaveString(llvm::toString(value, 10, /*isSigned=*/true)));
  }

 private:
  auto SaveString(std::string str) -> llvm::StringRef {
    auto& entry = allocated_strings_.emplace_back(
        std::make_unique<std::string>(std::move(str)));
    return *entry;
  }

  using FingerprintStore =
      FixedSizeValueStore<InstId, llvm::StringRef, Tag<CheckIRId>>;
  using FilesFingerprintStores =
      FixedSizeValueStore<CheckIRId, FingerprintStore>;
  FilesFingerprintStores fingerprints_;
  llvm::SmallVector<llvm::StringRef> contents_;
  llvm::SmallVector<std::unique_ptr<std::string>> allocated_strings_;
};

namespace {

template <typename StoreT>
struct Worklist {
  using ResultType = StoreT::ResultType;

  // The file containing the instruction we're currently processing.
  const File* sem_ir = nullptr;
  // The instructions we need to compute fingerprints for.
  llvm::SmallVector<std::pair<
      const File*, std::variant<InstId, InstBlockId, ImplId, CppOverloadSetId>>>
      todo;
  // Known cached instruction fingerprints.
  StoreT* store;

  // Finish fingerprinting and compute the fingerprint.
  auto Finish() -> ResultType { return store->Finish(); }

  // Gets the known fingerprint from the cache, or returns std::nullopt.
  auto GetFingerprint(const File* file, InstId inst_id)
      -> std::optional<ResultType> {
    return store->GetFingerprint(file, inst_id);
  }

  // Sets the fingerprint for an instruction in the cache.
  auto SetFingerprint(const File* file, InstId inst_id, ResultType fingerprint)
      -> void {
    store->SetFingerprint(file, inst_id, fingerprint);
  }

  // Add an invalid marker to the contents.
  auto AddInvalid() -> void { store->AddInvalid(); }

  // Add a string to the contents.
  auto AddString(llvm::StringRef string) -> void { store->AddString(string); }

  // Each of the following `Add` functions adds a typed argument to the contents
  // of the current instruction. If we don't yet have a fingerprint for the
  // argument, it instead adds that argument to the worklist instead.

  auto Add(InstKind kind) -> void {
    // TODO: Precompute or cache the hash of instruction IR names, or pick a
    // scheme that doesn't change when IR names change.
    AddString(kind.ir_name());
  }

  auto Add(IdentifierId ident_id) -> void {
    AddString(sem_ir->identifiers().Get(ident_id));
  }

  auto Add(StringLiteralValueId lit_id) -> void {
    AddString(sem_ir->string_literal_values().Get(lit_id));
  }

  auto Add(NameId name_id) -> void {
    AddString(sem_ir->names().GetIRBaseName(name_id));
  }

  auto Add(EntityNameId entity_name_id) -> void {
    if (!entity_name_id.has_value()) {
      AddInvalid();
      return;
    }
    const auto& entity_name = sem_ir->entity_names().Get(entity_name_id);
    if (entity_name.bind_index().has_value()) {
      Add(entity_name.bind_index());
      // Don't include the name. While it is part of the canonical identity of a
      // compile-time binding, renaming it (and its uses) is a compatible change
      // that we would like to not affect the fingerprint.
      //
      // Also don't include the `is_template` flag. Changing that flag should
      // also be a compatible change from the perspective of users of a generic.
    } else {
      Add(entity_name.name_id);
    }
    Add(entity_name.parent_scope_id);

    if (sem_ir->name_scopes().IsPrivateWithinNamespace(
            entity_name.name_id, entity_name.parent_scope_id)) {
      AddLibrary(sem_ir);
    }
  }

  auto AddInFile(const File* file, InstId inner_id) -> void {
    if (!inner_id.has_value()) {
      AddInvalid();
      return;
    }
    if (auto fingerprint = GetFingerprint(file, inner_id)) {
      store->AddFingerprint(*fingerprint);
      return;
    }
    todo.push_back({file, inner_id});
  }

  auto Add(InstId inner_id) -> void { AddInFile(sem_ir, inner_id); }

  auto Add(ConstantId constant_id) -> void {
    if (!constant_id.has_value()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->constant_values().GetInstId(constant_id));
  }

  auto Add(TypeId type_id) -> void {
    if (!type_id.has_value()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->types().GetTypeInstId(type_id));
  }

  template <typename T>
  auto AddBlock(llvm::ArrayRef<T> block) -> void {
    store->AddInteger(block.size());
    for (auto inner_id : block) {
      Add(inner_id);
    }
  }

  auto Add(InstBlockId inst_block_id) -> void {
    if (!inst_block_id.has_value()) {
      AddInvalid();
      return;
    }
    AddBlock(sem_ir->inst_blocks().Get(inst_block_id));
  }

  auto Add(StructTypeField field) -> void {
    Add(field.name_id);
    Add(field.type_inst_id);
  }

  auto Add(StructTypeFieldsId struct_type_fields_id) -> void {
    if (!struct_type_fields_id.has_value()) {
      AddInvalid();
      return;
    }
    AddBlock(sem_ir->struct_type_fields().Get(struct_type_fields_id));
  }

  auto Add(CustomLayoutId custom_layout_id) -> void {
    if (!custom_layout_id.has_value()) {
      AddInvalid();
      return;
    }
    auto block = sem_ir->custom_layouts().Get(custom_layout_id);
    store->AddInteger(block.size());
    for (auto size : block) {
      store->AddInteger(size.bits());
    }
  }

  auto AddPackage(NameScopeId name_scope_id) -> void {
    CARBON_CHECK(sem_ir->name_scopes().IsPackage(name_scope_id));
    Add(name_scope_id == NameScopeId::Package
            ? NameId::ForPackageName(sem_ir->package_id())
            : sem_ir->name_scopes().Get(name_scope_id).name_id());
  }

  auto Add(NameScopeId name_scope_id) -> void {
    if (!name_scope_id.has_value()) {
      AddInvalid();
      return;
    }

    // If this is the current package or an imported package, add the package
    // name.
    if (sem_ir->name_scopes().IsPackage(name_scope_id)) {
      AddPackage(name_scope_id);
      // Add a placeholder parent scope to match the fingerprinting we'd use for
      // a non-package scope.
      AddString("<package root>");
      return;
    }

    // For non-package scopes, add the name and parent scope.
    const auto& scope = sem_ir->name_scopes().Get(name_scope_id);
    Add(scope.name_id());
    if (scope.parent_scope_id().has_value()) {
      auto parent_id = scope.parent_scope_id();
      if (sem_ir->name_scopes().IsPackage(parent_id)) {
        AddPackage(parent_id);
      } else {
        Add(sem_ir->name_scopes().Get(parent_id).inst_id());
      }
    } else {
      // TODO: If the parent has no associated name scope, such as for a
      // function-local entity, we should still identify it uniquely somehow.
      AddInvalid();
    }
  }

  template <typename EntityT = EntityWithParamsBase>
  auto AddEntity(const std::type_identity_t<EntityT>& entity) -> void {
    Add(entity.name_id);
    Add(entity.parent_scope_id);

    if (sem_ir->name_scopes().IsPrivateWithinNamespace(
            entity.name_id, entity.parent_scope_id)) {
      AddLibrary(sem_ir);
    }
  }

  auto Add(FunctionId function_id) -> void {
    AddEntity(sem_ir->functions().Get(function_id));
  }

  auto Add(CppOverloadSetId cpp_overload_set_id) -> void {
    const CppOverloadSet& cpp_overload_set =
        sem_ir->cpp_overload_sets().Get(cpp_overload_set_id);
    Add(cpp_overload_set.name_id);
    Add(cpp_overload_set.parent_scope_id);
  }

  auto Add(ClangDeclId /*decl_id*/) -> void {
    // TODO: For `CppTemplateNameType` we don't need to fingerprint the
    // `decl_id`, because fingerprinting the `NameId` is sufficient to identify
    // the template, but this won't necessarily be true for other
    // `ClangDeclId`s.
    // See also: https://github.com/carbon-language/carbon-lang/issues/6728
  }

  auto Add(ClassId class_id) -> void {
    AddEntity(sem_ir->classes().Get(class_id));
    // Imported C++ classes are not uniquely identified by their name and parent
    // scope, so we also include the Clang mangled type name, which is computed
    // on demand. The Clang type name is unique, as it is the canonical C++
    // identity. Carbon classes rely on the name and scope from `AddEntity`.
    llvm::SmallString<128> cpp_mangled_name;
    llvm::raw_svector_ostream os(cpp_mangled_name);
    if (sem_ir->AppendCppMangledTypeName(class_id, os)) {
      AddString(cpp_mangled_name);
    } else {
      AddInvalid();
    }
  }

  auto Add(FieldId field_id) -> void {
    const auto& field = sem_ir->fields().Get(field_id);
    Add(field.index);
    Add(field.initializer_id);
  }

  auto Add(VtableId vtable_id) -> void {
    const auto& vtable = sem_ir->vtables().Get(vtable_id);
    if (vtable.class_id.has_value()) {
      Add(vtable.class_id);
    } else {
      AddInvalid();
    }
    Add(vtable.virtual_functions_id);
  }

  auto Add(InterfaceId interface_id) -> void {
    AddEntity(sem_ir->interfaces().Get(interface_id));
  }

  auto Add(NamedConstraintId named_constraint_id) -> void {
    AddEntity(sem_ir->named_constraints().Get(named_constraint_id));
  }

  auto Add(RequireImplsId require_id) -> void {
    CARBON_CHECK(require_id.has_value());
    const auto& require = sem_ir->require_impls().Get(require_id);
    Add(sem_ir->constant_values().Get(require.self_id));
    Add(sem_ir->constant_values().Get(require.facet_type_inst_id));
    store->AddInteger(require.extend_self);
    Add(require.parent_scope_id);
  }

  auto Add(AssociatedConstantId assoc_const_id) -> void {
    AddEntity<AssociatedConstant>(
        sem_ir->associated_constants().Get(assoc_const_id));
  }

  auto Add(ImplId impl_id) -> void {
    if (!impl_id.has_value()) {
      AddInvalid();
      return;
    }

    const auto& impl = sem_ir->impls().Get(impl_id);
    Add(sem_ir->constant_values().Get(impl.self_id));
    Add(sem_ir->constant_values().Get(impl.constraint_id));
    Add(impl.parent_scope_id);
  }

  auto Add(DeclInstBlockId /*block_id*/) -> void {
    // Intentionally exclude decl blocks from fingerprinting. Changes to the
    // decl block don't change the identity of the declaration.
  }

  auto Add(LabelId /*block_id*/) -> void {
    CARBON_FATAL("Should never fingerprint a label");
  }

  auto Add(FacetTypeId facet_type_id) -> void {
    const auto& facet_type = sem_ir->facet_types().Get(facet_type_id);
    auto add_constraints = [&](auto constraints) {
      store->AddInteger(constraints.size());
      for (auto [first, second] : constraints) {
        Add(first);
        Add(second);
      }
    };
    add_constraints(facet_type.extend_constraints);
    add_constraints(facet_type.self_impls_constraints);
    add_constraints(facet_type.rewrite_constraints);
    store->AddInteger(facet_type.other_requirements);
  }

  auto Add(GenericId generic_id) -> void {
    if (!generic_id.has_value()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->generics().Get(generic_id).decl_id);
  }

  auto Add(SpecificId specific_id) -> void {
    if (!specific_id.has_value()) {
      AddInvalid();
      return;
    }
    const auto& specific = sem_ir->specifics().Get(specific_id);
    Add(specific.generic_id);
    Add(specific.args_id);
  }

  auto Add(SpecificInterfaceId specific_interface_id) -> void {
    if (!specific_interface_id.has_value()) {
      AddInvalid();
      return;
    }
    const auto& interface =
        sem_ir->specific_interfaces().Get(specific_interface_id);
    Add(interface.interface_id);
    Add(interface.specific_id);
  }

  auto Add(const llvm::APInt& value) -> void { store->AddAPInt(value); }

  auto Add(IntId int_id) -> void { Add(sem_ir->ints().Get(int_id)); }

  auto Add(FloatId float_id) -> void {
    Add(sem_ir->floats().Get(float_id).bitcastToAPInt());
  }

  auto Add(RealId real_id) -> void {
    const auto& real = sem_ir->reals().Get(real_id);
    Add(real.mantissa);
    Add(real.exponent);
    store->AddInteger(real.is_decimal);
  }

  auto Add(PackageNameId package_id) -> void {
    Add(NameId::ForPackageName(package_id));
  }

  auto Add(LibraryNameId lib_name_id) -> void {
    if (lib_name_id == LibraryNameId::Default) {
      AddString("");
    } else if (lib_name_id == LibraryNameId::Error) {
      AddString("<error>");
    } else if (lib_name_id.has_value()) {
      Add(lib_name_id.AsStringLiteralValueId());
    } else {
      AddInvalid();
    }
  }

  // Adds just the library name to the fingerprint. Does not add the package
  // name, as it is typically already included as the outermost scope name. The
  // caller is responsible for ensuring the package name is added somehow.
  auto AddLibrary(const File* file) -> void {
    llvm::SaveAndRestore in_file(sem_ir, file);
    // Add a marker to prevent collisions with scope names.
    AddString("<library>");
    Add(file->library_id());
  }

  auto Add(ImportIRId ir_id) -> void {
    // TODO: Is the ImportIRId for an ImportRef always the same IR for the same
    // entity (eg, always the owning IR, or always the first-declaring IR)?
    const auto* file = sem_ir->import_irs().Get(ir_id).sem_ir;
    llvm::SaveAndRestore in_file(sem_ir, file);
    AddPackage(NameScopeId::Package);
    Add(file->library_id());
  }

  auto Add(ImportIRInstId ir_inst_id) -> void {
    auto ir_inst = sem_ir->import_ir_insts().Get(ir_inst_id);
    AddInFile(sem_ir->import_irs().Get(ir_inst.ir_id()).sem_ir,
              ir_inst.inst_id());
  }

  template <typename BundleT>
  auto Add(BundleId<BundleT> bundle_id) -> void {
    std::apply([&](auto... ids) { (..., Add(ids)); },
               sem_ir->bundles().GetAsTuple(bundle_id));
  }

  auto Add(RawBundleId bundle_id) -> void {
    CARBON_FATAL("Can't fingerprint untyped bundle ID {}", bundle_id);
  }

  template <typename T>
    requires(SameAsOneOf<T, BoolValue, CharId, CompileTimeBindIndex,
                         ElementIndex, FloatKind, IntKind, CallParamIndex>)
  auto Add(T arg) -> void {
    // Index-like ID: just include the value directly.
    store->AddInteger(arg.index);
  }

  template <typename T>
    requires(SameAsOneOf<T, AnyRawId, ExprRegionId, LocId>)
  auto Add(T /*arg*/) -> void {
    CARBON_FATAL("Unexpected instruction operand kind {0}", typeid(T).name());
  }

  auto Add(IdAndKind::InvalidType /*invalid*/) -> void {
    CARBON_FATAL("Unexpected invalid IdKind");
  }

  auto Add(IdAndKind::NoneType /*none*/) -> void {}

  // Add an instruction argument to the contents of the current instruction.
  auto AddWithKind(IdAndKind arg) -> void {
    arg.Dispatch<void>([&](auto id) { Add(id); });
  }

  // Ensure all the instructions on the todo list have fingerprints. To avoid a
  // re-lookup, returns the fingerprint of the first instruction on the todo
  // list, and requires the todo list to be non-empty.
  auto Run() -> ResultType {
    CARBON_CHECK(!todo.empty());
    while (true) {
      const size_t init_size = todo.size();
      auto [next_sem_ir, next] = todo.back();

      sem_ir = next_sem_ir;
      store->Prepare();

      if (!std::holds_alternative<InstId>(next)) {
        // Add the contents of the `next` instruction so they all contribute to
        // the `contents`.
        CARBON_KIND_SWITCH(next) {
          case CARBON_KIND(InstId _):
            CARBON_FATAL("InstId is checked for above.");
          case CARBON_KIND(ImplId impl_id):
            Add(impl_id);
            break;
          case CARBON_KIND(InstBlockId inst_block_id):
            Add(inst_block_id);
            break;
          case CARBON_KIND(CppOverloadSetId overload_set_id):
            Add(overload_set_id);
            break;
        }

        // If we didn't add any more work, then we have a fingerprint for the
        // `next` instruction, otherwise we wait until that work is completed.
        // If the `next` is the last thing in `todo`, we return the fingerprint.
        // Otherwise we would just discard it because we don't currently cache
        // the fingerprint for things other than `InstId`, but we really only
        // expect other `next` types to be at the bottom of the `todo` stack
        // since they are not added to `todo` during Run().
        if (todo.size() == init_size) {
          auto fingerprint = Finish();
          todo.pop_back();
          CARBON_CHECK(todo.empty(),
                       "A non-InstId was inserted into `todo` during Run()");
          return fingerprint;
        }

        // Move on to processing the instructions added above; we will come
        // back to this branch once they are done.
        continue;
      }

      auto next_inst_id = std::get<InstId>(next);

      // If we already have a fingerprint for this instruction, we have nothing
      // to do. Just pop it from `todo`.
      if (auto fingerprint = GetFingerprint(next_sem_ir, next_inst_id)) {
        todo.pop_back();
        if (todo.empty()) {
          return *fingerprint;
        }
        continue;
      }

      // Keep this instruction in `todo` for now. If we add more work, we'll
      // finish that work and process this instruction again, and if not, we'll
      // pop the instruction at the end of the loop.
      auto inst = next_sem_ir->insts().Get(next_inst_id);

      // Add the instruction's fields to the contents.
      Add(inst.kind());

      // Don't include the type if it's `type` or `<error>`, because those types
      // are self-referential.
      if (inst.type_id() != TypeType::TypeId &&
          inst.type_id() != ErrorInst::TypeId) {
        Add(inst.type_id());
      }

      AddWithKind(inst.arg0_and_kind());
      AddWithKind(inst.arg1_and_kind());

      // If we didn't add any work, we have a fingerprint for this instruction;
      // pop it from the todo list. Otherwise, we leave it on the todo list so
      // we can compute its fingerprint once we've finished the work we added.
      if (todo.size() == init_size) {
        ResultType fingerprint = Finish();
        SetFingerprint(next_sem_ir, next_inst_id, fingerprint);
        todo.pop_back();
        if (todo.empty()) {
          return fingerprint;
        }
      }
    }
  }
};

}  // namespace

template <typename StoreT, typename ResultT>
InstFingerprinterTemplate<StoreT, ResultT>::InstFingerprinterTemplate(
    int total_ir_count)
    : store_(std::make_unique<StoreT>(total_ir_count)) {}

template <typename StoreT, typename ResultT>
InstFingerprinterTemplate<StoreT, ResultT>::~InstFingerprinterTemplate() =
    default;

template <typename StoreT, typename ResultT>
auto InstFingerprinterTemplate<StoreT, ResultT>::GetOrCompute(const File* file,
                                                              InstId inst_id)
    -> ResultT {
  Worklist<StoreT> worklist = {.todo = {{file, inst_id}},
                               .store = store_.get()};
  return worklist.Run();
}

template <typename StoreT, typename ResultT>
auto InstFingerprinterTemplate<StoreT, ResultT>::GetOrCompute(
    const File* file, InstBlockId inst_block_id) -> ResultT {
  Worklist<StoreT> worklist = {.todo = {{file, inst_block_id}},
                               .store = store_.get()};
  return worklist.Run();
}

template <typename StoreT, typename ResultT>
auto InstFingerprinterTemplate<StoreT, ResultT>::GetOrCompute(const File* file,
                                                              ImplId impl_id)
    -> ResultT {
  Worklist<StoreT> worklist = {.todo = {{file, impl_id}},
                               .store = store_.get()};
  return worklist.Run();
}

template <typename StoreT, typename ResultT>
auto InstFingerprinterTemplate<StoreT, ResultT>::GetOrCompute(
    const File* file, CppOverloadSetId overload_set_id) -> ResultT {
  Worklist<StoreT> worklist = {.todo = {{file, overload_set_id}},
                               .store = store_.get()};
  return worklist.Run();
}

template class InstFingerprinterTemplate<HashFingerprintStore, uint64_t>;
template class InstFingerprinterTemplate<StringFingerprintStore,
                                         llvm::StringRef>;

}  // namespace Carbon::SemIR
