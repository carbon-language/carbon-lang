// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_fingerprinter.h"

#include <variant>

#include "common/ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "toolchain/base/value_ids.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

namespace {
struct Worklist {
  // The file containing the instruction we're currently processing.
  const File* sem_ir = nullptr;
  // The instructions we need to compute fingerprints for.
  llvm::SmallVector<
      std::pair<const File*, std::variant<InstId, InstBlockId, ImplId>>>
      todo;
  // The contents of the current instruction as accumulated so far. This is used
  // to build a Merkle tree containing a fingerprint for the current
  // instruction.
  llvm::SmallVector<llvm::stable_hash> contents = {};
  // Known cached instruction fingerprints. Each item in `todo` will be added to
  // the cache if not already present.
  Map<std::pair<const File*, InstId>, uint64_t>* fingerprints;

  // Finish fingerprinting and compute the fingerprint.
  auto Finish() -> uint64_t { return llvm::stable_hash_combine(contents); }

  // Add an invalid marker to the contents. This is used when the entity
  // contains a `None` ID. This uses an arbitrary fixed value that is assumed
  // to be unlikely to collide with a valid value.
  auto AddInvalid() -> void { contents.push_back(-1); }

  // Add a string to the contents.
  auto AddString(llvm::StringRef string) -> void {
    contents.push_back(llvm::stable_hash_name(string));
  }

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
    // TODO: Should we include the parent index?
  }

  auto AddInFile(const File* file, InstId inner_id) -> void {
    if (!inner_id.has_value()) {
      AddInvalid();
      return;
    }
    if (auto lookup = fingerprints->Lookup(std::pair(file, inner_id))) {
      contents.push_back(lookup.value());
    } else {
      todo.push_back({file, inner_id});
    }
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
    Add(sem_ir->types().GetInstId(type_id));
  }

  template <typename T>
  auto AddBlock(llvm::ArrayRef<T> block) -> void {
    contents.push_back(block.size());
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

  auto Add(TypeBlockId type_block_id) -> void {
    if (!type_block_id.has_value()) {
      AddInvalid();
      return;
    }
    AddBlock(sem_ir->type_blocks().Get(type_block_id));
  }

  auto Add(StructTypeField field) -> void {
    Add(field.name_id);
    Add(field.type_id);
  }

  auto Add(StructTypeFieldsId struct_type_fields_id) -> void {
    if (!struct_type_fields_id.has_value()) {
      AddInvalid();
      return;
    }
    AddBlock(sem_ir->struct_type_fields().Get(struct_type_fields_id));
  }

  auto Add(NameScopeId name_scope_id) -> void {
    if (!name_scope_id.has_value()) {
      AddInvalid();
      return;
    }
    const auto& scope = sem_ir->name_scopes().Get(name_scope_id);
    Add(scope.name_id());
    if (!sem_ir->name_scopes().IsPackage(name_scope_id) &&
        scope.parent_scope_id().has_value()) {
      Add(sem_ir->name_scopes().Get(scope.parent_scope_id()).inst_id());
    }
  }

  template <typename EntityT = EntityWithParamsBase>
  auto AddEntity(const std::type_identity_t<EntityT>& entity) -> void {
    Add(entity.name_id);
    if (entity.parent_scope_id.has_value()) {
      Add(sem_ir->name_scopes().Get(entity.parent_scope_id).inst_id());
    }
  }

  auto Add(FunctionId function_id) -> void {
    AddEntity(sem_ir->functions().Get(function_id));
  }

  auto Add(ClassId class_id) -> void {
    AddEntity(sem_ir->classes().Get(class_id));
  }

  auto Add(InterfaceId interface_id) -> void {
    AddEntity(sem_ir->interfaces().Get(interface_id));
  }

  auto Add(AssociatedConstantId assoc_const_id) -> void {
    AddEntity<AssociatedConstant>(
        sem_ir->associated_constants().Get(assoc_const_id));
  }

  auto Add(ImplId impl_id) -> void {
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
    for (auto [interface_id, specific_id] : facet_type.impls_constraints) {
      Add(interface_id);
      Add(specific_id);
    }
    for (auto [lhs_id, rhs_id] : facet_type.rewrite_constraints) {
      Add(lhs_id);
      Add(rhs_id);
    }
    contents.push_back(facet_type.other_requirements);
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

  auto Add(const llvm::APInt& value) -> void {
    contents.push_back(value.getBitWidth());
    for (auto word : llvm::seq((value.getBitWidth() + 63) / 64)) {
      // TODO: Is there a better way to copy the words from an APInt?
      contents.push_back(value.extractBitsAsZExtValue(64, 64 * word));
    }
  }

  auto Add(IntId int_id) -> void { Add(sem_ir->ints().Get(int_id)); }

  auto Add(FloatId float_id) -> void {
    Add(sem_ir->floats().Get(float_id).bitcastToAPInt());
  }

  auto Add(PackageNameId package_id) -> void {
    if (auto ident_id = package_id.AsIdentifierId(); ident_id.has_value()) {
      AddString(sem_ir->identifiers().Get(ident_id));
    } else {
      // TODO: May collide with a user package of the same name. Consider using
      // a different value.
      AddString(package_id.AsSpecialName());
    }
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

  auto Add(ImportIRId ir_id) -> void {
    const auto* ir = sem_ir->import_irs().Get(ir_id).sem_ir;
    Add(ir->package_id());
    Add(ir->library_id());
  }

  auto Add(ImportIRInstId ir_inst_id) -> void {
    auto ir_inst = sem_ir->import_ir_insts().Get(ir_inst_id);
    AddInFile(sem_ir->import_irs().Get(ir_inst.ir_id).sem_ir, ir_inst.inst_id);
  }

  template <typename T>
    requires(std::same_as<T, BoolValue> ||
             std::same_as<T, CompileTimeBindIndex> ||
             std::same_as<T, ElementIndex> || std::same_as<T, FloatKind> ||
             std::same_as<T, IntKind> || std::same_as<T, CallParamIndex>)
  auto Add(T arg) -> void {
    // Index-like ID: just include the value directly.
    contents.push_back(arg.index);
  }

  template <typename T>
    requires(std::same_as<T, AnyRawId> || std::same_as<T, ExprRegionId> ||
             std::same_as<T, LocId> || std::same_as<T, RealId>)
  auto Add(T /*arg*/) -> void {
    CARBON_FATAL("Unexpected instruction operand kind {0}", typeid(T).name());
  }

  // Add an instruction argument to the contents of the current instruction.
  template <typename... Types>
  auto AddWithKind(uint64_t arg, TypeEnum<Types...> kind) -> void {
    using AddFunction = void (*)(Worklist& worklist, uint64_t arg);
    using Kind = decltype(kind);

    // Build a lookup table to add an argument of the given kind.
    static constexpr std::array<AddFunction, Kind::NumTypes + 2> Table = [] {
      std::array<AddFunction, Kind::NumTypes + 2> table;
      table[Kind::None.ToIndex()] = [](Worklist& /*worklist*/,
                                       uint64_t /*arg*/) {};
      table[Kind::Invalid.ToIndex()] = [](Worklist& /*worklist*/,
                                          uint64_t /*arg*/) {
        CARBON_FATAL("Unexpected invalid argument kind");
      };
      ((table[Kind::template For<Types>.ToIndex()] =
            [](Worklist& worklist, uint64_t arg) {
              return worklist.Add(Inst::FromRaw<Types>(arg));
            }),
       ...);
      return table;
    }();

    Table[kind.ToIndex()](*this, arg);
  }

  // Ensure all the instructions on the todo list have fingerprints. To avoid a
  // re-lookup, returns the fingerprint of the first instruction on the todo
  // list, and requires the todo list to be non-empty.
  auto Run() -> uint64_t {
    CARBON_CHECK(!todo.empty());
    while (true) {
      const size_t init_size = todo.size();
      auto [next_sem_ir, next] = todo.back();

      sem_ir = next_sem_ir;
      contents.clear();

      if (!std::holds_alternative<InstId>(next)) {
        // Add the contents of the `next` instruction so they all contribute to
        // the `contents`.
        if (auto* impl_id = std::get_if<ImplId>(&next)) {
          Add(*impl_id);
        } else if (auto* inst_block_id = std::get_if<InstBlockId>(&next)) {
          Add(*inst_block_id);
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
      if (auto lookup =
              fingerprints->Lookup(std::pair(next_sem_ir, next_inst_id))) {
        todo.pop_back();
        if (todo.empty()) {
          return lookup.value();
        }
        continue;
      }

      // Keep this instruction in `todo` for now. If we add more work, we'll
      // finish that work and process this instruction again, and if not, we'll
      // pop the instruction at the end of the loop.
      auto inst = next_sem_ir->insts().Get(next_inst_id);
      auto [arg0_kind, arg1_kind] = inst.ArgKinds();

      // Add the instruction's fields to the contents.
      Add(inst.kind());

      // Don't include the type if it's `type` or `<error>`, because those types
      // are self-referential.
      if (inst.type_id() != TypeType::SingletonTypeId &&
          inst.type_id() != ErrorInst::SingletonTypeId) {
        Add(inst.type_id());
      }

      AddWithKind(inst.arg0(), arg0_kind);
      AddWithKind(inst.arg1(), arg1_kind);

      // If we didn't add any work, we have a fingerprint for this instruction;
      // pop it from the todo list. Otherwise, we leave it on the todo list so
      // we can compute its fingerprint once we've finished the work we added.
      if (todo.size() == init_size) {
        uint64_t fingerprint = Finish();
        fingerprints->Insert(std::pair(next_sem_ir, next_inst_id), fingerprint);
        todo.pop_back();
        if (todo.empty()) {
          return fingerprint;
        }
      }
    }
  }
};
}  // namespace

auto InstFingerprinter::GetOrCompute(const File* file, InstId inst_id)
    -> uint64_t {
  Worklist worklist = {.todo = {{file, inst_id}},
                       .fingerprints = &fingerprints_};
  return worklist.Run();
}

auto InstFingerprinter::GetOrCompute(const File* file,
                                     InstBlockId inst_block_id) -> uint64_t {
  Worklist worklist = {.todo = {{file, inst_block_id}},
                       .fingerprints = &fingerprints_};
  return worklist.Run();
}

auto InstFingerprinter::GetOrCompute(const File* file, ImplId impl_id)
    -> uint64_t {
  Worklist worklist = {.todo = {{file, impl_id}},
                       .fingerprints = &fingerprints_};
  return worklist.Run();
}

}  // namespace Carbon::SemIR
