// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_fingerprinter.h"

#include "common/ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"

namespace Carbon::SemIR {

namespace {
struct Worklist {
  const File* sem_ir;
  llvm::MutableArrayRef<uint64_t> fingerprints;
  llvm::SmallVector<InstId> todo;
  llvm::SmallVector<llvm::stable_hash> contents = {};

  // Add an invalid marker to the contents. This is used when the entity
  // contains an invalid ID.
  auto AddInvalid() -> void { contents.push_back(-1); }

  // Add a typed argument to the contents of the current instruction. If we
  // don't yet have a fingerprint for the argument, adds that argument to the
  // worklist instead.
  auto Add(InstKind kind) -> void {
    // TODO: Precompute or cache this.
    contents.push_back(llvm::stable_hash_name(kind.ir_name()));
  }
  auto Add(NameId name_id) -> void {
    contents.push_back(
        llvm::stable_hash_name(sem_ir->names().GetIRBaseName(name_id)));
  }
  auto Add(EntityNameId entity_name_id) -> void {
    if (!entity_name_id.is_valid()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->entity_names().Get(entity_name_id).name_id);
  }
  auto Add(InstId inner_id) -> void {
    if (!inner_id.is_valid()) {
      AddInvalid();
      return;
    }
    if (fingerprints[inner_id.index]) {
      contents.push_back(fingerprints[inner_id.index]);
    } else {
      todo.push_back(inner_id);
    }
  }
  auto Add(ConstantId constant_id) -> void {
    if (!constant_id.is_valid()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->constant_values().GetInstId(constant_id));
  }
  auto Add(TypeId type_id) -> void {
    if (!type_id.is_valid()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->types().GetInstId(type_id));
  }
  auto Add(InstBlockId inst_block_id) -> void {
    if (!inst_block_id.is_valid()) {
      AddInvalid();
      return;
    }
    auto block = sem_ir->inst_blocks().Get(inst_block_id);
    contents.push_back(block.size());
    for (auto inner_id : block) {
      Add(inner_id);
    }
  }
  auto Add(TypeBlockId type_block_id) -> void {
    if (!type_block_id.is_valid()) {
      AddInvalid();
      return;
    }
    auto block = sem_ir->type_blocks().Get(type_block_id);
    contents.push_back(block.size());
    for (auto inner_id : block) {
      Add(inner_id);
    }
  }
  auto Add(StructTypeFieldsId struct_type_fields_id) -> void {
    if (!struct_type_fields_id.is_valid()) {
      AddInvalid();
      return;
    }
    auto block = sem_ir->struct_type_fields().Get(struct_type_fields_id);
    contents.push_back(block.size());
    for (auto field : block) {
      Add(field.name_id);
      Add(field.type_id);
    }
  }
  auto Add(NameScopeId name_scope_id) -> void {
    if (!name_scope_id.is_valid()) {
      AddInvalid();
      return;
    }
    const auto& scope = sem_ir->name_scopes().Get(name_scope_id);
    Add(scope.name_id());
    if (!sem_ir->name_scopes().IsPackage(name_scope_id) &&
        scope.parent_scope_id().is_valid()) {
      Add(
          sem_ir->name_scopes().Get(scope.parent_scope_id()).inst_id());
    }
  }
  auto Add(const EntityWithParamsBase& entity) -> void {
    Add(entity.name_id);
    Add(entity.parent_scope_id);
  }
  auto Add(ImplId impl_id) -> void {
    const auto& impl = sem_ir->impls().Get(impl_id);
    Add(impl.self_id);
    Add(impl.constraint_id);
    Add(impl.parent_scope_id);
  }
  auto Add(GenericId generic_id) -> void {
    if (!generic_id.is_valid()) {
      AddInvalid();
      return;
    }
    Add(sem_ir->generics().Get(generic_id).decl_id);
  }
  auto Add(SpecificId specific_id) -> void {
    if (!specific_id.is_valid()) {
      AddInvalid();
      return;
    }
    const auto& specific = sem_ir->specifics().Get(specific_id);
    Add(specific.generic_id);
    Add(specific.args_id);
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

  // Add an instruction argument to the contents of the current instruction.
  auto AddWithKind(uint64_t arg, IdKind kind) -> void {
    // TODO: Generate this.
    switch (kind) {
      case IdKind::None:
        break;
      case IdKind::For<InstId>:
      case IdKind::For<AbsoluteInstId>:
        Add(InstId(arg));
        break;
      case IdKind::For<ConstantId>:
        Add(ConstantId(arg));
        break;
      case IdKind::For<TypeId>:
        Add(TypeId(arg));
        break;
      case IdKind::For<InstBlockId>:
      case IdKind::For<AbsoluteInstBlockId>:
        Add(InstBlockId(arg));
        break;
      case IdKind::For<TypeBlockId>:
        Add(TypeBlockId(arg));
        break;
      case IdKind::For<StructTypeFieldsId>:
        Add(StructTypeFieldsId(arg));
        break;
      case IdKind::For<NameId>:
        Add(NameId(arg));
        break;
      case IdKind::For<EntityNameId>:
        Add(EntityNameId(arg));
        break;
      case IdKind::For<NameScopeId>:
        Add(NameScopeId(arg));
        break;
      case IdKind::For<FunctionId>:
        Add(sem_ir->functions().Get(FunctionId(arg)));
        break;
      case IdKind::For<ClassId>:
        Add(sem_ir->classes().Get(ClassId(arg)));
        break;
      case IdKind::For<InterfaceId>:
        Add(sem_ir->interfaces().Get(InterfaceId(arg)));
        break;
      case IdKind::For<ImplId>:
        Add(ImplId(arg));
        break;
      case IdKind::For<GenericId>:
        Add(GenericId(arg));
        break;
      case IdKind::For<SpecificId>:
        Add(SpecificId(arg));
        break;
      case IdKind::For<BoolValue>:
      case IdKind::For<CompileTimeBindIndex>:
      case IdKind::For<ElementIndex>:
      case IdKind::For<FloatKind>:
      case IdKind::For<IntKind>:
      case IdKind::For<RuntimeParamIndex>:
        // Index-like ID: just include the value directly.
        contents.push_back(arg);
        break;
      case IdKind::For<IntId>:
        Add(IntId::MakeRaw(arg));
        break;
      case IdKind::For<FloatId>:
        Add(FloatId(arg));
        break;
      case IdKind::For<StringLiteralValueId>:
        contents.push_back(llvm::stable_hash_name(
            sem_ir->string_literal_values().Get(StringLiteralValueId(arg))));
        break;
      case IdKind::For<RealId>:
      case IdKind::For<FacetTypeId>:
      case IdKind::For<ImportIRId>:
      case IdKind::For<ImportIRInstId>:
      case IdKind::For<ExprRegionId>:
      case IdKind::For<LibraryNameId>:
        // TODO: Fingerprint more things.
        break;
      case IdKind::For<AnyRawId>:
      case IdKind::For<LocId>:
      case IdKind::Invalid:
        CARBON_FATAL("Unexpected instruction operand kind");
    }
  }
};
}

InstFingerprinter::InstFingerprinter(const File& sem_ir) : sem_ir_(&sem_ir) {
  fingerprints_.resize(sem_ir.insts().size(), 0);
}

auto InstFingerprinter::GetOrCompute(InstId inst_id) -> uint64_t {
  Worklist worklist = {
      .sem_ir = sem_ir_, .fingerprints = fingerprints_, .todo = {inst_id}};

  while (!worklist.todo.empty()) {
    if (worklist.fingerprints[worklist.todo.back().index]) {
      worklist.todo.pop_back();
      continue;
    }

    size_t init_size = worklist.todo.size();
    auto inst = sem_ir_->insts().Get(worklist.todo.back());
    auto [arg0_kind, arg1_kind] = inst.ArgKinds();

    worklist.contents.clear();
    worklist.Add(inst.kind());

    // Don't include the type if it's `type` or `<error>`, because those types
    // are self-referential.
    if (inst.type_id() != TypeType::SingletonTypeId &&
        inst.type_id() != ErrorInst::SingletonTypeId) {
      worklist.Add(inst.type_id());
    }

    for (auto [arg, kind] : {std::pair(inst.arg0(), arg0_kind),
                             std::pair(inst.arg1(), arg1_kind)}) {
      worklist.AddWithKind(arg, kind);
    }

    // If we didn't add any work, we have a fingerprint for this instruction.
    if (worklist.todo.size() == init_size) {
      auto fingerprint = llvm::stable_hash_combine(worklist.contents);
      // We use 0 to indicate we've not computed the fingerprint yet. In the
      // unlikely event we calculate a hash of 0, use a different hash.
      if (fingerprint == 0) {
        fingerprint = 1;
      }
      fingerprints_[worklist.todo.back().index] = fingerprint;
      worklist.todo.pop_back();
    }
  }
  return fingerprints_[inst_id.index];
}

}  // namespace Carbon::SemIR
