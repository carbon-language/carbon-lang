// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/class.h"

#include "clang/AST/Decl.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

static auto GetFoundationType(const File& file, SpecificId specific_id,
                              InstId inst_id) -> TypeId {
  if (!inst_id.has_value()) {
    return TypeId::None;
  }
  if (inst_id == ErrorInst::InstId) {
    return ErrorInst::TypeId;
  }
  return TypeId::ForTypeConstant(GetConstantValueInSpecific(
      file, specific_id,
      file.insts().GetAs<AnyFoundationDecl>(inst_id).foundation_type_inst_id));
}

auto Class::GetAdaptedType(const File& file, SpecificId specific_id) const
    -> TypeId {
  return GetFoundationType(file, specific_id, adapt_id);
}

auto Class::GetBaseType(const File& file, SpecificId specific_id) const
    -> TypeId {
  return GetFoundationType(file, specific_id, base_id);
}

auto Class::GetObjectRepr(const File& file, SpecificId specific_id) const
    -> TypeId {
  if (!complete_type_witness_id.has_value()) {
    return TypeId::None;
  }
  auto witness_id =
      GetConstantValueInSpecific(file, specific_id, complete_type_witness_id);
  if (witness_id == ErrorInst::ConstantId) {
    return ErrorInst::TypeId;
  }
  return file.types().GetTypeIdForTypeInstId(
      file.constant_values()
          .GetInstAs<CompleteTypeWitness>(witness_id)
          .object_repr_type_inst_id);
}

auto Class::GetStructTypeFields(const File& sem_ir) const
    -> llvm::ArrayRef<SemIR::StructTypeField> {
  if (adapt_id.has_value()) {
    // The representation of an adapter won't necessarily be a
    // struct. Return an empty array since adapters can't declare
    // fields.
    return {};
  }

  auto object_repr_type_id = GetObjectRepr(sem_ir, SemIR::SpecificId::None);
  if (object_repr_type_id == SemIR::ErrorInst::TypeId) {
    return {};
  }
  auto struct_type =
      sem_ir.types().GetAs<SemIR::StructType>(object_repr_type_id);
  return sem_ir.struct_type_fields().Get(struct_type.fields_id);
}

auto GetAsCarbonOwnedClass(const File& sem_ir, const clang::TagDecl* tag_decl)
    -> std::optional<std::pair<SemIR::TypeId, SemIR::ClassType>> {
  // Quickly check whether we could possibly own this class.
  // TODO: Once we multiplex with the ASTReader, handle
  // ASTReader::completeVisibleDeclsMap setting this to `false`.
  if (!tag_decl->hasExternalVisibleStorage()) {
    return std::nullopt;
  }

  auto key = SemIR::ClangDeclKey::ForNonFunctionDecl(
      const_cast<clang::TagDecl*>(tag_decl->getFirstDecl()));
  auto clang_decl_id = sem_ir.clang_decls().LookupId(key);
  if (!clang_decl_id.has_value()) {
    return std::nullopt;
  }

  auto inst_id = sem_ir.clang_decls().Get(clang_decl_id).inst_id;
  auto const_id = sem_ir.constant_values().Get(inst_id);
  if (!const_id.has_value()) {
    return std::nullopt;
  }

  auto class_type =
      sem_ir.constant_values().TryGetInstAs<SemIR::ClassType>(const_id);
  if (!class_type) {
    return std::nullopt;
  }

  // Determine whether this class was imported from C++.
  // TODO: This currently can't happen, because only Carbon classes have
  // external lexical storage, but will happen once we support importing C++
  // classes from AST files. Add a test once that is supported.
  // TODO: Consider setting `extern_library_id` on classes imported from C++ to
  // indicate the current file does not own them.
  const auto& class_info = sem_ir.classes().Get(class_type->class_id);
  if (class_info.parent_scope_id.has_value() &&
      sem_ir.name_scopes().Get(class_info.parent_scope_id).is_cpp_scope()) {
    return std::nullopt;
  }

  auto class_type_id = sem_ir.types().GetTypeIdForTypeConstantId(const_id);
  return std::make_pair(class_type_id, *class_type);
}

auto LookupClassFieldByStructField(const File& sem_ir,
                                   const SemIR::NameScope& class_scope,
                                   const SemIR::StructTypeField& struct_field)
    -> std::optional<SemIR::InstStore::GetAsWithIdResult<SemIR::FieldDecl>> {
  if (auto entry_id = class_scope.Lookup(struct_field.name_id)) {
    auto field_inst_id =
        class_scope.GetEntry(*entry_id).result.target_inst_id();
    return sem_ir.insts().TryGetAsWithId<SemIR::FieldDecl>(field_inst_id);
  }
  return std::nullopt;
}

}  // namespace Carbon::SemIR

namespace Carbon {
template class ValueStore<SemIR::ClassId, SemIR::Class, Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
