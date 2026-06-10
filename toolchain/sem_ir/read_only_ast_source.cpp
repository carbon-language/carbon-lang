// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/read_only_ast_source.h"

namespace Carbon::SemIR {

// TODO: dedup (from generate_ast.cpp)
static auto GetAsCarbonOwnedClass(const File& sem_ir,
                                  const clang::TagDecl* tag_decl)
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

// TODO: dedup (from export.cpp)
static auto GetStructTypeFields(const File& sem_ir,
                                const SemIR::Class& class_info)
    -> llvm::ArrayRef<SemIR::StructTypeField> {
  if (class_info.adapt_id.has_value()) {
    // The representation of an adapter won't necessarily be a
    // struct. Return an empty array since adapters can't declare
    // fields.
    return {};
  }

  auto object_repr_type_id =
      class_info.GetObjectRepr(sem_ir, SemIR::SpecificId::None);
  if (object_repr_type_id == SemIR::ErrorInst::TypeId) {
    return {};
  }
  auto struct_type =
      sem_ir.types().GetAs<SemIR::StructType>(object_repr_type_id);
  return sem_ir.struct_type_fields().Get(struct_type.fields_id);
}

// Get the field offset for each field in a class.
//
// Returns true on success, false if any error occurs.
static auto CalculateCppFieldOffsets(
    const File& sem_ir, SemIR::ClassId class_id,
    llvm::DenseMap<const clang::FieldDecl*, uint64_t>& field_offsets) -> bool {
  auto class_info = sem_ir.classes().Get(class_id);
  const auto& class_scope = sem_ir.name_scopes().Get(class_info.scope_id);

  auto class_layout = SemIR::ObjectLayout::Empty();
  for (const auto& struct_field : GetStructTypeFields(sem_ir, class_info)) {
    auto field_type_id =
        sem_ir.types().GetTypeIdForTypeInstId(struct_field.type_inst_id);
    auto field_layout =
        sem_ir.types().GetCompleteTypeInfo(field_type_id).object_layout;

    // Use the field's name to look up the corresponding entry in the
    // class. If it's a `FieldDecl`, write out the offset of the
    // corresponding `clang::FieldDecl`.
    auto class_field =
        LookupClassFieldByStructField(sem_ir, class_scope, struct_field);
    if (class_field) {
      const auto* clang_decl =
          sem_ir.clang_decls().Lookup(class_field->inst_id);
      if (!clang_decl) {
        return false;
      }

      auto cpp_field_decl = cast<clang::FieldDecl>(clang_decl->decl());
      field_offsets.insert(
          {cpp_field_decl, class_layout.FieldOffset(field_layout).bits()});
    }

    class_layout.AppendField(field_layout);
  }

  return true;
}

auto ReadOnlyASTSource::layoutRecordType(
    const clang::RecordDecl* record_decl, uint64_t& size, uint64_t& alignment,
    llvm::DenseMap<const clang::FieldDecl*, uint64_t>& field_offsets,
    llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>& base_offsets,
    llvm::DenseMap<const clang::CXXRecordDecl*, clang::CharUnits>&
        vbase_offsets) -> bool {
  auto carbon_class_info = GetAsCarbonOwnedClass(sem_ir_, record_decl);
  if (!carbon_class_info) {
    return false;
  }
  auto& [class_type_id, class_type] = *carbon_class_info;

  // Set the overall size and alignment. We round up the size to an integer
  // number of bytes in order to avoid surprising Clang too much.
  auto layout =
      sem_ir_.types().GetCompleteTypeInfo(class_type_id).object_layout;
  size = layout.size.bytes() * 8;
  alignment = layout.alignment.bits();

  // Fill in `field_offsets`.
  CalculateCppFieldOffsets(sem_ir_, class_type.class_id, field_offsets);

  // Add offset for base class, if any.
  if (const auto* class_decl = dyn_cast<clang::CXXRecordDecl>(record_decl);
      class_decl && !class_decl->bases().empty()) {
    CARBON_CHECK(class_decl->getNumBases() == 1,
                 "Carbon class with multiple bases");
    const auto& base = *class_decl->bases_begin();
    // TODO: If this class introduced a vptr, the base will be at an offset of
    // `sizeof(void*)`, not 0.
    base_offsets.insert(
        {base.getType()->getAsCXXRecordDecl()->getCanonicalDecl(),
         clang::CharUnits::Zero()});

    // TODO: Support deriving from a C++ class with virtual bases.
    CARBON_CHECK(class_decl->getNumVBases() == 0,
                 "Carbon class with multiple bases");
    static_cast<void>(vbase_offsets);
  }

  return true;
}

}  // namespace Carbon::SemIR
