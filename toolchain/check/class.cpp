// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/class.h"

#include "toolchain/check/context.h"
#include "toolchain/check/eval.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/inst.h"
#include "toolchain/check/type.h"

namespace Carbon::Check {

auto TryGetAsClass(Context& context, SemIR::TypeId type_id) -> SemIR::Class* {
  auto class_type = context.types().TryGetAs<SemIR::ClassType>(type_id);
  if (!class_type) {
    return nullptr;
  }
  return &context.classes().Get(class_type->class_id);
}

auto SetClassSelfType(Context& context, SemIR::ClassId class_id) -> void {
  auto& class_info = context.classes().Get(class_id);
  auto specific_id = context.generics().GetSelfSpecific(class_info.generic_id);
  class_info.self_type_id = GetClassType(context, class_id, specific_id);
}

auto StartClassDefinition(Context& context, SemIR::Class& class_info,
                          SemIR::InstId definition_id) -> void {
  // Track that this declaration is the definition.
  CARBON_CHECK(!class_info.has_definition_started());
  class_info.definition_id = definition_id;
  class_info.scope_id = context.name_scopes().Add(
      definition_id, SemIR::NameId::None, class_info.parent_scope_id);

  // Introduce `Self`.
  context.name_scopes().AddRequiredName(
      class_info.scope_id, SemIR::NameId::SelfType,
      context.types().GetInstId(class_info.self_type_id));
}

// Checks that the specified finished adapter definition is valid and builds and
// returns a corresponding complete type witness instruction.
static auto CheckCompleteAdapterClassType(
    Context& context, Parse::NodeId node_id, SemIR::ClassId class_id,
    llvm::ArrayRef<SemIR::InstId> field_decls,
    llvm::ArrayRef<SemIR::InstId> body) -> SemIR::InstId {
  const auto& class_info = context.classes().Get(class_id);
  if (class_info.base_id.has_value()) {
    CARBON_DIAGNOSTIC(AdaptWithBase, Error, "adapter with base class");
    CARBON_DIAGNOSTIC(AdaptWithBaseHere, Note, "`base` declaration is here");
    context.emitter()
        .Build(class_info.adapt_id, AdaptWithBase)
        .Note(class_info.base_id, AdaptWithBaseHere)
        .Emit();
    return SemIR::ErrorInst::SingletonInstId;
  }

  if (!field_decls.empty()) {
    CARBON_DIAGNOSTIC(AdaptWithFields, Error, "adapter with fields");
    CARBON_DIAGNOSTIC(AdaptWithFieldHere, Note,
                      "first field declaration is here");
    context.emitter()
        .Build(class_info.adapt_id, AdaptWithFields)
        .Note(field_decls.front(), AdaptWithFieldHere)
        .Emit();
    return SemIR::ErrorInst::SingletonInstId;
  }

  for (auto inst_id : body) {
    if (auto function_decl =
            context.insts().TryGetAs<SemIR::FunctionDecl>(inst_id)) {
      auto& function = context.functions().Get(function_decl->function_id);
      if (function.virtual_modifier ==
          SemIR::Function::VirtualModifier::Virtual) {
        CARBON_DIAGNOSTIC(AdaptWithVirtual, Error,
                          "adapter with virtual function");
        CARBON_DIAGNOSTIC(AdaptWithVirtualHere, Note,
                          "first virtual function declaration is here");
        context.emitter()
            .Build(class_info.adapt_id, AdaptWithVirtual)
            .Note(inst_id, AdaptWithVirtualHere)
            .Emit();
        return SemIR::ErrorInst::SingletonInstId;
      }
    }
  }

  // The object representation of the adapter is the object representation
  // of the adapted type.
  auto adapted_type_id =
      class_info.GetAdaptedType(context.sem_ir(), SemIR::SpecificId::None);
  auto object_repr_id = context.types().GetObjectRepr(adapted_type_id);

  return AddInst<SemIR::CompleteTypeWitness>(
      context, node_id,
      {.type_id =
           GetSingletonType(context, SemIR::WitnessType::SingletonInstId),
       .object_repr_id = object_repr_id});
}

static auto AddStructTypeFields(
    Context& context,
    llvm::SmallVector<SemIR::StructTypeField>& struct_type_fields,
    llvm::ArrayRef<SemIR::InstId> field_decls) -> SemIR::StructTypeFieldsId {
  for (auto field_decl_id : field_decls) {
    auto field_decl = context.insts().GetAs<SemIR::FieldDecl>(field_decl_id);
    field_decl.index =
        SemIR::ElementIndex{static_cast<int>(struct_type_fields.size())};
    ReplaceInstPreservingConstantValue(context, field_decl_id, field_decl);
    if (field_decl.type_id == SemIR::ErrorInst::SingletonTypeId) {
      struct_type_fields.push_back(
          {.name_id = field_decl.name_id,
           .type_id = SemIR::ErrorInst::SingletonTypeId});
      continue;
    }
    auto unbound_element_type =
        context.sem_ir().types().GetAs<SemIR::UnboundElementType>(
            field_decl.type_id);
    struct_type_fields.push_back(
        {.name_id = field_decl.name_id,
         .type_id = context.types().GetTypeIdForTypeInstId(
             unbound_element_type.element_type_inst_id)});
  }
  auto fields_id =
      context.struct_type_fields().AddCanonical(struct_type_fields);
  return fields_id;
}

// Builds and returns a vtable for the current class. Assumes that the virtual
// functions for the class are listed as the top element of the `vtable_stack`.
static auto BuildVtable(Context& context, Parse::NodeId node_id,
                        SemIR::InstId base_vtable_id,
                        llvm::ArrayRef<SemIR::InstId> vtable_contents)
    -> SemIR::InstId {
  llvm::SmallVector<SemIR::InstId> vtable;
  if (base_vtable_id.has_value()) {
    LoadImportRef(context, base_vtable_id);
    auto canonical_base_vtable_id =
        context.constant_values().GetConstantInstId(base_vtable_id);
    if (canonical_base_vtable_id == SemIR::ErrorInst::SingletonInstId) {
      return SemIR::ErrorInst::SingletonInstId;
    }
    auto base_vtable_inst_block = context.inst_blocks().Get(
        context.insts()
            .GetAs<SemIR::Vtable>(canonical_base_vtable_id)
            .virtual_functions_id);
    // TODO: Avoid quadratic search. Perhaps build a map from `NameId` to the
    // elements of the top of `vtable_stack`.
    for (auto fn_decl_id : base_vtable_inst_block) {
      auto fn_decl = GetCalleeFunction(context.sem_ir(), fn_decl_id);
      const auto& fn = context.functions().Get(fn_decl.function_id);
      for (auto override_fn_decl_id : vtable_contents) {
        auto override_fn_decl =
            context.insts().GetAs<SemIR::FunctionDecl>(override_fn_decl_id);
        const auto& override_fn =
            context.functions().Get(override_fn_decl.function_id);
        if (override_fn.virtual_modifier ==
                SemIR::FunctionFields::VirtualModifier::Impl &&
            override_fn.name_id == fn.name_id) {
          // TODO: Support generic base classes, rather than passing
          // `SpecificId::None`.
          CheckFunctionTypeMatches(context, override_fn, fn,
                                   SemIR::SpecificId::None,
                                   /*check_syntax=*/false,
                                   /*check_self=*/false);
          fn_decl_id = override_fn_decl_id;
        }
      }
      vtable.push_back(fn_decl_id);
    }
  }

  for (auto inst_id : vtable_contents) {
    auto fn_decl = context.insts().GetAs<SemIR::FunctionDecl>(inst_id);
    const auto& fn = context.functions().Get(fn_decl.function_id);
    if (fn.virtual_modifier != SemIR::FunctionFields::VirtualModifier::Impl) {
      vtable.push_back(inst_id);
    }
  }
  return AddInst<SemIR::Vtable>(
      context, node_id,
      {.type_id = GetSingletonType(context, SemIR::VtableType::SingletonInstId),
       .virtual_functions_id = context.inst_blocks().Add(vtable)});
}

// Checks that the specified finished class definition is valid and builds and
// returns a corresponding complete type witness instruction.
static auto CheckCompleteClassType(
    Context& context, Parse::NodeId node_id, SemIR::ClassId class_id,
    llvm::ArrayRef<SemIR::InstId> field_decls,
    llvm::ArrayRef<SemIR::InstId> vtable_contents,
    llvm::ArrayRef<SemIR::InstId> body) -> SemIR::InstId {
  auto& class_info = context.classes().Get(class_id);
  if (class_info.adapt_id.has_value()) {
    return CheckCompleteAdapterClassType(context, node_id, class_id,
                                         field_decls, body);
  }

  bool defining_vptr = class_info.is_dynamic;
  auto base_type_id =
      class_info.GetBaseType(context.sem_ir(), SemIR::SpecificId::None);
  SemIR::Class* base_class_info = nullptr;
  if (base_type_id.has_value()) {
    // TODO: If the base class is template dependent, we will need to decide
    // whether to add a vptr as part of instantiation.
    base_class_info = TryGetAsClass(context, base_type_id);
    if (base_class_info && base_class_info->is_dynamic) {
      defining_vptr = false;
    }
  }

  llvm::SmallVector<SemIR::StructTypeField> struct_type_fields;
  struct_type_fields.reserve(defining_vptr + class_info.base_id.has_value() +
                             field_decls.size());
  if (defining_vptr) {
    struct_type_fields.push_back(
        {.name_id = SemIR::NameId::Vptr,
         .type_id =
             GetPointerType(context, SemIR::VtableType::SingletonInstId)});
  }
  if (base_type_id.has_value()) {
    auto base_decl = context.insts().GetAs<SemIR::BaseDecl>(class_info.base_id);
    base_decl.index =
        SemIR::ElementIndex{static_cast<int>(struct_type_fields.size())};
    ReplaceInstPreservingConstantValue(context, class_info.base_id, base_decl);
    struct_type_fields.push_back(
        {.name_id = SemIR::NameId::Base, .type_id = base_type_id});
  }

  if (class_info.is_dynamic) {
    class_info.vtable_id = BuildVtable(
        context, node_id,
        defining_vptr ? SemIR::InstId::None : base_class_info->vtable_id,
        vtable_contents);
  }

  return AddInst<SemIR::CompleteTypeWitness>(
      context, node_id,
      {.type_id =
           GetSingletonType(context, SemIR::WitnessType::SingletonInstId),
       .object_repr_id = GetStructType(
           context,
           AddStructTypeFields(context, struct_type_fields, field_decls))});
}

auto ComputeClassObjectRepr(Context& context, Parse::NodeId node_id,
                            SemIR::ClassId class_id,
                            llvm::ArrayRef<SemIR::InstId> field_decls,
                            llvm::ArrayRef<SemIR::InstId> vtable_contents,
                            llvm::ArrayRef<SemIR::InstId> body) -> void {
  auto complete_type_witness_id = CheckCompleteClassType(
      context, node_id, class_id, field_decls, vtable_contents, body);
  auto& class_info = context.classes().Get(class_id);
  class_info.complete_type_witness_id = complete_type_witness_id;
}

}  // namespace Carbon::Check
