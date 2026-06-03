// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/sem_ir/dump.h"

#include <string>

#include "common/raw_string_ostream.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/stringify.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

LLVM_DUMP_METHOD auto Dump(const File& file) -> std::string {
  return PrintToString(file);
}

static auto DumpNameIfValid(const File& file, NameId name_id) -> std::string {
  RawStringOstream out;
  if (name_id.has_value()) {
    out << " `" << file.names().GetFormatted(name_id) << "`";
  }
  return out.TakeStr();
}

static auto DumpLocSummary(const File& file, LocId loc_id) -> std::string {
  RawStringOstream out;
  // TODO: If the canonical location is None but the original is an InstId,
  // should we dump the InstId anyway even though it has no location? Is that
  // ever useful?
  loc_id = file.insts().GetCanonicalLocId(loc_id);
  switch (loc_id.kind()) {
    case LocId::Kind::None: {
      out << "LocId(<none>)";
      break;
    }

    case LocId::Kind::ImportIRInstId: {
      auto import_ir_id =
          file.import_ir_insts().Get(loc_id.import_ir_inst_id()).ir_id();
      const auto* import_file = file.import_irs().Get(import_ir_id).sem_ir;
      out << "LocId(import from ";
      if (import_file == nullptr) {
        out << "unknown file";
      } else {
        out << "\"" << FormatEscaped(import_file->filename()) << "\"";
      }
      out << ")";
      break;
    }

    case LocId::Kind::NodeId: {
      auto token = file.parse_tree().node_token(loc_id.node_id());
      auto line = file.parse_tree().tokens().GetLineNumber(token);
      auto col = file.parse_tree().tokens().GetColumnNumber(token);
      const char* implicit = loc_id.is_desugared() ? " implicit" : "";
      out << "LocId(" << FormatEscaped(file.filename()) << ":" << line << ":"
          << col << implicit << ")";
      break;
    }

    case LocId::Kind::InstId:
      CARBON_FATAL("unexpected LocId kind");
  }
  return out.TakeStr();
}

static auto DumpConstantSummary(const File& file, ConstantId const_id)
    -> std::string {
  RawStringOstream out;
  out << const_id;
  if (!const_id.has_value()) {
    return out.TakeStr();
  }
  if (const_id.is_symbolic()) {
    out << ": " << file.constant_values().GetSymbolicConstant(const_id);
  } else if (const_id.is_concrete()) {
    out << ": "
        << file.insts().GetWithAttachedType(
               file.constant_values().GetInstId(const_id));
  }
  return out.TakeStr();
}

static auto DumpTypeSummary(const File& file, TypeId type_id) -> std::string {
  RawStringOstream out;
  out << type_id;
  if (!type_id.has_value()) {
    return out.TakeStr();
  }

  InstId inst_id = file.types().GetTypeInstId(type_id);
  out << ": " << StringifyConstantInst(file, inst_id) << "; "
      << file.insts().Get(inst_id);
  auto const_id = file.types().GetConstantId(type_id);
  if (const_id.is_symbolic()) {
    if (file.constant_values().IsAttached(const_id)) {
      out << " (attached symbolic)";
    } else {
      out << " (unattached symbolic)";
    }
  } else {
    out << " (concrete)";
  }
  return out.TakeStr();
}

static auto DumpInstSummary(const File& file, InstId inst_id) -> std::string {
  RawStringOstream out;
  out << inst_id;
  if (inst_id.has_value() && inst_id != InstId::InitTombstone &&
      inst_id != InstId::ImplWitnessTablePlaceholder) {
    out << ": " << file.insts().GetWithAttachedType(inst_id);
  }
  return out.TakeStr();
}

static auto DumpGenericSummary(const File& file, GenericId generic_id)
    -> std::string {
  RawStringOstream out;
  out << generic_id;
  if (!generic_id.has_value()) {
    return out.TakeStr();
  }
  const auto& generic = file.generics().Get(generic_id);
  out << ": " << generic
      << "\ndecl: " << DumpInstSummary(file, generic.decl_id);
  return out.TakeStr();
}

static auto DumpInterfaceSummary(const File& file, InterfaceId interface_id)
    -> std::string {
  RawStringOstream out;
  out << interface_id;
  if (interface_id.has_value()) {
    const auto& interface = file.interfaces().Get(interface_id);
    out << ": " << interface << DumpNameIfValid(file, interface.name_id);
  }
  return out.TakeStr();
}

static auto DumpNamedConstraintSummary(const File& file,
                                       NamedConstraintId named_constraint_id)
    -> std::string {
  RawStringOstream out;
  out << named_constraint_id;
  if (named_constraint_id.has_value()) {
    const auto& constraint = file.named_constraints().Get(named_constraint_id);
    out << ": " << constraint << DumpNameIfValid(file, constraint.name_id);
  }
  return out.TakeStr();
}

static auto DumpSpecificSummary(const File& file, SpecificId specific_id)
    -> std::string {
  RawStringOstream out;
  out << specific_id;
  if (specific_id.has_value()) {
    out << ": " << file.specifics().Get(specific_id);
  }
  return out.TakeStr();
}

static auto DumpRequireImplsSummary(const File& file,
                                    RequireImplsId require_impls_id)
    -> std::string {
  RawStringOstream out;
  out << require_impls_id;
  if (require_impls_id.has_value()) {
    out << ": " << file.require_impls().Get(require_impls_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, RawBundleId bundle_id)
    -> std::string {
  for (auto inst : file.insts().values()) {
    inst.CacheBundleDebugKinds(file.bundles());
  }
  RawStringOstream out;
  Print(out, file.bundles().OutputBundleYaml(bundle_id));
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, ClassId class_id) -> std::string {
  RawStringOstream out;
  out << class_id;
  if (class_id.has_value()) {
    const auto& class_obj = file.classes().Get(class_id);
    out << ": " << class_obj << DumpNameIfValid(file, class_obj.name_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, ConstantId const_id)
    -> std::string {
  RawStringOstream out;
  out << const_id;
  if (!const_id.has_value()) {
    return out.TakeStr();
  }
  if (const_id.is_symbolic()) {
    const auto& symbolic = file.constant_values().GetSymbolicConstant(const_id);
    out << ": " << symbolic << '\n'
        << DumpInstSummary(file, symbolic.inst_id) << '\n'
        << DumpGenericSummary(file, symbolic.generic_id);
  } else if (const_id.is_concrete()) {
    out << ": "
        << DumpInstSummary(file, file.constant_values().GetInstId(const_id));
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, EntityNameId entity_name_id)
    -> std::string {
  RawStringOstream out;
  out << entity_name_id;
  if (entity_name_id.has_value()) {
    auto entity_name = file.entity_names().Get(entity_name_id);
    out << ": " << entity_name << DumpNameIfValid(file, entity_name.name_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, FacetTypeId facet_type_id)
    -> std::string {
  RawStringOstream out;
  out << facet_type_id;
  if (!facet_type_id.has_value()) {
    return out.TakeStr();
  }

  const auto& facet_type = file.facet_types().Get(facet_type_id);
  out << ": " << facet_type;
  for (auto impls : facet_type.extend_constraints) {
    out << "\n  - " << DumpInterfaceSummary(file, impls.interface_id);
    if (impls.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, impls.specific_id);
    }
    out << " (extend)";
  }
  for (auto impls : facet_type.self_impls_constraints) {
    out << "\n  - " << DumpInterfaceSummary(file, impls.interface_id);
    if (impls.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, impls.specific_id);
    }
  }
  for (auto impls : facet_type.extend_named_constraints) {
    out << "\n  - "
        << DumpNamedConstraintSummary(file, impls.named_constraint_id);
    if (impls.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, impls.specific_id);
    }
    out << " (extend)";
  }
  for (auto impls : facet_type.self_impls_named_constraints) {
    out << "\n  - "
        << DumpNamedConstraintSummary(file, impls.named_constraint_id);
    if (impls.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, impls.specific_id);
    }
  }
  for (auto rewrite : facet_type.rewrite_constraints) {
    out << "\n"
        << "  - " << DumpInstSummary(file, rewrite.lhs_id) << "\n"
        << "  - " << DumpInstSummary(file, rewrite.rhs_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, FieldId field_id) -> std::string {
  RawStringOstream out;
  out << field_id;
  if (field_id.has_value()) {
    const auto& field = file.fields().Get(field_id);
    out << ": " << field;
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, FunctionId function_id)
    -> std::string {
  RawStringOstream out;
  out << function_id;
  if (function_id.has_value()) {
    const auto& function = file.functions().Get(function_id);
    out << ": " << function << DumpNameIfValid(file, function.name_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, GenericId generic_id)
    -> std::string {
  RawStringOstream out;
  out << DumpGenericSummary(file, generic_id);
  if (!generic_id.has_value()) {
    return out.TakeStr();
  }
  const auto& generic = file.generics().Get(generic_id);
  out << "\nbindings block: " << Dump(file, generic.bindings_id);
  if (generic.decl_block_id.has_value()) {
    out << "\ngeneric decl block: " << Dump(file, generic.decl_block_id);
  }
  if (generic.definition_block_id.has_value()) {
    out << "\ngeneric definition block: "
        << Dump(file, generic.definition_block_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           IdentifiedFacetTypeId identified_facet_type_id)
    -> std::string {
  RawStringOstream out;
  out << identified_facet_type_id;
  if (!identified_facet_type_id.has_value()) {
    return out.TakeStr();
  }

  const auto& identified_facet_type =
      file.identified_facet_types().Get(identified_facet_type_id);
  for (auto [i, req_impl] :
       llvm::enumerate(identified_facet_type.required_impls())) {
    auto [self, req_interface] = req_impl;
    out << "\n  - self: " << DumpConstantSummary(file, self);
    out << "\n    impls: "
        << DumpInterfaceSummary(file, req_interface.interface_id);
    if (req_interface.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, req_interface.specific_id);
    }
    if (req_interface == identified_facet_type.impl_as_target_interface()) {
      out << "\n     - `impl as` target";
    }
  }
  if (!identified_facet_type.is_valid_impl_as_target()) {
    out << "\n  - " << identified_facet_type.num_interfaces_to_impl()
        << " possible `impl as` targets";
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, ImplId impl_id) -> std::string {
  RawStringOstream out;
  out << impl_id;
  if (!impl_id.has_value()) {
    return out.TakeStr();
  }
  const auto& impl = file.impls().Get(impl_id);
  out << ": " << impl << '\n'
      << "  - interface_id: "
      << DumpInterfaceSummary(file, impl.interface.interface_id) << '\n'
      << "  - specific_id: "
      << DumpSpecificSummary(file, impl.interface.specific_id);
  if (impl.interface.specific_id.has_value()) {
    auto inst_block_id =
        file.specifics().Get(impl.interface.specific_id).args_id;
    out << '\n' << Dump(file, inst_block_id);
  }
  out << "\n  - witness loc: " << DumpLocSummary(file, LocId(impl.witness_id));
  return out.TakeStr();
}

static auto DumpInstCommonDetails(const File& file, const Inst& inst,
                                  RawStringOstream& out) -> void {
  if (inst.arg0_and_kind().kind() == IdKind::For<EntityNameId>) {
    auto entity_name_id = EntityNameId(inst.arg0());
    out << "\n  - name:"
        << DumpNameIfValid(file,
                           file.entity_names().Get(entity_name_id).name_id);
  }

  if (inst.type_id().has_value()) {
    out << "\n  - type: " << DumpTypeSummary(file, inst.type_id());
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, const Inst& inst) -> std::string {
  RawStringOstream out;
  out << inst;
  DumpInstCommonDetails(file, inst, out);
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, InstBlockId inst_block_id)
    -> std::string {
  RawStringOstream out;
  out << inst_block_id;
  if (inst_block_id.has_value()) {
    out << ":";
    auto inst_block = file.inst_blocks().GetOrEmpty(inst_block_id);
    for (auto inst_id : inst_block) {
      out << "\n  - " << DumpInstSummary(file, inst_id);
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, InstId inst_id) -> std::string {
  RawStringOstream out;
  out << DumpInstSummary(file, inst_id);
  if (!inst_id.has_value()) {
    return out.TakeStr();
  }

  Inst inst = file.insts().GetWithAttachedType(inst_id);
  DumpInstCommonDetails(file, inst, out);
  ConstantId const_id = file.constant_values().GetAttached(inst_id);
  if (const_id.has_value()) {
    InstId const_inst_id = file.constant_values().GetInstId(const_id);
    out << "\n  - value: ";
    if (const_inst_id == inst_id) {
      out << const_id;
    } else {
      out << DumpConstantSummary(file, const_id);
    }
  }
  out << "\n  - loc: " << DumpLocSummary(file, LocId(inst_id));
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, InterfaceId interface_id)
    -> std::string {
  RawStringOstream out;
  out << DumpInterfaceSummary(file, interface_id);
  if (interface_id.has_value()) {
    const auto& interface = file.interfaces().Get(interface_id);
    out << "\n  - complete: " << (interface.is_complete() ? "true" : "false");
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, LocId loc_id) -> std::string {
  return DumpLocSummary(file, loc_id);
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameId name_id) -> std::string {
  RawStringOstream out;
  out << name_id << DumpNameIfValid(file, name_id);
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, const NameScope& name_scope)
    -> std::string {
  RawStringOstream out;
  out << name_scope;
  if (name_scope.inst_id().has_value()) {
    out << ": " << DumpInstSummary(file, name_scope.inst_id());
  }
  out << "\n  - name: " << name_scope.name_id()
      << DumpNameIfValid(file, name_scope.name_id());
  for (const auto& entry : name_scope.entries()) {
    out << "\n  - " << entry.name_id << DumpNameIfValid(file, entry.name_id)
        << ": ";
    if (entry.result.is_poisoned()) {
      out << "<poisoned>";
    } else if (entry.result.is_found()) {
      switch (entry.result.access_kind()) {
        case AccessKind::Public:
          out << "public ";
          break;
        case AccessKind::Protected:
          out << "protected ";
          break;
        case AccessKind::Private:
          out << "private ";
          break;
        case AccessKind::Hidden:
          out << "hidden ";
          break;
      }
      out << DumpInstSummary(file, entry.result.target_inst_id());
    } else {
      out << "<not-found>";
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameScopeId name_scope_id)
    -> std::string {
  RawStringOstream out;
  out << name_scope_id;
  if (name_scope_id.has_value()) {
    out << ": " << Dump(file, file.name_scopes().Get(name_scope_id));
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           NamedConstraintId named_constraint_id)
    -> std::string {
  RawStringOstream out;
  out << DumpNamedConstraintSummary(file, named_constraint_id);
  if (named_constraint_id.has_value()) {
    const auto& constraint = file.named_constraints().Get(named_constraint_id);
    out << "\n  - complete: " << (constraint.is_complete() ? "true" : "false");
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           RequireImplsBlockId require_impls_block_id)
    -> std::string {
  RawStringOstream out;
  out << require_impls_block_id;
  if (require_impls_block_id.has_value()) {
    for (auto require_id :
         file.require_impls_blocks().Get(require_impls_block_id)) {
      out << "\n  - " << DumpRequireImplsSummary(file, require_id);
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, RequireImplsId require_impls_id)
    -> std::string {
  return DumpRequireImplsSummary(file, require_impls_id);
}

static auto DumpEvalBlock(const File& file, llvm::StringRef region_name,
                          InstBlockId block_id, bool has_error) -> std::string {
  RawStringOstream out;
  if (block_id.has_value()) {
    out << "\nspecific " << region_name << " block"
        << (has_error ? " (has error)" : "") << ": " << Dump(file, block_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, SpecificId specific_id)
    -> std::string {
  RawStringOstream out;
  out << DumpSpecificSummary(file, specific_id);
  if (specific_id.has_value()) {
    const auto& specific = file.specifics().Get(specific_id);
    out << '\n'
        << Dump(file, specific.args_id) << '\n'
        << DumpGenericSummary(file, specific.generic_id)
        << DumpEvalBlock(file, "decl", specific.decl_block_id,
                         specific.decl_block_has_error)
        << DumpEvalBlock(file, "definition", specific.definition_block_id,
                         specific.definition_block_has_error);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           SpecificInterfaceId specific_interface_id)
    -> std::string {
  RawStringOstream out;
  const auto& interface = file.specific_interfaces().Get(specific_interface_id);
  out << specific_interface_id << "\n"
      << "  - interface: " << DumpInterfaceSummary(file, interface.interface_id)
      << "\n"
      << "  - specific_id: "
      << DumpSpecificSummary(file, interface.specific_id);
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           StructTypeFieldsId struct_type_fields_id)
    -> std::string {
  RawStringOstream out;
  out << struct_type_fields_id;
  if (struct_type_fields_id.has_value()) {
    out << ":";
    auto block = file.struct_type_fields().Get(struct_type_fields_id);
    for (auto field : block) {
      out << "\n  - " << field << DumpNameIfValid(file, field.name_id);
      if (field.type_inst_id.has_value()) {
        out << ": " << StringifyConstantInst(file, field.type_inst_id);
      }
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, TypeId type_id) -> std::string {
  return DumpTypeSummary(file, type_id);
}

// Functions that can be used instead of the corresponding constructor, which is
// unavailable during debugging.
LLVM_DUMP_METHOD static auto MakeBundleId(int id) -> RawBundleId {
  return RawBundleId(id);
}
LLVM_DUMP_METHOD static auto MakeClassId(int id) -> ClassId {
  return ClassId(id);
}
LLVM_DUMP_METHOD static auto MakeConstantId(int id) -> ConstantId {
  return ConstantId(id);
}
LLVM_DUMP_METHOD auto MakeSymbolicConstantId(int id) -> ConstantId {
  return ConstantId::ForSymbolicConstantId(ConstantId::SymbolicId(id));
}
LLVM_DUMP_METHOD static auto MakeEntityNameId(int id) -> EntityNameId {
  return EntityNameId(id);
}
LLVM_DUMP_METHOD static auto MakeFacetTypeId(int id) -> FacetTypeId {
  return FacetTypeId(id);
}
LLVM_DUMP_METHOD static auto MakeFunctionId(int id) -> FunctionId {
  return FunctionId(id);
}
LLVM_DUMP_METHOD static auto MakeGenericId(int id) -> GenericId {
  return GenericId(id);
}
LLVM_DUMP_METHOD static auto MakeImplId(int id) -> ImplId { return ImplId(id); }
LLVM_DUMP_METHOD static auto MakeInstBlockId(int id) -> InstBlockId {
  return InstBlockId(id);
}
LLVM_DUMP_METHOD static auto MakeInstId(int id) -> InstId { return InstId(id); }
LLVM_DUMP_METHOD static auto MakeInterfaceId(int id) -> InterfaceId {
  return InterfaceId(id);
}
LLVM_DUMP_METHOD static auto MakeNameId(int id) -> NameId { return NameId(id); }
LLVM_DUMP_METHOD static auto MakeNameScopeId(int id) -> NameScopeId {
  return NameScopeId(id);
}
LLVM_DUMP_METHOD static auto MakeIdentifiedFacetTypeId(int id)
    -> IdentifiedFacetTypeId {
  return IdentifiedFacetTypeId(id);
}
LLVM_DUMP_METHOD static auto MakeNamedConstraintId(int id)
    -> NamedConstraintId {
  return NamedConstraintId(id);
}
LLVM_DUMP_METHOD static auto MakeRequireImplsBlockId(int id)
    -> RequireImplsBlockId {
  return RequireImplsBlockId(id);
}
LLVM_DUMP_METHOD static auto MakeRequireImplsId(int id) -> RequireImplsId {
  return RequireImplsId(id);
}
LLVM_DUMP_METHOD static auto MakeSpecificId(int id) -> SpecificId {
  return SpecificId(id);
}
LLVM_DUMP_METHOD static auto MakeSpecificInterfaceId(int id)
    -> SpecificInterfaceId {
  return SpecificInterfaceId(id);
}
LLVM_DUMP_METHOD static auto MakeStructTypeFieldsId(int id)
    -> StructTypeFieldsId {
  return StructTypeFieldsId(id);
}
LLVM_DUMP_METHOD static auto MakeTypeId(int id) -> TypeId { return TypeId(id); }

}  // namespace Carbon::SemIR

#endif  // NDEBUG
