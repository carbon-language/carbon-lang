// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/sem_ir/dump.h"

#include "common/raw_string_ostream.h"
#include "toolchain/sem_ir/stringify_type.h"

namespace Carbon::SemIR {

static auto DumpNameIfValid(const File& file, NameId name_id) -> std::string {
  RawStringOstream out;
  if (name_id.has_value()) {
    out << " `" << file.names().GetFormatted(name_id) << "`";
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
    out << ": " << file.insts().Get(file.constant_values().GetInstId(const_id));
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
  out << ": " << generic << "\ndecl: " << Dump(file, generic.decl_id);
  return out.TakeStr();
}

static auto DumpInstSummary(const File& file, InstId inst_id) -> std::string {
  RawStringOstream out;
  out << inst_id;
  if (inst_id.has_value()) {
    out << ": " << file.insts().Get(inst_id);
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
        << Dump(file, symbolic.inst_id) << '\n'
        << DumpGenericSummary(file, symbolic.generic_id);
  } else if (const_id.is_concrete()) {
    out << ": " << Dump(file, file.constant_values().GetInstId(const_id));
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
  for (auto impls : facet_type.impls_constraints) {
    out << "\n  - " << Dump(file, impls.interface_id);
    if (impls.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, impls.specific_id);
    }
  }
  for (auto rewrite : facet_type.rewrite_constraints) {
    out << "\n"
        << "  - " << DumpConstantSummary(file, rewrite.lhs_const_id) << "\n"
        << "  - " << DumpConstantSummary(file, rewrite.rhs_const_id);
  }
  if (auto complete_id = file.complete_facet_types().TryGetId(facet_type_id);
      complete_id.has_value()) {
    out << "\ncomplete: " << Dump(file, complete_id);
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

LLVM_DUMP_METHOD auto Dump(const File& file, ImplId impl_id) -> std::string {
  RawStringOstream out;
  out << impl_id;
  if (!impl_id.has_value()) {
    return out.TakeStr();
  }
  const auto& impl = file.impls().Get(impl_id);
  out << ": " << impl << '\n'
      << "  - interface_id: " << Dump(file, impl.interface.interface_id) << '\n'
      << "  - specific_id: "
      << DumpSpecificSummary(file, impl.interface.specific_id);
  if (impl.interface.specific_id.has_value()) {
    auto inst_block_id =
        file.specifics().Get(impl.interface.specific_id).args_id;
    out << '\n' << Dump(file, inst_block_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, InstBlockId inst_block_id)
    -> std::string {
  RawStringOstream out;
  out << inst_block_id;
  if (inst_block_id.has_value()) {
    out << ":";
    auto inst_block = file.inst_blocks().Get(inst_block_id);
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

  Inst inst = file.insts().Get(inst_id);
  if (inst.type_id().has_value()) {
    out << "\n  - type: " << Dump(file, inst.type_id());
  }
  ConstantId const_id = file.constant_values().Get(inst_id);
  if (const_id.has_value()) {
    InstId const_inst_id = file.constant_values().GetInstId(const_id);
    out << "\n  - value: ";
    if (const_inst_id == inst_id) {
      out << const_id;
    } else {
      out << DumpConstantSummary(file, const_id);
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, InterfaceId interface_id)
    -> std::string {
  RawStringOstream out;
  out << interface_id;
  if (interface_id.has_value()) {
    const auto& interface = file.interfaces().Get(interface_id);
    out << ": " << interface << DumpNameIfValid(file, interface.name_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameId name_id) -> std::string {
  RawStringOstream out;
  out << name_id << DumpNameIfValid(file, name_id);
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameScopeId name_scope_id)
    -> std::string {
  RawStringOstream out;
  out << name_scope_id;
  if (!name_scope_id.has_value()) {
    return out.TakeStr();
  }

  const auto& name_scope = file.name_scopes().Get(name_scope_id);
  out << ": " << name_scope;
  if (name_scope.inst_id().has_value()) {
    out << " " << file.insts().Get(name_scope.inst_id());
  }
  out << DumpNameIfValid(file, name_scope.name_id());
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
      }
      out << DumpInstSummary(file, entry.result.target_inst_id());
    } else {
      out << "<not-found>";
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           CompleteFacetTypeId complete_facet_type_id)
    -> std::string {
  RawStringOstream out;
  out << complete_facet_type_id;
  if (!complete_facet_type_id.has_value()) {
    return out.TakeStr();
  }

  const auto& complete_facet_type =
      file.complete_facet_types().Get(complete_facet_type_id);
  for (auto [i, req_interface] :
       llvm::enumerate(complete_facet_type.required_interfaces)) {
    out << "\n  - " << Dump(file, req_interface.interface_id);
    if (req_interface.specific_id.has_value()) {
      out << "; " << DumpSpecificSummary(file, req_interface.specific_id);
    }
    if (static_cast<int>(i) < complete_facet_type.num_to_impl) {
      out << " (to impl)";
    }
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
        << DumpGenericSummary(file, specific.generic_id);
    if (specific.decl_block_id.has_value()) {
      out << "\nspecific decl block: " << Dump(file, specific.decl_block_id);
    }
    if (specific.definition_block_id.has_value()) {
      out << "\nspecific definition block: "
          << Dump(file, specific.definition_block_id);
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           SpecificInterfaceId specific_interface_id)
    -> std::string {
  RawStringOstream out;
  const auto& interface = file.specific_interfaces().Get(specific_interface_id);
  out << specific_interface_id << "\n"
      << "  - interface: " << Dump(file, interface.interface_id) << "\n"
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
      if (field.type_id.has_value()) {
        InstId inst_id =
            file.constant_values().GetInstId(field.type_id.AsConstantId());
        out << ": " << StringifyTypeExpr(file, inst_id);
      }
    }
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, TypeBlockId type_block_id)
    -> std::string {
  RawStringOstream out;
  out << type_block_id;
  if (!type_block_id.has_value()) {
    return out.TakeStr();
  }

  out << ":";
  auto type_block = file.type_blocks().Get(type_block_id);
  for (auto type_id : type_block) {
    out << "\n  - " << Dump(file, type_id);
  }
  return out.TakeStr();
}

LLVM_DUMP_METHOD auto Dump(const File& file, TypeId type_id) -> std::string {
  RawStringOstream out;
  out << type_id;
  if (!type_id.has_value()) {
    return out.TakeStr();
  }

  InstId inst_id = file.constant_values().GetInstId(type_id.AsConstantId());
  out << ": " << StringifyTypeExpr(file, inst_id) << "; "
      << file.insts().Get(inst_id);
  return out.TakeStr();
}

// Functions that can be used instead of the corresponding constructor, which is
// unavailable during debugging.
LLVM_DUMP_METHOD static auto MakeClassId(int id) -> ClassId {
  return ClassId(id);
}
LLVM_DUMP_METHOD static auto MakeConstantId(int id) -> ConstantId {
  return ConstantId(id);
}
LLVM_DUMP_METHOD static auto MakeSymbolicConstantId(int id) -> ConstantId {
  return ConstantId::ForSymbolicConstantIndex(id);
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
LLVM_DUMP_METHOD static auto MakeCompleteFacetTypeId(int id)
    -> CompleteFacetTypeId {
  return CompleteFacetTypeId(id);
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
LLVM_DUMP_METHOD static auto MakeTypeBlockId(int id) -> TypeBlockId {
  return TypeBlockId(id);
}
LLVM_DUMP_METHOD static auto MakeTypeId(int id) -> TypeId { return TypeId(id); }

}  // namespace Carbon::SemIR

#endif  // NDEBUG
