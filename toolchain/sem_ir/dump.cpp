// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef NDEBUG

#include "toolchain/sem_ir/dump.h"

#include "common/ostream.h"
#include "toolchain/sem_ir/stringify_type.h"

namespace Carbon::SemIR {

static auto DumpNameIfValid(const File& file, NameId name_id) -> void {
  if (name_id.has_value()) {
    llvm::errs() << " `" << file.names().GetFormatted(name_id) << "`";
  }
}

static auto DumpNoNewline(const File& file, ConstantId const_id) -> void {
  llvm::errs() << const_id;
  if (const_id.is_symbolic()) {
    llvm::errs() << ": "
                 << file.constant_values().GetSymbolicConstant(const_id);
  } else if (const_id.is_concrete()) {
    llvm::errs() << ": "
                 << file.insts().Get(
                        file.constant_values().GetInstId(const_id));
  }
}

static auto DumpNoNewline(const File& file, InstId inst_id) -> void {
  llvm::errs() << inst_id;
  if (inst_id.has_value()) {
    llvm::errs() << ": " << file.insts().Get(inst_id);
  }
}

static auto DumpNoNewline(const File& file, InterfaceId interface_id) -> void {
  llvm::errs() << interface_id;
  if (interface_id.has_value()) {
    const auto& interface = file.interfaces().Get(interface_id);
    llvm::errs() << ": " << interface;
    DumpNameIfValid(file, interface.name_id);
  }
}

static auto DumpNoNewline(const File& file, SpecificId specific_id) -> void {
  llvm::errs() << specific_id;
  if (specific_id.has_value()) {
    llvm::errs() << ": " << file.specifics().Get(specific_id);
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, ClassId class_id) -> void {
  llvm::errs() << class_id;
  if (class_id.has_value()) {
    const auto& class_obj = file.classes().Get(class_id);
    llvm::errs() << ": " << class_obj;
    DumpNameIfValid(file, class_obj.name_id);
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, ConstantId const_id) -> void {
  DumpNoNewline(file, const_id);
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, EntityNameId entity_name_id)
    -> void {
  llvm::errs() << entity_name_id;
  if (entity_name_id.has_value()) {
    auto entity_name = file.entity_names().Get(entity_name_id);
    llvm::errs() << ": " << entity_name;
    DumpNameIfValid(file, entity_name.name_id);
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, FacetTypeId facet_type_id)
    -> void {
  llvm::errs() << facet_type_id;
  if (facet_type_id.has_value()) {
    const auto& facet_type = file.facet_types().Get(facet_type_id);
    llvm::errs() << ": " << facet_type << '\n';
    for (auto impls : facet_type.impls_constraints) {
      llvm::errs() << "  - ";
      DumpNoNewline(file, impls.interface_id);
      if (impls.specific_id.has_value()) {
        llvm::errs() << "; ";
        DumpNoNewline(file, impls.specific_id);
      }
      llvm::errs() << '\n';
    }
    for (auto rewrite : facet_type.rewrite_constraints) {
      llvm::errs() << "  - ";
      Dump(file, rewrite.lhs_const_id);
      llvm::errs() << "  - ";
      Dump(file, rewrite.rhs_const_id);
    }
    if (auto complete_id = file.complete_facet_types().TryGetId(facet_type_id);
        complete_id.has_value()) {
      llvm::errs() << "complete: ";
      Dump(file, complete_id);
    }
  } else {
    llvm::errs() << '\n';
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, FunctionId function_id) -> void {
  llvm::errs() << function_id;
  if (function_id.has_value()) {
    const auto& function = file.functions().Get(function_id);
    llvm::errs() << ": " << function;
    DumpNameIfValid(file, function.name_id);
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, GenericId generic_id) -> void {
  llvm::errs() << generic_id;
  if (generic_id.has_value()) {
    llvm::errs() << ": " << file.generics().Get(generic_id);
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, ImplId impl_id) -> void {
  llvm::errs() << impl_id;
  if (impl_id.has_value()) {
    llvm::errs() << ": " << file.impls().Get(impl_id);
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, InstBlockId inst_block_id)
    -> void {
  llvm::errs() << inst_block_id;
  if (inst_block_id.has_value()) {
    llvm::errs() << ":";
    auto inst_block = file.inst_blocks().Get(inst_block_id);
    for (auto inst_id : inst_block) {
      llvm::errs() << "\n  - ";
      DumpNoNewline(file, inst_id);
    }
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, InstId inst_id) -> void {
  DumpNoNewline(file, inst_id);
  llvm::errs() << '\n';
  if (inst_id.has_value()) {
    Inst inst = file.insts().Get(inst_id);
    if (inst.type_id().has_value()) {
      llvm::errs() << "  - type ";
      Dump(file, inst.type_id());
    }
    ConstantId const_id = file.constant_values().Get(inst_id);
    if (const_id.has_value()) {
      InstId const_inst_id = file.constant_values().GetInstId(const_id);
      llvm::errs() << "  - value ";
      if (const_inst_id == inst_id) {
        llvm::errs() << const_id << '\n';
      } else {
        Dump(file, const_id);
      }
    }
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, InterfaceId interface_id) -> void {
  DumpNoNewline(file, interface_id);
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameId name_id) -> void {
  llvm::errs() << name_id;
  DumpNameIfValid(file, name_id);
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, NameScopeId name_scope_id)
    -> void {
  llvm::errs() << name_scope_id;
  if (name_scope_id.has_value()) {
    const auto& name_scope = file.name_scopes().Get(name_scope_id);
    llvm::errs() << ": " << name_scope;
    if (name_scope.inst_id().has_value()) {
      llvm::errs() << " " << file.insts().Get(name_scope.inst_id());
    }
    DumpNameIfValid(file, name_scope.name_id());
    llvm::errs() << '\n';
    for (const auto& entry : name_scope.entries()) {
      llvm::errs() << "  - " << entry.name_id;
      DumpNameIfValid(file, entry.name_id);
      llvm::errs() << ": ";
      if (entry.result.is_poisoned()) {
        llvm::errs() << "<poisoned>\n";
      } else if (entry.result.is_found()) {
        switch (entry.result.access_kind()) {
          case AccessKind::Public:
            llvm::errs() << "public ";
            break;
          case AccessKind::Protected:
            llvm::errs() << "protected ";
            break;
          case AccessKind::Private:
            llvm::errs() << "private ";
            break;
        }
        DumpNoNewline(file, entry.result.target_inst_id());
        llvm::errs() << "\n";
      } else {
        llvm::errs() << "<not-found>\n";
      }
    }
  } else {
    llvm::errs() << '\n';
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           CompleteFacetTypeId complete_facet_type_id) -> void {
  llvm::errs() << complete_facet_type_id << "\n";
  if (complete_facet_type_id.has_value()) {
    const auto& complete_facet_type =
        file.complete_facet_types().Get(complete_facet_type_id);
    for (auto [i, req_interface] :
         llvm::enumerate(complete_facet_type.required_interfaces)) {
      llvm::errs() << "  - ";
      DumpNoNewline(file, req_interface.interface_id);
      if (req_interface.specific_id.has_value()) {
        llvm::errs() << "; ";
        DumpNoNewline(file, req_interface.specific_id);
      }
      if (static_cast<int>(i) < complete_facet_type.num_to_impl) {
        llvm::errs() << " (to impl)";
      }
      llvm::errs() << '\n';
    }
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, SpecificId specific_id) -> void {
  DumpNoNewline(file, specific_id);
  llvm::errs() << '\n';
  Dump(file, file.specifics().Get(specific_id).args_id);
}

LLVM_DUMP_METHOD auto Dump(const File& file,
                           StructTypeFieldsId struct_type_fields_id) -> void {
  llvm::errs() << struct_type_fields_id;
  if (struct_type_fields_id.has_value()) {
    llvm::errs() << ":";
    auto block = file.struct_type_fields().Get(struct_type_fields_id);
    for (auto field : block) {
      llvm::errs() << "\n  - " << field;
      DumpNameIfValid(file, field.name_id);
      if (field.type_id.has_value()) {
        InstId inst_id =
            file.constant_values().GetInstId(field.type_id.AsConstantId());
        llvm::errs() << ": " << StringifyTypeExpr(file, inst_id);
      }
    }
  }
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD auto Dump(const File& file, TypeBlockId type_block_id)
    -> void {
  llvm::errs() << type_block_id;
  if (type_block_id.has_value()) {
    llvm::errs() << ":\n";
    auto type_block = file.type_blocks().Get(type_block_id);
    for (auto type_id : type_block) {
      llvm::errs() << "  - ";
      Dump(file, type_id);
    }
  } else {
    llvm::errs() << '\n';
  }
}

LLVM_DUMP_METHOD auto Dump(const File& file, TypeId type_id) -> void {
  llvm::errs() << type_id;
  if (type_id.has_value()) {
    InstId inst_id = file.constant_values().GetInstId(type_id.AsConstantId());
    llvm::errs() << ": " << StringifyTypeExpr(file, inst_id) << "; "
                 << file.insts().Get(inst_id);
  }
  llvm::errs() << '\n';
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
