// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/mangler.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entry_point.h"

namespace Carbon::Lower {

auto Mangler::MangleInverseQualifiedNameScope(bool first_name_component,
                                              llvm::raw_ostream& os,
                                              SemIR::NameScopeId name_scope_id)
    -> void {
  if (!name_scope_id.is_valid() ||
      name_scope_id == SemIR::NameScopeId::Package) {
    return;
  }
  struct NameEntry {
    SemIR::NameScopeId name_id;
    char prefix = 0;
  };
  llvm::SmallVector<NameEntry> names_to_render = {
      {name_scope_id, first_name_component ? '\0' : '.'}};
  while (!names_to_render.empty()) {
    auto [name_scope_id, prefix] = names_to_render.back();
    names_to_render.pop_back();
    const auto& name_scope = sem_ir().name_scopes().Get(name_scope_id);
    if (prefix) {
      os << prefix;
    }
    CARBON_KIND_SWITCH(sem_ir().insts().Get(name_scope.inst_id)) {
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        const auto& impl = sem_ir().impls().Get(impl_decl.impl_id);
        CARBON_KIND_SWITCH(sem_ir().types().GetAsInst(impl.self_id)) {
          case CARBON_KIND(SemIR::ClassType class_type): {
            MangleInverseQualifiedNameScope(
                true, os, sem_ir().classes().Get(class_type.class_id).scope_id);
            break;
          }
          case CARBON_KIND(SemIR::BuiltinInst builtin_inst): {
            os << builtin_inst.builtin_inst_kind.label();
            break;
          }
          default:
            CARBON_FATAL() << "Attempting to mangle unsupported SemIR.";
            break;
        }
        os << ':';
        // FIXME: Qualify the interface name.
        auto opt_interface_constraint =
            types().GetAs<SemIR::InterfaceType>(impl.constraint_id);
        os << names().GetFormatted(
            sem_ir()
                .interfaces()
                .Get(opt_interface_constraint.interface_id)
                .name_id);
        return;
      }
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        os << names().GetFormatted(
            sem_ir().classes().Get(class_decl.class_id).name_id);
        break;
      }
      case SemIR::Namespace::Kind: {
        auto name = names().GetAsStringIfIdentifier(name_scope.name_id);
        CARBON_CHECK(name) << "Unexpected special name for namespace: "
                           << name_scope.name_id;
        os << *name;
        break;
      }
      default:
        CARBON_FATAL() << "Attempting to mangle unsupported SemIR.";
        break;
    }
    auto next_name_scope_id = name_scope.parent_scope_id;
    if (next_name_scope_id.is_valid() &&
        next_name_scope_id != SemIR::NameScopeId::Package) {
      names_to_render.push_back({next_name_scope_id, '.'});
    }
  }
}

auto Mangler::Mangle(SemIR::FunctionId function_id) -> std::string {
  const auto& function = sem_ir().functions().Get(function_id);
  if (SemIR::IsEntryPoint(sem_ir(), function_id)) {
    // TODO: Add an implicit `return 0` if `Run` doesn't return `i32`.
    return "main";
  }
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "_C";

  auto name = names().GetAsStringIfIdentifier(function.name_id);
  CARBON_CHECK(name) << "Unexpected special name for function: "
                     << function.name_id;
  os << name->str();

  MangleInverseQualifiedNameScope(false, os, function.parent_scope_id);

  return os.str();
}

}  // namespace Carbon::Lower
