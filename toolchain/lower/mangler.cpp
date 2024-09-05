// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/mangler.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entry_point.h"

namespace Carbon::Lower {

auto Mangler::MangleInverseQualifiedNameScope(llvm::raw_ostream& os,
                                              SemIR::NameScopeId name_scope_id)
    -> void {
  // Maintain a stack of names for delayed rendering of interface impls.
  struct NameEntry {
    SemIR::NameScopeId name_scope_id;
    char prefix;
  };
  llvm::SmallVector<NameEntry> names_to_render;
  auto add_scope = [&](SemIR::NameScopeId name_scope_id, char prefix) {
    if (name_scope_id.is_valid() &&
        name_scope_id != SemIR::NameScopeId::Package) {
      names_to_render.push_back({name_scope_id, prefix});
    }
  };
  add_scope(name_scope_id, '.');
  while (!names_to_render.empty()) {
    auto [name_scope_id, prefix] = names_to_render.back();
    names_to_render.pop_back();
    const auto& name_scope = sem_ir().name_scopes().Get(name_scope_id);
    if (prefix) {
      os << prefix;
    }
    add_scope(name_scope.parent_scope_id, '.');
    CARBON_KIND_SWITCH(sem_ir().insts().Get(name_scope.inst_id)) {
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        const auto& impl = sem_ir().impls().Get(impl_decl.impl_id);

        auto interface_type =
            types().GetAs<SemIR::InterfaceType>(impl.constraint_id);
        const auto& interface = sem_ir().interfaces().Get(interface_type.interface_id);
        names_to_render.push_back({interface.scope_id, ':'});

        CARBON_KIND_SWITCH(types().GetAsInst(impl.self_id)) {
          case CARBON_KIND(SemIR::ClassType class_type): {
            auto next_name_scope_id =
                sem_ir().classes().Get(class_type.class_id).scope_id;
            names_to_render.push_back({next_name_scope_id});
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
        break;
      }
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        os << names().GetAsStringIfIdentifier(
            sem_ir().classes().Get(class_decl.class_id).name_id);
        break;
      }
      case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
        os << names().GetAsStringIfIdentifier(
            sem_ir().interfaces().Get(interface_decl.interface_id).name_id);
        break;
      }
      case SemIR::Namespace::Kind: {
        os << names().GetAsStringIfIdentifier(name_scope.name_id);
        break;
      }
      default:
        CARBON_FATAL() << "Attempting to mangle unsupported SemIR.";
        break;
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

  os << names().GetAsStringIfIdentifier(function.name_id);

  MangleInverseQualifiedNameScope(os, function.parent_scope_id);

  return os.str();
}

}  // namespace Carbon::Lower
