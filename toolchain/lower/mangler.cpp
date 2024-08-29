// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/mangler.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entry_point.h"

namespace Carbon::Lower {

auto Mangler::Mangle(SemIR::FunctionId function_id) -> std::string {
  auto& sem_ir = file_context_.sem_ir();
  auto names = sem_ir.names();
  const auto& function = sem_ir.functions().Get(function_id);
  if (SemIR::IsEntryPoint(sem_ir, function_id)) {
    // TODO: Add an implicit `return 0` if `Run` doesn't return `i32`.
    return "main";
  }
  // TODO: Decide on a name mangling scheme.
  auto name = names.GetAsStringIfIdentifier(function.name_id);
  CARBON_CHECK(name) << "Unexpected special name for function: "
                     << function.name_id;
  SemIR::NameScopeId parent_scope_id = function.parent_scope_id;
  std::string result = "_C" + name->str();
  while (parent_scope_id.is_valid()) {
    const auto& parent = sem_ir.name_scopes().Get(parent_scope_id);
    if (parent.inst_id == SemIR::InstId::PackageNamespace) {
      break;
    }
    std::string name_component;
    CARBON_KIND_SWITCH(sem_ir.insts().Get(parent.inst_id)) {
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        name_component = names.GetFormatted(
            sem_ir.classes().Get(class_decl.class_id).name_id);
        break;
      }
      case SemIR::Namespace::Kind: {
        auto name = names.GetAsStringIfIdentifier(parent.name_id);
        if (!name) {
          break;
        }
        CARBON_CHECK(name) << "Unexpected special name for function scope: "
                           << function.name_id;
        name_component = *name;
        break;
      }
      default:
        break;
    }
    if (name_component.empty()) {
      break;
    }
    result += '.';
    result += name_component;
    parent_scope_id = parent.parent_scope_id;
  }
  return result;
}

}  // namespace Carbon::Lower
