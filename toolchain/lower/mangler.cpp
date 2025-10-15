// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/mangler.h"

#include <string>

#include "common/raw_string_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/lower/clang_global_decl.h"
#include "toolchain/sem_ir/clang_decl.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/pattern.h"

namespace Carbon::Lower {

auto Mangler::MangleNameId(llvm::raw_ostream& os, SemIR::NameId name_id)
    -> void {
  CARBON_CHECK(name_id.AsIdentifierId().has_value(),
               "Mangling non-identifier name {0}", name_id);
  os << names().GetAsStringIfIdentifier(name_id);
}

auto Mangler::MangleInverseQualifiedNameScope(llvm::raw_ostream& os,
                                              SemIR::NameScopeId name_scope_id)
    -> void {
  // Maintain a stack of names for delayed rendering of interface impls.
  struct NameEntry {
    SemIR::NameScopeId name_scope_id;

    // The prefix emitted before this name component. If '\0', no prefix will be
    // emitted.
    // - Namespace components are separated by '.'.
    // - The two components of an interface are separated by ':'.
    char prefix;
  };
  llvm::SmallVector<NameEntry> names_to_render;
  names_to_render.push_back({.name_scope_id = name_scope_id, .prefix = '.'});
  while (!names_to_render.empty()) {
    auto [name_scope_id, prefix] = names_to_render.pop_back_val();
    if (prefix) {
      os << prefix;
    }
    if (name_scope_id == SemIR::NameScopeId::Package) {
      auto package_id = sem_ir().package_id();
      if (auto ident_id = package_id.AsIdentifierId(); ident_id.has_value()) {
        os << sem_ir().identifiers().Get(ident_id);
      } else {
        // TODO: Handle name conflicts between special package names and raw
        // identifier package names. Note that any change here will also require
        // a change to namespace mangling for imported packages.
        os << package_id.AsSpecialName();
      }
      continue;
    }
    const auto& name_scope = sem_ir().name_scopes().Get(name_scope_id);
    CARBON_KIND_SWITCH(sem_ir().insts().Get(name_scope.inst_id())) {
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        const auto& impl = sem_ir().impls().Get(impl_decl.impl_id);

        auto facet_type = insts().GetAs<SemIR::FacetType>(
            constant_values().GetConstantInstId(impl.constraint_id));
        const auto& facet_type_info =
            sem_ir().facet_types().Get(facet_type.facet_type_id);
        CARBON_CHECK(facet_type_info.extend_constraints.size() == 1,
                     "Mangling of an impl of something other than a single "
                     "interface is not yet supported.");
        auto interface_type = facet_type_info.extend_constraints.front();
        const auto& interface =
            sem_ir().interfaces().Get(interface_type.interface_id);
        names_to_render.push_back(
            {.name_scope_id = interface.scope_id, .prefix = ':'});

        auto self_const_inst_id =
            constant_values().GetConstantInstId(impl.self_id);
        auto self_inst = insts().Get(self_const_inst_id);
        CARBON_KIND_SWITCH(self_inst) {
          case CARBON_KIND(SemIR::ClassType class_type): {
            auto next_name_scope_id =
                sem_ir().classes().Get(class_type.class_id).scope_id;
            names_to_render.push_back(
                {.name_scope_id = next_name_scope_id, .prefix = '\0'});
            break;
          }
          case SemIR::AutoType::Kind:
          case SemIR::BoolType::Kind:
          case SemIR::BoundMethodType::Kind:
          case SemIR::FloatLiteralType::Kind:
          case SemIR::IntLiteralType::Kind:
          case SemIR::NamespaceType::Kind:
          case SemIR::SpecificFunctionType::Kind:
          case SemIR::TypeType::Kind:
          case SemIR::VtableType::Kind:
          case SemIR::WitnessType::Kind: {
            os << self_inst.kind().ir_name();
            break;
          }
          case CARBON_KIND(SemIR::IntType int_type): {
            os << (int_type.int_kind == SemIR::IntKind::Signed ? "i" : "u")
               << sem_ir().ints().Get(
                      sem_ir()
                          .insts()
                          .GetAs<SemIR::IntValue>(int_type.bit_width_id)
                          .int_id);
            break;
          }
          default: {
            // Fall back to including a fingerprint.
            llvm::write_hex(
                os, fingerprinter_.GetOrCompute(&sem_ir(), self_const_inst_id),
                llvm::HexPrintStyle::Lower, 16);
            break;
          }
        }
        // Skip the tail of the loop that adds the parent name scope to the
        // stack - the scope in which the impl was defined is not part of the
        // mangling, the constraint and interface alone uniquelify identify an
        // impl.
        continue;
      }
      case CARBON_KIND(SemIR::ClassDecl class_decl): {
        MangleNameId(os, sem_ir().classes().Get(class_decl.class_id).name_id);
        break;
      }
      case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
        MangleNameId(
            os, sem_ir().interfaces().Get(interface_decl.interface_id).name_id);
        break;
      }
      case SemIR::Namespace::Kind: {
        os << names().GetIRBaseName(name_scope.name_id());
        break;
      }
      default:
        CARBON_FATAL("Attempting to mangle unsupported SemIR.");
        break;
    }
    if (!name_scope.is_imported_package()) {
      names_to_render.push_back(
          {.name_scope_id = name_scope.parent_scope_id(), .prefix = '.'});
    }
  }
}

auto Mangler::Mangle(SemIR::FunctionId function_id,
                     SemIR::SpecificId specific_id) -> std::string {
  const auto& function = sem_ir().functions().Get(function_id);
  if (SemIR::IsEntryPoint(sem_ir(), function_id)) {
    CARBON_CHECK(!specific_id.has_value(), "entry point should not be generic");
    return "main";
  }
  if (function.clang_decl_id.has_value()) {
    CARBON_CHECK(function.special_function_kind !=
                     SemIR::Function::SpecialFunctionKind::HasCppThunk,
                 "Shouldn't mangle C++ function that uses a thunk");
    const auto& clang_decl = sem_ir().clang_decls().Get(function.clang_decl_id);
    return MangleCppClang(cast<clang::NamedDecl>(clang_decl.key.decl));
  }
  RawStringOstream os;
  os << "_C";

  MangleNameId(os, function.name_id);

  // For a special function, add a marker to disambiguate.
  switch (function.special_function_kind) {
    case SemIR::Function::SpecialFunctionKind::None:
      break;
    case SemIR::Function::SpecialFunctionKind::Builtin:
      CARBON_FATAL("Attempting to mangle declaration of builtin function {0}",
                   function.builtin_function_kind());
    case SemIR::Function::SpecialFunctionKind::Thunk:
      os << ":thunk";
      break;
    case SemIR::Function::SpecialFunctionKind::HasCppThunk:
      CARBON_FATAL("C++ functions should have been handled earlier");
  }

  // TODO: If the function is private, also include the library name as part of
  // the mangling.
  MangleInverseQualifiedNameScope(os, function.parent_scope_id);

  MangleSpecificId(os, specific_id);

  return os.TakeStr();
}

auto Mangler::MangleSpecificId(llvm::raw_ostream& os,
                               SemIR::SpecificId specific_id) -> void {
  // TODO: Add proper support for mangling generic entities. For now we use a
  // fingerprint of the specific arguments, which should be stable across files,
  // but isn't necessarily stable across toolchain changes.
  if (specific_id.has_value()) {
    os << ".";
    llvm::write_hex(
        os,
        fingerprinter_.GetOrCompute(
            &sem_ir(), sem_ir().specifics().Get(specific_id).args_id),
        llvm::HexPrintStyle::Lower, 16);
  }
}

auto Mangler::MangleGlobalVariable(SemIR::InstId pattern_id) -> std::string {
  // Use the name of the first binding in the variable as its mangled name.
  auto var_name_id =
      SemIR::GetFirstBindingNameFromPatternId(sem_ir(), pattern_id);
  if (!var_name_id.has_value()) {
    return std::string();
  }

  SemIR::CppGlobalVarId cpp_global_var_id =
      sem_ir().cpp_global_vars().Lookup({.entity_name_id = var_name_id});
  if (cpp_global_var_id.has_value()) {
    SemIR::ClangDeclId clang_decl_id =
        sem_ir().cpp_global_vars().Get(cpp_global_var_id).clang_decl_id;
    CARBON_CHECK(clang_decl_id.has_value(),
                 "CppGlobalVar should have a clang_decl_id");
    return MangleCppClang(cast<clang::NamedDecl>(
        sem_ir().clang_decls().Get(clang_decl_id).key.decl));
  }

  RawStringOstream os;
  os << "_C";

  auto var_name = sem_ir().entity_names().Get(var_name_id);
  MangleNameId(os, var_name.name_id);
  // TODO: If the variable is private, also include the library name as part of
  // the mangling.
  MangleInverseQualifiedNameScope(os, var_name.parent_scope_id);
  return os.TakeStr();
}

auto Mangler::MangleCppClang(const clang::NamedDecl* decl) -> std::string {
  return file_context_.cpp_code_generator()
      .GetMangledName(CreateGlobalDecl(decl))
      .str();
}

auto Mangler::MangleVTable(const SemIR::Class& class_info,
                           SemIR::SpecificId specific_id) -> std::string {
  RawStringOstream os;
  os << "_C";

  MangleNameId(os, class_info.name_id);
  // TODO: If the class is private, also include the library name as part of the
  // mangling.
  MangleInverseQualifiedNameScope(os, class_info.parent_scope_id);

  os << ".$vtable";

  MangleSpecificId(os, specific_id);

  return os.TakeStr();
}

}  // namespace Carbon::Lower
