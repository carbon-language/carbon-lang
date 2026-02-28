// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/mangler.h"

#include <string>

#include "common/raw_string_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"
#include "toolchain/sem_ir/typed_insts.h"

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
    SemIR::SpecificId specific_id;

    // The prefix emitted before this name component. If '\0', no prefix will be
    // emitted.
    // - Namespace components are separated by '.'.
    // - The two components of an interface are separated by ':'.
    char prefix;
  };
  llvm::SmallVector<NameEntry> names_to_render;
  names_to_render.push_back({.name_scope_id = name_scope_id,
                             .specific_id = SemIR::SpecificId::None,
                             .prefix = '.'});
  while (!names_to_render.empty()) {
    auto [name_scope_id, specific_id, prefix] = names_to_render.pop_back_val();
    if (!name_scope_id.has_value()) {
      // TODO: Include something in the mangling to identify the scope for a
      // function-local class, function, or similar. We may need to number
      // these within the enclosing function, as their name need not be unique.
      os << ":enclosed";
      continue;
    }
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

        auto identified_facet_type_id =
            sem_ir().identified_facet_types().Lookup(
                {.facet_type_id = facet_type.facet_type_id,
                 .self_const_id =
                     sem_ir().constant_values().Get(impl.self_id)});
        CARBON_CHECK(identified_facet_type_id.has_value(),
                     "ImplDecl with unidentified facet type constraint");
        const auto& identified =
            sem_ir().identified_facet_types().Get(identified_facet_type_id);
        auto impl_target = identified.impl_as_target_interface();
        const auto& interface =
            sem_ir().interfaces().Get(impl_target.interface_id);
        names_to_render.push_back(
            // We mangle names in an interface without `Self` in the specific
            // since it would just add noise and `Self` is not part of how you
            // name the entities syntactically.
            {.name_scope_id = interface.scope_without_self_id,
             .specific_id = impl_target.specific_id,
             .prefix = ':'});

        auto self_const_inst_id =
            constant_values().GetConstantInstId(impl.self_id);
        auto self_inst = insts().Get(self_const_inst_id);
        CARBON_KIND_SWITCH(self_inst) {
          case CARBON_KIND(SemIR::ClassType class_type): {
            const auto& class_info =
                sem_ir().classes().Get(class_type.class_id);

            names_to_render.push_back(
                {.name_scope_id = class_info.parent_scope_id,
                 .specific_id = class_type.specific_id,
                 .prefix = '.'});

            MangleUnqualifiedClass(os, class_info, class_type.specific_id);
            break;
          }
          case SemIR::AutoType::Kind:
          case SemIR::BoolType::Kind:
          case SemIR::BoundMethodType::Kind:
          case SemIR::FloatLiteralType::Kind:
          case SemIR::IntLiteralType::Kind:
          case SemIR::NamespaceType::Kind:
          case SemIR::RequireSpecificDefinitionType::Kind:
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
        MangleUnqualifiedClass(os, sem_ir().classes().Get(class_decl.class_id),
                               specific_id);
        break;
      }
      case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
        MangleNameId(
            os, sem_ir().interfaces().Get(interface_decl.interface_id).name_id);
        MangleSpecificId(os, specific_id);
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
      names_to_render.push_back({.name_scope_id = name_scope.parent_scope_id(),
                                 .specific_id = SemIR::SpecificId::None,
                                 .prefix = '.'});
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

  // Clang should emit C++ function declarations for us.
  CARBON_CHECK(!function.clang_decl_id.has_value(),
               "Shouldn't mangle C++ function");

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

  CARBON_CHECK(!sem_ir()
                    .cpp_global_vars()
                    .Lookup({.entity_name_id = var_name_id})
                    .has_value(),
               "Mangling a C++ variable");

  RawStringOstream os;
  os << "_C";

  auto var_name = sem_ir().entity_names().Get(var_name_id);
  MangleNameId(os, var_name.name_id);
  // TODO: If the variable is private, also include the library name as part of
  // the mangling.
  MangleInverseQualifiedNameScope(os, var_name.parent_scope_id);
  return os.TakeStr();
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

auto Mangler::MangleUnqualifiedClass(llvm::raw_ostream& os,
                                     const SemIR::Class& class_info,
                                     SemIR::SpecificId specific_id) -> void {
  MangleNameId(os, class_info.name_id);
  MangleSpecificId(os, specific_id);
}
}  // namespace Carbon::Lower
