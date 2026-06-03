// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/mangler.h"

#include <string>

#include "common/raw_string_ostream.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/sem_ir/entry_point.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/pattern.h"
#include "toolchain/sem_ir/specific_interface.h"
#include "toolchain/sem_ir/specific_named_constraint.h"
#include "toolchain/sem_ir/thunk.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

Mangler::Mangler(const SemIR::File& sem_ir, int total_ir_count,
                 bool use_string_fingerprint)
    : sem_ir_(sem_ir),
      fingerprinter_(
          use_string_fingerprint
              ? std::variant<HashInstFingerprinter, StringInstFingerprinter>(
                    std::in_place_type<StringInstFingerprinter>, total_ir_count)
              : std::variant<HashInstFingerprinter, StringInstFingerprinter>(
                    std::in_place_type<HashInstFingerprinter>,
                    total_ir_count)) {}

auto Mangler::MangleNameId(llvm::raw_ostream& os, SemIR::NameId name_id)
    -> void {
  CARBON_CHECK(name_id.AsIdentifierId().has_value(),
               "Mangling non-identifier name {0}", name_id);
  os << names().GetAsStringIfIdentifier(name_id);
}

auto Mangler::MangleInverseQualifiedNameScope(llvm::raw_ostream& os,
                                              SemIR::NameScopeId name_scope_id,
                                              SemIR::SpecificId specific_id,
                                              char initial_prefix) -> void {
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
                             .specific_id = specific_id,
                             .prefix = initial_prefix});
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
    auto next_scope_id = name_scope.parent_scope_id();
    CARBON_KIND_SWITCH(sem_ir().insts().Get(name_scope.inst_id())) {
      case CARBON_KIND(SemIR::ImplDecl impl_decl): {
        const auto& impl = sem_ir().impls().Get(impl_decl.impl_id);
        const auto& interface =
            sem_ir().interfaces().Get(impl.interface.interface_id);
        names_to_render.push_back(
            // We mangle names in an interface without `Self` in the specific
            // since it would just add noise and `Self` is not part of how you
            // name the entities syntactically.
            {.name_scope_id = interface.scope_without_self_id,
             .specific_id = impl.interface.specific_id,
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

            MangleUnqualifiedName(os, class_info, class_type.specific_id);
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
            MangleFingerprint(os, &sem_ir(), self_const_inst_id);
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
        MangleUnqualifiedName(os, sem_ir().classes().Get(class_decl.class_id),
                              specific_id);
        break;
      }
      case CARBON_KIND(SemIR::InterfaceDecl interface_decl): {
        MangleUnqualifiedName(
            os, sem_ir().interfaces().Get(interface_decl.interface_id),
            specific_id);
        break;
      }
      case CARBON_KIND(SemIR::InterfaceWithSelfDecl interface_with_self_decl): {
        MangleUnqualifiedName(
            os,
            sem_ir().interfaces().Get(interface_with_self_decl.interface_id),
            specific_id);
        // Skip the enclosing `InterfaceDecl` as we already mangled its name.
        next_scope_id =
            sem_ir().name_scopes().Get(next_scope_id).parent_scope_id();
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
      names_to_render.push_back({.name_scope_id = next_scope_id,
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
  if (function_id != sem_ir().global_ctor_id()) {
    auto clang_decl_id =
        sem_ir().clang_decls().Lookup(function.first_decl_id());
    CARBON_CHECK(!clang_decl_id.has_value() ||
                     !sem_ir().clang_decls().Get(clang_decl_id).is_imported,
                 "Shouldn't mangle C++ function");
  }

  RawStringOstream os;
  os << "_C";

  MangleNameId(os, function.name_id);
  char separator = '.';

  // For a special function, add a marker to disambiguate.
  switch (function.special_function_kind) {
    case SemIR::Function::SpecialFunctionKind::None:
    case SemIR::Function::SpecialFunctionKind::CppThunk:
      break;

    case SemIR::Function::SpecialFunctionKind::CoreWitness:
      os << ".";
      MangleFingerprint(os, &sem_ir(), function.self_param_id);
      os << ":core";
      break;
    case SemIR::Function::SpecialFunctionKind::Thunk:
      os << ":thunk";
      if (function.thunk_id().has_value()) {
        const auto& thunk_info = sem_ir().thunks().Get(function.thunk_id());
        if (thunk_info.signature_id.has_value()) {
          const auto& sig_fn =
              sem_ir().functions().Get(thunk_info.signature_id);
          // If this ever becomes possible, we'll need to include the signature
          // name in the mangling too.
          CARBON_CHECK(sig_fn.name_id == function.name_id,
                       "Thunk name mismatches signature name");
          MangleInverseQualifiedNameScope(os, sig_fn.parent_scope_id,
                                          thunk_info.specific_id, ':');
          separator = ':';
        }
      }
      break;

    case SemIR::Function::SpecialFunctionKind::Builtin:
      CARBON_FATAL("Attempting to mangle declaration of builtin function {0}",
                   function.builtin_function_kind());
    case SemIR::Function::SpecialFunctionKind::HasCppThunk:
      CARBON_FATAL("C++ functions should have been handled earlier");
  }

  MangleInverseQualifiedNameScope(os, function.parent_scope_id,
                                  SemIR::SpecificId::None, separator);

  if (sem_ir().name_scopes().IsPrivateToLibrary(function.name_id,
                                                function.parent_scope_id)) {
    os << ".";
    MangleFingerprint(os, &sem_ir(), function.first_decl_id());
  }

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
    MangleFingerprint(os, &sem_ir(),
                      sem_ir().specifics().Get(specific_id).args_id);
  }
}

auto Mangler::MangleGlobalVariable(SemIR::InstId pattern_id) -> std::string {
  // Use the name of the first binding in the variable as its mangled name.
  auto var_name_id =
      SemIR::GetFirstBindingNameFromPatternId(sem_ir(), pattern_id);
  if (!var_name_id.has_value()) {
    return std::string();
  }

  auto clang_decl_id = sem_ir().clang_decls().Lookup(pattern_id);
  if (clang_decl_id.has_value()) {
    CARBON_CHECK(!sem_ir().clang_decls().Get(clang_decl_id).is_imported,
                 "Mangling a C++ variable");
  }

  RawStringOstream os;
  os << "_C";

  auto var_name = sem_ir().entity_names().Get(var_name_id);
  MangleNameId(os, var_name.name_id);
  MangleInverseQualifiedNameScope(os, var_name.parent_scope_id);

  if (sem_ir().name_scopes().IsPrivateToLibrary(var_name.name_id,
                                                var_name.parent_scope_id)) {
    os << ".";
    MangleFingerprint(os, &sem_ir(), pattern_id);
  }
  return os.TakeStr();
}

auto Mangler::MangleVTable(const SemIR::Class& class_info,
                           SemIR::SpecificId specific_id) -> std::string {
  RawStringOstream os;
  os << "_C";

  MangleNameId(os, class_info.name_id);
  MangleInverseQualifiedNameScope(os, class_info.parent_scope_id);

  if (sem_ir().name_scopes().IsPrivateToLibrary(class_info.name_id,
                                                class_info.parent_scope_id)) {
    os << ".";
    MangleFingerprint(os, &sem_ir(), class_info.first_decl_id());
  }

  os << ".$vtable";

  MangleSpecificId(os, specific_id);

  return os.TakeStr();
}

auto Mangler::MangleUnqualifiedName(llvm::raw_ostream& os,
                                    const SemIR::EntityWithParamsBase& entity,
                                    SemIR::SpecificId specific_id) -> void {
  MangleNameId(os, entity.name_id);
  MangleSpecificId(os, specific_id);
}
}  // namespace Carbon::SemIR
