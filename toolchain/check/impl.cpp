// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/check/context.h"
#include "toolchain/check/function.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/subst.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Adds the location of the associated function to a diagnostic.
static auto NoteAssociatedFunction(Context& context,
                                   Context::DiagnosticBuilder& builder,
                                   SemIR::FunctionId function_id) -> void {
  CARBON_DIAGNOSTIC(ImplAssociatedFunctionHere, Note,
                    "Associated function {0} declared here.", SemIR::NameId);
  const auto& function = context.functions().Get(function_id);
  builder.Note(function.decl_id, ImplAssociatedFunctionHere, function.name_id);
}

// Checks that `impl_function_id` is a valid implementation of the function
// described in the interface as `interface_function_id`. Returns the value to
// put into the corresponding slot in the witness table, which can be
// `BuiltinError` if the function is not usable.
static auto CheckAssociatedFunctionImplementation(
    Context& context, SemIR::FunctionId interface_function_id,
    SemIR::InstId impl_decl_id, Substitutions substitutions) -> SemIR::InstId {
  auto impl_function_decl =
      context.insts().TryGetAs<SemIR::FunctionDecl>(impl_decl_id);
  if (!impl_function_decl) {
    CARBON_DIAGNOSTIC(ImplFunctionWithNonFunction, Error,
                      "Associated function {0} implemented by non-function.",
                      SemIR::NameId);
    auto builder = context.emitter().Build(
        impl_decl_id, ImplFunctionWithNonFunction,
        context.functions().Get(interface_function_id).name_id);
    NoteAssociatedFunction(context, builder, interface_function_id);
    builder.Emit();

    return SemIR::InstId::BuiltinError;
  }

  // TODO: This should be a semantic check rather than a syntactic one. The
  // functions should be allowed to have different signatures as long as we can
  // synthesize a suitable thunk.
  if (!CheckFunctionTypeMatches(context, impl_function_decl->function_id,
                                interface_function_id, substitutions)) {
    return SemIR::InstId::BuiltinError;
  }
  return impl_decl_id;
}

// Builds a witness that the specified impl implements the given interface.
static auto BuildInterfaceWitness(
    Context& context, const SemIR::Impl& impl, SemIR::InterfaceId interface_id,
    llvm::SmallVectorImpl<SemIR::InstId>& used_decl_ids) -> SemIR::InstId {
  const auto& interface = context.interfaces().Get(interface_id);
  if (!interface.is_defined()) {
    CARBON_DIAGNOSTIC(ImplOfUndefinedInterface, Error,
                      "Implementation of undefined interface {0}.",
                      SemIR::NameId);
    auto builder = context.emitter().Build(
        impl.definition_id, ImplOfUndefinedInterface, interface.name_id);
    context.NoteUndefinedInterface(interface_id, builder);
    builder.Emit();
    return SemIR::InstId::BuiltinError;
  }

  auto& impl_scope = context.name_scopes().Get(impl.scope_id);

  llvm::SmallVector<SemIR::InstId> table;
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  table.reserve(assoc_entities.size());

  // Substitute `Self` with the impl's self type when associated functions.
  Substitution substitutions[1] = {
      {.bind_id = interface.self_param_id,
       .replacement_id = context.types().GetConstantId(impl.self_id)}};

  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id, impl.definition_id);
    auto const_id = context.constant_values().Get(decl_id);
    CARBON_CHECK(const_id.is_constant()) << "Non-constant associated entity";
    auto decl = context.insts().Get(const_id.inst_id());
    if (auto fn_decl = decl.TryAs<SemIR::FunctionDecl>()) {
      auto& fn = context.functions().Get(fn_decl->function_id);
      auto impl_decl_id = context.LookupNameInExactScope(
          decl_id, fn.name_id, impl_scope, /*mark_imports_used=*/true);
      if (impl_decl_id.is_valid()) {
        used_decl_ids.push_back(impl_decl_id);
        table.push_back(CheckAssociatedFunctionImplementation(
            context, fn_decl->function_id, impl_decl_id, substitutions));
      } else {
        CARBON_DIAGNOSTIC(
            ImplMissingFunction, Error,
            "Missing implementation of {0} in impl of interface {1}.",
            SemIR::NameId, SemIR::NameId);
        auto builder =
            context.emitter().Build(impl.definition_id, ImplMissingFunction,
                                    fn.name_id, interface.name_id);
        NoteAssociatedFunction(context, builder, fn_decl->function_id);
        builder.Emit();

        table.push_back(SemIR::InstId::BuiltinError);
      }
    } else if (auto const_decl = decl.TryAs<SemIR::AssociatedConstantDecl>()) {
      // TODO: Check we have a value for this constant in the constraint.
      context.TODO(impl.definition_id,
                   "impl of interface with associated constant");
      return SemIR::InstId::BuiltinError;
    } else {
      CARBON_FATAL() << "Unexpected kind of associated entity " << decl;
    }
  }

  auto table_id = context.inst_blocks().Add(table);
  return context.AddInst(SemIR::InterfaceWitness{
      context.GetBuiltinType(SemIR::BuiltinKind::WitnessType), table_id});
}

auto BuildImplWitness(Context& context, SemIR::ImplId impl_id)
    -> SemIR::InstId {
  auto& impl = context.impls().Get(impl_id);
  CARBON_CHECK(impl.is_being_defined());

  // TODO: Handle non-interface constraints.
  auto interface_type =
      context.types().TryGetAs<SemIR::InterfaceType>(impl.constraint_id);
  if (!interface_type) {
    context.TODO(impl.definition_id, "impl as non-interface");
    return SemIR::InstId::BuiltinError;
  }

  llvm::SmallVector<SemIR::InstId> used_decl_ids;

  auto witness_id = BuildInterfaceWitness(
      context, impl, interface_type->interface_id, used_decl_ids);

  // TODO: Diagnose if any declarations in the impl are not in used_decl_ids.

  return witness_id;
}

}  // namespace Carbon::Check
