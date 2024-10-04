// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/base/kind_switch.h"
#include "toolchain/check/context.h"
#include "toolchain/check/function.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/sem_ir/generic.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Adds the location of the associated function to a diagnostic.
static auto NoteAssociatedFunction(Context& context,
                                   Context::DiagnosticBuilder& builder,
                                   SemIR::FunctionId function_id) -> void {
  CARBON_DIAGNOSTIC(ImplAssociatedFunctionHere, Note,
                    "associated function {0} declared here", SemIR::NameId);
  const auto& function = context.functions().Get(function_id);
  builder.Note(function.latest_decl_id(), ImplAssociatedFunctionHere,
               function.name_id);
}

// Gets the self specific of a generic declaration that is an interface member,
// given a specific for an enclosing generic, plus a type to use as `Self`.
static auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIR::SpecificId enclosing_specific_id,
    SemIR::GenericId generic_id, SemIR::TypeId self_type_id)
    -> SemIR::SpecificId {
  const auto& generic = context.generics().Get(generic_id);
  auto bindings = context.inst_blocks().Get(generic.bindings_id);

  llvm::SmallVector<SemIR::InstId> arg_ids;
  arg_ids.reserve(bindings.size());

  // Start with the enclosing arguments.
  if (enclosing_specific_id.is_valid()) {
    auto enclosing_specific_args_id =
        context.specifics().Get(enclosing_specific_id).args_id;
    auto enclosing_specific_args =
        context.inst_blocks().Get(enclosing_specific_args_id);
    arg_ids.assign(enclosing_specific_args.begin(),
                   enclosing_specific_args.end());
  }

  // Add the `Self` argument.
  CARBON_CHECK(
      context.entity_names()
              .Get(context.insts()
                       .GetAs<SemIR::BindSymbolicName>(bindings[arg_ids.size()])
                       .entity_name_id)
              .name_id == SemIR::NameId::SelfType,
      "Expected a Self binding, found {0}",
      context.insts().Get(bindings[arg_ids.size()]));
  arg_ids.push_back(context.types().GetInstId(self_type_id));

  // Take any trailing argument values from the self specific.
  // TODO: If these refer to outer arguments, for example in their types, we may
  // need to perform extra substitutions here.
  auto self_specific_args = context.inst_blocks().Get(
      context.specifics().Get(generic.self_specific_id).args_id);
  for (auto arg_id : self_specific_args.drop_front(arg_ids.size())) {
    arg_ids.push_back(context.constant_values().GetConstantInstId(arg_id));
  }

  auto args_id = context.inst_blocks().AddCanonical(arg_ids);
  return MakeSpecific(context, generic_id, args_id);
}

// Checks that `impl_function_id` is a valid implementation of the function
// described in the interface as `interface_function_id`. Returns the value to
// put into the corresponding slot in the witness table, which can be
// `BuiltinError` if the function is not usable.
static auto CheckAssociatedFunctionImplementation(
    Context& context, SemIR::FunctionType interface_function_type,
    SemIR::InstId impl_decl_id, SemIR::TypeId self_type_id) -> SemIR::InstId {
  auto impl_function_decl =
      context.insts().TryGetAs<SemIR::FunctionDecl>(impl_decl_id);
  if (!impl_function_decl) {
    CARBON_DIAGNOSTIC(ImplFunctionWithNonFunction, Error,
                      "associated function {0} implemented by non-function",
                      SemIR::NameId);
    auto builder = context.emitter().Build(
        impl_decl_id, ImplFunctionWithNonFunction,
        context.functions().Get(interface_function_type.function_id).name_id);
    NoteAssociatedFunction(context, builder,
                           interface_function_type.function_id);
    builder.Emit();

    return SemIR::InstId::BuiltinError;
  }

  // Map from the specific for the function type to the specific for the
  // function signature. The function signature may have additional generic
  // parameters.
  auto interface_function_specific_id =
      GetSelfSpecificForInterfaceMemberWithSelfType(
          context, interface_function_type.specific_id,
          context.functions()
              .Get(interface_function_type.function_id)
              .generic_id,
          self_type_id);

  // TODO: This should be a semantic check rather than a syntactic one. The
  // functions should be allowed to have different signatures as long as we can
  // synthesize a suitable thunk.
  if (!CheckFunctionTypeMatches(
          context, context.functions().Get(impl_function_decl->function_id),
          context.functions().Get(interface_function_type.function_id),
          interface_function_specific_id,
          /*check_syntax=*/false)) {
    return SemIR::InstId::BuiltinError;
  }
  return impl_decl_id;
}

// Builds a witness that the specified impl implements the given interface.
static auto BuildInterfaceWitness(
    Context& context, const SemIR::Impl& impl, SemIR::TypeId interface_type_id,
    SemIR::InterfaceType interface_type,
    llvm::SmallVectorImpl<SemIR::InstId>& used_decl_ids) -> SemIR::InstId {
  const auto& interface = context.interfaces().Get(interface_type.interface_id);
  if (!context.TryToDefineType(interface_type_id, [&] {
        CARBON_DIAGNOSTIC(ImplOfUndefinedInterface, Error,
                          "implementation of undefined interface {0}",
                          SemIR::NameId);
        return context.emitter().Build(
            impl.definition_id, ImplOfUndefinedInterface, interface.name_id);
      })) {
    return SemIR::InstId::BuiltinError;
  }

  auto& impl_scope = context.name_scopes().Get(impl.scope_id);
  auto self_type_id = context.GetTypeIdForTypeInst(impl.self_id);

  llvm::SmallVector<SemIR::InstId> table;
  auto assoc_entities =
      context.inst_blocks().Get(interface.associated_entities_id);
  table.reserve(assoc_entities.size());

  for (auto decl_id : assoc_entities) {
    LoadImportRef(context, decl_id);
    decl_id =
        context.constant_values().GetInstId(SemIR::GetConstantValueInSpecific(
            context.sem_ir(), interface_type.specific_id, decl_id));
    CARBON_CHECK(decl_id.is_valid(), "Non-constant associated entity");
    auto decl = context.insts().Get(decl_id);
    CARBON_KIND_SWITCH(decl) {
      case CARBON_KIND(SemIR::StructValue struct_value): {
        if (struct_value.type_id == SemIR::TypeId::Error) {
          return SemIR::InstId::BuiltinError;
        }
        auto type_inst = context.types().GetAsInst(struct_value.type_id);
        auto fn_type = type_inst.TryAs<SemIR::FunctionType>();
        if (!fn_type) {
          CARBON_FATAL("Unexpected type: {0}", type_inst);
        }
        auto& fn = context.functions().Get(fn_type->function_id);
        auto [impl_decl_id, _] = context.LookupNameInExactScope(
            decl_id, fn.name_id, impl.scope_id, impl_scope);
        if (impl_decl_id.is_valid()) {
          used_decl_ids.push_back(impl_decl_id);
          table.push_back(CheckAssociatedFunctionImplementation(
              context, *fn_type, impl_decl_id, self_type_id));
        } else {
          CARBON_DIAGNOSTIC(
              ImplMissingFunction, Error,
              "missing implementation of {0} in impl of interface {1}",
              SemIR::NameId, SemIR::NameId);
          auto builder =
              context.emitter().Build(impl.definition_id, ImplMissingFunction,
                                      fn.name_id, interface.name_id);
          NoteAssociatedFunction(context, builder, fn_type->function_id);
          builder.Emit();

          table.push_back(SemIR::InstId::BuiltinError);
        }
        break;
      }
      case SemIR::AssociatedConstantDecl::Kind:
        // TODO: Check we have a value for this constant in the constraint.
        context.TODO(impl.definition_id,
                     "impl of interface with associated constant");
        return SemIR::InstId::BuiltinError;
      default:
        CARBON_CHECK(decl_id == SemIR::InstId::BuiltinError,
                     "Unexpected kind of associated entity {0}", decl);
        table.push_back(SemIR::InstId::BuiltinError);
        break;
    }
  }

  auto table_id = context.inst_blocks().Add(table);
  return context.AddInst<SemIR::InterfaceWitness>(
      context.insts().GetLocId(impl.definition_id),
      {.type_id = context.GetBuiltinType(SemIR::BuiltinInstKind::WitnessType),
       .elements_id = table_id});
}

auto BuildImplWitness(Context& context, SemIR::ImplId impl_id)
    -> SemIR::InstId {
  auto& impl = context.impls().Get(impl_id);
  CARBON_CHECK(impl.is_being_defined());

  // TODO: Handle non-interface constraints.
  auto interface_type_id = context.GetTypeIdForTypeInst(impl.constraint_id);
  auto interface_type =
      context.types().TryGetAs<SemIR::InterfaceType>(interface_type_id);
  if (!interface_type) {
    context.TODO(impl.definition_id, "impl as non-interface");
    return SemIR::InstId::BuiltinError;
  }

  llvm::SmallVector<SemIR::InstId> used_decl_ids;

  auto witness_id = BuildInterfaceWitness(context, impl, interface_type_id,
                                          *interface_type, used_decl_ids);

  // TODO: Diagnose if any declarations in the impl are not in used_decl_ids.

  return witness_id;
}

}  // namespace Carbon::Check
