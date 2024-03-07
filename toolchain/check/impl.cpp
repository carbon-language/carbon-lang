// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl.h"

#include "toolchain/check/context.h"
#include "toolchain/check/function.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/impl.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Checks that `impl_function_id` is a valid implementation of the function
// described in the interface as `interface_function_id`. Returns the value to
// put into the corresponding slot in the witness table, which can be
// `BuiltinError` if the function is not usable.
static auto CheckAssociatedFunctionImplementation(
    Context& context, SemIR::FunctionId interface_function_id,
    SemIR::InstId impl_decl_id) -> SemIR::InstId {
  auto decl_node = context.insts().GetParseNode(impl_decl_id);
  auto impl_function_decl =
      context.insts().TryGetAs<SemIR::FunctionDecl>(impl_decl_id);
  if (!impl_function_decl) {
    context.TODO(decl_node, "diagnose non-function implementing function");
    return SemIR::InstId::BuiltinError;
  }

  // TODO: Substitute the `Self` from the `impl` into the type in the interface
  // before checking. Also, this should be a semantic check rather than a
  // syntactic one. The functions should be allowed to have different signatures
  // as long as we can synthesize a suitable thunk.
  if (!CheckFunctionRedecl(context, impl_function_decl->function_id,
                           interface_function_id)) {
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
    context.TODO(context.insts().GetParseNode(impl.definition_id),
                 "impl of non-defined interface");
    return SemIR::InstId::BuiltinError;
  }

  auto& impl_scope = context.name_scopes().Get(impl.scope_id);

  llvm::SmallVector<SemIR::InstId, 32> table;
  auto assoc_entities = context.inst_blocks().Get(interface.associated_entities_id);
  table.reserve(assoc_entities.size());

  for (auto decl_id : assoc_entities) {
    auto decl = context.insts().Get(decl_id);
    if (auto fn_decl = decl.TryAs<SemIR::FunctionDecl>()) {
      auto& fn = context.functions().Get(fn_decl->function_id);
      auto impl_decl_id = context.LookupNameInExactScope(fn.name_id, impl_scope);
      if (impl_decl_id.is_valid()) {
        used_decl_ids.push_back(impl_decl_id);
        table.push_back(CheckAssociatedFunctionImplementation(
            context, fn_decl->function_id, impl_decl_id));
      } else {
        context.TODO(context.insts().GetParseNode(impl.definition_id),
                     "diagnose missing decl in impl");
        table.push_back(SemIR::InstId::BuiltinError);
      }
    } else if (auto const_decl = decl.TryAs<SemIR::AssociatedConstantDecl>()) {
      // TODO: Check we have a value for this constant in the constraint.
      context.TODO(context.insts().GetParseNode(impl.definition_id),
                   "impl of interface with associated constant");
      return SemIR::InstId::BuiltinError;
    } else {
      CARBON_FATAL() << "Unexpected kind of associated entity " << decl;
    }
  }

  auto table_id = context.inst_blocks().Add(table);
  return context.AddInst(SemIR::InterfaceWitness{
      context.GetBuiltinType(SemIR::BuiltinKind::WitnessType), interface_id,
      table_id});
}

auto BuildImplWitness(Context& context, SemIR::ImplId impl_id)
	-> SemIR::InstId {
  auto& impl = context.impls().Get(impl_id);
  CARBON_CHECK(impl.is_being_defined());

  // TODO: Handle non-interface constraints.
  auto interface_type =
      context.types().TryGetAs<SemIR::InterfaceType>(impl.constraint_id);
  if (!interface_type) {
    context.TODO(context.insts().GetParseNode(impl.definition_id),
                 "impl as non-interface");
    return SemIR::InstId::BuiltinError;
  }

  llvm::SmallVector<SemIR::InstId, 32> used_decl_ids;

  auto witness_id =
      BuildInterfaceWitness(context, impl, interface_type->interface_id, used_decl_ids);

  // TODO: Diagnose if any declarations in the impl are not in used_decl_ids.

  return witness_id;
}

}  // namespace Carbon::Check
