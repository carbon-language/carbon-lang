// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INTERFACE_SUPPORT_H_
#define CARBON_TOOLCHAIN_CHECK_INTERFACE_SUPPORT_H_

#include <optional>

#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/name_component.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Given a search result for `name_context`, return the previous valid
// declaration of `name_context` if there is one. Otherwise, produce diagnostics
// if needed and return nullopt.
//
// `try_get_entity` should return the entity pointer for the Inst only if the
// Inst matches the expected type. For instance, for a `SemIR::InterfaceDecl`,
// it may return the `SemIR::Interface`.
auto GetExistingDeclOrDiagnoseMismatch(
    Context& context, Parse::NodeId node_id, const NameComponent& name,
    const DeclNameStack::NameContext& name_context,
    const SemIR::EntityWithParamsBase& entity, bool is_definition,
    llvm::function_ref<auto(SemIR::Inst)->const SemIR::EntityWithParamsBase*>
        try_get_entity,
    SemIR::ScopeLookupResult lookup_result) -> std::optional<SemIR::Inst>;

// Create a symbolic binding for `Self` of type `type_id` in the scope of
// `scope_id`, and add the name `Self` for the binding.
//
// Returns the symbolic binding instruction.
auto GetSelfParameter(Context& context, SemIR::TypeId type_id,
                      SemIR::NameScopeId scope_id, bool is_template)
    -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INTERFACE_SUPPORT_H_
