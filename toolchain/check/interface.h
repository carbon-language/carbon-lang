// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_INTERFACE_H_
#define CARBON_TOOLCHAIN_CHECK_INTERFACE_H_

#include <optional>

#include "toolchain/check/context.h"
#include "toolchain/check/decl_name_stack.h"
#include "toolchain/check/name_component.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/name_scope.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

// Builds and returns an associated entity for `interface_id` corresponding to
// the declaration `decl_id`, which can be an associated function or an
// associated constant. Registers the associated entity in the list for the
// interface.
auto BuildAssociatedEntity(Context& context, SemIR::InterfaceId interface_id,
                           SemIR::InstId decl_id) -> SemIR::InstId;

// Gets the self specific of a generic declaration that is an interface member,
// given a specific for the interface plus a type to use as `Self`.
auto GetSelfSpecificForInterfaceMemberWithSelfType(
    Context& context, SemIR::LocId loc_id,
    SemIR::SpecificId interface_specific_id, SemIR::GenericId generic_id,
    SemIR::SpecificId enclosing_specific_id, SemIR::TypeId self_type_id,
    SemIR::InstId witness_inst_id) -> SemIR::SpecificId;

// Gets the type of the specified associated entity, given the specific for the
// interface and the type of `Self`.
auto GetTypeForSpecificAssociatedEntity(Context& context, SemIR::LocId loc_id,
                                        SemIR::SpecificId interface_specific_id,
                                        SemIR::InstId decl_id,
                                        SemIR::TypeId self_type_id,
                                        SemIR::InstId self_witness_id)
    -> SemIR::TypeId;

// Creates a symbolic binding for `Self` of type `type_id` in the scope of
// `scope_id`, and add the name `Self` for the binding.
//
// Returns the symbolic binding instruction.
auto GetSelfParameter(Context& context, SemIR::TypeId type_id,
                      SemIR::NameScopeId scope_id, bool is_template)
    -> SemIR::InstId;

// Given a search result `lookup_result` for `name_context`, returns the
// previous valid declaration of `name_context` if there is one. Otherwise,
// produces diagnostics if needed and returns ErrorInst.
//
// `try_get_entity` should return the entity pointer for the Inst only if the
// Inst matches the expected type. For instance, for a `SemIR::InterfaceDecl`,
// it may return the `SemIR::Interface`. Otherwise, it should return nullptr
// which will be diagnosed as a redeclaration of a different eniuty type.
auto TryGetExistingDecl(
    Context& context, SemIR::LocId loc_id, const NameComponent& name,
    const DeclNameStack::NameContext& name_context,
    Lex::TokenKind decl_token_kind, const SemIR::EntityWithParamsBase& entity,
    bool is_definition,
    llvm::function_ref<auto(SemIR::Inst)->const SemIR::EntityWithParamsBase*>
        try_get_entity,
    SemIR::ScopeLookupResult lookup_result) -> std::optional<SemIR::Inst>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INTERFACE_H_
