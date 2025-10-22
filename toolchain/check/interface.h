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
// `scope_id`, and add the name `Self` for the compile time binding.
//
// Returns the symbolic binding instruction.
auto AddSelfGenericParameter(Context& context, SemIR::TypeId type_id,
                             SemIR::NameScopeId scope_id, bool is_template)
    -> SemIR::InstId;

// Given a search result `lookup_result` for `name`, returns the previous valid
// declaration of `name` if there is one. The `entity` is a new decl of the same
// `name`, and the existing decl need to be of the same entity type. Otherwise,
// produces diagnostics if needed and returns nullopt.
template <typename EntityT>
  requires SameAsOneOf<EntityT, SemIR::Interface, SemIR::NamedConstraint>
auto TryGetExistingDecl(Context& context, const NameComponent& name,
                        SemIR::ScopeLookupResult lookup_result,
                        const EntityT& entity, bool is_definition)
    -> std::optional<SemIR::Inst>;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_INTERFACE_H_
