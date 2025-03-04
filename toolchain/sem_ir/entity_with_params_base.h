// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_ENTITY_WITH_PARAMS_BASE_H_
#define CARBON_TOOLCHAIN_SEM_IR_ENTITY_WITH_PARAMS_BASE_H_

#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Common entity fields.
//
// `EntityWithParamsBase` would be a base class of entities like `Function`,
// except that then we couldn't use named initialization (or would need to
// disable warnings about mixing named and unnamed initialization) due to how
// C++ handles initialization of base structs. Instead, this is composed with a
// `Fields` struct to provide an entity's actual struct.
//
// For example:
//   struct FunctionFields {
//     ... data members ...
//   };
//
//   struct Function : public EntityWithParamsBase,
//                     public FunctionFields, public Printable<Function> {
//     ... methods ...
//   };
//
// This achieves a few things:
//   - Allows named initialization, such as:
//     `{{.name_id = ...}, {.function_field = ...}}`
//   - Makes `entity.name_id` access work.
//   - Allows passing a `EntityWithParamsBase*` when only common fields are
//     needed.
//   - Does all this in a way that's vanilla C++.
struct EntityWithParamsBase {
  auto PrintBaseFields(llvm::raw_ostream& out) const -> void {
    out << "name: " << name_id << ", parent_scope: " << parent_scope_id;
  }

  // When merging a declaration and definition, prefer things which would point
  // at the definition for diagnostics.
  auto MergeDefinition(const EntityWithParamsBase& definition) -> void {
    first_param_node_id = definition.first_param_node_id;
    last_param_node_id = definition.last_param_node_id;
    pattern_block_id = definition.pattern_block_id;
    implicit_param_patterns_id = definition.implicit_param_patterns_id;
    param_patterns_id = definition.param_patterns_id;
    call_params_id = definition.call_params_id;
    definition_id = definition.definition_id;
  }

  // Determines whether the definition of this entity has begun. This is false
  // until we reach the `{` of the definition.
  auto has_definition_started() const -> bool {
    return definition_id.has_value();
  }

  // Returns the instruction for the first declaration.
  auto first_decl_id() const -> SemIR::InstId {
    if (non_owning_decl_id.has_value()) {
      return non_owning_decl_id;
    }
    CARBON_CHECK(first_owning_decl_id.has_value());
    return first_owning_decl_id;
  }

  // Returns the instruction for the latest declaration.
  auto latest_decl_id() const -> SemIR::InstId {
    if (has_definition_started()) {
      return definition_id;
    }
    if (first_owning_decl_id.has_value()) {
      return first_owning_decl_id;
    }
    return non_owning_decl_id;
  }

  // Determines whether this entity has any parameter lists.
  auto has_parameters() const -> bool {
    return implicit_param_patterns_id.has_value() ||
           param_patterns_id.has_value();
  }

  // The following members always have values, and do not change throughout the
  // lifetime of the entity.

  // The entity's name.
  NameId name_id;

  // The parent scope.
  NameScopeId parent_scope_id;

  // If this is a generic entity, information about the generic.
  GenericId generic_id;

  // Parse tree bounds for the parameters, including both implicit and explicit
  // parameters. These will be compared to match between declaration and
  // definition.
  Parse::NodeId first_param_node_id;
  Parse::NodeId last_param_node_id;

  // A block containing the pattern insts for the parameter lists.
  InstBlockId pattern_block_id;

  // A block containing, for each implicit parameter, a reference to the
  // instruction in the entity's pattern block that depends on all other
  // pattern insts pertaining to that parameter.
  InstBlockId implicit_param_patterns_id;
  // A block containing, for each element of the explicit parameter list tuple
  // pattern, a reference to the root pattern inst for that element.
  InstBlockId param_patterns_id;

  // If this entity is a function, this block consists of references to the
  // `AnyParam` insts that represent the function's `Call` parameters. The
  // "`Call` parameters" are the parameters corresponding to the arguments that
  // are passed to a `Call` inst, so they do not include compile-time
  // parameters, but they do include the return slot.
  //
  // The parameters appear in declaration order: `self` (if present), then the
  // explicit runtime parameters, then the return slot (which is "declared" by
  // the function's return type declaration). This is not populated on imported
  // functions, because it is relevant only for a function definition.
  //
  // TODO: Can this be moved to `Function`, since it is not applicable to other
  // kinds of entities?
  InstBlockId call_params_id;

  // True if declarations are `extern`.
  bool is_extern;

  // For an `extern library` declaration, the library name.
  SemIR::LibraryNameId extern_library_id;

  // The non-owning declaration of the entity, if present. This will be a
  // <entity>Decl.
  InstId non_owning_decl_id;

  // The first owning declaration of the entity, if present. This will be a
  // <entity>Decl. It may either be a forward declaration, or the same as
  // `definition_id`.
  InstId first_owning_decl_id;

  // The following members are set at the `{` of the definition.

  // The definition of the entity. This will be a <entity>Decl.
  InstId definition_id = InstId::None;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_ENTITY_WITH_PARAMS_BASE_H_
