// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_NAME_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_NAME_LOOKUP_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Information about a scope in which we can perform name lookup.
struct LookupScope {
  // The name scope in which names are searched.
  SemIR::NameScopeId name_scope_id;
  // The specific for the name scope, or `None` if the name scope is not
  // defined by a generic or we should perform lookup into the generic itself.
  SemIR::SpecificId specific_id;
};

// A result produced by name lookup.
struct LookupResult {
  // The specific in which the lookup result was found. `None` if the result
  // was not found in a specific.
  SemIR::SpecificId specific_id;

  // The result from the lookup in the scope.
  SemIR::ScopeLookupResult scope_result;
};

// Information about an access.
struct AccessInfo {
  // The constant being accessed.
  SemIR::ConstantId constant_id;

  // The highest allowed access for a lookup. For example, `Protected` allows
  // access to `Public` and `Protected` names, but not `Private`.
  SemIR::AccessKind highest_allowed_access;
};

// Adds a name to name lookup. Prints a diagnostic for name conflicts. If
// specified, `scope_index` specifies which lexical scope the name is inserted
// into, otherwise the name is inserted into the current scope.
auto AddNameToLookup(Context& context, SemIR::NameId name_id,
                     SemIR::InstId target_id,
                     ScopeIndex scope_index = ScopeIndex::None) -> void;

// Performs name lookup in a specified scope for a name appearing in a
// declaration. If scope_id is `None`, performs lookup into the lexical scope
// specified by scope_index instead.
auto LookupNameInDecl(Context& context, SemIR::LocId loc_id,
                      SemIR::NameId name_id, SemIR::NameScopeId scope_id,
                      ScopeIndex scope_index) -> SemIR::ScopeLookupResult;

// Performs an unqualified name lookup, returning the referenced `InstId`.
auto LookupUnqualifiedName(Context& context, Parse::NodeId node_id,
                           SemIR::NameId name_id, bool required = true)
    -> LookupResult;

// Performs a name lookup in a specified scope, returning the referenced
// `InstId`. Does not look into extended scopes. Returns `InstId::None` if the
// name is not found.
//
// If `is_being_declared` is false, then this is a regular name lookup, and
// the name will be poisoned if not found so that later lookups will fail; a
// poisoned name will be treated as if it is not declared. Otherwise, this is
// a lookup for a name being declared, so the name will not be poisoned, but
// poison will be returned if it's already been looked up.
//
// If `name_id` is not an identifier, the name will not be poisoned.
auto LookupNameInExactScope(Context& context, SemIR::LocId loc_id,
                            SemIR::NameId name_id, SemIR::NameScopeId scope_id,
                            SemIR::NameScope& scope,
                            bool is_being_declared = false)
    -> SemIR::ScopeLookupResult;

// Appends the lookup scopes corresponding to `base_const_id` to `*scopes`.
// Returns `false` if not a scope. On invalid scopes, prints a diagnostic, but
// still updates `*scopes` and returns `true`.
auto AppendLookupScopesForConstant(Context& context, SemIR::LocId loc_id,
                                   SemIR::ConstantId base_const_id,
                                   llvm::SmallVector<LookupScope>* scopes)
    -> bool;

// Performs a qualified name lookup in a specified scopes and in scopes that
// they extend, returning the referenced `InstId`.
auto LookupQualifiedName(Context& context, SemIR::LocId loc_id,
                         SemIR::NameId name_id,
                         llvm::ArrayRef<LookupScope> lookup_scopes,
                         bool required = true,
                         std::optional<AccessInfo> access_info = std::nullopt)
    -> LookupResult;

// Returns the `InstId` corresponding to a name in the core package, or
// BuiltinErrorInst if not found.
auto LookupNameInCore(Context& context, SemIR::LocId loc_id,
                      llvm::StringRef name) -> SemIR::InstId;

// Prints a diagnostic for a duplicate name.
auto DiagnoseDuplicateName(Context& context, SemIR::NameId name_id,
                           SemIRLoc dup_def, SemIRLoc prev_def) -> void;

// Prints a diagnostic for a poisoned name when it's later declared.
auto DiagnosePoisonedName(Context& context, SemIR::NameId name_id,
                          SemIR::LocId poisoning_loc_id,
                          SemIR::LocId decl_name_loc_id) -> void;

// Prints a diagnostic for a missing name.
auto DiagnoseNameNotFound(Context& context, SemIRLoc loc, SemIR::NameId name_id)
    -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_NAME_LOOKUP_H_
