// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_GENERIC_H_
#define CARBON_TOOLCHAIN_CHECK_GENERIC_H_

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDecl(Context& /*context*/) -> void;

// Start processing a declaration or definition that might be a generic entity.
auto StartGenericDefinition(Context& /*context*/,
                            SemIR::GenericId /*generic_id*/) -> void;

// Finish processing a potentially generic declaration and produce a
// corresponding generic object. Returns SemIR::GenericId::Invalid if this
// declaration is not actually generic.
auto FinishGenericDecl(Context& context,
                       SemIR::InstId decl_id) -> SemIR::GenericId;

// Merge a redeclaration of an entity that might be a generic into the original
// declaration.
auto FinishGenericRedecl(Context& context, SemIR::InstId decl_id,
                         SemIR::GenericId generic_id) -> void;

// Finish processing a potentially generic definition.
auto FinishGenericDefinition(Context& context,
                             SemIR::GenericId generic_id) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_GENERIC_H_
