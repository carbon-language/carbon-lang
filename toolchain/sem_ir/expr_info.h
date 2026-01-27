// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_
#define CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_

#include <cstdint>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst_kind.h"

namespace Carbon::SemIR {

// Returns the expression category for an instruction.
auto GetExprCategory(const File& file, InstId inst_id) -> ExprCategory;

// Returns whether the given expression category is for a reference expression.
inline auto IsRefCategory(ExprCategory cat) -> bool {
  return cat == ExprCategory::DurableRef || cat == ExprCategory::EphemeralRef;
}

// Returns whether the given expression category is for an initializer
// (see inst_kind.h for background).
inline auto IsInitializerCategory(ExprCategory cat) -> bool {
  return cat == ExprCategory::ReprInitializing ||
         cat == ExprCategory::InPlaceInitializing;
}

// If `init_id` is an initializer, find the inst ID that specifies the storage
// to initialize, if any. If `allow_transitive` is true, the result may be an
// argument to some other inst whose outcome is forwarded by `init_id`;
// otherwise the result must be an argument to `init_id` itself. Returns `None`
// if there is no such storage argument. When `allow_transitive` is true, this
// can only return `None` if `init_id` is known not to perform in-place
// initialization; i.e. its type's initializing representation is not in-place,
// and its category is `Initializing`.
auto FindStorageArgForInitializer(const File& sem_ir, InstId init_id,
                                  bool allow_transitive = true) -> InstId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_EXPR_INFO_H_
