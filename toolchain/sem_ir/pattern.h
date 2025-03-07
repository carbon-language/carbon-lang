// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_PATTERN_H_
#define CARBON_TOOLCHAIN_SEM_IR_PATTERN_H_

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Returns true if `pattern_id` is a `self` parameter pattern, such as
// `self: Foo` or `addr self: Self*`.
auto IsSelfPattern(const File& sem_ir, InstId pattern_id) -> bool;

// If `pattern_id` is a declaration of a single name, this returns that name,
// and otherwise returns `None`. This tries to "see through" wrappers like
// `AddrPattern` and `*ParamPattern`, so this may return the same name for
// different insts if one is an ancestor of the other (or if they represent
// separate declarations of the same name).
//
// This should only be used for decorative purposes such as SemIR
// pretty-printing or LLVM parameter naming.
auto GetPrettyNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> SemIR::NameId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_PATTERN_H_
