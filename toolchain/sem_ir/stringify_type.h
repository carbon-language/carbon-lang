// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_STRINGIFY_TYPE_H_
#define CARBON_TOOLCHAIN_SEM_IR_STRINGIFY_TYPE_H_

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Produces a string version of a type expression. Generally, this should not be
// called directly. To format a string into a diagnostic, use a diagnostic
// parameter of type `InstIdAsType`, `InstIdAsRawType`, or `TypeOfInstId` where
// possible, or of type `TypeId` or `TypeIdAsRawType` if you don't have an
// expression describing the type.
auto StringifyTypeExpr(const File& sem_ir, InstId outer_inst_id) -> std::string;

// Produces a string version of the name of a specific. Generally, this should
// not be called directly. To format a string into a diagnostic, use a
// diagnostic parameter of type `SemIR::SpecificId`.
auto StringifySpecific(const File& sem_ir, SpecificId specific_id)
    -> std::string;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_STRINGIFY_TYPE_H_
