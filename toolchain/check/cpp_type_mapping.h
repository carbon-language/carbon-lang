// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_TYPE_MAPPING_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_TYPE_MAPPING_H_

#include "clang/AST/Type.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Maps a Carbon type to a C++ type. Accepts an InstId, representing a value
// whose type is mapped to a C++ type. Returns `clang::QualType` if the mapping
// succeeds, or `clang::QualType::isNull()` if the type is not supported.
auto MapToCppType(Context& context, SemIR::InstId inst_id) -> clang::QualType;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_TYPE_MAPPING_H_
