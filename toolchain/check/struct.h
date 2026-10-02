// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_STRUCT_H_
#define CARBON_TOOLCHAIN_CHECK_STRUCT_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// Struct kinds, for diagnostics. Converted to an int for a format select.
enum class StructKind : uint8_t {
  StructLiteral = 0,
  StructTypeLiteral = 1,
  StructPattern = 2
};

// Pops the names of each field from the stack. These will have been left while
// handling struct fields.
auto PopStructFieldNameNodes(Context& context, size_t field_count)
    -> llvm::SmallVector<Parse::NodeId>;

// Diagnoses and returns true if there's a duplicate name.
auto DiagnoseDuplicateNames(Context& context,
                            llvm::ArrayRef<Parse::NodeId> field_name_nodes,
                            llvm::ArrayRef<SemIR::StructTypeField> fields,
                            StructKind struct_kind_for_diagnostic) -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_STRUCT_H_
