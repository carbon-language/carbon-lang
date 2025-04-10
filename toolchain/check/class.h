// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CLASS_H_
#define CARBON_TOOLCHAIN_CHECK_CLASS_H_

#include "toolchain/check/context.h"

namespace Carbon::Check {

// If `type_id` is a class type, get its corresponding `SemIR::Class` object.
// Otherwise returns `nullptr`.
auto TryGetAsClass(Context& context, SemIR::TypeId type_id) -> SemIR::Class*;

// Sets the `Self` type for the class.
auto SetClassSelfType(Context& context, SemIR::ClassId class_id) -> void;

// Starts the class definition, adding `Self` to name lookup.
auto StartClassDefinition(Context& context, SemIR::Class& class_info,
                          SemIR::InstId definition_id) -> void;

// Computes the object representation for a fully defined class.
auto ComputeClassObjectRepr(Context& context, Parse::NodeId node_id,
                            SemIR::ClassId class_id,
                            llvm::ArrayRef<SemIR::InstId> field_decls,
                            llvm::ArrayRef<SemIR::InstId> vtable_contents,
                            llvm::ArrayRef<SemIR::InstId> body) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CLASS_H_
