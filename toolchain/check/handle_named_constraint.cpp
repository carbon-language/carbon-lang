// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"

namespace Carbon::Check {

auto HandleNamedConstraintDecl(Context& context,
                               Parse::NamedConstraintDeclId node_id) -> bool {
  return context.TODO(node_id, "HandleNamedConstraintDecl");
}

auto HandleNamedConstraintDefinition(Context& context,
                                     Parse::NamedConstraintDefinitionId node_id)
    -> bool {
  // Note that the decl_name_stack will be popped by `ProcessNodeIds`.
  return context.TODO(node_id, "HandleNamedConstraintDefinition");
}

auto HandleNamedConstraintDefinitionStart(
    Context& context, Parse::NamedConstraintDefinitionStartId node_id) -> bool {
  return context.TODO(node_id, "HandleNamedConstraintDefinitionStart");
}

auto HandleNamedConstraintIntroducer(Context& context,
                                     Parse::NamedConstraintIntroducerId node_id)
    -> bool {
  return context.TODO(node_id, "HandleNamedConstraintIntroducer");
}

}  // namespace Carbon::Check
