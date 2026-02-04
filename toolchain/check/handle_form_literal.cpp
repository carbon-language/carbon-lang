// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/check/handle.h"
#include "toolchain/parse/node_category.h"

namespace Carbon::Check {

auto HandleParseNode(Context& context, Parse::RefCategoryModifierId node_id)
    -> bool {
  return context.TODO(node_id, "Implement form literals");
}

auto HandleParseNode(Context& context, Parse::ValCategoryModifierId node_id)
    -> bool {
  return context.TODO(node_id, "Implement form literals");
}

auto HandleParseNode(Context& context, Parse::VarCategoryModifierId node_id)
    -> bool {
  return context.TODO(node_id, "Implement form literals");
}

auto HandleParseNode(Context& context, Parse::FormLiteralKeywordId node_id)
    -> bool {
  return context.TODO(node_id, "Implement form literals");
}

auto HandleParseNode(Context& context, Parse::FormLiteralOpenParenId node_id)
    -> bool {
  return context.TODO(node_id, "Implement form literals");
}

auto HandleParseNode(Context& context, Parse::FormLiteralId node_id) -> bool {
  return context.TODO(node_id, "Implement form literals");
}

}  // namespace Carbon::Check
