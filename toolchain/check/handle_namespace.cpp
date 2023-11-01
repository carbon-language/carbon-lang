// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleNamespaceStart(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  context.declaration_name_stack().Push();
  return true;
}

auto HandleNamespace(Context& context, Parse::Node parse_node) -> bool {
  auto name_context = context.declaration_name_stack().Pop();
  auto namespace_id = context.AddInst(SemIR::Namespace{
      parse_node, context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
      context.name_scopes().Add()});
  context.declaration_name_stack().AddNameToLookup(name_context, namespace_id);
  return true;
}

}  // namespace Carbon::Check
