// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/context.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::Check {

auto HandleNamespaceStart(Context& context, Parse::Node /*parse_node*/)
    -> bool {
  context.decl_name_stack().PushScopeAndStartName();
  return true;
}

auto HandleNamespace(Context& context, Parse::Node parse_node) -> bool {
  auto name_context = context.decl_name_stack().FinishName();
  auto namespace_id = context.AddInst(SemIR::Namespace{
      parse_node, context.GetBuiltinType(SemIR::BuiltinKind::NamespaceType),
      context.name_scopes().Add()});
  context.decl_name_stack().AddNameToLookup(name_context, namespace_id);
  context.decl_name_stack().PopScope();
  return true;
}

}  // namespace Carbon::Check
