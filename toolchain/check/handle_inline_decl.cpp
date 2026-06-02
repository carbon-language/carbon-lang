// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "toolchain/check/context.h"
#include "toolchain/check/cpp/generate_ast.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/handle.h"
#include "toolchain/check/inst.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/typed_nodes.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::Check {

auto HandleParseNode(Context& /*context*/,
                     Parse::InlineIntroducerId /*node_id*/) -> bool {
  return true;
}

auto HandleParseNode(Context& context, Parse::InlineCppDeclId node_id) -> bool {
  auto body_id = context.node_stack()
                     .PopForSoloNodeId<Parse::NodeKind::InlineImportBody>();

  // If there are no Cpp imports prior to this point, the `Cpp` expression after
  // `import` will have already produced an error.
  //
  // TODO: It'd be nice to produce a clearer error saying to insert an `import
  // Cpp` in a file that uses `inline Cpp` and doesn't otherwise import anything
  // from package `Cpp`.
  auto cpp_id = context.constant_values().Get(
      context.node_stack().Pop<Parse::NodeKind::CppNameExpr>());
  if (cpp_id == SemIR::ErrorInst::ConstantId) {
    return true;
  }

  // If Clang initialization catastrophically failed, skip the inline fragment.
  if (!context.cpp_context()) {
    // We should have already diagnosed an error initializing Clang.
    auto cpp_scope_id = context.constant_values()
                            .GetInstAs<SemIR::Namespace>(cpp_id)
                            .name_scope_id;
    CARBON_CHECK(context.name_scopes().Get(cpp_scope_id).has_error(),
                 "Have valid `Cpp` scope but no Cpp context");
    return true;
  }

  auto string_token = context.parse_tree().node_token(body_id);
  auto string_value_id = context.tokens().GetStringLiteralValue(string_token);
  auto string_value = context.string_literal_values().Get(string_value_id);

  if (context.scope_stack().PeekIndex() != ScopeIndex::Package) {
    CARBON_DIAGNOSTIC(InlineDeclNotAtFileScope, Error,
                      "`inline Cpp` declaration not at file scope");
    context.emitter().Emit(node_id, InlineDeclNotAtFileScope);
    return true;
  }

  InjectAstFromInlineCode(context, SemIR::LocId(body_id), string_value);
  AddInst<SemIR::InlineCppDecl>(context, node_id, {.text_id = string_value_id});
  return true;
}

}  // namespace Carbon::Check
