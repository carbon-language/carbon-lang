// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Declare handlers for each parse state.
#define CARBON_PARSE_STATE(Name) auto Handle##Name(Context& context) -> void;
#include "toolchain/parse/state.def"

auto HandleInvalid(Context& context) -> void {
  CARBON_FATAL() << "The Invalid state shouldn't be on the stack: "
                 << context.PopState();
}

auto Parse(Lex::TokenizedBuffer& tokens, DiagnosticConsumer& consumer,
           llvm::raw_ostream* vlog_stream) -> Tree {
  Lex::TokenLocationTranslator translator(&tokens);
  Lex::TokenDiagnosticEmitter emitter(translator, consumer);

  // Delegate to the parser.
  Tree tree(tokens);
  Context context(tree, tokens, emitter, vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  context.AddLeafNode(NodeKind::FileStart,
                      context.ConsumeChecked(Lex::TokenKind::FileStart));

  context.PushState(State::DeclScopeLoop);

  while (!context.state_stack().empty()) {
    switch (context.state_stack().back().state) {
#define CARBON_PARSE_STATE(Name) \
  case State::Name:              \
    Handle##Name(context);       \
    break;
#include "toolchain/parse/state.def"
    }
  }

  context.AddLeafNode(NodeKind::FileEnd, *context.position());

  if (auto verify = tree.Verify(); !verify.ok()) {
    // TODO: This is temporarily printing to stderr directly during development.
    // If we can, restrict this to a subtree with the error and add it to the
    // stack trace (such as with PrettyStackTraceFunction). Otherwise, switch
    // back to vlog_stream prior to broader distribution so that end users are
    // hopefully comfortable copy-pasting stderr when there are bugs in tree
    // construction.
    tree.Print(llvm::errs());
    CARBON_FATAL() << "Invalid tree returned by Parse(): " << verify.error();
  }
  return tree;
}

}  // namespace Carbon::Parse
