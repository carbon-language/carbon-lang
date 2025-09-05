// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/parse.h"

#include "common/check.h"
#include "common/pretty_stack_trace_function.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"
#include "toolchain/parse/tree_and_subtrees.h"

namespace Carbon::Parse {

auto HandleInvalid(Context& context) -> void {
  CARBON_FATAL("The Invalid state shouldn't be on the stack: {0}",
               context.PopState());
}

auto Parse(Lex::TokenizedBuffer& tokens, ParseOptions options) -> Tree {
  auto* consumer =
      options.consumer ? options.consumer : &Diagnostics::ConsoleConsumer();

  // Delegate to the parser.
  Tree tree(tokens);
  Context context(&tree, &tokens, consumer, options.vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  context.AddLeafNode(NodeKind::FileStart,
                      context.ConsumeChecked(Lex::TokenKind::FileStart));

  context.PushState(StateKind::DeclScopeLoopAsRegular);

  while (!context.state_stack().empty()) {
    switch (context.state_stack().back().kind) {
#define CARBON_PARSE_STATE(Name) \
  case StateKind::Name:          \
    Handle##Name(context);       \
    break;
#include "toolchain/parse/state.def"
    }
  }

  context.AddLeafNode(NodeKind::FileEnd, *context.position());

  // Mark the tree as potentially having errors if there were errors coming in
  // from the tokenized buffer or we diagnosed new errors.
  tree.set_has_errors(tokens.has_errors() || context.has_errors());

  if (options.vlog_stream || options.dump_stream) {
    // Flush diagnostics before printing.
    consumer->Flush();
  }
  CARBON_VLOG_TO(options.vlog_stream, "*** Parse::Tree ***\n{0}", tree);
  if (options.dump_stream) {
    Parse::TreeAndSubtrees tree_and_subtrees(tokens, tree);
    if (options.dump_preorder_parse_tree) {
      tree_and_subtrees.PrintPreorder(*options.dump_stream);
    } else {
      tree_and_subtrees.Print(*options.dump_stream);
    }
  }

  if (auto verify = tree.Verify(); !verify.ok()) {
    // TODO: Consider printing a subtree as part of the error.
    CARBON_FATAL("Invalid tree returned by Parse(): {0}", verify.error());
  }
  return tree;
}

}  // namespace Carbon::Parse
