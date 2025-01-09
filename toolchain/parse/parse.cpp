// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/parse.h"

#include "common/check.h"
#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

namespace {
// Applies the `position_` to the `last_byte_offset` returned by `ConvertLoc`.
class TokenDiagnosticConverterForParse : public Lex::TokenDiagnosticConverter {
 public:
  explicit TokenDiagnosticConverterForParse(Lex::TokenizedBuffer* tokens,
                                            Lex::TokenIterator* position)
      : Lex::TokenDiagnosticConverter(tokens), position_(position) {}

  auto ConvertLoc(Lex::TokenIndex token, ContextFnT context_fn) const
      -> std::pair<DiagnosticLoc, int32_t> override {
    auto loc = Lex::TokenDiagnosticConverter::ConvertLoc(token, context_fn);
    loc.second = std::max(loc.second, tokens().GetByteOffset(**position_));
    return loc;
  }

 private:
  // The position in `Parse()`.
  Lex::TokenIterator* position_;
};
}  // namespace

auto HandleInvalid(Context& context) -> void {
  CARBON_FATAL("The Invalid state shouldn't be on the stack: {0}",
               context.PopState());
}

auto Parse(Lex::TokenizedBuffer& tokens, DiagnosticConsumer& consumer,
           llvm::raw_ostream* vlog_stream) -> Tree {
  // This owns the parse position so that the converter can know where the
  // context is.
  Lex::TokenIterator position = tokens.tokens().begin();

  TokenDiagnosticConverterForParse converter(&tokens, &position);
  ErrorTrackingDiagnosticConsumer err_tracker(consumer);
  Lex::TokenDiagnosticEmitter emitter(converter, err_tracker);

  // Delegate to the parser.
  Tree tree(tokens);
  Context context(tree, tokens, emitter, &position, vlog_stream);
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

  // Mark the tree as potentially having errors if there were errors coming in
  // from the tokenized buffer or we diagnosed new errors.
  tree.set_has_errors(tokens.has_errors() || err_tracker.seen_error());

  if (auto verify = tree.Verify(); !verify.ok()) {
    // TODO: This is temporarily printing to stderr directly during development.
    // If we can, restrict this to a subtree with the error and add it to the
    // stack trace (such as with PrettyStackTraceFunction). Otherwise, switch
    // back to vlog_stream prior to broader distribution so that end users are
    // hopefully comfortable copy-pasting stderr when there are bugs in tree
    // construction.
    tree.Print(llvm::errs());
    CARBON_FATAL("Invalid tree returned by Parse(): {0}", verify.error());
  }
  return tree;
}

}  // namespace Carbon::Parse
