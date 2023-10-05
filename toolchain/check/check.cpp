// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/check.h"

#include "toolchain/base/pretty_stack_trace_function.h"
#include "toolchain/check/context.h"
#include "toolchain/parse/tree_node_location_translator.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

auto CheckParseTree(const SemIR::File& builtin_ir,
                    const Lex::TokenizedBuffer& tokens,
                    const Parse::Tree& parse_tree, DiagnosticConsumer& consumer,
                    llvm::raw_ostream* vlog_stream) -> SemIR::File {
  auto semantics_ir = SemIR::File(tokens.filename().str(), &builtin_ir);

  Parse::NodeLocationTranslator translator(&tokens, &parse_tree);
  ErrorTrackingDiagnosticConsumer err_tracker(consumer);
  DiagnosticEmitter<Parse::Node> emitter(translator, err_tracker);

  Check::Context context(tokens, emitter, parse_tree, semantics_ir,
                         vlog_stream);
  PrettyStackTraceFunction context_dumper(
      [&](llvm::raw_ostream& output) { context.PrintForStackDump(output); });

  // Add a block for the Parse::Tree.
  context.node_block_stack().Push();
  context.PushScope();

  // Loops over all nodes in the tree. On some errors, this may return early,
  // for example if an unrecoverable state is encountered.
  for (auto parse_node : parse_tree.postorder()) {
    // clang warns on unhandled enum values; clang-tidy is incorrect here.
    // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
    switch (auto parse_kind = parse_tree.node_kind(parse_node)) {
#define CARBON_PARSE_NODE_KIND(Name)                                         \
  case Parse::NodeKind::Name: {                                              \
    if (!Check::Handle##Name(context, parse_node)) {                         \
      CARBON_CHECK(err_tracker.seen_error())                                 \
          << "Handle" #Name " returned false without printing a diagnostic"; \
      semantics_ir.set_has_errors(true);                                     \
      return semantics_ir;                                                   \
    }                                                                        \
    break;                                                                   \
  }
#include "toolchain/parse/node_kind.def"
    }
  }

  // Pop information for the file-level scope.
  semantics_ir.set_top_node_block_id(context.node_block_stack().Pop());
  context.PopScope();

  context.VerifyOnFinish();

  semantics_ir.set_has_errors(err_tracker.seen_error());

#ifndef NDEBUG
  if (auto verify = semantics_ir.Verify(); !verify.ok()) {
    CARBON_FATAL() << semantics_ir
                   << "Built invalid semantics IR: " << verify.error() << "\n";
  }
#endif

  return semantics_ir;
}

}  // namespace Carbon::Check
