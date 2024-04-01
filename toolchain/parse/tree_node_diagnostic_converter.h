// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TREE_NODE_DIAGNOSTIC_CONVERTER_H_
#define CARBON_TOOLCHAIN_PARSE_TREE_NODE_DIAGNOSTIC_CONVERTER_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

class NodeLoc {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor)
  NodeLoc(NodeId node_id) : NodeLoc(node_id, false) {}
  NodeLoc(NodeId node_id, bool token_only)
      : node_id_(node_id), token_only_(token_only) {}
  // TODO: Have some other way of representing diagnostic that applies to a file
  // as a whole.
  // NOLINTNEXTLINE(google-explicit-constructor)
  NodeLoc(InvalidNodeId node_id) : NodeLoc(node_id, false) {}

  auto node_id() const -> NodeId { return node_id_; }
  auto token_only() const -> bool { return token_only_; }

 private:
  NodeId node_id_;
  bool token_only_;
};

class NodeLocConverter : public DiagnosticConverter<NodeLoc> {
 public:
  explicit NodeLocConverter(const Lex::TokenizedBuffer* tokens,
                            llvm::StringRef filename, const Tree* parse_tree)
      : token_converter_(tokens),
        filename_(filename),
        parse_tree_(parse_tree) {}

  // Map the given token into a diagnostic location.
  auto ConvertLoc(NodeLoc node_loc, ContextFnT context_fn) const
      -> DiagnosticLoc override {
    // Support the invalid token as a way to emit only the filename, when there
    // is no line association.
    if (!node_loc.node_id().is_valid()) {
      return {.filename = filename_};
    }

    if (node_loc.token_only()) {
      return token_converter_.ConvertLoc(
          parse_tree_->node_token(node_loc.node_id()), context_fn);
    }

    // Construct a location that encompasses all tokens that descend from this
    // node (including the root).
    Lex::TokenIndex start_token = parse_tree_->node_token(node_loc.node_id());
    Lex::TokenIndex end_token = start_token;
    for (NodeId desc : parse_tree_->postorder(node_loc.node_id())) {
      Lex::TokenIndex desc_token = parse_tree_->node_token(desc);
      if (!desc_token.is_valid()) {
        continue;
      }
      if (desc_token < start_token) {
        start_token = desc_token;
      } else if (desc_token > end_token) {
        end_token = desc_token;
      }
    }
    DiagnosticLoc start_loc =
        token_converter_.ConvertLoc(start_token, context_fn);
    if (start_token == end_token) {
      return start_loc;
    }
    DiagnosticLoc end_loc = token_converter_.ConvertLoc(end_token, context_fn);
    // For multiline locations we simply return the rest of the line for now
    // since true multiline locations are not yet supported.
    if (start_loc.line_number != end_loc.line_number) {
      start_loc.length = start_loc.line.size() - start_loc.column_number + 1;
    } else {
      if (start_loc.column_number != end_loc.column_number) {
        start_loc.length =
            end_loc.column_number + end_loc.length - start_loc.column_number;
      }
    }
    return start_loc;
  }

 private:
  Lex::TokenDiagnosticConverter token_converter_;
  llvm::StringRef filename_;
  const Tree* parse_tree_;
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TREE_NODE_DIAGNOSTIC_CONVERTER_H_
