// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
#define CARBON_TOOLCHAIN_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_

#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

class NodeLocationTranslator : public DiagnosticLocationTranslator<NodeId> {
 public:
  explicit NodeLocationTranslator(const Lex::TokenizedBuffer* tokens,
                                  llvm::StringRef filename,
                                  const Tree* parse_tree)
      : token_translator_(tokens),
        filename_(filename),
        parse_tree_(parse_tree) {}

  // Map the given token into a diagnostic location.
  auto GetLocation(NodeId node) -> DiagnosticLocation override {
    // Support the invalid token as a way to emit only the filename, when there
    // is no line association.
    if (!node.is_valid()) {
      return {.file_name = filename_};
    }

    // Retrieve all tokens that descend from this node (including the root).
    // This will always represent a contiguous chunk of source b/c XXXX. Then
    // construct a location that represents this range.
    DiagnosticLocation start_loc =
        token_translator_.GetLocation(parse_tree_->node_token(node));
    DiagnosticLocation end_loc = start_loc;
    for (Node descendant : parse_tree_->postorder(node)) {
      Lex::Token token = parse_tree_->node_token(descendant);
      if (!token.is_valid()) {
        continue;
      }
      DiagnosticLocation loc = token_translator_.GetLocation(token);
      if (loc.line_number < start_loc.line_number ||
          (loc.line_number == start_loc.line_number &&
           loc.column_number < start_loc.column_number)) {
        start_loc = loc;
      } else if (loc.line_number > end_loc.line_number ||
                 (loc.line_number == end_loc.line_number &&
                  loc.column_number > end_loc.column_number)) {
        end_loc = loc;
      }
    }

    // For multiline locations we simply return the rest of the line for now
    // since true multiline locations are not yet supported.
    if (start_loc.line_number != end_loc.line_number) {
      start_loc.length = start_loc.line.size() - start_loc.column_number + 1;
    } else {
      if (start_loc.column_number != end_loc.column_number) {
        start_loc.length = end_loc.column_number - start_loc.column_number + 1;
      }
    }
    return start_loc;
  }

 private:
  Lex::TokenLocationTranslator token_translator_;
  llvm::StringRef filename_;
  const Tree* parse_tree_;
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
