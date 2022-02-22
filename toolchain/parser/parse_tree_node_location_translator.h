// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_PARSER_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
#define TOOLCHAIN_PARSER_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_

#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon {

// Provides diagnostic locations for a ParseTree Node.
class ParseTreeNodeLocationTranslator
    : public DiagnosticLocationTranslator<ParseTree::Node> {
 public:
  explicit ParseTreeNodeLocationTranslator(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  // Translate a particular node to a location.
  auto GetLocation(ParseTree::Node loc) -> Diagnostic::Location override {
    auto token = parse_tree_->GetNodeToken(loc);
    TokenizedBuffer::TokenLocationTranslator translator(
        *parse_tree_->tokens_, /*last_line_lexed_to_column=*/nullptr);
    return translator.GetLocation(token);
  }

 private:
  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_PARSER_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
