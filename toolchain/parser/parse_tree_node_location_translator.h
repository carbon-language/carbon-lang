// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSER_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
#define CARBON_TOOLCHAIN_PARSER_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_

#include "toolchain/parser/parse_tree.h"

namespace Carbon {

class ParseTreeNodeLocationTranslator
    : public DiagnosticLocationTranslator<ParseTree::Node> {
 public:
  explicit ParseTreeNodeLocationTranslator(const TokenizedBuffer* tokens,
                                           const ParseTree* parse_tree)
      : token_translator_(tokens, nullptr), parse_tree_(parse_tree) {}

  // Map the given token into a diagnostic location.
  auto GetLocation(ParseTree::Node node) -> DiagnosticLocation override {
    return token_translator_.GetLocation(parse_tree_->node_token(node));
  }

 private:
  TokenizedBuffer::TokenLocationTranslator token_translator_;
  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_PARSER_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
