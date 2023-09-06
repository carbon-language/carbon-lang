// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
#define CARBON_TOOLCHAIN_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_

#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

class NodeLocationTranslator : public DiagnosticLocationTranslator<Node> {
 public:
  explicit NodeLocationTranslator(const Lex::TokenizedBuffer* tokens,
                                  const Tree* parse_tree)
      : token_translator_(tokens), parse_tree_(parse_tree) {}

  // Map the given token into a diagnostic location.
  auto GetLocation(Node node) -> DiagnosticLocation override {
    return token_translator_.GetLocation(parse_tree_->node_token(node));
  }

 private:
  Lex::TokenLocationTranslator token_translator_;
  const Tree* parse_tree_;
};

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TREE_NODE_LOCATION_TRANSLATOR_H_
