// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"

namespace Carbon {

void SemanticsIR::Block::Add(llvm::StringRef name, Node named_entity) {
  ordering_.push_back(named_entity);
  name_lookup_.insert({name, named_entity});
}

auto SemanticsIR::AddFunction(Block& block, ParseTree::Node decl_node,
                              ParseTree::Node name_node)
    -> Semantics::Function& {
  int32_t index = functions_.size();
  functions_.push_back(Semantics::Function(decl_node, name_node));
  block.Add(parse_tree_->GetNodeText(name_node),
            Node(Node::Kind::Function, index));
  return functions_[index];
}

}  // namespace Carbon
