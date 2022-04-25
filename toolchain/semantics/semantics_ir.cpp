// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics_ir.h"

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/semantics/declared_name.h"

namespace Carbon {

void SemanticsIR::Block::Add(llvm::StringRef name, Node named_entity) {
  nodes_.push_back(named_entity);
  name_lookup_.insert({name, named_entity});
}

void SemanticsIR::AddFunction(Block& block, Semantics::Function function) {
  int32_t index = functions_.size();
  functions_.push_back(function);
  block.Add(function.name().str(), Node(Node::Kind::Function, index));
}

}  // namespace Carbon
