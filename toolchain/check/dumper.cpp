// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/dumper.h"

#include "llvm/Support/raw_ostream.h"

namespace Carbon::Check {

auto Dumper::Dump(const Carbon::Lex::TokenizedBuffer& tokens,
                  const Parse::Tree& parse_tree, SemIR::LocId loc_id) -> void {
  if (loc_id.is_node_id()) {
    auto token = parse_tree.node_token(loc_id.node_id());
    auto line = tokens.GetLineNumber(token);
    auto col = tokens.GetColumnNumber(token);
    llvm::errs() << "LocId(line: " << line << ", col: " << col << ")\n";
  } else {
    llvm::errs() << "LocId(invalid)\n";
  }
}

}  // namespace Carbon::Check
