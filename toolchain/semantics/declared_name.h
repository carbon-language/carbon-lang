// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_DECLARED_NAME_H_
#define TOOLCHAIN_SEMANTICS_DECLARED_NAME_H_

#include "common/ostream.h"
#include "toolchain/parser/parse_tree.h"

namespace Carbon::Semantics {

// Semantic information for a function.
class DeclaredName {
 public:
  DeclaredName(llvm::StringRef str, ParseTree::Node node)
      : str_(str), node_(node) {}

  void Print(llvm::raw_ostream& out) const { out << str_; }

  auto str() const -> llvm::StringRef { return str_; }

 private:
  llvm::StringRef str_;
  ParseTree::Node node_;
};

}  // namespace Carbon::Semantics

#endif  // TOOLCHAIN_SEMANTICS_DECLARED_NAME_H_
