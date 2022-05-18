// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
#define CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_

#include "llvm/ADT/SmallVector.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/semantics/nodes/expression_statement.h"
#include "toolchain/semantics/nodes/function.h"
#include "toolchain/semantics/nodes/infix_operator.h"
#include "toolchain/semantics/nodes/literal.h"
#include "toolchain/semantics/nodes/meta_node_block.h"
#include "toolchain/semantics/nodes/pattern_binding.h"
#include "toolchain/semantics/nodes/return.h"

namespace Carbon::Testing {
class SemanticsIRForTest;
}  // namespace Carbon::Testing

namespace Carbon {

// Provides semantic analysis on a ParseTree.
class SemanticsIR {
 public:
  // File-level declarations.
  auto root_block() const -> const Semantics::DeclarationBlock& {
    return *root_block_;
  }

  // Debug printer for the parse tree.
  void Print(llvm::raw_ostream& out, ParseTree::Node node) const;

  // Debug printers for meta nodes.
  void Print(llvm::raw_ostream& out, Semantics::Declaration decl) const;
  void Print(llvm::raw_ostream& out, Semantics::Expression expr) const;
  void Print(llvm::raw_ostream& out, Semantics::Statement stmt) const;

  // Debug printers for other nodes.
  void Print(llvm::raw_ostream& out, const Semantics::DeclaredName& name) const;
  void Print(llvm::raw_ostream& out,
             const Semantics::ExpressionStatement& expr) const;
  void Print(llvm::raw_ostream& out, const Semantics::Function& function) const;
  void Print(llvm::raw_ostream& out, const Semantics::InfixOperator& op) const;
  void Print(llvm::raw_ostream& out, const Semantics::Literal& literal) const;
  void Print(llvm::raw_ostream& out,
             const Semantics::PatternBinding& binding) const;
  void Print(llvm::raw_ostream& out, const Semantics::Return& ret) const;
  void Print(llvm::raw_ostream& out,
             const Semantics::StatementBlock& block) const;

 private:
  friend class SemanticsIRFactory;
  friend class Testing::SemanticsIRForTest;

  explicit SemanticsIR(const ParseTree& parse_tree)
      : parse_tree_(&parse_tree) {}

  Semantics::DeclarationStore declarations_;
  Semantics::ExpressionStore expressions_;
  Semantics::StatementStore statements_;

  // The file-level block. Only assigned after initialization is complete.
  llvm::Optional<Semantics::DeclarationBlock> root_block_;

  const ParseTree* parse_tree_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEMANTICS_SEMANTICS_IR_H_
