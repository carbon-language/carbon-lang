// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TOOLCHAIN_SEMANTICS_SEMANTICS_ANALYZER_H_
#define TOOLCHAIN_SEMANTICS_SEMANTICS_ANALYZER_H_

#include <optional>

#include "llvm/ADT/StringMap.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"
#include "toolchain/parser/parse_tree.h"
#include "toolchain/parser/parse_tree_node_location_translator.h"
#include "toolchain/semantics/semantics.h"

namespace Carbon {

// The main semantic analysis entry.
class SemanticAnalyzer {
 public:
  // Runs semantic analysis on a ParseTree in order to generate the resulting
  // Semantics.
  static auto Analyze(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      -> Semantics;

 private:
  SemanticAnalyzer(const ParseTree& parse_tree, DiagnosticConsumer& consumer)
      : semantics_(parse_tree),
        translator_(parse_tree),
        emitter_(translator_, consumer) {}

  // Processes the roots of the ParseTree into the Semantics, transitively
  // handling children.
  void ProcessRoots();

  // Turns a function node from the parse tree into a semantic function node,
  // adding it to the containing scope.
  void ProcessFunctionNode(llvm::StringMap<Semantics::NamedEntity>& name_scope,
                           ParseTree::Node fn_node);

  // Returns the location of an entity. This assists diagnostic output where
  // supplemental locations are provided in formatting.
  auto GetEntityLocation(Semantics::NamedEntity entity) -> Diagnostic::Location;

  Semantics semantics_;
  ParseTreeNodeLocationTranslator translator_;
  DiagnosticEmitter<ParseTree::Node> emitter_;
};

}  // namespace Carbon

#endif  // TOOLCHAIN_SEMANTICS_SEMANTICS_ANALYZER_H_
