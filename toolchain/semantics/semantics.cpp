// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics.h"

#include <optional>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_tree_node_location_translator.h"

namespace Carbon {

struct NameConflict : DiagnosticBase<NameConflict> {
  static constexpr llvm::StringLiteral ShortName = "semantics-name-conflict";

  auto Format() -> std::string {
    return llvm::formatv(
        "Name conflict for `{0}`; previously declared at {1}:{2}:{3}.", name,
        conflict.file_name, conflict.line_number, conflict.column_number);
  }

  llvm::StringRef name;
  Diagnostic::Location conflict;
};

auto Semantics::AddFunction(DiagnosticEmitter<ParseTree::Node>& emitter,
                            llvm::StringMap<NamedEntity>& name_scope,
                            ParseTree::Node decl_node,
                            ParseTree::Node name_node) -> Function& {
  int32_t index = functions_.size();
  functions_.push_back(Function(decl_node, name_node));
  llvm::StringRef name = parse_tree_->GetNodeText(name_node);
  fprintf(stderr, "Adding %s\n", name.str().c_str());
  auto [it, success] = name_scope.insert(
      {name, NamedEntity(NamedEntity::Kind::Function, index)});
  // TODO: Probably need to distinguish between declaration and definition.
  if (!success) {
    emitter.EmitError<NameConflict>(
        name_node,
        {.name = name, .conflict = GetEntityLocation(name_scope[name])});
  }
  return functions_[index];
}

auto Semantics::GetEntityLocation(NamedEntity entity) -> Diagnostic::Location {
  switch (entity.kind_) {
    case NamedEntity::Kind::Function:
      return ParseTreeNodeLocationTranslator(*parse_tree_)
          .GetLocation(functions_[entity.index_].name_node());
    case NamedEntity::Kind::Invalid:
      FATAL() << "Encountered invalid NamedEntity";
  }
}

}  // namespace Carbon
