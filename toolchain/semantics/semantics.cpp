// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/semantics/semantics.h"

#include <optional>

#include "common/check.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lexer/tokenized_buffer.h"
#include "toolchain/parser/parse_node_kind.h"

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

auto Semantics::AddFunction(DiagnosticEmitter<ParseTree::Node> emitter,
                            llvm::StringMap<NamedEntity>& name_scope,
                            llvm::StringRef name) -> Function& {
  int32_t index = functions_.size();
  functions_.resize(index + 1);
  auto [it, success] = name_scope.insert(
      {name, {.kind = NamedEntity::Kind::Function, .index = index}});
  if (!success) {
    emitter.EmitError<NameConflict>(
        fn.name_node,
        {.name = fn_name, .conflict = GetEntityLocation(name_scope[fn_name])});
    return;
  }
  semantics_->functions_.push_back(fn);
}

auto Semantics::Analyzer::GetEntityLocation(NamedEntity entity)
    -> Diagnostic::Location {
  switch (entity.kind) {
    case NamedEntity::Kind::Function: {
      Function fn = semantics_->functions_[entity.index];
      return translator_.GetLocation(fn.name_node);
    }
  }
}

}  // namespace Carbon
