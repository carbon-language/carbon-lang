// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/parse/extract.h"

#include <initializer_list>
#include <optional>

#include "common/find.h"
#include "llvm/ADT/StringExtras.h"
#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/node_category.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

auto NodeExtractor::MatchesNodeIdForKind(NodeKind expected_kind) const -> bool {
  if (at_end()) {
    MaybeTrace("NodeIdForKind error: no more children, expected {0}\n",
               expected_kind);
    return false;
  } else if (kind() != expected_kind) {
    MaybeTrace("NodeIdForKind error: wrong kind {0}, expected {1}\n", kind(),
               expected_kind);
    return false;
  }
  MaybeTrace("NodeIdForKind: {0} consumed\n", expected_kind);
  return true;
}

auto NodeExtractor::MatchesNodeIdInCategory(NodeCategory category) const
    -> bool {
  if (at_end()) {
    MaybeTrace("NodeIdInCategory {0} error: no more children\n", category);
    return false;
  } else if (!kind().category().HasAnyOf(category)) {
    MaybeTrace("NodeIdInCategory {0} error: kind {1} doesn't match\n", category,
               kind());
    return false;
  }
  MaybeTrace("NodeIdInCategory {0}: kind {1} consumed\n", category, kind());
  return true;
}

auto NodeExtractor::MatchesNodeIdOneOf(
    std::initializer_list<NodeKind> kinds) const -> bool {
  auto trace_kinds = [&] {
    llvm::ListSeparator sep(" or ");
    for (auto kind : kinds) {
      *trace_ << sep << kind;
    }
  };
  auto node_kind = kind();
  if (at_end()) {
    if (trace_) {
      *trace_ << "NodeIdOneOf error: no more children, expected ";
      trace_kinds();
      *trace_ << "\n";
    }
    return false;
  } else if (!Contains(kinds, node_kind)) {
    if (trace_) {
      *trace_ << "NodeIdOneOf error: wrong kind " << node_kind << ", expected ";
      trace_kinds();
      *trace_ << "\n";
    }
    return false;
  }
  if (trace_) {
    *trace_ << "NodeIdOneOf ";
    trace_kinds();
    *trace_ << ": " << node_kind << " consumed\n";
  }
  return true;
}

auto NodeExtractor::MatchesTokenKind(Lex::TokenKind expected_kind) const
    -> bool {
  if (!node_id_.has_value()) {
    MaybeTrace("Token {0} expected but processing root node\n", expected_kind);
    return false;
  }
  if (token_kind() != expected_kind) {
    if (trace_) {
      *trace_ << "Token " << expected_kind << " expected for "
              << tree_->tree().node_kind(node_id_) << ", found " << token_kind()
              << "\n";
    }
    return false;
  }
  return true;
}

// Instantiate for `File`, which is the root container type.
template auto TreeAndSubtrees::TryExtractNodeFromChildren<File>(
    NodeId node_id,
    llvm::iterator_range<TreeAndSubtrees::SiblingIterator> children,
    ErrorBuilder* trace) const -> std::optional<File>;

auto TreeAndSubtrees::ExtractFile() const -> File {
  return ExtractNodeFromChildren<File>(NodeId::None, roots());
}

}  // namespace Carbon::Parse
