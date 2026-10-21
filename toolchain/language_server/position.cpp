// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/language_server/position.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "toolchain/language_server/sem_ir_index.h"
#include "toolchain/parse/tree.h"
#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::LanguageServer {

auto GetTokenRange(const Lex::TokenizedBuffer& tokens, Lex::TokenIndex start,
                   Lex::TokenIndex end) -> clang::clangd::Range {
  auto start_line = tokens.GetLine(start);
  auto start_col = tokens.GetColumnNumber(start);
  auto [end_line, end_col] = tokens.GetEndLoc(end);
  return clang::clangd::Range{
      .start = {.line = start_line.index, .character = start_col - 1},
      .end = {.line = end_line.index, .character = end_col - 1},
  };
}

// Returns the 0-based (line, column) where a token starts, for comparison
// against an LSP position.
static auto GetTokenStart(const Lex::TokenizedBuffer& tokens,
                          Lex::TokenIndex token) -> std::pair<int, int> {
  return {tokens.GetLine(token).index, tokens.GetColumnNumber(token) - 1};
}

auto FindToken(const Lex::TokenizedBuffer& tokens,
               const clang::clangd::Position& position) -> Lex::TokenIndex {
  std::pair<int, int> target = {position.line, position.character};

  // Tokens are in source order, so find the first one starting after
  // `position`; only the token before that can contain it. Searching on
  // (line, column) avoids converting the position to a byte offset, which would
  // mean rescanning the text for line boundaries.
  //
  // `tokens()` excludes the recovery tokens added after lexing finished, which
  // matters here: those are appended rather than inserted in source order, so
  // including them would break the ordering this search relies on.
  auto range = tokens.tokens();
  auto after = llvm::partition_point(range, [&](Lex::TokenIndex token) {
    return GetTokenStart(tokens, token) <= target;
  });
  if (after == range.begin()) {
    return Lex::TokenIndex::None;
  }

  // The position is in that token only if it's also before the token's end;
  // otherwise it falls in whitespace or a comment.
  auto token = *std::prev(after);
  auto [end_line, end_col] = tokens.GetEndLoc(token);
  if (target < std::pair<int, int>(end_line.index, end_col - 1)) {
    return token;
  }
  return Lex::TokenIndex::None;
}

auto FindPositionInfo(const Context::File& file,
                      const clang::clangd::Position& position) -> PositionInfo {
  const auto* sem_ir = file.sem_ir();
  const auto* index = file.sem_ir_index();
  if (!sem_ir || !index) {
    return {};
  }

  auto token = FindToken(file.tokens(), position);
  if (!token.has_value()) {
    return {};
  }

  PositionInfo info = {.file = &file, .token = token};
  auto insts = index->InstsForToken(token);
  for (auto inst_id : insts) {
    // Prefer a name reference: a token such as the name in `fn F()` also has
    // instructions for the declaration itself, but a request at a name is about
    // the name.
    if (sem_ir->insts().Is<SemIR::NameRef>(inst_id)) {
      info.inst_id = inst_id;
      return info;
    }
  }
  if (!insts.empty()) {
    info.inst_id = insts.front();
  }
  return info;
}

auto GetReferencedInst(const SemIR::File& sem_ir, SemIR::InstId inst_id)
    -> SemIR::InstId {
  if (auto name_ref = sem_ir.insts().TryGetAs<SemIR::NameRef>(inst_id)) {
    return name_ref->value_id;
  }
  return inst_id;
}

auto GetInstNameToken(const Context::File& file, SemIR::InstId inst_id)
    -> Lex::TokenIndex {
  const auto* sem_ir = file.sem_ir();
  if (!sem_ir || !inst_id.has_value()) {
    return Lex::TokenIndex::None;
  }
  auto loc_id = sem_ir->insts().GetCanonicalLocId(inst_id);
  if (loc_id.kind() != SemIR::LocId::Kind::NodeId) {
    // Imported from another file, which we can't yet name a location in.
    return Lex::TokenIndex::None;
  }
  auto node_id = loc_id.node_id();
  if (!node_id.has_value()) {
    return Lex::TokenIndex::None;
  }
  return GetNameToken(file.tree_and_subtrees(), node_id);
}

auto GetInstLocation(const Context::File& file, SemIR::InstId inst_id)
    -> std::optional<clang::clangd::Location> {
  auto token = GetInstNameToken(file, inst_id);
  if (!token.has_value()) {
    return std::nullopt;
  }
  return clang::clangd::Location{.uri = file.uri(),
                                 .range = GetTokenRange(file.tokens(), token)};
}

}  // namespace Carbon::LanguageServer
