// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_POSITION_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_POSITION_H_

#include "clang-tools-extra/clangd/Protocol.h"
#include "toolchain/language_server/context.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::LanguageServer {

// Returns the range covering a closed interval of tokens.
auto GetTokenRange(const Lex::TokenizedBuffer& tokens, Lex::TokenIndex start,
                   Lex::TokenIndex end) -> clang::clangd::Range;

// Returns the range covering a single token.
inline auto GetTokenRange(const Lex::TokenizedBuffer& tokens,
                          Lex::TokenIndex token) -> clang::clangd::Range {
  return GetTokenRange(tokens, token, token);
}

// Returns the token containing `position`, or `None` if the position isn't
// within a token. Positions in whitespace and comments produce `None`, which is
// why hovering over blank space yields no result rather than the nearest token.
auto FindToken(const Lex::TokenizedBuffer& tokens,
               const clang::clangd::Position& position) -> Lex::TokenIndex;

// What a request at a source position refers to.
struct PositionInfo {
  // The token at the position, or `None` if there isn't one.
  Lex::TokenIndex token = Lex::TokenIndex::None;

  // The instruction to answer the request from, or `None` if the position has
  // no instruction. A position can have several instructions; this is the one
  // that names something, if any, because that's what these requests are about.
  SemIR::InstId inst_id = SemIR::InstId::None;

  auto has_inst() const -> bool { return inst_id.has_value(); }
};

// Resolves a position to the token and instruction it refers to. Returns an
// empty result if the file has no checked IR, or the position isn't in a token,
// or the token produced no instructions.
auto FindPositionInfo(const Context::File& file,
                      const clang::clangd::Position& position) -> PositionInfo;

// Returns the instruction that `inst_id` names, which for a name reference is
// the referenced entity and otherwise is `inst_id` itself. This is what
// `definition` and `references` are both anchored on: it gives every mention of
// an entity, including its declaration, the same identity.
auto GetReferencedInst(const SemIR::File& sem_ir, SemIR::InstId inst_id)
    -> SemIR::InstId;

// Returns the token naming `inst_id`, or `None` if it isn't located in this
// file. Two instructions naming the same token denote the same entity, which is
// how a declaration and the references to it are matched up: they don't share
// an instruction, but they do share a name.
auto GetInstNameToken(const Context::File& file, SemIR::InstId inst_id)
    -> Lex::TokenIndex;

// Returns the location of `inst_id` in `file`, or nullopt if it isn't located
// in this file. Instructions imported from elsewhere have no location here.
auto GetInstLocation(const Context::File& file, SemIR::InstId inst_id)
    -> std::optional<clang::clangd::Location>;

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_POSITION_H_
