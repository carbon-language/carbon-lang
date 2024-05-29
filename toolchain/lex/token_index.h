// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_TOKEN_INDEX_H_
#define CARBON_TOOLCHAIN_LEX_TOKEN_INDEX_H_

#include "toolchain/base/index_base.h"

namespace Carbon::Lex {

// A lightweight handle to a lexed token in a `TokenizedBuffer`.
//
// `TokenIndex` objects are designed to be passed by value, not reference or
// pointer. They are also designed to be small and efficient to store in data
// structures.
//
// `TokenIndex` objects from the same `TokenizedBuffer` can be compared with
// each other, both for being the same token within the buffer, and to establish
// relative position within the token stream that has been lexed out of the
// buffer. `TokenIndex` objects from different `TokenizedBuffer`s cannot be
// meaningfully compared.
//
// All other APIs to query a `TokenIndex` are on the `TokenizedBuffer`.
struct TokenIndex : public IndexBase {
  static const TokenIndex Invalid;
  // Comments aren't tokenized, so this is the first token after FileStart.
  static const TokenIndex FirstNonCommentToken;
  using IndexBase::IndexBase;
};

constexpr TokenIndex TokenIndex::Invalid(TokenIndex::InvalidIndex);
constexpr TokenIndex TokenIndex::FirstNonCommentToken(1);

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_TOKEN_INDEX_H_
