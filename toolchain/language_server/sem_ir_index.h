// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LANGUAGE_SERVER_SEM_IR_INDEX_H_
#define CARBON_TOOLCHAIN_LANGUAGE_SERVER_SEM_IR_INDEX_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/tree_and_subtrees.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::LanguageServer {

// Returns the token that identifies `node_id`: the name it declares, if it
// declares one, and otherwise the node's own token.
//
// A declaration's own token is wherever the parser finished it -- the `{` of a
// function body, or the `:` of a binding -- which is not what a user points at
// to mean that declaration. Requests are answered at the name instead, so both
// the index and the locations we report are keyed on it.
auto GetNameToken(const Parse::TreeAndSubtrees& tree_and_subtrees,
                  Parse::NodeId node_id) -> Lex::TokenIndex;

// Maps each token to the instructions that were checked from it.
//
// Position-based requests such as `hover` and `definition` all start from a
// source position, so this is keyed by token rather than by parse node: the
// caller has to find the token for a position anyway, and the parse node is
// recoverable from an instruction's `LocId`. Instructions imported from other
// files are excluded, because their locations refer to another file's parse
// tree rather than to a token in this one.
//
// Build this lazily, on the first query after the file's text changes. Building
// costs the same single pass over the instructions that an unindexed scan
// would, and most text changes are never followed by a query, so building
// eagerly would only add work to the latency-sensitive path that produces
// diagnostics.
class SemIRIndex {
 public:
  explicit SemIRIndex(const SemIR::File& sem_ir,
                      const Parse::TreeAndSubtrees& tree_and_subtrees);

  // Returns the instructions checked from `token`, in `InstId` order. Returns
  // an empty list for a token that produced no instructions, which is common:
  // punctuation and keywords usually contribute to an enclosing instruction
  // rather than producing one of their own.
  auto InstsForToken(Lex::TokenIndex token) const
      -> llvm::ArrayRef<SemIR::InstId>;

 private:
  // Instructions grouped by token, in the compressed-sparse-row layout: the
  // group for token `i` is `insts_[token_starts_[i] .. token_starts_[i + 1])`.
  // `token_starts_` therefore has one more entry than there are tokens.
  //
  // Token indices are dense, so this is built by counting sort in a single pass
  // over the instructions, and looked up in constant time. A hash map would
  // need to handle the many-instructions-per-token case explicitly; here it
  // falls out of the layout.
  llvm::SmallVector<SemIR::InstId, 0> insts_;
  llvm::SmallVector<int32_t, 0> token_starts_;
};

}  // namespace Carbon::LanguageServer

#endif  // CARBON_TOOLCHAIN_LANGUAGE_SERVER_SEM_IR_INDEX_H_
