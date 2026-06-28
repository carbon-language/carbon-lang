// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
#define CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_

#include <cstdint>

#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Format {

// The syntactic role of a token, used for formatting decisions that can't be
// made from the token's kind alone. Roles are read from the parse tree, as the
// kind of the parse node that the token is the root of, so the formatter never
// has to guess the way a token-only formatter such as clang-format must.
//
// This starts with only the roles the formatter currently needs, and grows as
// more formatting behavior is added.
enum class TokenRole : uint8_t {
  // No distinguishing role; spacing falls back to token-kind rules.
  Unknown,
  // An opening `(` or `[` that binds tightly to the preceding token: a call, an
  // explicit or implicit parameter list, or a subscript. It takes no space
  // before, for example `F(x)`, `x[i]`, or `F[T: type]`. This is in contrast
  // to a grouping or control-flow `(` such as in `if (cond)`.
  PostfixBracket,
  // A prefix unary operator, the root of a `PrefixOperator*` node. A symbolic
  // one (`*p`, `-x`, `&v`) binds tightly to its operand, taking no space after;
  // a word-based one (`not x`, `const T`) keeps its space.
  PrefixOperator,
  // A postfix unary operator, the root of a `PostfixOperator*` node, binding
  // tightly to its operand and taking no space before, for example the pointer
  // type `p*`.
  PostfixOperator,
  // A member-access `.` or `->`, binding tightly on both sides: `a.b`, `p->x`.
  // This is how the dual `->` is told apart without heuristics: a
  // `PointerMemberAccessExpr` `->` is a member access, while a `ReturnType`
  // `->` keeps the default space on each side, even though they share a token
  // kind.
  MemberAccess,
};

// Maps a parse node kind to the role of the token at the root of that node.
auto RoleForNodeKind(Parse::NodeKind kind) -> TokenRole;

// The formatting information about one token that formatting decisions need.
// This currently holds only what spacing requires, and grows with the
// formatter. Information the tokenized buffer already provides, such as the
// token's kind, is read from it rather than copied here.
struct TokenInfo {
  // The token's syntactic role, derived from the parse tree.
  TokenRole role = TokenRole::Unknown;
};

// The per-token formatting information, indexed by the token. The `Formatter`
// fills the store up front; there is no formatter-specific token type, and a
// `Lex::TokenIndex` plus this store stands in wherever clang-format would pass
// its own `FormatToken`.
using TokenInfoStore = FixedSizeValueStore<Lex::TokenIndex, TokenInfo>;

// Returns the number of spaces to place between the adjacent tokens `left` and
// `right` when they are on the same line, reading kinds from `tokens` and the
// formatter's annotations from `token_infos`. Indentation and line breaks are
// handled separately.
auto SpacesBefore(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right) -> int;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
