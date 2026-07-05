// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
#define CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/format/style.h"
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

// Formatting information about a binary-operator parse node, derived from its
// node kind. See `OperatorInfoForNodeKind`.
struct OperatorInfo {
  // The penalty for breaking the line after this operator (before its right
  // operand), or -1 if the node is not a binary operator. Looser-binding
  // operators get a lower penalty, so the solver breaks the loosest one first,
  // mirroring clang-format, which uses the operator precedence level as the
  // penalty. Assignment operators use the assignment break penalty.
  int break_penalty = -1;
  // Whether the operator's operands are aligned with each other when the
  // expression wraps (clang-format's `AlignOperands`). True for every binary
  // operator except the assignment family, whose right-hand side is instead
  // continuation-indented.
  bool aligns_operands = false;
};

// Returns the formatting information for a binary-operator node kind (any
// `InfixOperator*` or `ShortCircuitOperator*`). For every other node kind the
// result's `break_penalty` is -1.
auto OperatorInfoForNodeKind(Parse::NodeKind kind, const Style& style)
    -> OperatorInfo;

// If `kind` is a member-access node (`MemberAccessExpr` for `.` or
// `PointerMemberAccessExpr` for `->`), returns the penalty for breaking the
// line before its `.`/`->` token, so a long call chain wraps before each member
// rather than splitting `.member` apart. Returns -1 for any other node kind.
auto MemberAccessBreakPenalty(Parse::NodeKind kind) -> int;

// The break-before penalty for the final link in a member-access chain, which
// is cheaper than a mid-chain link (clang-format's 35 vs 150) so a chain that
// must wrap prefers to break at its end.
constexpr int MemberAccessLastLinkBreakPenalty = 35;

// The formatting information about one token that formatting decisions need.
// This currently holds only what spacing and wrapping require, and grows with
// the formatter. Information the tokenized buffer already provides, such as
// the token's kind, is read from it rather than copied here.
struct TokenInfo {
  // The token's syntactic role, derived from the parse tree.
  TokenRole role = TokenRole::Unknown;
  // The display width, in columns, the token occupies on the line it starts:
  // the byte length of its text, or of just its first physical line for a
  // multi-line token (such as a multi-line string literal).
  // TODO: Use the true display width (encoding-aware) for wide or multi-byte
  // tokens, the way clang-format's `column_width` does.
  // TODO: Also track a multi-line token's last physical line width (the analog
  // of clang-format's `LastLineColumnWidth`): the column after such a token is
  // currently computed from `column_width`, so later tokens on its unwrapped
  // line are placed from a fictitious column.
  int column_width = 0;
  // The number of operator-precedence operand groups (the analog of
  // clang-format's fake parentheses) that open at this token because it is the
  // first token of their operand span, and that close at it because it is the
  // last. Each open starts an operand-alignment scope in the wrapping solver.
  int open_scopes = 0;
  int close_scopes = 0;
  // If non-negative, this token is an infix operator after which a line break
  // is the canonical split point, and the value is the penalty for breaking
  // there; a break *before* such a token is disallowed. -1 means the token is
  // not such an operator; in particular an initializer's `=` is not an
  // infix-operator node and is priced by a `SplitPenalty` fallback instead.
  int break_penalty_after = -1;
  // If non-negative, this token is a member-access `.`/`->` before which a line
  // break is the canonical split point (a break *after* it is disallowed by the
  // token-kind rule); the value is the penalty for breaking there. The last
  // member access in a chain gets the cheaper 35; the rest get 150. -1 means
  // the token is not a member-access operator.
  int break_penalty_before = -1;
  // For a member-access `.`/`->` token, the identity of the call chain it
  // belongs to: the token index of the chain's receiver root (shared by every
  // member access in the chain, since the chain is left-nested). This groups a
  // chain for receiver-anchored indentation and fluent all-or-nothing breaking.
  // -1 means the token is not a member-access operator.
  int member_chain_id = -1;
  // Whether the token is the string literal body of an `inline Cpp` (or
  // `import Cpp inline`) declaration. Such a literal holds C++ and is
  // reformatted even without a `'''cpp` file type indicator.
  bool is_cpp_string = false;
  // Whether the token lies in a minimal error subtree of the parse tree. Such
  // a region is emitted with its original source text (each token's leading
  // gap within the region is copied verbatim rather than reformatted),
  // preserving author intent where the parse is unreliable. See `Formatter`'s
  // constructor.
  bool is_verbatim = false;
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

// Returns whether a line break is allowed before `right`, given the preceding
// token `left`. This rules out only the positions where a break is never
// allowed; the split penalty discourages the merely-undesirable ones.
auto CanBreakBefore(const Lex::TokenizedBuffer& tokens,
                    const TokenInfoStore& token_infos, Lex::TokenIndex left,
                    Lex::TokenIndex right) -> bool;

// Returns the penalty for breaking the line before `right`, given the preceding
// token `left`. Lower is cheaper, so the layout solver prefers low-penalty
// break points. These are starting values, tuned against the test corpus;
// `style` supplies the ones that are configurable.
auto SplitPenalty(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right, const Style& style) -> int;

// Returns the width, in columns, of `line` laid out on a single line: the sum
// of the token widths and the spaces between them, excluding indentation.
auto RenderedWidth(const Lex::TokenizedBuffer& tokens,
                   const TokenInfoStore& token_infos,
                   llvm::ArrayRef<Lex::TokenIndex> line) -> int;

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
