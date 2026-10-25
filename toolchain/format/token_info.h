// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
#define CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "toolchain/base/fixed_size_value_store.h"
#include "toolchain/format/style.h"
#include "toolchain/lex/token_index.h"
#include "toolchain/lex/tokenized_buffer.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Format {

// The syntactic role of a token, derived from the parse node whose root is that
// token, for formatting decisions that depend on syntax beyond `TokenKind`.
enum class TokenRole : uint8_t {
  // No distinguishing role; spacing falls back to token-kind rules.
  Unknown,
  // An opening `(` or `[` attached to the preceding token, such as a call,
  // parameter list, type constructor, or subscript: `F(x)`, `array(T, N)`,
  // `x[i]`, `F[T: type]`.
  PostfixBracket,
  // A prefix unary operator or leading `.` designator (`*p`, `-x`, `&v`,
  // `.Red`, `not x`, `const T`). Symbolic forms take no trailing space; word
  // forms keep a space.
  PrefixOperator,
  // A postfix unary operator (`p*`), taking no leading space.
  PostfixOperator,
  // A member-access or name-qualifier `.` or `->` (`a.b`, `p->x`, `fn A.F()`),
  // taking no space on either side.
  MemberAccess,
  // An opening `{` of a declaration or statement block (`fn F() {`,
  // `class C {`, `if (c) {`, `match (x) {`). Its matching `}` terminates the
  // line unless followed by `else` or a separator.
  BlockOpen,
  // An opening `{` of a lambda body (`fn() {`). Its matching `}` stays on the
  // enclosing expression's line.
  LambdaOpen,
  // An opening `{` of an inline struct literal, struct type literal, or struct
  // pattern (`{.a = 1}`, `{.a: i32}`, `{}*`). Always paired with a matching
  // `StructClose` `}`; neither brace ends the line or takes interior spaces.
  StructOpen,
  // A closing `}` paired with `StructOpen`.
  StructClose,
  // An opening `{` of a struct literal, struct type literal, or struct pattern
  // that expands across lines (trailing comma, empty `{ // comment }`, or
  // unclosed recovery brace). Its matching `}` stays on the enclosing line.
  MultilineStructOpen,
  // A comma that ends a line inside an expanded brace list (`choice`
  // alternatives or fields of a trailing-comma struct).
  BlockComma,
};

// Maps a parse node kind to the role of the token at the root of that node.
auto RoleForNodeKind(Parse::NodeKind kind) -> TokenRole;

// Wrapping behavior for an infix or constraint operator parse node.
struct OperatorInfo {
  // Penalty for breaking after this operator (before its right operand), or -1
  // if the node is not such an operator. Looser-binding operators receive lower
  // penalties so the solver breaks outer operators before tighter inner ones.
  int break_penalty = -1;
  // Whether wrapped operands of this node's subtree align with its first
  // operand rather than using continuation indentation.
  bool aligns_operands = false;
};

// Returns operator wrapping metadata for binary and constraint operator nodes,
// or `{.break_penalty = -1}` for all other node kinds.
auto OperatorInfoForNodeKind(Parse::NodeKind kind, const Style& style)
    -> OperatorInfo;

// Returns the break-before penalty for a member-access expression node
// (`MemberAccessExpr` or `PointerMemberAccessExpr`), or -1 for other node
// kinds.
auto MemberAccessBreakPenalty(Parse::NodeKind kind) -> int;

// Break-before penalty for the outermost link of a multi-link member-access
// chain (`>= 2` links), lower than an interior link's 150 so a plain field
// chain that must wrap prefers breaking before its final field.
inline constexpr int MemberAccessLastLinkBreakPenalty = 35;

// Per-token formatting annotations derived from the parse tree.
struct TokenInfo {
  TokenRole role = TokenRole::Unknown;
  // For a member-access `.` or `->` token, whether the accessed member is
  // directly called or indexed (`obj.Method()` or `obj.items[i]`).
  bool is_call_link = false;
  // Width in bytes of the token on the line where it starts (its first line
  // for multi-line tokens such as block string literals).
  // TODO: Compute display width for multi-byte UTF-8 characters, and track the
  // last line's width separately for multi-line tokens.
  int column_width = 0;
  // Number of operand-alignment scopes opening or closing at this token.
  int open_scopes = 0;
  int close_scopes = 0;
  // If non-negative, this token is an operator that stays on the preceding line
  // when wrapped, and `break_penalty_after` is the split penalty after it.
  int break_penalty_after = -1;
  // If non-negative, this token is a member-access `.` or `->` with this split
  // penalty before it, or -1 otherwise.
  int break_penalty_before = -1;
  // For a member-access `.` or `->` token, the token index of its chain's
  // receiver root, grouping all links of a left-nested chain for
  // receiver-anchored indentation and all-or-nothing builder wrapping; -1
  // otherwise.
  int member_chain_id = -1;
  // Identifier of the outermost error subtree containing this token, or -1 for
  // well-formed tokens, file boundaries, and synthesized recovery tokens.
  // Adjacent tokens sharing a non-negative `verbatim_region_id` keep the raw
  // source gap between them verbatim instead of reformatting it.
  int32_t verbatim_region_id = -1;
};

// Formatting annotations indexed by `Lex::TokenIndex`.
using TokenInfoStore = FixedSizeValueStore<Lex::TokenIndex, TokenInfo>;

// Returns the number of spaces to place between adjacent tokens `left` and
// `right` on the same line.
auto SpacesBefore(const Lex::TokenizedBuffer& tokens,
                  const TokenInfoStore& token_infos, Lex::TokenIndex left,
                  Lex::TokenIndex right) -> int;

// If `text` is a single-line `#`-delimited string literal starting with `"""`
// that would re-lex as a `"""` multi-line introducer once followed by a
// newline, returns an equivalent literal with the third `"` escaped; otherwise
// returns `std::nullopt`.
auto EscapeTripleQuoteString(llvm::StringRef text)
    -> std::optional<std::string>;

// Returns whether a line break is allowed before `right`, given the preceding
// token `left`.
auto CanBreakBefore(const Lex::TokenizedBuffer& tokens,
                    const TokenInfoStore& token_infos, Lex::TokenIndex left,
                    Lex::TokenIndex right) -> bool;

// Returns the penalty for breaking the line before `right`, given the preceding
// token `left` and `style`. Lower penalties are preferred by the line-wrapping
// solver.
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
