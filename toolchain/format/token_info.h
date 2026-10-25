// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
#define CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "toolchain/base/fixed_size_value_store.h"
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

// Per-token formatting annotations derived from the parse tree.
struct TokenInfo {
  TokenRole role = TokenRole::Unknown;
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

}  // namespace Carbon::Format

#endif  // CARBON_TOOLCHAIN_FORMAT_TOKEN_INFO_H_
