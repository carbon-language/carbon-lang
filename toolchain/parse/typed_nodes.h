// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
#define CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_

#include "toolchain/parse/extract.h"
#include "toolchain/parse/tree.h"

namespace Carbon::Parse {

// This class provides a shorthand for defining parse node kinds for leaf nodes.
template <const NodeKind& KindT>
struct LeafNode {
  static constexpr auto Kind = KindT;
};

// A pair of a list item and its optional following comma.
template <typename Comma>
struct ListItem {
  AnyNode value;
  Optional<Comma> comma;
};

// A list of items, parameterized by the kind of the comma and the opening
// bracket.
template <typename Comma, typename Bracket>
using CommaSeparatedList = BracketedList<ListItem<Comma>, Bracket>;

// Each type defined below corresponds to a parse node kind, and describes the
// expected child structure of that parse node.

// An invalid parse. Used to balance the parse tree. This type is here only to
// ensure we have a type for each parse node kind. This node kind always has an
// error, so can never be extracted.
using InvalidParse = LeafNode<NodeKind::InvalidParse>;

// The start of the file.
using FileStart = LeafNode<NodeKind::FileStart>;

// The end of the file.
using FileEnd = LeafNode<NodeKind::FileEnd>;

// A complete source file. Note that there is no corresponding parse node for
// the file. The file is instead the complete contents of the parse tree.
struct File {
  Required<FileStart> start;
  BracketedList<AnyDecl, FileStart> decls;
  Required<FileEnd> end;

  static auto Make(const Tree* tree) -> File {
    return tree->ExtractNodeFromChildren<File>(tree->roots());
  }
};

// An empty declaration, such as `;`.
using EmptyDecl = LeafNode<NodeKind::EmptyDecl>;

// A name in a non-expression context, such as a declaration.
using Name = LeafNode<NodeKind::Name>;

// A name in an expression context.
using NameExpr = LeafNode<NodeKind::NameExpr>;

// TODO: Library, package, import (likely to change soon).

using NamespaceStart = LeafNode<NodeKind::NamespaceStart>;

// A namespace: `namespace N;`.
struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace;
  NamespaceStart introducer;
  // Name or QualifiedDecl.
  AnyNode name;
};

using CodeBlockStart = LeafNode<NodeKind::CodeBlockStart>;

// A code block: `{ statement; statement; ... }`.
struct CodeBlock {
  static constexpr auto Kind = NodeKind::CodeBlock;
  Required<CodeBlockStart> left_brace;
  BracketedList<AnyStatement, CodeBlockStart> statements;
};

using VariableIntroducer = LeafNode<NodeKind::VariableIntroducer>;

using ParameterListStart = LeafNode<NodeKind::ParameterListStart>;
using ImplicitParameterListStart =
    LeafNode<NodeKind::ImplicitParameterListStart>;
using ParameterListComma = LeafNode<NodeKind::ParameterListComma>;

// A parameter list: `(a: i32, b: i32)`.
struct ParameterList {
  static constexpr auto Kind = NodeKind::ParameterList;
  Required<ParameterListStart> left_paren;
  CommaSeparatedList<ParameterListComma, ParameterListStart> parameters;
};

// An implicit parameter list: `[T:! type, self: Self]`.
struct ImplicitParameterList {
  static constexpr auto Kind = NodeKind::ImplicitParameterList;
  Required<ImplicitParameterListStart> left_square;
  CommaSeparatedList<ParameterListComma, ImplicitParameterListStart> parameters;
};

using FunctionIntroducer = LeafNode<NodeKind::FunctionIntroducer>;

// A return type: `-> i32`.
struct ReturnType {
  static constexpr auto Kind = NodeKind::ReturnType;
  AnyExpr type;
};

// A function signature: `fn F() -> i32`.
template <const NodeKind& KindT>
struct FunctionSignature {
  static constexpr auto Kind = KindT;
  Required<FunctionIntroducer> introducer;
  Required<Name> name;
  Required<ParameterList> parameters;
  Optional<ReturnType> return_type;
};

using FunctionDecl = FunctionSignature<NodeKind::FunctionDecl>;
using FunctionDefinitionStart =
    FunctionSignature<NodeKind::FunctionDefinitionStart>;

// A function definition: `fn F() -> i32 { ... }`.
struct FunctionDefinition {
  static constexpr auto Kind = NodeKind::FunctionDefinition;
  Required<FunctionDefinitionStart> signature;
  BracketedList<AnyStatement, FunctionDefinitionStart> body;
};

using ArrayExprStart = LeafNode<NodeKind::ArrayExprStart>;

// The start of an array type, `[i32;`.
//
// TODO: Consider flattening this into `ArrayExpr`.
struct ArrayExprSemi {
  static constexpr auto Kind = NodeKind::ArrayExprSemi;
  Required<ArrayExprStart> left_square;
  AnyExpr type;
};

// An array type, such as  `[i32; 3]` or `[i32;]`.
struct ArrayExpr {
  Required<ArrayExprSemi> start;
  OptionalNot<ArrayExprSemi> bound;
};

// A pattern binding, such as `name: Type`.
struct PatternBinding {
  static constexpr auto Kind = NodeKind::PatternBinding;
  // Either `Name` or `SelfValueName`.
  AnyNode name;
  AnyExpr type;
};

// An address-of binding: `addr self: Self*`.
struct Address {
  static constexpr auto Kind = NodeKind::Address;
  AnyPattern inner;
};

// A template binding: `template T:! type`.
struct Template {
  static constexpr auto Kind = NodeKind::Template;
  // This is a Required<GenericPatternBinding> in any valid program.
  // TODO: Should the parser enforce that?
  AnyPattern inner;
};

using LetIntroducer = LeafNode<NodeKind::LetIntroducer>;
using LetInitializer = LeafNode<NodeKind::LetInitializer>;

// A `let` declaration: `let a: i32 = 5;`.
struct LetDecl {
  static constexpr auto Kind = NodeKind::LetDecl;
  Required<LetIntroducer> introducer;
  AnyPattern pattern;
  Required<LetInitializer> equals;
  AnyExpr initializer;
};

using VariableIntroducer = LeafNode<NodeKind::VariableIntroducer>;
using ReturnedSpecifier = LeafNode<NodeKind::ReturnedSpecifier>;
using VariableInitializer = LeafNode<NodeKind::VariableInitializer>;

// A `var` declaration: `var a: i32;` or `var a: i32 = 5;`.
struct VariableDecl {
  static constexpr auto Kind = NodeKind::VariableDecl;
  Required<VariableIntroducer> introducer;
  Optional<ReturnedSpecifier> returned;
  AnyPattern pattern;

  struct Initializer {
    Required<VariableInitializer> equals;
    AnyExpr value;
  };
  std::optional<Initializer> initializer;
};

// An expression statement: `F(x);`.
struct ExprStatement {
  static constexpr auto Kind = NodeKind::ExprStatement;
  AnyExpr expr;
};

using BreakStatementStart = LeafNode<NodeKind::BreakStatementStart>;

// A break statement: `break;`.
struct BreakStatement {
  static constexpr auto Kind = NodeKind::BreakStatement;
  Required<BreakStatementStart> introducer;
};

using ContinueStatementStart = LeafNode<NodeKind::ContinueStatementStart>;

// A continue statement: `continue;`.
struct ContinueStatement {
  static constexpr auto Kind = NodeKind::ContinueStatement;
  Required<ContinueStatementStart> introducer;
};

using ReturnStatementStart = LeafNode<NodeKind::ReturnStatementStart>;
using ReturnVarSpecifier = LeafNode<NodeKind::ReturnVarSpecifier>;

// A return statement: `return;` or `return expr;` or `return var;`.
struct ReturnStatement {
  static constexpr auto Kind = NodeKind::ReturnStatement;
  Required<ReturnStatementStart> introducer;
  OptionalNot<ReturnStatementStart> expr;
  Optional<ReturnVarSpecifier> var;
};

using ForHeaderStart = LeafNode<NodeKind::ForHeaderStart>;

// The `var ... in` portion of a `for` statement.
struct ForIn {
  static constexpr auto Kind = NodeKind::ForIn;
  Required<VariableIntroducer> introducer;
  AnyPattern pattern;
};

// The `for (var ... in ...)` portion of a `for` statement.
struct ForHeader {
  static constexpr auto Kind = NodeKind::ForHeader;
  Required<ForHeaderStart> introducer;
  Required<ForIn> var;
  AnyExpr range;
};

// A complete `for (...) { ... }` statement.
struct ForStatement {
  static constexpr auto Kind = NodeKind::ForStatement;
  Required<ForHeader> header;
  Required<CodeBlock> body;
};

using IfConditionStart = LeafNode<NodeKind::IfConditionStart>;

// The condition portion of an `if` statement: `(expr)`.
struct IfCondition {
  static constexpr auto Kind = NodeKind::IfCondition;
  Required<IfConditionStart> left_paren;
  AnyExpr condition;
};

using IfStatementElse = LeafNode<NodeKind::IfStatementElse>;

// An `if` statement: `if (expr) { ... } else { ... }`.
struct IfStatement {
  static constexpr auto Kind = NodeKind::IfStatement;
  Required<IfCondition> head;
  Required<CodeBlock> then;

  struct Else {
    Required<IfStatementElse> else_token;
    // Either a CodeBlock or an IfStatement.
    AnyStatement statement;
  };
  std::optional<Else> else_clause;
};

using WhileConditionStart = LeafNode<NodeKind::WhileConditionStart>;

// The condition portion of a `while` statement: `(expr)`.
struct WhileCondition {
  static constexpr auto Kind = NodeKind::WhileCondition;
  Required<WhileConditionStart> left_paren;
  AnyExpr condition;
};

// A `while` statement: `while (expr) { ... }`.
struct WhileStatement {
  static constexpr auto Kind = NodeKind::WhileStatement;
  Required<WhileCondition> head;
  Required<CodeBlock> body;
};

// The opening portion of an indexing expression: `a[`.
//
// TODO: Consider flattening this into `IndexExpr`.
struct IndexExprStart {
  static constexpr auto Kind = NodeKind::IndexExprStart;
  AnyExpr sequence;
};

// An indexing expression, such as `a[1]`.
struct IndexExpr {
  static constexpr auto Kind = NodeKind::IndexExpr;
  Required<IndexExprStart> start;
  AnyExpr index;
};

using ParenExprOrTupleLiteralStart =
    LeafNode<NodeKind::ParenExprOrTupleLiteralStart>;

// A parenthesized expression: `(a)`.
struct ParenExpr {
  static constexpr auto Kind = NodeKind::ParenExpr;
  Required<ParenExprOrTupleLiteralStart> left_paren;
  AnyExpr expr;
};

using TupleLiteralComma = LeafNode<NodeKind::TupleLiteralComma>;

// A tuple literal: `()`, `(a, b, c)`, or `(a,)`.
struct TupleLiteral {
  static constexpr auto Kind = NodeKind::TupleLiteral;
  Required<ParenExprOrTupleLiteralStart> left_paren;
  CommaSeparatedList<TupleLiteralComma, ParenExprOrTupleLiteralStart> elements;
};

// The opening portion of a call expression: `F(`.
//
// TODO: Consider flattening this into `CallExpr`.
struct CallExprStart {
  static constexpr auto Kind = NodeKind::CallExprStart;
  AnyExpr callee;
};

using CallExprComma = LeafNode<NodeKind::CallExprComma>;

// A call expression: `F(a, b, c)`.
struct CallExpr {
  static constexpr auto Kind = NodeKind::CallExpr;
  Required<CallExprStart> start;
  CommaSeparatedList<CallExprComma, CallExprStart> arguments;
};

// A qualified name: `A.B`.
//
// TODO: This is not a declaration. Rename this parse node.
struct QualifiedDecl {
  static constexpr auto Kind = NodeKind::QualifiedDecl;

  // For now, this is either a Name or a QualifiedDecl.
  AnyNode lhs;

  // TODO: This will eventually need to support more general expressions, for
  // example `GenericType(type_args).ChildType(child_type_args).Name`.
  Required<Name> rhs;
};

// A simple member access expression: `a.b`.
struct MemberAccessExpr {
  static constexpr auto Kind = NodeKind::MemberAccessExpr;
  AnyExpr lhs;
  Required<Name> rhs;
};

// A simple indirect member access expression: `a->b`.
struct PointerMemberAccessExpr {
  static constexpr auto Kind = NodeKind::PointerMemberAccessExpr;
  AnyExpr lhs;
  Required<Name> rhs;
};

// A literal.
using Literal = LeafNode<NodeKind::Literal>;

// A prefix operator expression.
struct PrefixOperator {
  static constexpr auto Kind = NodeKind::PrefixOperator;
  AnyExpr operand;
};

// The first operand of a short-circuiting infix operator: `a and` or `a or`.
// The complete operator expression will be an InfixOperator with this as the
// `lhs`.
struct ShortCircuitOperand {
  static constexpr auto Kind = NodeKind::ShortCircuitOperand;
  AnyExpr operand;
};

// An infix operator expression.
struct InfixOperator {
  static constexpr auto Kind = NodeKind::InfixOperator;
  AnyExpr lhs;
  AnyExpr rhs;
};

// A postfix operator expression.
struct PostfixOperator {
  static constexpr auto Kind = NodeKind::PostfixOperator;
  AnyExpr operand;
};

// The `if` portion of an `if` expression: `if expr`.
struct IfExprIf {
  static constexpr auto Kind = NodeKind::IfExprIf;
  AnyExpr condition;
};

// The `then` portion of an `if` expression: `then expr`.
struct IfExprThen {
  static constexpr auto Kind = NodeKind::IfExprThen;
  AnyExpr result;
};

// A full `if` expression: `if expr then expr else expr`.
struct IfExprElse {
  static constexpr auto Kind = NodeKind::IfExprElse;
  Required<IfExprIf> start;
  Required<IfExprIf> then;
  AnyExpr else_result;
};

// TODO: StructLiteral onwards

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
