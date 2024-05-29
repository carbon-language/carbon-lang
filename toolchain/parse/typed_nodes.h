// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
#define CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_

#include <optional>

#include "toolchain/parse/node_ids.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// Helpers for defining different kinds of parse nodes.
// ----------------------------------------------------

// A pair of a list item and its optional following comma.
template <typename Element, typename Comma>
struct ListItem {
  Element value;
  std::optional<Comma> comma;
};

// A list of items, parameterized by the kind of the elements and comma.
template <typename Element, typename Comma>
using CommaSeparatedList = llvm::SmallVector<ListItem<Element, Comma>>;

// This class provides a shorthand for defining parse node kinds for leaf nodes.
template <const NodeKind& KindT,
          NodeCategory::RawEnumType Category = NodeCategory::None>
struct LeafNode {
  static constexpr auto Kind =
      KindT.Define({.category = Category, .child_count = 0});
};

// ----------------------------------------------------------------------------
// Each node kind (in node_kind.def) should have a corresponding type defined
// here which describes the expected child structure of that parse node.
//
// Each of these types should start with a `static constexpr Kind` member
// initialized by calling `Define` on the corresponding `NodeKind`, and passing
// in the `NodeCategory` of that kind.  This will both associate the category
// with the node kind and create the necessary kind object for the typed node.
//
// This should be followed by field declarations that describe the child nodes,
// in order, that occur in the parse tree. The `Extract...` functions on the
// parse tree use struct reflection on these fields to guide the extraction of
// the child nodes from the tree into an object of this type with these fields
// for convenient access.
//
// The types of these fields are special and describe the specific child node
// structure of the parse node. Many of these types are defined in `node_ids.h`.
//
// Valid primitive types here are:
// - `NodeId` to match any single child node
// - `FooId` to require that child to have kind `NodeKind::Foo`
// - `AnyCatId` to require that child to have a kind in category `Cat`
// - `NodeIdOneOf<A, B>` to require the child to have kind `NodeKind::A` or
// `NodeKind::B`
// - `NodeIdNot<A>` to match any single child whose kind is not `NodeKind::A`
//
// There a few, restricted composite field types allowed that compose types in
// various ways, where all of the `T`s and `U`s below are themselves valid field
// types:
// - `llvm::SmallVector<T>` to match any number of children matching `T`
// - `std::optional<T>` to match 0 or 1 children matching `T`
// - `std::tuple<T...>` to match children matching `T...`
// - Any provided `Aggregate` type that is a simple aggregate type such as
// `struct Aggregate { T x; U y; }`,
//   to match children with types `T` and `U`.
// ----------------------------------------------------------------------------

// Error nodes
// -----------

// An invalid parse. Used to balance the parse tree. This type is here only to
// ensure we have a type for each parse node kind. This node kind always has an
// error, so can never be extracted.
using InvalidParse =
    LeafNode<NodeKind::InvalidParse, NodeCategory::Decl | NodeCategory::Expr>;

// An invalid subtree. Always has an error so can never be extracted.
using InvalidParseStart = LeafNode<NodeKind::InvalidParseStart>;
struct InvalidParseSubtree {
  static constexpr auto Kind = NodeKind::InvalidParseSubtree.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = InvalidParseStart::Kind});

  InvalidParseStartId start;
  llvm::SmallVector<NodeIdNot<InvalidParseStart>> extra;
};

// A placeholder node to be replaced; it will never exist in a valid parse tree.
// Its token kind is not enforced even when valid.
using Placeholder = LeafNode<NodeKind::Placeholder>;

// File nodes
// ----------

// The start of the file.
using FileStart = LeafNode<NodeKind::FileStart>;

// The end of the file.
using FileEnd = LeafNode<NodeKind::FileEnd>;

// General-purpose nodes
// ---------------------

// An empty declaration, such as `;`.
using EmptyDecl =
    LeafNode<NodeKind::EmptyDecl, NodeCategory::Decl | NodeCategory::Statement>;

// A name in a non-expression context, such as a declaration.
using IdentifierName =
    LeafNode<NodeKind::IdentifierName, NodeCategory::MemberName>;

// A name in an expression context.
using IdentifierNameExpr =
    LeafNode<NodeKind::IdentifierNameExpr, NodeCategory::Expr>;

// The `self` value and `Self` type identifier keywords. Typically of the form
// `self: Self`.
using SelfValueName = LeafNode<NodeKind::SelfValueName>;
using SelfValueNameExpr =
    LeafNode<NodeKind::SelfValueNameExpr, NodeCategory::Expr>;
using SelfTypeNameExpr =
    LeafNode<NodeKind::SelfTypeNameExpr, NodeCategory::Expr>;

// The `base` value keyword, introduced by `base: B`. Typically referenced in
// an expression, as in `x.base` or `{.base = ...}`, but can also be used as a
// declared name, as in `{.base: partial B}`.
using BaseName = LeafNode<NodeKind::BaseName, NodeCategory::MemberName>;

// An unqualified name and optionally a following sequence of parameters.
// For example, `A`, `A(n: i32)`, or `A[T:! type](n: T)`.
struct NameAndParams {
  IdentifierNameId name;
  std::optional<ImplicitParamListId> implicit_params;
  std::optional<TuplePatternId> params;
};

// A name qualifier: `A.`, `A(T:! type).`, or `A[T:! type](N:! T).`.
struct NameQualifier {
  static constexpr auto Kind =
      NodeKind::NameQualifier.Define({.bracketed_by = IdentifierName::Kind});

  NameAndParams name_and_params;
};

// A complete name in a declaration: `A.C(T:! type).F(n: i32)`.
// Note that this includes the parameters of the entity itself.
struct DeclName {
  llvm::SmallVector<NameQualifierId> qualifiers;
  NameAndParams name_and_params;
};

// Library, package, import, export
// --------------------------------

// The `package` keyword in an expression.
using PackageExpr = LeafNode<NodeKind::PackageExpr, NodeCategory::Expr>;

// The name of a package or library for `package`, `import`, and `library`.
using PackageName = LeafNode<NodeKind::PackageName>;
using LibraryName = LeafNode<NodeKind::LibraryName>;
using DefaultLibrary = LeafNode<NodeKind::DefaultLibrary>;

using PackageIntroducer = LeafNode<NodeKind::PackageIntroducer>;

// `library` in `package` or `import`.
struct LibrarySpecifier {
  static constexpr auto Kind =
      NodeKind::LibrarySpecifier.Define({.child_count = 1});

  NodeIdOneOf<LibraryName, DefaultLibrary> name;
};

// First line of the file, such as:
//   `impl package MyPackage library "MyLibrary";`
struct PackageDecl {
  static constexpr auto Kind =
      NodeKind::PackageDecl.Define({.category = NodeCategory::Decl,
                                    .bracketed_by = PackageIntroducer::Kind});

  PackageIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
};

// `import TheirPackage library "TheirLibrary";`
using ImportIntroducer = LeafNode<NodeKind::ImportIntroducer>;
struct ImportDecl {
  static constexpr auto Kind = NodeKind::ImportDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = ImportIntroducer::Kind});

  ImportIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
};

// `library` as declaration.
using LibraryIntroducer = LeafNode<NodeKind::LibraryIntroducer>;
struct LibraryDecl {
  static constexpr auto Kind =
      NodeKind::LibraryDecl.Define({.category = NodeCategory::Decl,
                                    .bracketed_by = LibraryIntroducer::Kind});

  LibraryIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  NodeIdOneOf<LibraryName, DefaultLibrary> library_name;
};

// `export` as a declaration.
using ExportIntroducer = LeafNode<NodeKind::ExportIntroducer>;
struct ExportDecl {
  static constexpr auto Kind = NodeKind::ExportDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = ExportIntroducer::Kind});

  ExportIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
};

// Namespace nodes
// ---------------

using NamespaceStart = LeafNode<NodeKind::NamespaceStart>;

// A namespace: `namespace N;`.
struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace.Define(
      {.category = NodeCategory::Decl, .bracketed_by = NamespaceStart::Kind});

  NamespaceStartId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
};

// Pattern nodes
// -------------

// A pattern binding, such as `name: Type`.
struct BindingPattern {
  static constexpr auto Kind = NodeKind::BindingPattern.Define(
      {.category = NodeCategory::Pattern, .child_count = 2});

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  AnyExprId type;
};

// `name:! Type`
struct CompileTimeBindingPattern {
  static constexpr auto Kind = NodeKind::CompileTimeBindingPattern.Define(
      {.category = NodeCategory::Pattern, .child_count = 2});

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  AnyExprId type;
};

// An address-of binding: `addr self: Self*`.
struct Addr {
  static constexpr auto Kind = NodeKind::Addr.Define(
      {.category = NodeCategory::Pattern, .child_count = 1});

  AnyPatternId inner;
};

// A template binding: `template T:! type`.
struct Template {
  static constexpr auto Kind = NodeKind::Template.Define(
      {.category = NodeCategory::Pattern, .child_count = 1});

  // This is a CompileTimeBindingPatternId in any valid program.
  // TODO: Should the parser enforce that?
  AnyPatternId inner;
};

using TuplePatternStart = LeafNode<NodeKind::TuplePatternStart>;
using PatternListComma = LeafNode<NodeKind::PatternListComma>;

// A parameter list or tuple pattern: `(a: i32, b: i32)`.
struct TuplePattern {
  static constexpr auto Kind =
      NodeKind::TuplePattern.Define({.category = NodeCategory::Pattern,
                                     .bracketed_by = TuplePatternStart::Kind});

  TuplePatternStartId left_paren;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
};

using ImplicitParamListStart = LeafNode<NodeKind::ImplicitParamListStart>;

// An implicit parameter list: `[T:! type, self: Self]`.
struct ImplicitParamList {
  static constexpr auto Kind = NodeKind::ImplicitParamList.Define(
      {.bracketed_by = ImplicitParamListStart::Kind});

  ImplicitParamListStartId left_square;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
};

// Function nodes
// --------------

using FunctionIntroducer = LeafNode<NodeKind::FunctionIntroducer>;

// A return type: `-> i32`.
struct ReturnType {
  static constexpr auto Kind = NodeKind::ReturnType.Define({.child_count = 1});

  AnyExprId type;
};

// A function signature: `fn F() -> i32`.
template <const NodeKind& KindT, NodeCategory::RawEnumType Category>
struct FunctionSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = FunctionIntroducer::Kind});

  FunctionIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  std::optional<ReturnTypeId> return_type;
};

using FunctionDecl =
    FunctionSignature<NodeKind::FunctionDecl, NodeCategory::Decl>;
using FunctionDefinitionStart =
    FunctionSignature<NodeKind::FunctionDefinitionStart, NodeCategory::None>;

// A function definition: `fn F() -> i32 { ... }`.
struct FunctionDefinition {
  static constexpr auto Kind = NodeKind::FunctionDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = FunctionDefinitionStart::Kind});

  FunctionDefinitionStartId signature;
  llvm::SmallVector<AnyStatementId> body;
};

using BuiltinFunctionDefinitionStart =
    FunctionSignature<NodeKind::BuiltinFunctionDefinitionStart,
                      NodeCategory::None>;
using BuiltinName = LeafNode<NodeKind::BuiltinName>;

// A builtin function definition: `fn F() -> i32 = "builtin name";`
struct BuiltinFunctionDefinition {
  static constexpr auto Kind = NodeKind::BuiltinFunctionDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = BuiltinFunctionDefinitionStart::Kind});

  BuiltinFunctionDefinitionStartId signature;
  BuiltinNameId builtin_name;
};

// `alias` nodes
// -------------

using AliasIntroducer = LeafNode<NodeKind::AliasIntroducer>;
using AliasInitializer = LeafNode<NodeKind::AliasInitializer>;

// An `alias` declaration: `alias a = b;`.
struct Alias {
  static constexpr auto Kind = NodeKind::Alias.Define(
      {.category = NodeCategory::Decl | NodeCategory::Statement,
       .bracketed_by = AliasIntroducer::Kind});

  AliasIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  AliasInitializerId equals;
  AnyExprId initializer;
};

// `let` nodes
// -----------

using LetIntroducer = LeafNode<NodeKind::LetIntroducer>;
using LetInitializer = LeafNode<NodeKind::LetInitializer>;

// A `let` declaration: `let a: i32 = 5;`.
struct LetDecl {
  static constexpr auto Kind = NodeKind::LetDecl.Define(
      {.category = NodeCategory::Decl | NodeCategory::Statement,
       .bracketed_by = LetIntroducer::Kind});

  LetIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyPatternId pattern;

  struct Initializer {
    LetInitializerId equals;
    AnyExprId initializer;
  };
  std::optional<Initializer> initializer;
};

// `var` nodes
// -----------

using VariableIntroducer = LeafNode<NodeKind::VariableIntroducer>;
using ReturnedModifier = LeafNode<NodeKind::ReturnedModifier>;
using VariableInitializer = LeafNode<NodeKind::VariableInitializer>;

// A `var` declaration: `var a: i32;` or `var a: i32 = 5;`.
struct VariableDecl {
  static constexpr auto Kind = NodeKind::VariableDecl.Define(
      {.category = NodeCategory::Decl | NodeCategory::Statement,
       .bracketed_by = VariableIntroducer::Kind});

  VariableIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<ReturnedModifierId> returned;
  AnyPatternId pattern;

  struct Initializer {
    VariableInitializerId equals;
    AnyExprId value;
  };
  std::optional<Initializer> initializer;
};

// Statement nodes
// ---------------

using CodeBlockStart = LeafNode<NodeKind::CodeBlockStart>;

// A code block: `{ statement; statement; ... }`.
struct CodeBlock {
  static constexpr auto Kind =
      NodeKind::CodeBlock.Define({.bracketed_by = CodeBlockStart::Kind});

  CodeBlockStartId left_brace;
  llvm::SmallVector<AnyStatementId> statements;
};

// An expression statement: `F(x);`.
struct ExprStatement {
  static constexpr auto Kind = NodeKind::ExprStatement.Define(
      {.category = NodeCategory::Statement, .child_count = 1});

  AnyExprId expr;
};

using BreakStatementStart = LeafNode<NodeKind::BreakStatementStart>;

// A break statement: `break;`.
struct BreakStatement {
  static constexpr auto Kind = NodeKind::BreakStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = BreakStatementStart::Kind,
       .child_count = 1});

  BreakStatementStartId introducer;
};

using ContinueStatementStart = LeafNode<NodeKind::ContinueStatementStart>;

// A continue statement: `continue;`.
struct ContinueStatement {
  static constexpr auto Kind = NodeKind::ContinueStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = ContinueStatementStart::Kind,
       .child_count = 1});

  ContinueStatementStartId introducer;
};

using ReturnStatementStart = LeafNode<NodeKind::ReturnStatementStart>;
using ReturnVarModifier = LeafNode<NodeKind::ReturnVarModifier>;

// A return statement: `return;` or `return expr;` or `return var;`.
struct ReturnStatement {
  static constexpr auto Kind = NodeKind::ReturnStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = ReturnStatementStart::Kind});

  ReturnStatementStartId introducer;
  std::optional<AnyExprId> expr;
  std::optional<ReturnVarModifierId> var;
};

using ForHeaderStart = LeafNode<NodeKind::ForHeaderStart>;

// The `var ... in` portion of a `for` statement.
struct ForIn {
  static constexpr auto Kind = NodeKind::ForIn.Define(
      {.bracketed_by = VariableIntroducer::Kind, .child_count = 2});

  VariableIntroducerId introducer;
  AnyPatternId pattern;
};

// The `for (var ... in ...)` portion of a `for` statement.
struct ForHeader {
  static constexpr auto Kind =
      NodeKind::ForHeader.Define({.bracketed_by = ForHeaderStart::Kind});

  ForHeaderStartId introducer;
  ForInId var;
  AnyExprId range;
};

// A complete `for (...) { ... }` statement.
struct ForStatement {
  static constexpr auto Kind =
      NodeKind::ForStatement.Define({.category = NodeCategory::Statement,
                                     .bracketed_by = ForHeader::Kind,
                                     .child_count = 2});

  ForHeaderId header;
  CodeBlockId body;
};

using IfConditionStart = LeafNode<NodeKind::IfConditionStart>;

// The condition portion of an `if` statement: `(expr)`.
struct IfCondition {
  static constexpr auto Kind = NodeKind::IfCondition.Define(
      {.bracketed_by = IfConditionStart::Kind, .child_count = 2});

  IfConditionStartId left_paren;
  AnyExprId condition;
};

using IfStatementElse = LeafNode<NodeKind::IfStatementElse>;

// An `if` statement: `if (expr) { ... } else { ... }`.
struct IfStatement {
  static constexpr auto Kind = NodeKind::IfStatement.Define(
      {.category = NodeCategory::Statement, .bracketed_by = IfCondition::Kind});

  IfConditionId head;
  CodeBlockId then;

  struct Else {
    IfStatementElseId else_token;
    NodeIdOneOf<CodeBlock, IfStatement> body;
  };
  std::optional<Else> else_clause;
};

using WhileConditionStart = LeafNode<NodeKind::WhileConditionStart>;

// The condition portion of a `while` statement: `(expr)`.
struct WhileCondition {
  static constexpr auto Kind = NodeKind::WhileCondition.Define(
      {.bracketed_by = WhileConditionStart::Kind, .child_count = 2});

  WhileConditionStartId left_paren;
  AnyExprId condition;
};

// A `while` statement: `while (expr) { ... }`.
struct WhileStatement {
  static constexpr auto Kind =
      NodeKind::WhileStatement.Define({.category = NodeCategory::Statement,
                                       .bracketed_by = WhileCondition::Kind,
                                       .child_count = 2});

  WhileConditionId head;
  CodeBlockId body;
};

using MatchConditionStart = LeafNode<NodeKind::MatchConditionStart>;

struct MatchCondition {
  static constexpr auto Kind = NodeKind::MatchCondition.Define(
      {.bracketed_by = MatchConditionStart::Kind, .child_count = 2});

  MatchConditionStartId left_paren;
  AnyExprId condition;
};

using MatchIntroducer = LeafNode<NodeKind::MatchIntroducer>;
struct MatchStatementStart {
  static constexpr auto Kind = NodeKind::MatchStatementStart.Define(
      {.bracketed_by = MatchIntroducer::Kind, .child_count = 2});

  MatchIntroducerId introducer;
  MatchConditionId left_brace;
};

using MatchCaseIntroducer = LeafNode<NodeKind::MatchCaseIntroducer>;
using MatchCaseGuardIntroducer = LeafNode<NodeKind::MatchCaseGuardIntroducer>;
using MatchCaseGuardStart = LeafNode<NodeKind::MatchCaseGuardStart>;

struct MatchCaseGuard {
  static constexpr auto Kind = NodeKind::MatchCaseGuard.Define(
      {.bracketed_by = MatchCaseGuardIntroducer::Kind, .child_count = 3});

  MatchCaseGuardIntroducerId introducer;
  MatchCaseGuardStartId left_paren;
  AnyExprId condition;
};

using MatchCaseEqualGreater = LeafNode<NodeKind::MatchCaseEqualGreater>;

struct MatchCaseStart {
  static constexpr auto Kind = NodeKind::MatchCaseStart.Define(
      {.bracketed_by = MatchCaseIntroducer::Kind});

  MatchCaseIntroducerId introducer;
  AnyPatternId pattern;
  std::optional<MatchCaseGuardId> guard;
  MatchCaseEqualGreaterId equal_greater_token;
};

struct MatchCase {
  static constexpr auto Kind =
      NodeKind::MatchCase.Define({.bracketed_by = MatchCaseStart::Kind});

  MatchCaseStartId head;
  llvm::SmallVector<AnyStatementId> statements;
};

using MatchDefaultIntroducer = LeafNode<NodeKind::MatchDefaultIntroducer>;
using MatchDefaultEqualGreater = LeafNode<NodeKind::MatchDefaultEqualGreater>;

struct MatchDefaultStart {
  static constexpr auto Kind = NodeKind::MatchDefaultStart.Define(
      {.bracketed_by = MatchDefaultIntroducer::Kind, .child_count = 2});

  MatchDefaultIntroducerId introducer;
  MatchDefaultEqualGreaterId equal_greater_token;
};

struct MatchDefault {
  static constexpr auto Kind =
      NodeKind::MatchDefault.Define({.bracketed_by = MatchDefaultStart::Kind});

  MatchDefaultStartId introducer;
  llvm::SmallVector<AnyStatementId> statements;
};

// A `match` statement: `match (expr) { case (...) => {...} default => {...}}`.
struct MatchStatement {
  static constexpr auto Kind = NodeKind::MatchStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = MatchStatementStart::Kind});

  MatchStatementStartId head;

  llvm::SmallVector<MatchCaseId> cases;
  std::optional<MatchDefaultId> default_case;
};

// Expression nodes
// ----------------

using ArrayExprStart = LeafNode<NodeKind::ArrayExprStart>;

// The start of an array type, `[i32;`.
//
// TODO: Consider flattening this into `ArrayExpr`.
struct ArrayExprSemi {
  static constexpr auto Kind = NodeKind::ArrayExprSemi.Define(
      {.bracketed_by = ArrayExprStart::Kind, .child_count = 2});

  ArrayExprStartId left_square;
  AnyExprId type;
};

// An array type, such as  `[i32; 3]` or `[i32;]`.
struct ArrayExpr {
  static constexpr auto Kind = NodeKind::ArrayExpr.Define(
      {.category = NodeCategory::Expr, .bracketed_by = ArrayExprSemi::Kind});

  ArrayExprSemiId start;
  std::optional<AnyExprId> bound;
};

// The opening portion of an indexing expression: `a[`.
//
// TODO: Consider flattening this into `IndexExpr`.
struct IndexExprStart {
  static constexpr auto Kind =
      NodeKind::IndexExprStart.Define({.child_count = 1});

  AnyExprId sequence;
};

// An indexing expression, such as `a[1]`.
struct IndexExpr {
  static constexpr auto Kind =
      NodeKind::IndexExpr.Define({.category = NodeCategory::Expr,
                                  .bracketed_by = IndexExprStart::Kind,
                                  .child_count = 2});

  IndexExprStartId start;
  AnyExprId index;
};

using ParenExprStart = LeafNode<NodeKind::ParenExprStart>;

// A parenthesized expression: `(a)`.
struct ParenExpr {
  static constexpr auto Kind = NodeKind::ParenExpr.Define(
      {.category = NodeCategory::Expr | NodeCategory::MemberExpr,
       .bracketed_by = ParenExprStart::Kind,
       .child_count = 2});

  ParenExprStartId start;
  AnyExprId expr;
};

using TupleLiteralStart = LeafNode<NodeKind::TupleLiteralStart>;
using TupleLiteralComma = LeafNode<NodeKind::TupleLiteralComma>;

// A tuple literal: `()`, `(a, b, c)`, or `(a,)`.
struct TupleLiteral {
  static constexpr auto Kind =
      NodeKind::TupleLiteral.Define({.category = NodeCategory::Expr,
                                     .bracketed_by = TupleLiteralStart::Kind});

  TupleLiteralStartId start;
  CommaSeparatedList<AnyExprId, TupleLiteralCommaId> elements;
};

// The opening portion of a call expression: `F(`.
//
// TODO: Consider flattening this into `CallExpr`.
struct CallExprStart {
  static constexpr auto Kind =
      NodeKind::CallExprStart.Define({.child_count = 1});

  AnyExprId callee;
};

using CallExprComma = LeafNode<NodeKind::CallExprComma>;

// A call expression: `F(a, b, c)`.
struct CallExpr {
  static constexpr auto Kind = NodeKind::CallExpr.Define(
      {.category = NodeCategory::Expr, .bracketed_by = CallExprStart::Kind});

  CallExprStartId start;
  CommaSeparatedList<AnyExprId, CallExprCommaId> arguments;
};

// A member access expression: `a.b` or `a.(b)`.
struct MemberAccessExpr {
  static constexpr auto Kind = NodeKind::MemberAccessExpr.Define(
      {.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  AnyMemberNameOrMemberExprId rhs;
};

// An indirect member access expression: `a->b` or `a->(b)`.
struct PointerMemberAccessExpr {
  static constexpr auto Kind = NodeKind::PointerMemberAccessExpr.Define(
      {.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  AnyMemberNameOrMemberExprId rhs;
};

// A prefix operator expression.
template <const NodeKind& KindT>
struct PrefixOperator {
  static constexpr auto Kind =
      KindT.Define({.category = NodeCategory::Expr, .child_count = 1});

  AnyExprId operand;
};

// An infix operator expression.
template <const NodeKind& KindT>
struct InfixOperator {
  static constexpr auto Kind =
      KindT.Define({.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  AnyExprId rhs;
};

// A postfix operator expression.
template <const NodeKind& KindT>
struct PostfixOperator {
  static constexpr auto Kind =
      KindT.Define({.category = NodeCategory::Expr, .child_count = 1});

  AnyExprId operand;
};

// Literals, operators, and modifiers

#define CARBON_PARSE_NODE_KIND(...)
#define CARBON_PARSE_NODE_KIND_TOKEN_LITERAL(Name, ...) \
  using Name = LeafNode<NodeKind::Name, NodeCategory::Expr>;
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name, ...) \
  using Name##Modifier =                                 \
      LeafNode<NodeKind::Name##Modifier, NodeCategory::Modifier>;
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name, ...) \
  using PrefixOperator##Name = PrefixOperator<NodeKind::PrefixOperator##Name>;
#define CARBON_PARSE_NODE_KIND_INFIX_OPERATOR(Name, ...) \
  using InfixOperator##Name = InfixOperator<NodeKind::InfixOperator##Name>;
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name, ...) \
  using PostfixOperator##Name =                            \
      PostfixOperator<NodeKind::PostfixOperator##Name>;
#include "toolchain/parse/node_kind.def"

// The first operand of a short-circuiting infix operator: `a and` or `a or`.
// The complete operator expression will be an InfixOperator with this as the
// `lhs`.
// TODO: Make this be a template if we ever need to write generic code to cover
// both cases at once, say in check.
struct ShortCircuitOperandAnd {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperandAnd.Define({.child_count = 1});

  AnyExprId operand;
};

struct ShortCircuitOperandOr {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperandOr.Define({.child_count = 1});

  AnyExprId operand;
};

struct ShortCircuitOperatorAnd {
  static constexpr auto Kind = NodeKind::ShortCircuitOperatorAnd.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = ShortCircuitOperandAnd::Kind,
       .child_count = 2});

  ShortCircuitOperandAndId lhs;
  AnyExprId rhs;
};

struct ShortCircuitOperatorOr {
  static constexpr auto Kind = NodeKind::ShortCircuitOperatorOr.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = ShortCircuitOperandOr::Kind,
       .child_count = 2});

  ShortCircuitOperandOrId lhs;
  AnyExprId rhs;
};

// The `if` portion of an `if` expression: `if expr`.
struct IfExprIf {
  static constexpr auto Kind = NodeKind::IfExprIf.Define({.child_count = 1});

  AnyExprId condition;
};

// The `then` portion of an `if` expression: `then expr`.
struct IfExprThen {
  static constexpr auto Kind = NodeKind::IfExprThen.Define({.child_count = 1});

  AnyExprId result;
};

// A full `if` expression: `if expr then expr else expr`.
struct IfExprElse {
  static constexpr auto Kind =
      NodeKind::IfExprElse.Define({.category = NodeCategory::Expr,
                                   .bracketed_by = IfExprIf::Kind,
                                   .child_count = 3});

  IfExprIfId start;
  IfExprThenId then;
  AnyExprId else_result;
};

// Choice nodes
// ------------

using ChoiceIntroducer = LeafNode<NodeKind::ChoiceIntroducer>;

struct ChoiceSignature {
  static constexpr auto Kind = NodeKind::ChoiceDefinitionStart.Define(
      {.category = NodeCategory::None, .bracketed_by = ChoiceIntroducer::Kind});

  ChoiceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
};

using ChoiceDefinitionStart = ChoiceSignature;

using ChoiceAlternativeListComma =
    LeafNode<NodeKind::ChoiceAlternativeListComma>;

struct ChoiceDefinition {
  static constexpr auto Kind = NodeKind::ChoiceDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ChoiceDefinitionStart::Kind});

  ChoiceDefinitionStartId signature;
  struct Alternative {
    IdentifierNameId name;
    std::optional<TuplePatternId> parameters;
  };
  CommaSeparatedList<Alternative, ChoiceAlternativeListCommaId> alternatives;
};

// Struct type and value literals
// ----------------------------------------

// `{`
using StructLiteralStart = LeafNode<NodeKind::StructLiteralStart>;
using StructTypeLiteralStart = LeafNode<NodeKind::StructTypeLiteralStart>;
// `,`
using StructComma = LeafNode<NodeKind::StructComma>;

// `.a`
struct StructFieldDesignator {
  static constexpr auto Kind =
      NodeKind::StructFieldDesignator.Define({.child_count = 1});

  NodeIdOneOf<IdentifierName, BaseName> name;
};

// `.a = 0`
struct StructField {
  static constexpr auto Kind = NodeKind::StructField.Define(
      {.bracketed_by = StructFieldDesignator::Kind, .child_count = 2});

  StructFieldDesignatorId designator;
  AnyExprId expr;
};

// `.a: i32`
struct StructTypeField {
  static constexpr auto Kind = NodeKind::StructTypeField.Define(
      {.bracketed_by = StructFieldDesignator::Kind, .child_count = 2});

  StructFieldDesignatorId designator;
  AnyExprId type_expr;
};

// Struct literals, such as `{.a = 0}`.
struct StructLiteral {
  static constexpr auto Kind = NodeKind::StructLiteral.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = StructLiteralStart::Kind});

  StructLiteralStartId start;
  CommaSeparatedList<StructFieldId, StructCommaId> fields;
};

// Struct type literals, such as `{.a: i32}`.
struct StructTypeLiteral {
  static constexpr auto Kind = NodeKind::StructTypeLiteral.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = StructTypeLiteralStart::Kind});

  StructTypeLiteralStartId start;
  CommaSeparatedList<StructTypeFieldId, StructCommaId> fields;
};

// `class` declarations and definitions
// ------------------------------------

// `class`
using ClassIntroducer = LeafNode<NodeKind::ClassIntroducer>;

// A class signature `class C`
template <const NodeKind& KindT, NodeCategory::RawEnumType Category>
struct ClassSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = ClassIntroducer::Kind});

  ClassIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
};

// `class C;`
using ClassDecl = ClassSignature<NodeKind::ClassDecl, NodeCategory::Decl>;
// `class C {`
using ClassDefinitionStart =
    ClassSignature<NodeKind::ClassDefinitionStart, NodeCategory::None>;

// `class C { ... }`
struct ClassDefinition {
  static constexpr auto Kind = NodeKind::ClassDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ClassDefinitionStart::Kind});

  ClassDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// Adapter declaration
// -------------------

// `adapt`
using AdaptIntroducer = LeafNode<NodeKind::AdaptIntroducer>;
// `adapt SomeType;`
struct AdaptDecl {
  static constexpr auto Kind = NodeKind::AdaptDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = AdaptIntroducer::Kind});

  AdaptIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyExprId adapted_type;
};

// Base class declaration
// ----------------------

// `base`
using BaseIntroducer = LeafNode<NodeKind::BaseIntroducer>;
using BaseColon = LeafNode<NodeKind::BaseColon>;
// `extend base: BaseClass;`
struct BaseDecl {
  static constexpr auto Kind = NodeKind::BaseDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = BaseIntroducer::Kind});

  BaseIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  BaseColonId colon;
  AnyExprId base_class;
};

// Interface declarations and definitions
// --------------------------------------

// `interface`
using InterfaceIntroducer = LeafNode<NodeKind::InterfaceIntroducer>;

// `interface I`
template <const NodeKind& KindT, NodeCategory::RawEnumType Category>
struct InterfaceSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = InterfaceIntroducer::Kind});

  InterfaceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
};

// `interface I;`
using InterfaceDecl =
    InterfaceSignature<NodeKind::InterfaceDecl, NodeCategory::Decl>;
// `interface I {`
using InterfaceDefinitionStart =
    InterfaceSignature<NodeKind::InterfaceDefinitionStart, NodeCategory::None>;

// `interface I { ... }`
struct InterfaceDefinition {
  static constexpr auto Kind = NodeKind::InterfaceDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = InterfaceDefinitionStart::Kind});

  InterfaceDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// `impl`...`as` declarations and definitions
// ------------------------------------------

// `impl`
using ImplIntroducer = LeafNode<NodeKind::ImplIntroducer>;

// `forall [...]`
struct ImplForall {
  static constexpr auto Kind = NodeKind::ImplForall.Define({.child_count = 1});

  ImplicitParamListId params;
};

// `as` with no type before it
using DefaultSelfImplAs =
    LeafNode<NodeKind::DefaultSelfImplAs, NodeCategory::ImplAs>;

// `<type> as`
struct TypeImplAs {
  static constexpr auto Kind = NodeKind::TypeImplAs.Define(
      {.category = NodeCategory::ImplAs, .child_count = 1});

  AnyExprId type_expr;
};

// `impl T as I`
template <const NodeKind& KindT, NodeCategory::RawEnumType Category>
struct ImplSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = ImplIntroducer::Kind});

  ImplIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<ImplForallId> forall;
  AnyImplAsId as;
  AnyExprId interface;
};

// `impl T as I;`
using ImplDecl = ImplSignature<NodeKind::ImplDecl, NodeCategory::Decl>;
// `impl T as I {`
using ImplDefinitionStart =
    ImplSignature<NodeKind::ImplDefinitionStart, NodeCategory::None>;

// `impl T as I { ... }`
struct ImplDefinition {
  static constexpr auto Kind = NodeKind::ImplDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ImplDefinitionStart::Kind});

  ImplDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// Named constraint declarations and definitions
// ---------------------------------------------

// `constraint`
using NamedConstraintIntroducer = LeafNode<NodeKind::NamedConstraintIntroducer>;

// `constraint NC`
template <const NodeKind& KindT, NodeCategory::RawEnumType Category>
struct NamedConstraintSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = NamedConstraintIntroducer::Kind});

  NamedConstraintIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
};

// `constraint NC;`
using NamedConstraintDecl =
    NamedConstraintSignature<NodeKind::NamedConstraintDecl, NodeCategory::Decl>;
// `constraint NC {`
using NamedConstraintDefinitionStart =
    NamedConstraintSignature<NodeKind::NamedConstraintDefinitionStart,
                             NodeCategory::None>;

// `constraint NC { ... }`
struct NamedConstraintDefinition {
  static constexpr auto Kind = NodeKind::NamedConstraintDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = NamedConstraintDefinitionStart::Kind});

  NamedConstraintDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// ---------------------------------------------------------------------------

// A complete source file. Note that there is no corresponding parse node for
// the file. The file is instead the complete contents of the parse tree.
struct File {
  FileStartId start;
  llvm::SmallVector<AnyDeclId> decls;
  FileEndId end;
};

// Define `Foo` as the node type for the ID type `FooId`.
#define CARBON_PARSE_NODE_KIND(KindName) \
  template <>                            \
  struct NodeForId<KindName##Id> {       \
    using TypedNode = KindName;          \
  };
#include "toolchain/parse/node_kind.def"

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
