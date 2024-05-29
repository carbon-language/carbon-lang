// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
#define CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_

#include <optional>

#include "toolchain/lex/token_kind.h"
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
template <const NodeKind& KindT, typename TokenKind,
          NodeCategory Category = NodeCategory::None>
struct LeafNode {
  static constexpr auto Kind =
      KindT.Define({.category = Category, .child_count = 0});

  TokenKind token;
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
using InvalidParse = LeafNode<NodeKind::InvalidParse, AnyToken,
                              NodeCategory::Decl | NodeCategory::Expr>;

// An invalid subtree. Always has an error so can never be extracted.
using InvalidParseStart = LeafNode<NodeKind::InvalidParseStart, AnyToken>;
struct InvalidParseSubtree {
  static constexpr auto Kind = NodeKind::InvalidParseSubtree.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = InvalidParseStart::Kind});

  InvalidParseStartId start;
  llvm::SmallVector<NodeIdNot<InvalidParseStart>> extra;
  AnyToken token;
};

// A placeholder node to be replaced; it will never exist in a valid parse tree.
// Its token kind is not enforced even when valid.
using Placeholder = LeafNode<NodeKind::Placeholder, AnyToken>;

// File nodes
// ----------

// The start of the file.
using FileStart =
    LeafNode<NodeKind::FileStart, Token<Lex::TokenKind::FileStart>>;

// The end of the file.
using FileEnd = LeafNode<NodeKind::FileEnd, Token<Lex::TokenKind::FileEnd>>;

// General-purpose nodes
// ---------------------

// An empty declaration, such as `;`.
using EmptyDecl =
    LeafNode<NodeKind::EmptyDecl, TokenIfValid<Lex::TokenKind::Semi>,
             NodeCategory::Decl | NodeCategory::Statement>;

// A name in a non-expression context, such as a declaration.
using IdentifierName =
    LeafNode<NodeKind::IdentifierName, TokenIfValid<Lex::TokenKind::Identifier>,
             NodeCategory::MemberName>;

// A name in an expression context.
using IdentifierNameExpr =
    LeafNode<NodeKind::IdentifierNameExpr, Token<Lex::TokenKind::Identifier>,
             NodeCategory::Expr>;

// The `self` value and `Self` type identifier keywords. Typically of the form
// `self: Self`.
using SelfValueName = LeafNode<NodeKind::SelfValueName,
                               Token<Lex::TokenKind::SelfValueIdentifier>>;
using SelfValueNameExpr =
    LeafNode<NodeKind::SelfValueNameExpr,
             Token<Lex::TokenKind::SelfValueIdentifier>, NodeCategory::Expr>;
using SelfTypeNameExpr =
    LeafNode<NodeKind::SelfTypeNameExpr,
             Token<Lex::TokenKind::SelfTypeIdentifier>, NodeCategory::Expr>;

// The `base` value keyword, introduced by `base: B`. Typically referenced in
// an expression, as in `x.base` or `{.base = ...}`, but can also be used as a
// declared name, as in `{.base: partial B}`.
using BaseName = LeafNode<NodeKind::BaseName, Token<Lex::TokenKind::Base>,
                          NodeCategory::MemberName>;

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
  Token<Lex::TokenKind::Period> token;
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
using PackageExpr =
    LeafNode<NodeKind::PackageExpr, Token<Lex::TokenKind::Package>,
             NodeCategory::Expr>;

// The name of a package or library for `package`, `import`, and `library`.
using PackageName =
    LeafNode<NodeKind::PackageName, Token<Lex::TokenKind::Identifier>>;
using LibraryName =
    LeafNode<NodeKind::LibraryName, Token<Lex::TokenKind::StringLiteral>>;
using DefaultLibrary =
    LeafNode<NodeKind::DefaultLibrary, Token<Lex::TokenKind::Default>>;

using PackageIntroducer =
    LeafNode<NodeKind::PackageIntroducer, Token<Lex::TokenKind::Package>>;

// `library` in `package` or `import`.
struct LibrarySpecifier {
  static constexpr auto Kind =
      NodeKind::LibrarySpecifier.Define({.child_count = 1});

  Token<Lex::TokenKind::Library> token;
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
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// `import TheirPackage library "TheirLibrary";`
using ImportIntroducer =
    LeafNode<NodeKind::ImportIntroducer, Token<Lex::TokenKind::Import>>;
struct ImportDecl {
  static constexpr auto Kind = NodeKind::ImportDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = ImportIntroducer::Kind});

  ImportIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// `library` as declaration.
using LibraryIntroducer =
    LeafNode<NodeKind::LibraryIntroducer, Token<Lex::TokenKind::Library>>;
struct LibraryDecl {
  static constexpr auto Kind =
      NodeKind::LibraryDecl.Define({.category = NodeCategory::Decl,
                                    .bracketed_by = LibraryIntroducer::Kind});

  LibraryIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  NodeIdOneOf<LibraryName, DefaultLibrary> library_name;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// `export` as a declaration.
using ExportIntroducer =
    LeafNode<NodeKind::ExportIntroducer, Token<Lex::TokenKind::Export>>;
struct ExportDecl {
  static constexpr auto Kind = NodeKind::ExportDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = ExportIntroducer::Kind});

  ExportIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// Namespace nodes
// ---------------

using NamespaceStart =
    LeafNode<NodeKind::NamespaceStart, Token<Lex::TokenKind::Namespace>>;

// A namespace: `namespace N;`.
struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace.Define(
      {.category = NodeCategory::Decl, .bracketed_by = NamespaceStart::Kind});

  NamespaceStartId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// Pattern nodes
// -------------

// A pattern binding, such as `name: Type`.
struct BindingPattern {
  static constexpr auto Kind = NodeKind::BindingPattern.Define(
      {.category = NodeCategory::Pattern, .child_count = 2});

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  TokenIfValid<Lex::TokenKind::Colon> token;
  AnyExprId type;
};

// `name:! Type`
struct CompileTimeBindingPattern {
  static constexpr auto Kind = NodeKind::CompileTimeBindingPattern.Define(
      {.category = NodeCategory::Pattern, .child_count = 2});

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  Token<Lex::TokenKind::ColonExclaim> token;
  AnyExprId type;
};

// An address-of binding: `addr self: Self*`.
struct Addr {
  static constexpr auto Kind = NodeKind::Addr.Define(
      {.category = NodeCategory::Pattern, .child_count = 1});

  Token<Lex::TokenKind::Addr> token;
  AnyPatternId inner;
};

// A template binding: `template T:! type`.
struct Template {
  static constexpr auto Kind = NodeKind::Template.Define(
      {.category = NodeCategory::Pattern, .child_count = 1});

  Token<Lex::TokenKind::Template> token;
  // This is a CompileTimeBindingPatternId in any valid program.
  // TODO: Should the parser enforce that?
  AnyPatternId inner;
};

using TuplePatternStart =
    LeafNode<NodeKind::TuplePatternStart, Token<Lex::TokenKind::OpenParen>>;
using PatternListComma =
    LeafNode<NodeKind::PatternListComma, Token<Lex::TokenKind::Comma>>;

// A parameter list or tuple pattern: `(a: i32, b: i32)`.
struct TuplePattern {
  static constexpr auto Kind =
      NodeKind::TuplePattern.Define({.category = NodeCategory::Pattern,
                                     .bracketed_by = TuplePatternStart::Kind});

  TuplePatternStartId left_paren;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
  Token<Lex::TokenKind::CloseParen> token;
};

using ImplicitParamListStart =
    LeafNode<NodeKind::ImplicitParamListStart,
             Token<Lex::TokenKind::OpenSquareBracket>>;

// An implicit parameter list: `[T:! type, self: Self]`.
struct ImplicitParamList {
  static constexpr auto Kind = NodeKind::ImplicitParamList.Define(
      {.bracketed_by = ImplicitParamListStart::Kind});

  ImplicitParamListStartId left_square;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
  Token<Lex::TokenKind::CloseSquareBracket> token;
};

// Function nodes
// --------------

using FunctionIntroducer =
    LeafNode<NodeKind::FunctionIntroducer, Token<Lex::TokenKind::Fn>>;

// A return type: `-> i32`.
struct ReturnType {
  static constexpr auto Kind = NodeKind::ReturnType.Define({.child_count = 1});

  Token<Lex::TokenKind::MinusGreater> token;
  AnyExprId type;
};

// A function signature: `fn F() -> i32`.
template <const NodeKind& KindT, typename TokenKind, NodeCategory Category>
struct FunctionSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = FunctionIntroducer::Kind});

  FunctionIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  std::optional<ReturnTypeId> return_type;
  TokenKind token;
};

using FunctionDecl =
    FunctionSignature<NodeKind::FunctionDecl,
                      TokenIfValid<Lex::TokenKind::Semi>, NodeCategory::Decl>;
using FunctionDefinitionStart =
    FunctionSignature<NodeKind::FunctionDefinitionStart,
                      Token<Lex::TokenKind::OpenCurlyBrace>,
                      NodeCategory::None>;

// A function definition: `fn F() -> i32 { ... }`.
struct FunctionDefinition {
  static constexpr auto Kind = NodeKind::FunctionDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = FunctionDefinitionStart::Kind});

  FunctionDefinitionStartId signature;
  llvm::SmallVector<AnyStatementId> body;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
};

using BuiltinFunctionDefinitionStart =
    FunctionSignature<NodeKind::BuiltinFunctionDefinitionStart,
                      Token<Lex::TokenKind::Equal>, NodeCategory::None>;
using BuiltinName =
    LeafNode<NodeKind::BuiltinName, Token<Lex::TokenKind::StringLiteral>>;

// A builtin function definition: `fn F() -> i32 = "builtin name";`
struct BuiltinFunctionDefinition {
  static constexpr auto Kind = NodeKind::BuiltinFunctionDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = BuiltinFunctionDefinitionStart::Kind});

  BuiltinFunctionDefinitionStartId signature;
  BuiltinNameId builtin_name;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// `alias` nodes
// -------------

using AliasIntroducer =
    LeafNode<NodeKind::AliasIntroducer, Token<Lex::TokenKind::Alias>>;
using AliasInitializer =
    LeafNode<NodeKind::AliasInitializer, Token<Lex::TokenKind::Equal>>;

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
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// `let` nodes
// -----------

using LetIntroducer =
    LeafNode<NodeKind::LetIntroducer, Token<Lex::TokenKind::Let>>;
using LetInitializer =
    LeafNode<NodeKind::LetInitializer, Token<Lex::TokenKind::Equal>>;

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
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// `var` nodes
// -----------

using VariableIntroducer =
    LeafNode<NodeKind::VariableIntroducer, TokenIfValid<Lex::TokenKind::Var>>;
using ReturnedModifier =
    LeafNode<NodeKind::ReturnedModifier, Token<Lex::TokenKind::Returned>>;
using VariableInitializer =
    LeafNode<NodeKind::VariableInitializer, Token<Lex::TokenKind::Equal>>;

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
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// Statement nodes
// ---------------

using CodeBlockStart = LeafNode<NodeKind::CodeBlockStart,
                                TokenIfValid<Lex::TokenKind::OpenCurlyBrace>>;

// A code block: `{ statement; statement; ... }`.
struct CodeBlock {
  static constexpr auto Kind =
      NodeKind::CodeBlock.Define({.bracketed_by = CodeBlockStart::Kind});

  CodeBlockStartId left_brace;
  llvm::SmallVector<AnyStatementId> statements;
  TokenIfValid<Lex::TokenKind::CloseCurlyBrace> token;
};

// An expression statement: `F(x);`.
struct ExprStatement {
  static constexpr auto Kind = NodeKind::ExprStatement.Define(
      {.category = NodeCategory::Statement, .child_count = 1});

  AnyExprId expr;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

using BreakStatementStart =
    LeafNode<NodeKind::BreakStatementStart, Token<Lex::TokenKind::Break>>;

// A break statement: `break;`.
struct BreakStatement {
  static constexpr auto Kind = NodeKind::BreakStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = BreakStatementStart::Kind,
       .child_count = 1});

  BreakStatementStartId introducer;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

using ContinueStatementStart =
    LeafNode<NodeKind::ContinueStatementStart, Token<Lex::TokenKind::Continue>>;

// A continue statement: `continue;`.
struct ContinueStatement {
  static constexpr auto Kind = NodeKind::ContinueStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = ContinueStatementStart::Kind,
       .child_count = 1});

  ContinueStatementStartId introducer;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

using ReturnStatementStart =
    LeafNode<NodeKind::ReturnStatementStart, Token<Lex::TokenKind::Return>>;
using ReturnVarModifier =
    LeafNode<NodeKind::ReturnVarModifier, Token<Lex::TokenKind::Var>>;

// A return statement: `return;` or `return expr;` or `return var;`.
struct ReturnStatement {
  static constexpr auto Kind = NodeKind::ReturnStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = ReturnStatementStart::Kind});

  ReturnStatementStartId introducer;
  std::optional<AnyExprId> expr;
  std::optional<ReturnVarModifierId> var;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

using ForHeaderStart =
    LeafNode<NodeKind::ForHeaderStart, TokenIfValid<Lex::TokenKind::OpenParen>>;

// The `var ... in` portion of a `for` statement.
struct ForIn {
  static constexpr auto Kind = NodeKind::ForIn.Define(
      {.bracketed_by = VariableIntroducer::Kind, .child_count = 2});

  VariableIntroducerId introducer;
  TokenIfValid<Lex::TokenKind::In> token;
  AnyPatternId pattern;
};

// The `for (var ... in ...)` portion of a `for` statement.
struct ForHeader {
  static constexpr auto Kind =
      NodeKind::ForHeader.Define({.bracketed_by = ForHeaderStart::Kind});

  ForHeaderStartId introducer;
  ForInId var;
  AnyExprId range;
  TokenIfValid<Lex::TokenKind::CloseParen> token;
};

// A complete `for (...) { ... }` statement.
struct ForStatement {
  static constexpr auto Kind =
      NodeKind::ForStatement.Define({.category = NodeCategory::Statement,
                                     .bracketed_by = ForHeader::Kind,
                                     .child_count = 2});

  Token<Lex::TokenKind::For> token;
  ForHeaderId header;
  CodeBlockId body;
};

using IfConditionStart = LeafNode<NodeKind::IfConditionStart,
                                  TokenIfValid<Lex::TokenKind::OpenParen>>;

// The condition portion of an `if` statement: `(expr)`.
struct IfCondition {
  static constexpr auto Kind = NodeKind::IfCondition.Define(
      {.bracketed_by = IfConditionStart::Kind, .child_count = 2});

  IfConditionStartId left_paren;
  AnyExprId condition;
  TokenIfValid<Lex::TokenKind::OpenParen> token;
};

using IfStatementElse =
    LeafNode<NodeKind::IfStatementElse, Token<Lex::TokenKind::Else>>;

// An `if` statement: `if (expr) { ... } else { ... }`.
struct IfStatement {
  static constexpr auto Kind = NodeKind::IfStatement.Define(
      {.category = NodeCategory::Statement, .bracketed_by = IfCondition::Kind});

  Token<Lex::TokenKind::If> token;
  IfConditionId head;
  CodeBlockId then;

  struct Else {
    IfStatementElseId else_token;
    NodeIdOneOf<CodeBlock, IfStatement> body;
  };
  std::optional<Else> else_clause;
};

using WhileConditionStart = LeafNode<NodeKind::WhileConditionStart,
                                     TokenIfValid<Lex::TokenKind::OpenParen>>;

// The condition portion of a `while` statement: `(expr)`.
struct WhileCondition {
  static constexpr auto Kind = NodeKind::WhileCondition.Define(
      {.bracketed_by = WhileConditionStart::Kind, .child_count = 2});

  WhileConditionStartId left_paren;
  AnyExprId condition;
  TokenIfValid<Lex::TokenKind::CloseParen> token;
};

// A `while` statement: `while (expr) { ... }`.
struct WhileStatement {
  static constexpr auto Kind =
      NodeKind::WhileStatement.Define({.category = NodeCategory::Statement,
                                       .bracketed_by = WhileCondition::Kind,
                                       .child_count = 2});

  Token<Lex::TokenKind::While> token;
  WhileConditionId head;
  CodeBlockId body;
};

using MatchConditionStart = LeafNode<NodeKind::MatchConditionStart,
                                     TokenIfValid<Lex::TokenKind::OpenParen>>;

struct MatchCondition {
  static constexpr auto Kind = NodeKind::MatchCondition.Define(
      {.bracketed_by = MatchConditionStart::Kind, .child_count = 2});

  MatchConditionStartId left_paren;
  AnyExprId condition;
  TokenIfValid<Lex::TokenKind::CloseParen> token;
};

using MatchIntroducer =
    LeafNode<NodeKind::MatchIntroducer, Token<Lex::TokenKind::Match>>;
struct MatchStatementStart {
  static constexpr auto Kind = NodeKind::MatchStatementStart.Define(
      {.bracketed_by = MatchIntroducer::Kind, .child_count = 2});

  MatchIntroducerId introducer;
  MatchConditionId condition;
  Token<Lex::TokenKind::OpenCurlyBrace> token;
};

using MatchCaseIntroducer =
    LeafNode<NodeKind::MatchCaseIntroducer, Token<Lex::TokenKind::Case>>;
using MatchCaseGuardIntroducer =
    LeafNode<NodeKind::MatchCaseGuardIntroducer, Token<Lex::TokenKind::If>>;
using MatchCaseGuardStart = LeafNode<NodeKind::MatchCaseGuardStart,
                                     TokenIfValid<Lex::TokenKind::OpenParen>>;

struct MatchCaseGuard {
  static constexpr auto Kind = NodeKind::MatchCaseGuard.Define(
      {.bracketed_by = MatchCaseGuardIntroducer::Kind, .child_count = 3});

  MatchCaseGuardIntroducerId introducer;
  MatchCaseGuardStartId left_paren;
  AnyExprId condition;
  TokenIfValid<Lex::TokenKind::CloseParen> token;
};

using MatchCaseEqualGreater =
    LeafNode<NodeKind::MatchCaseEqualGreater,
             TokenIfValid<Lex::TokenKind::EqualGreater>>;

struct MatchCaseStart {
  static constexpr auto Kind = NodeKind::MatchCaseStart.Define(
      {.bracketed_by = MatchCaseIntroducer::Kind});

  MatchCaseIntroducerId introducer;
  AnyPatternId pattern;
  std::optional<MatchCaseGuardId> guard;
  MatchCaseEqualGreaterId equal_greater_token;
  TokenIfValid<Lex::TokenKind::OpenCurlyBrace> token;
};

struct MatchCase {
  static constexpr auto Kind =
      NodeKind::MatchCase.Define({.bracketed_by = MatchCaseStart::Kind});

  MatchCaseStartId head;
  llvm::SmallVector<AnyStatementId> statements;
  TokenIfValid<Lex::TokenKind::CloseCurlyBrace> token;
};

using MatchDefaultIntroducer =
    LeafNode<NodeKind::MatchDefaultIntroducer, Token<Lex::TokenKind::Default>>;
using MatchDefaultEqualGreater =
    LeafNode<NodeKind::MatchDefaultEqualGreater,
             TokenIfValid<Lex::TokenKind::EqualGreater>>;

struct MatchDefaultStart {
  static constexpr auto Kind = NodeKind::MatchDefaultStart.Define(
      {.bracketed_by = MatchDefaultIntroducer::Kind, .child_count = 2});

  MatchDefaultIntroducerId introducer;
  MatchDefaultEqualGreaterId equal_greater_token;
  TokenIfValid<Lex::TokenKind::OpenCurlyBrace> token;
};

struct MatchDefault {
  static constexpr auto Kind =
      NodeKind::MatchDefault.Define({.bracketed_by = MatchDefaultStart::Kind});

  MatchDefaultStartId introducer;
  llvm::SmallVector<AnyStatementId> statements;
  TokenIfValid<Lex::TokenKind::CloseCurlyBrace> token;
};

// A `match` statement: `match (expr) { case (...) => {...} default => {...}}`.
struct MatchStatement {
  static constexpr auto Kind = NodeKind::MatchStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = MatchStatementStart::Kind});

  MatchStatementStartId head;

  llvm::SmallVector<MatchCaseId> cases;
  std::optional<MatchDefaultId> default_case;
  TokenIfValid<Lex::TokenKind::CloseCurlyBrace> token;
};

// Expression nodes
// ----------------

using ArrayExprStart = LeafNode<NodeKind::ArrayExprStart,
                                Token<Lex::TokenKind::OpenSquareBracket>>;

// The start of an array type, `[i32;`.
//
// TODO: Consider flattening this into `ArrayExpr`.
struct ArrayExprSemi {
  static constexpr auto Kind = NodeKind::ArrayExprSemi.Define(
      {.bracketed_by = ArrayExprStart::Kind, .child_count = 2});

  ArrayExprStartId left_square;
  AnyExprId type;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// An array type, such as  `[i32; 3]` or `[i32;]`.
struct ArrayExpr {
  static constexpr auto Kind = NodeKind::ArrayExpr.Define(
      {.category = NodeCategory::Expr, .bracketed_by = ArrayExprSemi::Kind});

  ArrayExprSemiId start;
  std::optional<AnyExprId> bound;
  Token<Lex::TokenKind::CloseSquareBracket> token;
};

// The opening portion of an indexing expression: `a[`.
//
// TODO: Consider flattening this into `IndexExpr`.
struct IndexExprStart {
  static constexpr auto Kind =
      NodeKind::IndexExprStart.Define({.child_count = 1});

  AnyExprId sequence;
  Token<Lex::TokenKind::OpenSquareBracket> token;
};

// An indexing expression, such as `a[1]`.
struct IndexExpr {
  static constexpr auto Kind =
      NodeKind::IndexExpr.Define({.category = NodeCategory::Expr,
                                  .bracketed_by = IndexExprStart::Kind,
                                  .child_count = 2});

  IndexExprStartId start;
  AnyExprId index;
  Token<Lex::TokenKind::CloseSquareBracket> token;
};

using ParenExprStart =
    LeafNode<NodeKind::ParenExprStart, Token<Lex::TokenKind::OpenParen>>;

// A parenthesized expression: `(a)`.
struct ParenExpr {
  static constexpr auto Kind = NodeKind::ParenExpr.Define(
      {.category = NodeCategory::Expr | NodeCategory::MemberExpr,
       .bracketed_by = ParenExprStart::Kind,
       .child_count = 2});

  ParenExprStartId start;
  AnyExprId expr;
  Token<Lex::TokenKind::CloseParen> token;
};

using TupleLiteralStart =
    LeafNode<NodeKind::TupleLiteralStart, Token<Lex::TokenKind::OpenParen>>;
using TupleLiteralComma =
    LeafNode<NodeKind::TupleLiteralComma, Token<Lex::TokenKind::Comma>>;

// A tuple literal: `()`, `(a, b, c)`, or `(a,)`.
struct TupleLiteral {
  static constexpr auto Kind =
      NodeKind::TupleLiteral.Define({.category = NodeCategory::Expr,
                                     .bracketed_by = TupleLiteralStart::Kind});

  TupleLiteralStartId start;
  CommaSeparatedList<AnyExprId, TupleLiteralCommaId> elements;
  Token<Lex::TokenKind::CloseParen> token;
};

// The opening portion of a call expression: `F(`.
//
// TODO: Consider flattening this into `CallExpr`.
struct CallExprStart {
  static constexpr auto Kind =
      NodeKind::CallExprStart.Define({.child_count = 1});

  AnyExprId callee;
  Token<Lex::TokenKind::OpenParen> token;
};

using CallExprComma =
    LeafNode<NodeKind::CallExprComma, Token<Lex::TokenKind::Comma>>;

// A call expression: `F(a, b, c)`.
struct CallExpr {
  static constexpr auto Kind = NodeKind::CallExpr.Define(
      {.category = NodeCategory::Expr, .bracketed_by = CallExprStart::Kind});

  CallExprStartId start;
  CommaSeparatedList<AnyExprId, CallExprCommaId> arguments;
  Token<Lex::TokenKind::CloseParen> token;
};

// A member access expression: `a.b` or `a.(b)`.
struct MemberAccessExpr {
  static constexpr auto Kind = NodeKind::MemberAccessExpr.Define(
      {.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  Token<Lex::TokenKind::Period> token;
  AnyMemberNameOrMemberExprId rhs;
};

// An indirect member access expression: `a->b` or `a->(b)`.
struct PointerMemberAccessExpr {
  static constexpr auto Kind = NodeKind::PointerMemberAccessExpr.Define(
      {.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  Token<Lex::TokenKind::MinusGreater> token;
  AnyMemberNameOrMemberExprId rhs;
};

// A prefix operator expression.
template <const NodeKind& KindT, typename TokenKind>
struct PrefixOperator {
  static constexpr auto Kind =
      KindT.Define({.category = NodeCategory::Expr, .child_count = 1});

  TokenKind token;
  AnyExprId operand;
};

// An infix operator expression.
template <const NodeKind& KindT, typename TokenKind>
struct InfixOperator {
  static constexpr auto Kind =
      KindT.Define({.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  TokenKind token;
  AnyExprId rhs;
};

// A postfix operator expression.
template <const NodeKind& KindT, typename TokenKind>
struct PostfixOperator {
  static constexpr auto Kind =
      KindT.Define({.category = NodeCategory::Expr, .child_count = 1});

  AnyExprId operand;
  TokenKind token;
};

// Literals, operators, and modifiers

#define CARBON_PARSE_NODE_KIND(...)
#define CARBON_PARSE_NODE_KIND_TOKEN_LITERAL(Name, LexTokenKind)             \
  using Name = LeafNode<NodeKind::Name, Token<Lex::TokenKind::LexTokenKind>, \
                        NodeCategory::Expr>;
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name)                   \
  using Name##Modifier =                                              \
      LeafNode<NodeKind::Name##Modifier, Token<Lex::TokenKind::Name>, \
               NodeCategory::Modifier>;
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name)                          \
  using PrefixOperator##Name = PrefixOperator<NodeKind::PrefixOperator##Name, \
                                              Token<Lex::TokenKind::Name>>;
#define CARBON_PARSE_NODE_KIND_INFIX_OPERATOR(Name)                        \
  using InfixOperator##Name = InfixOperator<NodeKind::InfixOperator##Name, \
                                            Token<Lex::TokenKind::Name>>;
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name)  \
  using PostfixOperator##Name =                        \
      PostfixOperator<NodeKind::PostfixOperator##Name, \
                      Token<Lex::TokenKind::Name>>;
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
  VirtualToken<Lex::TokenKind::And> token;
};

struct ShortCircuitOperandOr {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperandOr.Define({.child_count = 1});

  AnyExprId operand;
  VirtualToken<Lex::TokenKind::Or> token;
};

struct ShortCircuitOperatorAnd {
  static constexpr auto Kind = NodeKind::ShortCircuitOperatorAnd.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = ShortCircuitOperandAnd::Kind,
       .child_count = 2});

  ShortCircuitOperandAndId lhs;
  Token<Lex::TokenKind::And> token;
  AnyExprId rhs;
};

struct ShortCircuitOperatorOr {
  static constexpr auto Kind = NodeKind::ShortCircuitOperatorOr.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = ShortCircuitOperandOr::Kind,
       .child_count = 2});

  ShortCircuitOperandOrId lhs;
  Token<Lex::TokenKind::Or> token;
  AnyExprId rhs;
};

// The `if` portion of an `if` expression: `if expr`.
struct IfExprIf {
  static constexpr auto Kind = NodeKind::IfExprIf.Define({.child_count = 1});

  Token<Lex::TokenKind::If> token;
  AnyExprId condition;
};

// The `then` portion of an `if` expression: `then expr`.
struct IfExprThen {
  static constexpr auto Kind = NodeKind::IfExprThen.Define({.child_count = 1});

  Token<Lex::TokenKind::Then> token;
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
  TokenIfValid<Lex::TokenKind::Else> token;
  AnyExprId else_result;
};

// Choice nodes
// ------------

using ChoiceIntroducer =
    LeafNode<NodeKind::ChoiceIntroducer, Token<Lex::TokenKind::Choice>>;

struct ChoiceSignature {
  static constexpr auto Kind = NodeKind::ChoiceDefinitionStart.Define(
      {.category = NodeCategory::None, .bracketed_by = ChoiceIntroducer::Kind});

  ChoiceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenIfValid<Lex::TokenKind::OpenCurlyBrace> token;
};

using ChoiceDefinitionStart = ChoiceSignature;

using ChoiceAlternativeListComma =
    LeafNode<NodeKind::ChoiceAlternativeListComma,
             Token<Lex::TokenKind::Comma>>;

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
  TokenIfValid<Lex::TokenKind::CloseCurlyBrace> token;
};

// Struct type and value literals
// ----------------------------------------

// `{`
using StructLiteralStart = LeafNode<NodeKind::StructLiteralStart,
                                    Token<Lex::TokenKind::OpenCurlyBrace>>;
using StructTypeLiteralStart = LeafNode<NodeKind::StructTypeLiteralStart,
                                        Token<Lex::TokenKind::OpenCurlyBrace>>;
// `,`
using StructComma =
    LeafNode<NodeKind::StructComma, Token<Lex::TokenKind::Comma>>;

// `.a`
struct StructFieldDesignator {
  static constexpr auto Kind =
      NodeKind::StructFieldDesignator.Define({.child_count = 1});

  Token<Lex::TokenKind::Period> token;
  NodeIdOneOf<IdentifierName, BaseName> name;
};

// `.a = 0`
struct StructField {
  static constexpr auto Kind = NodeKind::StructField.Define(
      {.bracketed_by = StructFieldDesignator::Kind, .child_count = 2});

  StructFieldDesignatorId designator;
  Token<Lex::TokenKind::Equal> token;
  AnyExprId expr;
};

// `.a: i32`
struct StructTypeField {
  static constexpr auto Kind = NodeKind::StructTypeField.Define(
      {.bracketed_by = StructFieldDesignator::Kind, .child_count = 2});

  StructFieldDesignatorId designator;
  Token<Lex::TokenKind::Colon> token;
  AnyExprId type_expr;
};

// Struct literals, such as `{.a = 0}`.
struct StructLiteral {
  static constexpr auto Kind = NodeKind::StructLiteral.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = StructLiteralStart::Kind});

  StructLiteralStartId start;
  CommaSeparatedList<StructFieldId, StructCommaId> fields;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
};

// Struct type literals, such as `{.a: i32}`.
struct StructTypeLiteral {
  static constexpr auto Kind = NodeKind::StructTypeLiteral.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = StructTypeLiteralStart::Kind});

  StructTypeLiteralStartId start;
  CommaSeparatedList<StructTypeFieldId, StructCommaId> fields;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
};

// `class` declarations and definitions
// ------------------------------------

// `class`
using ClassIntroducer =
    LeafNode<NodeKind::ClassIntroducer, Token<Lex::TokenKind::Class>>;

// A class signature `class C`
template <const NodeKind& KindT, typename TokenKind, NodeCategory Category>
struct ClassSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = ClassIntroducer::Kind});

  ClassIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenKind token;
};

// `class C;`
using ClassDecl =
    ClassSignature<NodeKind::ClassDecl, TokenIfValid<Lex::TokenKind::Semi>,
                   NodeCategory::Decl>;
// `class C {`
using ClassDefinitionStart =
    ClassSignature<NodeKind::ClassDefinitionStart,
                   Token<Lex::TokenKind::OpenCurlyBrace>, NodeCategory::None>;

// `class C { ... }`
struct ClassDefinition {
  static constexpr auto Kind = NodeKind::ClassDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ClassDefinitionStart::Kind});

  ClassDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
};

// Adapter declaration
// -------------------

// `adapt`
using AdaptIntroducer =
    LeafNode<NodeKind::AdaptIntroducer, Token<Lex::TokenKind::Adapt>>;
// `adapt SomeType;`
struct AdaptDecl {
  static constexpr auto Kind = NodeKind::AdaptDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = AdaptIntroducer::Kind});

  AdaptIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyExprId adapted_type;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// Base class declaration
// ----------------------

// `base`
using BaseIntroducer =
    LeafNode<NodeKind::BaseIntroducer, Token<Lex::TokenKind::Base>>;
using BaseColon = LeafNode<NodeKind::BaseColon, Token<Lex::TokenKind::Colon>>;
// `extend base: BaseClass;`
struct BaseDecl {
  static constexpr auto Kind = NodeKind::BaseDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = BaseIntroducer::Kind});

  BaseIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  BaseColonId colon;
  AnyExprId base_class;
  TokenIfValid<Lex::TokenKind::Semi> token;
};

// Interface declarations and definitions
// --------------------------------------

// `interface`
using InterfaceIntroducer =
    LeafNode<NodeKind::InterfaceIntroducer, Token<Lex::TokenKind::Interface>>;

// `interface I`
template <const NodeKind& KindT, typename TokenKind, NodeCategory Category>
struct InterfaceSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = InterfaceIntroducer::Kind});

  InterfaceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenKind token;
};

// `interface I;`
using InterfaceDecl =
    InterfaceSignature<NodeKind::InterfaceDecl,
                       TokenIfValid<Lex::TokenKind::Semi>, NodeCategory::Decl>;
// `interface I {`
using InterfaceDefinitionStart =
    InterfaceSignature<NodeKind::InterfaceDefinitionStart,
                       Token<Lex::TokenKind::OpenCurlyBrace>,
                       NodeCategory::None>;

// `interface I { ... }`
struct InterfaceDefinition {
  static constexpr auto Kind = NodeKind::InterfaceDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = InterfaceDefinitionStart::Kind});

  InterfaceDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
};

// `impl`...`as` declarations and definitions
// ------------------------------------------

// `impl`
using ImplIntroducer =
    LeafNode<NodeKind::ImplIntroducer, Token<Lex::TokenKind::Impl>>;

// `forall [...]`
struct ImplForall {
  static constexpr auto Kind = NodeKind::ImplForall.Define({.child_count = 1});

  Token<Lex::TokenKind::Forall> token;
  ImplicitParamListId params;
};

// `as` with no type before it
using DefaultSelfImplAs =
    LeafNode<NodeKind::DefaultSelfImplAs, Token<Lex::TokenKind::As>,
             NodeCategory::ImplAs>;

// `<type> as`
struct TypeImplAs {
  static constexpr auto Kind = NodeKind::TypeImplAs.Define(
      {.category = NodeCategory::ImplAs, .child_count = 1});

  AnyExprId type_expr;
  Token<Lex::TokenKind::As> token;
};

// `impl T as I`
template <const NodeKind& KindT, typename TokenKind, NodeCategory Category>
struct ImplSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = ImplIntroducer::Kind});

  ImplIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<ImplForallId> forall;
  AnyImplAsId as;
  AnyExprId interface;
  TokenKind token;
};

// `impl T as I;`
using ImplDecl =
    ImplSignature<NodeKind::ImplDecl, TokenIfValid<Lex::TokenKind::Semi>,
                  NodeCategory::Decl>;
// `impl T as I {`
using ImplDefinitionStart =
    ImplSignature<NodeKind::ImplDefinitionStart,
                  Token<Lex::TokenKind::OpenCurlyBrace>, NodeCategory::None>;

// `impl T as I { ... }`
struct ImplDefinition {
  static constexpr auto Kind = NodeKind::ImplDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ImplDefinitionStart::Kind});

  ImplDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
};

// Named constraint declarations and definitions
// ---------------------------------------------

// `constraint`
using NamedConstraintIntroducer = LeafNode<NodeKind::NamedConstraintIntroducer,
                                           Token<Lex::TokenKind::Constraint>>;

// `constraint NC`
template <const NodeKind& KindT, typename TokenKind, NodeCategory Category>
struct NamedConstraintSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = NamedConstraintIntroducer::Kind});

  NamedConstraintIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenKind token;
};

// `constraint NC;`
using NamedConstraintDecl =
    NamedConstraintSignature<NodeKind::NamedConstraintDecl,
                             TokenIfValid<Lex::TokenKind::Semi>,
                             NodeCategory::Decl>;
// `constraint NC {`
using NamedConstraintDefinitionStart =
    NamedConstraintSignature<NodeKind::NamedConstraintDefinitionStart,
                             Token<Lex::TokenKind::OpenCurlyBrace>,
                             NodeCategory::None>;

// `constraint NC { ... }`
struct NamedConstraintDefinition {
  static constexpr auto Kind = NodeKind::NamedConstraintDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = NamedConstraintDefinitionStart::Kind});

  NamedConstraintDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Token<Lex::TokenKind::CloseCurlyBrace> token;
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
