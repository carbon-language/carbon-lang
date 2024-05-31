// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
#define CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_

#include <optional>

#include "toolchain/lex/token_index.h"
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
          NodeCategory::RawEnumType Category = NodeCategory::None>
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
//
// Each parse node should also have exactly one field of type `Lex::*TokenIndex`
// that matches the token corresponding to the parse node itself. This is not a
// child node in the parse tree, but specifies the token kind associated with
// the parse node. The location of the field relative to the child nodes
// indicates the location within the corresponding grammar production where the
// token appears.
//
// There should be exactly one `Lex::*TokenIndex` field in each parse node, and
// it should always have the name `token`. In the case where a parse node can
// correspond to any kind of token, `Lex::TokenIndex` can be used instead. This
// should only be used when the node kind is either not used in a finished tree,
// such as `Placeholder`, or is always invalid, such as `InvalidParse`.
// ----------------------------------------------------------------------------

// Error nodes
// -----------

// An invalid parse. Used to balance the parse tree. This type is here only to
// ensure we have a type for each parse node kind. This node kind always has an
// error, so can never be extracted.
using InvalidParse = LeafNode<NodeKind::InvalidParse, Lex::TokenIndex,
                              NodeCategory::Decl | NodeCategory::Expr>;

// An invalid subtree. Always has an error so can never be extracted.
using InvalidParseStart =
    LeafNode<NodeKind::InvalidParseStart, Lex::TokenIndex>;
struct InvalidParseSubtree {
  static constexpr auto Kind = NodeKind::InvalidParseSubtree.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = InvalidParseStart::Kind});

  InvalidParseStartId start;
  llvm::SmallVector<NodeIdNot<InvalidParseStart>> extra;
  Lex::TokenIndex token;
};

// A placeholder node to be replaced; it will never exist in a valid parse tree.
// Its token kind is not enforced even when valid.
using Placeholder = LeafNode<NodeKind::Placeholder, Lex::TokenIndex>;

// File nodes
// ----------

// The start of the file.
using FileStart = LeafNode<NodeKind::FileStart, Lex::FileStartTokenIndex>;

// The end of the file.
using FileEnd = LeafNode<NodeKind::FileEnd, Lex::FileEndTokenIndex>;

// General-purpose nodes
// ---------------------

// An empty declaration, such as `;`.
using EmptyDecl = LeafNode<NodeKind::EmptyDecl, Lex::SemiTokenIndex,
                           NodeCategory::Decl | NodeCategory::Statement>;

// A name in a non-expression context, such as a declaration.
using IdentifierName =
    LeafNode<NodeKind::IdentifierName, Lex::IdentifierTokenIndex,
             NodeCategory::MemberName>;

// A name in an expression context.
using IdentifierNameExpr =
    LeafNode<NodeKind::IdentifierNameExpr, Lex::IdentifierTokenIndex,
             NodeCategory::Expr>;

// The `self` value and `Self` type identifier keywords. Typically of the form
// `self: Self`.
using SelfValueName =
    LeafNode<NodeKind::SelfValueName, Lex::SelfValueIdentifierTokenIndex>;
using SelfValueNameExpr =
    LeafNode<NodeKind::SelfValueNameExpr, Lex::SelfValueIdentifierTokenIndex,
             NodeCategory::Expr>;
using SelfTypeNameExpr =
    LeafNode<NodeKind::SelfTypeNameExpr, Lex::SelfTypeIdentifierTokenIndex,
             NodeCategory::Expr>;

// The `base` value keyword, introduced by `base: B`. Typically referenced in
// an expression, as in `x.base` or `{.base = ...}`, but can also be used as a
// declared name, as in `{.base: partial B}`.
using BaseName =
    LeafNode<NodeKind::BaseName, Lex::BaseTokenIndex, NodeCategory::MemberName>;

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
  Lex::PeriodTokenIndex token;
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
    LeafNode<NodeKind::PackageExpr, Lex::PackageTokenIndex, NodeCategory::Expr>;

// The name of a package or library for `package`, `import`, and `library`.
using PackageName = LeafNode<NodeKind::PackageName, Lex::IdentifierTokenIndex>;
using LibraryName =
    LeafNode<NodeKind::LibraryName, Lex::StringLiteralTokenIndex>;
using DefaultLibrary =
    LeafNode<NodeKind::DefaultLibrary, Lex::DefaultTokenIndex>;

using PackageIntroducer =
    LeafNode<NodeKind::PackageIntroducer, Lex::PackageTokenIndex>;

// `library` in `package` or `import`.
struct LibrarySpecifier {
  static constexpr auto Kind =
      NodeKind::LibrarySpecifier.Define({.child_count = 1});

  Lex::LibraryTokenIndex token;
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
  Lex::SemiTokenIndex token;
};

// `import TheirPackage library "TheirLibrary";`
using ImportIntroducer =
    LeafNode<NodeKind::ImportIntroducer, Lex::ImportTokenIndex>;
struct ImportDecl {
  static constexpr auto Kind = NodeKind::ImportDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = ImportIntroducer::Kind});

  ImportIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
  Lex::SemiTokenIndex token;
};

// `library` as declaration.
using LibraryIntroducer =
    LeafNode<NodeKind::LibraryIntroducer, Lex::LibraryTokenIndex>;
struct LibraryDecl {
  static constexpr auto Kind =
      NodeKind::LibraryDecl.Define({.category = NodeCategory::Decl,
                                    .bracketed_by = LibraryIntroducer::Kind});

  LibraryIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  NodeIdOneOf<LibraryName, DefaultLibrary> library_name;
  Lex::SemiTokenIndex token;
};

// `export` as a declaration.
using ExportIntroducer =
    LeafNode<NodeKind::ExportIntroducer, Lex::ExportTokenIndex>;
struct ExportDecl {
  static constexpr auto Kind = NodeKind::ExportDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = ExportIntroducer::Kind});

  ExportIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  Lex::SemiTokenIndex token;
};

// Namespace nodes
// ---------------

using NamespaceStart =
    LeafNode<NodeKind::NamespaceStart, Lex::NamespaceTokenIndex>;

// A namespace: `namespace N;`.
struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace.Define(
      {.category = NodeCategory::Decl, .bracketed_by = NamespaceStart::Kind});

  NamespaceStartId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  Lex::SemiTokenIndex token;
};

// Pattern nodes
// -------------

// A pattern binding, such as `name: Type`.
struct BindingPattern {
  static constexpr auto Kind = NodeKind::BindingPattern.Define(
      {.category = NodeCategory::Pattern, .child_count = 2});

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  Lex::ColonTokenIndex token;
  AnyExprId type;
};

// `name:! Type`
struct CompileTimeBindingPattern {
  static constexpr auto Kind = NodeKind::CompileTimeBindingPattern.Define(
      {.category = NodeCategory::Pattern, .child_count = 2});

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  Lex::ColonExclaimTokenIndex token;
  AnyExprId type;
};

// An address-of binding: `addr self: Self*`.
struct Addr {
  static constexpr auto Kind = NodeKind::Addr.Define(
      {.category = NodeCategory::Pattern, .child_count = 1});

  Lex::AddrTokenIndex token;
  AnyPatternId inner;
};

// A template binding: `template T:! type`.
struct Template {
  static constexpr auto Kind = NodeKind::Template.Define(
      {.category = NodeCategory::Pattern, .child_count = 1});

  Lex::TemplateTokenIndex token;
  // This is a CompileTimeBindingPatternId in any valid program.
  // TODO: Should the parser enforce that?
  AnyPatternId inner;
};

using TuplePatternStart =
    LeafNode<NodeKind::TuplePatternStart, Lex::OpenParenTokenIndex>;
using PatternListComma =
    LeafNode<NodeKind::PatternListComma, Lex::CommaTokenIndex>;

// A parameter list or tuple pattern: `(a: i32, b: i32)`.
struct TuplePattern {
  static constexpr auto Kind =
      NodeKind::TuplePattern.Define({.category = NodeCategory::Pattern,
                                     .bracketed_by = TuplePatternStart::Kind});

  TuplePatternStartId left_paren;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
  Lex::CloseParenTokenIndex token;
};

using ImplicitParamListStart = LeafNode<NodeKind::ImplicitParamListStart,
                                        Lex::OpenSquareBracketTokenIndex>;

// An implicit parameter list: `[T:! type, self: Self]`.
struct ImplicitParamList {
  static constexpr auto Kind = NodeKind::ImplicitParamList.Define(
      {.bracketed_by = ImplicitParamListStart::Kind});

  ImplicitParamListStartId left_square;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
  Lex::CloseSquareBracketTokenIndex token;
};

// Function nodes
// --------------

using FunctionIntroducer =
    LeafNode<NodeKind::FunctionIntroducer, Lex::FnTokenIndex>;

// A return type: `-> i32`.
struct ReturnType {
  static constexpr auto Kind = NodeKind::ReturnType.Define({.child_count = 1});

  Lex::MinusGreaterTokenIndex token;
  AnyExprId type;
};

// A function signature: `fn F() -> i32`.
template <const NodeKind& KindT, typename TokenKind,
          NodeCategory::RawEnumType Category>
struct FunctionSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = FunctionIntroducer::Kind});

  FunctionIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  std::optional<ReturnTypeId> return_type;
  TokenKind token;
};

using FunctionDecl = FunctionSignature<NodeKind::FunctionDecl,
                                       Lex::SemiTokenIndex, NodeCategory::Decl>;
using FunctionDefinitionStart =
    FunctionSignature<NodeKind::FunctionDefinitionStart,
                      Lex::OpenCurlyBraceTokenIndex, NodeCategory::None>;

// A function definition: `fn F() -> i32 { ... }`.
struct FunctionDefinition {
  static constexpr auto Kind = NodeKind::FunctionDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = FunctionDefinitionStart::Kind});

  FunctionDefinitionStartId signature;
  llvm::SmallVector<AnyStatementId> body;
  Lex::CloseCurlyBraceTokenIndex token;
};

using BuiltinFunctionDefinitionStart =
    FunctionSignature<NodeKind::BuiltinFunctionDefinitionStart,
                      Lex::EqualTokenIndex, NodeCategory::None>;
using BuiltinName =
    LeafNode<NodeKind::BuiltinName, Lex::StringLiteralTokenIndex>;

// A builtin function definition: `fn F() -> i32 = "builtin name";`
struct BuiltinFunctionDefinition {
  static constexpr auto Kind = NodeKind::BuiltinFunctionDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = BuiltinFunctionDefinitionStart::Kind});

  BuiltinFunctionDefinitionStartId signature;
  BuiltinNameId builtin_name;
  Lex::SemiTokenIndex token;
};

// `alias` nodes
// -------------

using AliasIntroducer =
    LeafNode<NodeKind::AliasIntroducer, Lex::AliasTokenIndex>;
using AliasInitializer =
    LeafNode<NodeKind::AliasInitializer, Lex::EqualTokenIndex>;

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
  Lex::SemiTokenIndex token;
};

// `let` nodes
// -----------

using LetIntroducer = LeafNode<NodeKind::LetIntroducer, Lex::LetTokenIndex>;
using LetInitializer = LeafNode<NodeKind::LetInitializer, Lex::EqualTokenIndex>;

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
  Lex::SemiTokenIndex token;
};

// `var` nodes
// -----------

using VariableIntroducer =
    LeafNode<NodeKind::VariableIntroducer, Lex::VarTokenIndex>;
using ReturnedModifier =
    LeafNode<NodeKind::ReturnedModifier, Lex::ReturnedTokenIndex>;
using VariableInitializer =
    LeafNode<NodeKind::VariableInitializer, Lex::EqualTokenIndex>;

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
  Lex::SemiTokenIndex token;
};

// Statement nodes
// ---------------

using CodeBlockStart =
    LeafNode<NodeKind::CodeBlockStart, Lex::OpenCurlyBraceTokenIndex>;

// A code block: `{ statement; statement; ... }`.
struct CodeBlock {
  static constexpr auto Kind =
      NodeKind::CodeBlock.Define({.bracketed_by = CodeBlockStart::Kind});

  CodeBlockStartId left_brace;
  llvm::SmallVector<AnyStatementId> statements;
  Lex::CloseCurlyBraceTokenIndex token;
};

// An expression statement: `F(x);`.
struct ExprStatement {
  static constexpr auto Kind = NodeKind::ExprStatement.Define(
      {.category = NodeCategory::Statement, .child_count = 1});

  AnyExprId expr;
  Lex::SemiTokenIndex token;
};

using BreakStatementStart =
    LeafNode<NodeKind::BreakStatementStart, Lex::BreakTokenIndex>;

// A break statement: `break;`.
struct BreakStatement {
  static constexpr auto Kind = NodeKind::BreakStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = BreakStatementStart::Kind,
       .child_count = 1});

  BreakStatementStartId introducer;
  Lex::SemiTokenIndex token;
};

using ContinueStatementStart =
    LeafNode<NodeKind::ContinueStatementStart, Lex::ContinueTokenIndex>;

// A continue statement: `continue;`.
struct ContinueStatement {
  static constexpr auto Kind = NodeKind::ContinueStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = ContinueStatementStart::Kind,
       .child_count = 1});

  ContinueStatementStartId introducer;
  Lex::SemiTokenIndex token;
};

using ReturnStatementStart =
    LeafNode<NodeKind::ReturnStatementStart, Lex::ReturnTokenIndex>;
using ReturnVarModifier =
    LeafNode<NodeKind::ReturnVarModifier, Lex::VarTokenIndex>;

// A return statement: `return;` or `return expr;` or `return var;`.
struct ReturnStatement {
  static constexpr auto Kind = NodeKind::ReturnStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = ReturnStatementStart::Kind});

  ReturnStatementStartId introducer;
  std::optional<AnyExprId> expr;
  std::optional<ReturnVarModifierId> var;
  Lex::SemiTokenIndex token;
};

using ForHeaderStart =
    LeafNode<NodeKind::ForHeaderStart, Lex::OpenParenTokenIndex>;

// The `var ... in` portion of a `for` statement.
struct ForIn {
  static constexpr auto Kind = NodeKind::ForIn.Define(
      {.bracketed_by = VariableIntroducer::Kind, .child_count = 2});

  VariableIntroducerId introducer;
  Lex::InTokenIndex token;
  AnyPatternId pattern;
};

// The `for (var ... in ...)` portion of a `for` statement.
struct ForHeader {
  static constexpr auto Kind =
      NodeKind::ForHeader.Define({.bracketed_by = ForHeaderStart::Kind});

  ForHeaderStartId introducer;
  ForInId var;
  AnyExprId range;
  Lex::CloseParenTokenIndex token;
};

// A complete `for (...) { ... }` statement.
struct ForStatement {
  static constexpr auto Kind =
      NodeKind::ForStatement.Define({.category = NodeCategory::Statement,
                                     .bracketed_by = ForHeader::Kind,
                                     .child_count = 2});

  Lex::ForTokenIndex token;
  ForHeaderId header;
  CodeBlockId body;
};

using IfConditionStart =
    LeafNode<NodeKind::IfConditionStart, Lex::OpenParenTokenIndex>;

// The condition portion of an `if` statement: `(expr)`.
struct IfCondition {
  static constexpr auto Kind = NodeKind::IfCondition.Define(
      {.bracketed_by = IfConditionStart::Kind, .child_count = 2});

  IfConditionStartId left_paren;
  AnyExprId condition;
  Lex::CloseParenTokenIndex token;
};

using IfStatementElse =
    LeafNode<NodeKind::IfStatementElse, Lex::ElseTokenIndex>;

// An `if` statement: `if (expr) { ... } else { ... }`.
struct IfStatement {
  static constexpr auto Kind = NodeKind::IfStatement.Define(
      {.category = NodeCategory::Statement, .bracketed_by = IfCondition::Kind});

  Lex::IfTokenIndex token;
  IfConditionId head;
  CodeBlockId then;

  struct Else {
    IfStatementElseId else_token;
    NodeIdOneOf<CodeBlock, IfStatement> body;
  };
  std::optional<Else> else_clause;
};

using WhileConditionStart =
    LeafNode<NodeKind::WhileConditionStart, Lex::OpenParenTokenIndex>;

// The condition portion of a `while` statement: `(expr)`.
struct WhileCondition {
  static constexpr auto Kind = NodeKind::WhileCondition.Define(
      {.bracketed_by = WhileConditionStart::Kind, .child_count = 2});

  WhileConditionStartId left_paren;
  AnyExprId condition;
  Lex::CloseParenTokenIndex token;
};

// A `while` statement: `while (expr) { ... }`.
struct WhileStatement {
  static constexpr auto Kind =
      NodeKind::WhileStatement.Define({.category = NodeCategory::Statement,
                                       .bracketed_by = WhileCondition::Kind,
                                       .child_count = 2});

  Lex::WhileTokenIndex token;
  WhileConditionId head;
  CodeBlockId body;
};

using MatchConditionStart =
    LeafNode<NodeKind::MatchConditionStart, Lex::OpenParenTokenIndex>;

struct MatchCondition {
  static constexpr auto Kind = NodeKind::MatchCondition.Define(
      {.bracketed_by = MatchConditionStart::Kind, .child_count = 2});

  MatchConditionStartId left_paren;
  AnyExprId condition;
  Lex::CloseParenTokenIndex token;
};

using MatchIntroducer =
    LeafNode<NodeKind::MatchIntroducer, Lex::MatchTokenIndex>;
struct MatchStatementStart {
  static constexpr auto Kind = NodeKind::MatchStatementStart.Define(
      {.bracketed_by = MatchIntroducer::Kind, .child_count = 2});

  MatchIntroducerId introducer;
  MatchConditionId condition;
  Lex::OpenCurlyBraceTokenIndex token;
};

using MatchCaseIntroducer =
    LeafNode<NodeKind::MatchCaseIntroducer, Lex::CaseTokenIndex>;
using MatchCaseGuardIntroducer =
    LeafNode<NodeKind::MatchCaseGuardIntroducer, Lex::IfTokenIndex>;
using MatchCaseGuardStart =
    LeafNode<NodeKind::MatchCaseGuardStart, Lex::OpenParenTokenIndex>;

struct MatchCaseGuard {
  static constexpr auto Kind = NodeKind::MatchCaseGuard.Define(
      {.bracketed_by = MatchCaseGuardIntroducer::Kind, .child_count = 3});

  MatchCaseGuardIntroducerId introducer;
  MatchCaseGuardStartId left_paren;
  AnyExprId condition;
  Lex::CloseParenTokenIndex token;
};

using MatchCaseEqualGreater =
    LeafNode<NodeKind::MatchCaseEqualGreater, Lex::EqualGreaterTokenIndex>;

struct MatchCaseStart {
  static constexpr auto Kind = NodeKind::MatchCaseStart.Define(
      {.bracketed_by = MatchCaseIntroducer::Kind});

  MatchCaseIntroducerId introducer;
  AnyPatternId pattern;
  std::optional<MatchCaseGuardId> guard;
  MatchCaseEqualGreaterId equal_greater_token;
  Lex::OpenCurlyBraceTokenIndex token;
};

struct MatchCase {
  static constexpr auto Kind =
      NodeKind::MatchCase.Define({.bracketed_by = MatchCaseStart::Kind});

  MatchCaseStartId head;
  llvm::SmallVector<AnyStatementId> statements;
  Lex::CloseCurlyBraceTokenIndex token;
};

using MatchDefaultIntroducer =
    LeafNode<NodeKind::MatchDefaultIntroducer, Lex::DefaultTokenIndex>;
using MatchDefaultEqualGreater =
    LeafNode<NodeKind::MatchDefaultEqualGreater, Lex::EqualGreaterTokenIndex>;

struct MatchDefaultStart {
  static constexpr auto Kind = NodeKind::MatchDefaultStart.Define(
      {.bracketed_by = MatchDefaultIntroducer::Kind, .child_count = 2});

  MatchDefaultIntroducerId introducer;
  MatchDefaultEqualGreaterId equal_greater_token;
  Lex::OpenCurlyBraceTokenIndex token;
};

struct MatchDefault {
  static constexpr auto Kind =
      NodeKind::MatchDefault.Define({.bracketed_by = MatchDefaultStart::Kind});

  MatchDefaultStartId introducer;
  llvm::SmallVector<AnyStatementId> statements;
  Lex::CloseCurlyBraceTokenIndex token;
};

// A `match` statement: `match (expr) { case (...) => {...} default => {...}}`.
struct MatchStatement {
  static constexpr auto Kind = NodeKind::MatchStatement.Define(
      {.category = NodeCategory::Statement,
       .bracketed_by = MatchStatementStart::Kind});

  MatchStatementStartId head;

  llvm::SmallVector<MatchCaseId> cases;
  std::optional<MatchDefaultId> default_case;
  Lex::CloseCurlyBraceTokenIndex token;
};

// Expression nodes
// ----------------

using ArrayExprStart =
    LeafNode<NodeKind::ArrayExprStart, Lex::OpenSquareBracketTokenIndex>;

// The start of an array type, `[i32;`.
//
// TODO: Consider flattening this into `ArrayExpr`.
struct ArrayExprSemi {
  static constexpr auto Kind = NodeKind::ArrayExprSemi.Define(
      {.bracketed_by = ArrayExprStart::Kind, .child_count = 2});

  ArrayExprStartId left_square;
  AnyExprId type;
  Lex::SemiTokenIndex token;
};

// An array type, such as  `[i32; 3]` or `[i32;]`.
struct ArrayExpr {
  static constexpr auto Kind = NodeKind::ArrayExpr.Define(
      {.category = NodeCategory::Expr, .bracketed_by = ArrayExprSemi::Kind});

  ArrayExprSemiId start;
  std::optional<AnyExprId> bound;
  Lex::CloseSquareBracketTokenIndex token;
};

// The opening portion of an indexing expression: `a[`.
//
// TODO: Consider flattening this into `IndexExpr`.
struct IndexExprStart {
  static constexpr auto Kind =
      NodeKind::IndexExprStart.Define({.child_count = 1});

  AnyExprId sequence;
  Lex::OpenSquareBracketTokenIndex token;
};

// An indexing expression, such as `a[1]`.
struct IndexExpr {
  static constexpr auto Kind =
      NodeKind::IndexExpr.Define({.category = NodeCategory::Expr,
                                  .bracketed_by = IndexExprStart::Kind,
                                  .child_count = 2});

  IndexExprStartId start;
  AnyExprId index;
  Lex::CloseSquareBracketTokenIndex token;
};

using ParenExprStart =
    LeafNode<NodeKind::ParenExprStart, Lex::OpenParenTokenIndex>;

// A parenthesized expression: `(a)`.
struct ParenExpr {
  static constexpr auto Kind = NodeKind::ParenExpr.Define(
      {.category = NodeCategory::Expr | NodeCategory::MemberExpr,
       .bracketed_by = ParenExprStart::Kind,
       .child_count = 2});

  ParenExprStartId start;
  AnyExprId expr;
  Lex::CloseParenTokenIndex token;
};

using TupleLiteralStart =
    LeafNode<NodeKind::TupleLiteralStart, Lex::OpenParenTokenIndex>;
using TupleLiteralComma =
    LeafNode<NodeKind::TupleLiteralComma, Lex::CommaTokenIndex>;

// A tuple literal: `()`, `(a, b, c)`, or `(a,)`.
struct TupleLiteral {
  static constexpr auto Kind =
      NodeKind::TupleLiteral.Define({.category = NodeCategory::Expr,
                                     .bracketed_by = TupleLiteralStart::Kind});

  TupleLiteralStartId start;
  CommaSeparatedList<AnyExprId, TupleLiteralCommaId> elements;
  Lex::CloseParenTokenIndex token;
};

// The opening portion of a call expression: `F(`.
//
// TODO: Consider flattening this into `CallExpr`.
struct CallExprStart {
  static constexpr auto Kind =
      NodeKind::CallExprStart.Define({.child_count = 1});

  AnyExprId callee;
  Lex::OpenParenTokenIndex token;
};

using CallExprComma = LeafNode<NodeKind::CallExprComma, Lex::CommaTokenIndex>;

// A call expression: `F(a, b, c)`.
struct CallExpr {
  static constexpr auto Kind = NodeKind::CallExpr.Define(
      {.category = NodeCategory::Expr, .bracketed_by = CallExprStart::Kind});

  CallExprStartId start;
  CommaSeparatedList<AnyExprId, CallExprCommaId> arguments;
  Lex::CloseParenTokenIndex token;
};

// A member access expression: `a.b` or `a.(b)`.
struct MemberAccessExpr {
  static constexpr auto Kind = NodeKind::MemberAccessExpr.Define(
      {.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  Lex::PeriodTokenIndex token;
  AnyMemberNameOrMemberExprId rhs;
};

// An indirect member access expression: `a->b` or `a->(b)`.
struct PointerMemberAccessExpr {
  static constexpr auto Kind = NodeKind::PointerMemberAccessExpr.Define(
      {.category = NodeCategory::Expr, .child_count = 2});

  AnyExprId lhs;
  Lex::MinusGreaterTokenIndex token;
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
#define CARBON_PARSE_NODE_KIND_TOKEN_LITERAL(Name, LexTokenKind)       \
  using Name = LeafNode<NodeKind::Name, Lex::LexTokenKind##TokenIndex, \
                        NodeCategory::Expr>;
#define CARBON_PARSE_NODE_KIND_TOKEN_MODIFIER(Name)             \
  using Name##Modifier =                                        \
      LeafNode<NodeKind::Name##Modifier, Lex::Name##TokenIndex, \
               NodeCategory::Modifier>;
#define CARBON_PARSE_NODE_KIND_PREFIX_OPERATOR(Name) \
  using PrefixOperator##Name =                       \
      PrefixOperator<NodeKind::PrefixOperator##Name, Lex::Name##TokenIndex>;
#define CARBON_PARSE_NODE_KIND_INFIX_OPERATOR(Name) \
  using InfixOperator##Name =                       \
      InfixOperator<NodeKind::InfixOperator##Name, Lex::Name##TokenIndex>;
#define CARBON_PARSE_NODE_KIND_POSTFIX_OPERATOR(Name) \
  using PostfixOperator##Name =                       \
      PostfixOperator<NodeKind::PostfixOperator##Name, Lex::Name##TokenIndex>;
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
  // This is a virtual token. The `and` token is owned by the
  // ShortCircuitOperatorAnd node.
  Lex::AndTokenIndex token;
};

struct ShortCircuitOperandOr {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperandOr.Define({.child_count = 1});

  AnyExprId operand;
  // This is a virtual token. The `or` token is owned by the
  // ShortCircuitOperatorOr node.
  Lex::OrTokenIndex token;
};

struct ShortCircuitOperatorAnd {
  static constexpr auto Kind = NodeKind::ShortCircuitOperatorAnd.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = ShortCircuitOperandAnd::Kind,
       .child_count = 2});

  ShortCircuitOperandAndId lhs;
  Lex::AndTokenIndex token;
  AnyExprId rhs;
};

struct ShortCircuitOperatorOr {
  static constexpr auto Kind = NodeKind::ShortCircuitOperatorOr.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = ShortCircuitOperandOr::Kind,
       .child_count = 2});

  ShortCircuitOperandOrId lhs;
  Lex::OrTokenIndex token;
  AnyExprId rhs;
};

// The `if` portion of an `if` expression: `if expr`.
struct IfExprIf {
  static constexpr auto Kind = NodeKind::IfExprIf.Define({.child_count = 1});

  Lex::IfTokenIndex token;
  AnyExprId condition;
};

// The `then` portion of an `if` expression: `then expr`.
struct IfExprThen {
  static constexpr auto Kind = NodeKind::IfExprThen.Define({.child_count = 1});

  Lex::ThenTokenIndex token;
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
  Lex::ElseTokenIndex token;
  AnyExprId else_result;
};

// Choice nodes
// ------------

using ChoiceIntroducer =
    LeafNode<NodeKind::ChoiceIntroducer, Lex::ChoiceTokenIndex>;

struct ChoiceSignature {
  static constexpr auto Kind = NodeKind::ChoiceDefinitionStart.Define(
      {.category = NodeCategory::None, .bracketed_by = ChoiceIntroducer::Kind});

  ChoiceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  Lex::OpenCurlyBraceTokenIndex token;
};

using ChoiceDefinitionStart = ChoiceSignature;

using ChoiceAlternativeListComma =
    LeafNode<NodeKind::ChoiceAlternativeListComma, Lex::CommaTokenIndex>;

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
  Lex::CloseCurlyBraceTokenIndex token;
};

// Struct type and value literals
// ----------------------------------------

// `{`
using StructLiteralStart =
    LeafNode<NodeKind::StructLiteralStart, Lex::OpenCurlyBraceTokenIndex>;
using StructTypeLiteralStart =
    LeafNode<NodeKind::StructTypeLiteralStart, Lex::OpenCurlyBraceTokenIndex>;
// `,`
using StructComma = LeafNode<NodeKind::StructComma, Lex::CommaTokenIndex>;

// `.a`
struct StructFieldDesignator {
  static constexpr auto Kind =
      NodeKind::StructFieldDesignator.Define({.child_count = 1});

  Lex::PeriodTokenIndex token;
  NodeIdOneOf<IdentifierName, BaseName> name;
};

// `.a = 0`
struct StructField {
  static constexpr auto Kind = NodeKind::StructField.Define(
      {.bracketed_by = StructFieldDesignator::Kind, .child_count = 2});

  StructFieldDesignatorId designator;
  Lex::EqualTokenIndex token;
  AnyExprId expr;
};

// `.a: i32`
struct StructTypeField {
  static constexpr auto Kind = NodeKind::StructTypeField.Define(
      {.bracketed_by = StructFieldDesignator::Kind, .child_count = 2});

  StructFieldDesignatorId designator;
  Lex::ColonTokenIndex token;
  AnyExprId type_expr;
};

// Struct literals, such as `{.a = 0}`.
struct StructLiteral {
  static constexpr auto Kind = NodeKind::StructLiteral.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = StructLiteralStart::Kind});

  StructLiteralStartId start;
  CommaSeparatedList<StructFieldId, StructCommaId> fields;
  Lex::CloseCurlyBraceTokenIndex token;
};

// Struct type literals, such as `{.a: i32}`.
struct StructTypeLiteral {
  static constexpr auto Kind = NodeKind::StructTypeLiteral.Define(
      {.category = NodeCategory::Expr,
       .bracketed_by = StructTypeLiteralStart::Kind});

  StructTypeLiteralStartId start;
  CommaSeparatedList<StructTypeFieldId, StructCommaId> fields;
  Lex::CloseCurlyBraceTokenIndex token;
};

// `class` declarations and definitions
// ------------------------------------

// `class`
using ClassIntroducer =
    LeafNode<NodeKind::ClassIntroducer, Lex::ClassTokenIndex>;

// A class signature `class C`
template <const NodeKind& KindT, typename TokenKind,
          NodeCategory::RawEnumType Category>
struct ClassSignature {
  static constexpr auto Kind = KindT.Define(
      {.category = Category, .bracketed_by = ClassIntroducer::Kind});

  ClassIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  DeclName name;
  TokenKind token;
};

// `class C;`
using ClassDecl = ClassSignature<NodeKind::ClassDecl, Lex::SemiTokenIndex,
                                 NodeCategory::Decl>;
// `class C {`
using ClassDefinitionStart =
    ClassSignature<NodeKind::ClassDefinitionStart,
                   Lex::OpenCurlyBraceTokenIndex, NodeCategory::None>;

// `class C { ... }`
struct ClassDefinition {
  static constexpr auto Kind = NodeKind::ClassDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ClassDefinitionStart::Kind});

  ClassDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Lex::CloseCurlyBraceTokenIndex token;
};

// Adapter declaration
// -------------------

// `adapt`
using AdaptIntroducer =
    LeafNode<NodeKind::AdaptIntroducer, Lex::AdaptTokenIndex>;
// `adapt SomeType;`
struct AdaptDecl {
  static constexpr auto Kind = NodeKind::AdaptDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = AdaptIntroducer::Kind});

  AdaptIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyExprId adapted_type;
  Lex::SemiTokenIndex token;
};

// Base class declaration
// ----------------------

// `base`
using BaseIntroducer = LeafNode<NodeKind::BaseIntroducer, Lex::BaseTokenIndex>;
using BaseColon = LeafNode<NodeKind::BaseColon, Lex::ColonTokenIndex>;
// `extend base: BaseClass;`
struct BaseDecl {
  static constexpr auto Kind = NodeKind::BaseDecl.Define(
      {.category = NodeCategory::Decl, .bracketed_by = BaseIntroducer::Kind});

  BaseIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  BaseColonId colon;
  AnyExprId base_class;
  Lex::SemiTokenIndex token;
};

// Interface declarations and definitions
// --------------------------------------

// `interface`
using InterfaceIntroducer =
    LeafNode<NodeKind::InterfaceIntroducer, Lex::InterfaceTokenIndex>;

// `interface I`
template <const NodeKind& KindT, typename TokenKind,
          NodeCategory::RawEnumType Category>
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
    InterfaceSignature<NodeKind::InterfaceDecl, Lex::SemiTokenIndex,
                       NodeCategory::Decl>;
// `interface I {`
using InterfaceDefinitionStart =
    InterfaceSignature<NodeKind::InterfaceDefinitionStart,
                       Lex::OpenCurlyBraceTokenIndex, NodeCategory::None>;

// `interface I { ... }`
struct InterfaceDefinition {
  static constexpr auto Kind = NodeKind::InterfaceDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = InterfaceDefinitionStart::Kind});

  InterfaceDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Lex::CloseCurlyBraceTokenIndex token;
};

// `impl`...`as` declarations and definitions
// ------------------------------------------

// `impl`
using ImplIntroducer = LeafNode<NodeKind::ImplIntroducer, Lex::ImplTokenIndex>;

// `forall [...]`
struct ImplForall {
  static constexpr auto Kind = NodeKind::ImplForall.Define({.child_count = 1});

  Lex::ForallTokenIndex token;
  ImplicitParamListId params;
};

// `as` with no type before it
using DefaultSelfImplAs = LeafNode<NodeKind::DefaultSelfImplAs,
                                   Lex::AsTokenIndex, NodeCategory::ImplAs>;

// `<type> as`
struct TypeImplAs {
  static constexpr auto Kind = NodeKind::TypeImplAs.Define(
      {.category = NodeCategory::ImplAs, .child_count = 1});

  AnyExprId type_expr;
  Lex::AsTokenIndex token;
};

// `impl T as I`
template <const NodeKind& KindT, typename TokenKind,
          NodeCategory::RawEnumType Category>
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
    ImplSignature<NodeKind::ImplDecl, Lex::SemiTokenIndex, NodeCategory::Decl>;
// `impl T as I {`
using ImplDefinitionStart =
    ImplSignature<NodeKind::ImplDefinitionStart, Lex::OpenCurlyBraceTokenIndex,
                  NodeCategory::None>;

// `impl T as I { ... }`
struct ImplDefinition {
  static constexpr auto Kind = NodeKind::ImplDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = ImplDefinitionStart::Kind});

  ImplDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Lex::CloseCurlyBraceTokenIndex token;
};

// Named constraint declarations and definitions
// ---------------------------------------------

// `constraint`
using NamedConstraintIntroducer =
    LeafNode<NodeKind::NamedConstraintIntroducer, Lex::ConstraintTokenIndex>;

// `constraint NC`
template <const NodeKind& KindT, typename TokenKind,
          NodeCategory::RawEnumType Category>
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
    NamedConstraintSignature<NodeKind::NamedConstraintDecl, Lex::SemiTokenIndex,
                             NodeCategory::Decl>;
// `constraint NC {`
using NamedConstraintDefinitionStart =
    NamedConstraintSignature<NodeKind::NamedConstraintDefinitionStart,
                             Lex::OpenCurlyBraceTokenIndex, NodeCategory::None>;

// `constraint NC { ... }`
struct NamedConstraintDefinition {
  static constexpr auto Kind = NodeKind::NamedConstraintDefinition.Define(
      {.category = NodeCategory::Decl,
       .bracketed_by = NamedConstraintDefinitionStart::Kind});

  NamedConstraintDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
  Lex::CloseCurlyBraceTokenIndex token;
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
