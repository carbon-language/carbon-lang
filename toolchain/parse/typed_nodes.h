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
template <const NodeKind& KindT, NodeCategory Category = NodeCategory::None>
struct LeafNode {
  static constexpr auto Kind = KindT.Define(Category);
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
  static constexpr auto Kind =
      NodeKind::InvalidParseSubtree.Define(NodeCategory::Decl);

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
    LeafNode<NodeKind::IdentifierName,
             NodeCategory::NameComponent | NodeCategory::MemberName>;

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

// A qualified name: `A.B`.
struct QualifiedName {
  static constexpr auto Kind =
      NodeKind::QualifiedName.Define(NodeCategory::NameComponent);

  // For now, this is either an IdentifierName or a QualifiedName.
  AnyNameComponentId lhs;

  // TODO: This will eventually need to support more general expressions, for
  // example `GenericType(type_args).ChildType(child_type_args).Name`.
  IdentifierNameId rhs;
};

// Library, package, import
// ------------------------

// The `package` keyword in an expression.
using PackageExpr = LeafNode<NodeKind::PackageExpr, NodeCategory::Expr>;

// The name of a package or library for `package`, `import`, and `library`.
using PackageName = LeafNode<NodeKind::PackageName>;
using LibraryName = LeafNode<NodeKind::LibraryName>;
using DefaultLibrary = LeafNode<NodeKind::DefaultLibrary>;

using PackageIntroducer = LeafNode<NodeKind::PackageIntroducer>;
using PackageApi = LeafNode<NodeKind::PackageApi>;
using PackageImpl = LeafNode<NodeKind::PackageImpl>;

// `library` in `package` or `import`.
struct LibrarySpecifier {
  static constexpr auto Kind = NodeKind::LibrarySpecifier.Define();

  NodeIdOneOf<LibraryName, DefaultLibrary> name;
};

// First line of the file, such as:
//   `package MyPackage library "MyLibrary" impl;`
struct PackageDirective {
  static constexpr auto Kind =
      NodeKind::PackageDirective.Define(NodeCategory::Decl);

  PackageIntroducerId introducer;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
  NodeIdOneOf<PackageApi, PackageImpl> api_or_impl;
};

// `import TheirPackage library "TheirLibrary";`
using ImportIntroducer = LeafNode<NodeKind::ImportIntroducer>;
struct ImportDirective {
  static constexpr auto Kind =
      NodeKind::ImportDirective.Define(NodeCategory::Decl);

  ImportIntroducerId introducer;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
};

// `library` as directive.
using LibraryIntroducer = LeafNode<NodeKind::LibraryIntroducer>;
struct LibraryDirective {
  static constexpr auto Kind =
      NodeKind::LibraryDirective.Define(NodeCategory::Decl);

  LibraryIntroducerId introducer;
  NodeIdOneOf<LibraryName, DefaultLibrary> library_name;
  NodeIdOneOf<PackageApi, PackageImpl> api_or_impl;
};

// Namespace nodes
// ---------------

using NamespaceStart = LeafNode<NodeKind::NamespaceStart>;

// A namespace: `namespace N;`.
struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace.Define(NodeCategory::Decl);

  NamespaceStartId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  NodeIdOneOf<IdentifierName, QualifiedName> name;
};

// Pattern nodes
// -------------

// A pattern binding, such as `name: Type`.
struct BindingPattern {
  static constexpr auto Kind =
      NodeKind::BindingPattern.Define(NodeCategory::Pattern);

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  AnyExprId type;
};

// `name:! Type`
struct CompileTimeBindingPattern {
  static constexpr auto Kind =
      NodeKind::CompileTimeBindingPattern.Define(NodeCategory::Pattern);

  NodeIdOneOf<IdentifierName, SelfValueName> name;
  AnyExprId type;
};

// An address-of binding: `addr self: Self*`.
struct Addr {
  static constexpr auto Kind = NodeKind::Addr.Define(NodeCategory::Pattern);

  AnyPatternId inner;
};

// A template binding: `template T:! type`.
struct Template {
  static constexpr auto Kind = NodeKind::Template.Define(NodeCategory::Pattern);

  // This is a CompileTimeBindingPatternId in any valid program.
  // TODO: Should the parser enforce that?
  AnyPatternId inner;
};

using TuplePatternStart = LeafNode<NodeKind::TuplePatternStart>;
using PatternListComma = LeafNode<NodeKind::PatternListComma>;

// A parameter list or tuple pattern: `(a: i32, b: i32)`.
struct TuplePattern {
  static constexpr auto Kind =
      NodeKind::TuplePattern.Define(NodeCategory::Pattern);

  TuplePatternStartId left_paren;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
};

using ImplicitParamListStart = LeafNode<NodeKind::ImplicitParamListStart>;

// An implicit parameter list: `[T:! type, self: Self]`.
struct ImplicitParamList {
  static constexpr auto Kind = NodeKind::ImplicitParamList.Define();

  ImplicitParamListStartId left_square;
  CommaSeparatedList<AnyPatternId, PatternListCommaId> params;
};

// Function nodes
// --------------

using FunctionIntroducer = LeafNode<NodeKind::FunctionIntroducer>;

// A return type: `-> i32`.
struct ReturnType {
  static constexpr auto Kind = NodeKind::ReturnType.Define();

  AnyExprId type;
};

// A function signature: `fn F() -> i32`.
template <const NodeKind& KindT, NodeCategory Category>
struct FunctionSignature {
  static constexpr auto Kind = KindT.Define(Category);

  FunctionIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  // For now, this is either an IdentifierName or a QualifiedName.
  AnyNameComponentId name;
  std::optional<ImplicitParamListId> implicit_params;
  TuplePatternId params;
  std::optional<ReturnTypeId> return_type;
};

using FunctionDecl =
    FunctionSignature<NodeKind::FunctionDecl, NodeCategory::Decl>;
using FunctionDefinitionStart =
    FunctionSignature<NodeKind::FunctionDefinitionStart, NodeCategory::None>;

// A function definition: `fn F() -> i32 { ... }`.
struct FunctionDefinition {
  static constexpr auto Kind =
      NodeKind::FunctionDefinition.Define(NodeCategory::Decl);

  FunctionDefinitionStartId signature;
  llvm::SmallVector<AnyStatementId> body;
};

using BuiltinFunctionDefinitionStart =
    FunctionSignature<NodeKind::BuiltinFunctionDefinitionStart,
                      NodeCategory::None>;
using BuiltinName = LeafNode<NodeKind::BuiltinName>;

// A builtin function definition: `fn F() -> i32 = "builtin name";`
struct BuiltinFunctionDefinition {
  static constexpr auto Kind =
      NodeKind::BuiltinFunctionDefinition.Define(NodeCategory::Decl);

  BuiltinFunctionDefinitionStartId signature;
  BuiltinNameId builtin_name;
};

// `alias` nodes
// -------------

using AliasIntroducer = LeafNode<NodeKind::AliasIntroducer>;
using AliasInitializer = LeafNode<NodeKind::AliasInitializer>;

// An `alias` declaration: `alias a = b;`.
struct Alias {
  static constexpr auto Kind =
      NodeKind::Alias.Define(NodeCategory::Decl | NodeCategory::Statement);

  AliasIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  // For now, this is either an IdentifierName or a QualifiedName.
  AnyNameComponentId name;
  AliasInitializerId equals;
  AnyExprId initializer;
};

// `let` nodes
// -----------

using LetIntroducer = LeafNode<NodeKind::LetIntroducer>;
using LetInitializer = LeafNode<NodeKind::LetInitializer>;

// A `let` declaration: `let a: i32 = 5;`.
struct LetDecl {
  static constexpr auto Kind =
      NodeKind::LetDecl.Define(NodeCategory::Decl | NodeCategory::Statement);

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
      NodeCategory::Decl | NodeCategory::Statement);

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
  static constexpr auto Kind = NodeKind::CodeBlock.Define();

  CodeBlockStartId left_brace;
  llvm::SmallVector<AnyStatementId> statements;
};

// An expression statement: `F(x);`.
struct ExprStatement {
  static constexpr auto Kind =
      NodeKind::ExprStatement.Define(NodeCategory::Statement);

  AnyExprId expr;
};

using BreakStatementStart = LeafNode<NodeKind::BreakStatementStart>;

// A break statement: `break;`.
struct BreakStatement {
  static constexpr auto Kind =
      NodeKind::BreakStatement.Define(NodeCategory::Statement);

  BreakStatementStartId introducer;
};

using ContinueStatementStart = LeafNode<NodeKind::ContinueStatementStart>;

// A continue statement: `continue;`.
struct ContinueStatement {
  static constexpr auto Kind =
      NodeKind::ContinueStatement.Define(NodeCategory::Statement);

  ContinueStatementStartId introducer;
};

using ReturnStatementStart = LeafNode<NodeKind::ReturnStatementStart>;
using ReturnVarModifier = LeafNode<NodeKind::ReturnVarModifier>;

// A return statement: `return;` or `return expr;` or `return var;`.
struct ReturnStatement {
  static constexpr auto Kind =
      NodeKind::ReturnStatement.Define(NodeCategory::Statement);

  ReturnStatementStartId introducer;
  std::optional<AnyExprId> expr;
  std::optional<ReturnVarModifierId> var;
};

using ForHeaderStart = LeafNode<NodeKind::ForHeaderStart>;

// The `var ... in` portion of a `for` statement.
struct ForIn {
  static constexpr auto Kind = NodeKind::ForIn.Define();

  VariableIntroducerId introducer;
  AnyPatternId pattern;
};

// The `for (var ... in ...)` portion of a `for` statement.
struct ForHeader {
  static constexpr auto Kind = NodeKind::ForHeader.Define();

  ForHeaderStartId introducer;
  ForInId var;
  AnyExprId range;
};

// A complete `for (...) { ... }` statement.
struct ForStatement {
  static constexpr auto Kind =
      NodeKind::ForStatement.Define(NodeCategory::Statement);

  ForHeaderId header;
  CodeBlockId body;
};

using IfConditionStart = LeafNode<NodeKind::IfConditionStart>;

// The condition portion of an `if` statement: `(expr)`.
struct IfCondition {
  static constexpr auto Kind = NodeKind::IfCondition.Define();

  IfConditionStartId left_paren;
  AnyExprId condition;
};

using IfStatementElse = LeafNode<NodeKind::IfStatementElse>;

// An `if` statement: `if (expr) { ... } else { ... }`.
struct IfStatement {
  static constexpr auto Kind =
      NodeKind::IfStatement.Define(NodeCategory::Statement);

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
  static constexpr auto Kind = NodeKind::WhileCondition.Define();

  WhileConditionStartId left_paren;
  AnyExprId condition;
};

// A `while` statement: `while (expr) { ... }`.
struct WhileStatement {
  static constexpr auto Kind =
      NodeKind::WhileStatement.Define(NodeCategory::Statement);

  WhileConditionId head;
  CodeBlockId body;
};

using MatchConditionStart = LeafNode<NodeKind::MatchConditionStart>;

struct MatchCondition {
  static constexpr auto Kind = NodeKind::MatchCondition.Define();

  MatchConditionStartId left_paren;
  AnyExprId condition;
};

using MatchIntroducer = LeafNode<NodeKind::MatchIntroducer>;
struct MatchStatementStart {
  static constexpr auto Kind = NodeKind::MatchStatementStart.Define();

  MatchIntroducerId introducer;
  MatchConditionId left_brace;
};

using MatchCaseIntroducer = LeafNode<NodeKind::MatchCaseIntroducer>;
using MatchCaseGuardIntroducer = LeafNode<NodeKind::MatchCaseGuardIntroducer>;
using MatchCaseGuardStart = LeafNode<NodeKind::MatchCaseGuardStart>;

struct MatchCaseGuard {
  static constexpr auto Kind = NodeKind::MatchCaseGuard.Define();
  MatchCaseGuardIntroducerId introducer;
  MatchCaseGuardStartId left_paren;
  AnyExprId condition;
};

using MatchCaseEqualGreater = LeafNode<NodeKind::MatchCaseEqualGreater>;

struct MatchCaseStart {
  static constexpr auto Kind = NodeKind::MatchCaseStart.Define();
  MatchCaseIntroducerId introducer;
  AnyPatternId pattern;
  std::optional<MatchCaseGuardId> guard;
  MatchCaseEqualGreaterId equal_greater_token;
};

struct MatchCase {
  static constexpr auto Kind = NodeKind::MatchCase.Define();
  MatchCaseStartId head;
  llvm::SmallVector<AnyStatementId> statements;
};

using MatchDefaultIntroducer = LeafNode<NodeKind::MatchDefaultIntroducer>;
using MatchDefaultEqualGreater = LeafNode<NodeKind::MatchDefaultEqualGreater>;

struct MatchDefaultStart {
  static constexpr auto Kind = NodeKind::MatchDefaultStart.Define();
  MatchDefaultIntroducerId introducer;
  MatchDefaultEqualGreaterId equal_greater_token;
};

struct MatchDefault {
  static constexpr auto Kind = NodeKind::MatchDefault.Define();

  MatchDefaultStartId introducer;
  llvm::SmallVector<AnyStatementId> statements;
};

// A `match` statement: `match (expr) { case (...) => {...} default => {...}}`.
struct MatchStatement {
  static constexpr auto Kind =
      NodeKind::MatchStatement.Define(NodeCategory::Statement);

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
  static constexpr auto Kind = NodeKind::ArrayExprSemi.Define();

  ArrayExprStartId left_square;
  AnyExprId type;
};

// An array type, such as  `[i32; 3]` or `[i32;]`.
struct ArrayExpr {
  static constexpr auto Kind = NodeKind::ArrayExpr.Define(NodeCategory::Expr);

  ArrayExprSemiId start;
  std::optional<AnyExprId> bound;
};

// The opening portion of an indexing expression: `a[`.
//
// TODO: Consider flattening this into `IndexExpr`.
struct IndexExprStart {
  static constexpr auto Kind = NodeKind::IndexExprStart.Define();

  AnyExprId sequence;
};

// An indexing expression, such as `a[1]`.
struct IndexExpr {
  static constexpr auto Kind = NodeKind::IndexExpr.Define(NodeCategory::Expr);

  IndexExprStartId start;
  AnyExprId index;
};

using ParenExprStart = LeafNode<NodeKind::ParenExprStart>;

// A parenthesized expression: `(a)`.
struct ParenExpr {
  static constexpr auto Kind =
      NodeKind::ParenExpr.Define(NodeCategory::Expr | NodeCategory::MemberExpr);

  ParenExprStartId start;
  AnyExprId expr;
};

using TupleLiteralStart = LeafNode<NodeKind::TupleLiteralStart>;
using TupleLiteralComma = LeafNode<NodeKind::TupleLiteralComma>;

// A tuple literal: `()`, `(a, b, c)`, or `(a,)`.
struct TupleLiteral {
  static constexpr auto Kind =
      NodeKind::TupleLiteral.Define(NodeCategory::Expr);

  TupleLiteralStartId start;
  CommaSeparatedList<AnyExprId, TupleLiteralCommaId> elements;
};

// The opening portion of a call expression: `F(`.
//
// TODO: Consider flattening this into `CallExpr`.
struct CallExprStart {
  static constexpr auto Kind = NodeKind::CallExprStart.Define();

  AnyExprId callee;
};

using CallExprComma = LeafNode<NodeKind::CallExprComma>;

// A call expression: `F(a, b, c)`.
struct CallExpr {
  static constexpr auto Kind = NodeKind::CallExpr.Define(NodeCategory::Expr);

  CallExprStartId start;
  CommaSeparatedList<AnyExprId, CallExprCommaId> arguments;
};

// A member access expression: `a.b` or `a.(b)`.
struct MemberAccessExpr {
  static constexpr auto Kind =
      NodeKind::MemberAccessExpr.Define(NodeCategory::Expr);

  AnyExprId lhs;
  AnyMemberNameOrMemberExprId rhs;
};

// An indirect member access expression: `a->b` or `a->(b)`.
struct PointerMemberAccessExpr {
  static constexpr auto Kind =
      NodeKind::PointerMemberAccessExpr.Define(NodeCategory::Expr);

  AnyExprId lhs;
  AnyMemberNameOrMemberExprId rhs;
};

// A prefix operator expression.
template <const NodeKind& KindT>
struct PrefixOperator {
  static constexpr auto Kind = KindT.Define(NodeCategory::Expr);

  AnyExprId operand;
};

// An infix operator expression.
template <const NodeKind& KindT>
struct InfixOperator {
  static constexpr auto Kind = KindT.Define(NodeCategory::Expr);

  AnyExprId lhs;
  AnyExprId rhs;
};

// A postfix operator expression.
template <const NodeKind& KindT>
struct PostfixOperator {
  static constexpr auto Kind = KindT.Define(NodeCategory::Expr);

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
  static constexpr auto Kind = NodeKind::ShortCircuitOperandAnd.Define();

  AnyExprId operand;
};

struct ShortCircuitOperandOr {
  static constexpr auto Kind = NodeKind::ShortCircuitOperandOr.Define();

  AnyExprId operand;
};

struct ShortCircuitOperatorAnd {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperatorAnd.Define(NodeCategory::Expr);

  ShortCircuitOperandAndId lhs;
  AnyExprId rhs;
};

struct ShortCircuitOperatorOr {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperatorOr.Define(NodeCategory::Expr);

  ShortCircuitOperandOrId lhs;
  AnyExprId rhs;
};

// The `if` portion of an `if` expression: `if expr`.
struct IfExprIf {
  static constexpr auto Kind = NodeKind::IfExprIf.Define();

  AnyExprId condition;
};

// The `then` portion of an `if` expression: `then expr`.
struct IfExprThen {
  static constexpr auto Kind = NodeKind::IfExprThen.Define();

  AnyExprId result;
};

// A full `if` expression: `if expr then expr else expr`.
struct IfExprElse {
  static constexpr auto Kind = NodeKind::IfExprElse.Define(NodeCategory::Expr);

  IfExprIfId start;
  IfExprThenId then;
  AnyExprId else_result;
};

// Choice nodes
// ------------

using ChoiceIntroducer = LeafNode<NodeKind::ChoiceIntroducer>;

struct ChoiceSignature {
  static constexpr auto Kind =
      NodeKind::ChoiceDefinitionStart.Define(NodeCategory::None);

  ChoiceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyNameComponentId name;
  std::optional<ImplicitParamListId> implicit_params;
  std::optional<TuplePatternId> params;
};

using ChoiceDefinitionStart = ChoiceSignature;

using ChoiceAlternativeListComma =
    LeafNode<NodeKind::ChoiceAlternativeListComma>;

struct ChoiceDefinition {
  static constexpr auto Kind =
      NodeKind::ChoiceDefinition.Define(NodeCategory::Decl);

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
  static constexpr auto Kind = NodeKind::StructFieldDesignator.Define();

  NodeIdOneOf<IdentifierName, BaseName> name;
};

// `.a = 0`
struct StructField {
  static constexpr auto Kind = NodeKind::StructField.Define();

  StructFieldDesignatorId designator;
  AnyExprId expr;
};

// `.a: i32`
struct StructTypeField {
  static constexpr auto Kind = NodeKind::StructTypeField.Define();

  StructFieldDesignatorId designator;
  AnyExprId type_expr;
};

// Struct literals, such as `{.a = 0}`.
struct StructLiteral {
  static constexpr auto Kind =
      NodeKind::StructLiteral.Define(NodeCategory::Expr);

  StructLiteralStartId start;
  CommaSeparatedList<StructFieldId, StructCommaId> fields;
};

// Struct type literals, such as `{.a: i32}`.
struct StructTypeLiteral {
  static constexpr auto Kind =
      NodeKind::StructTypeLiteral.Define(NodeCategory::Expr);

  StructTypeLiteralStartId start;
  CommaSeparatedList<StructTypeFieldId, StructCommaId> fields;
};

// `class` declarations and definitions
// ------------------------------------

// `class`
using ClassIntroducer = LeafNode<NodeKind::ClassIntroducer>;

// A class signature `class C`
template <const NodeKind& KindT, NodeCategory Category>
struct ClassSignature {
  static constexpr auto Kind = KindT.Define(Category);

  ClassIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyNameComponentId name;
  std::optional<ImplicitParamListId> implicit_params;
  std::optional<TuplePatternId> params;
};

// `class C;`
using ClassDecl = ClassSignature<NodeKind::ClassDecl, NodeCategory::Decl>;
// `class C {`
using ClassDefinitionStart =
    ClassSignature<NodeKind::ClassDefinitionStart, NodeCategory::None>;

// `class C { ... }`
struct ClassDefinition {
  static constexpr auto Kind =
      NodeKind::ClassDefinition.Define(NodeCategory::Decl);

  ClassDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// Adapter declaration
// -------------------

// `adapt`
using AdaptIntroducer = LeafNode<NodeKind::AdaptIntroducer>;
// `adapt SomeType;`
struct AdaptDecl {
  static constexpr auto Kind = NodeKind::AdaptDecl.Define(NodeCategory::Decl);

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
  static constexpr auto Kind = NodeKind::BaseDecl.Define(NodeCategory::Decl);

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
template <const NodeKind& KindT, NodeCategory Category>
struct InterfaceSignature {
  static constexpr auto Kind = KindT.Define(Category);

  InterfaceIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyNameComponentId name;
  std::optional<ImplicitParamListId> implicit_params;
  std::optional<TuplePatternId> params;
};

// `interface I;`
using InterfaceDecl =
    InterfaceSignature<NodeKind::InterfaceDecl, NodeCategory::Decl>;
// `interface I {`
using InterfaceDefinitionStart =
    InterfaceSignature<NodeKind::InterfaceDefinitionStart, NodeCategory::None>;

// `interface I { ... }`
struct InterfaceDefinition {
  static constexpr auto Kind =
      NodeKind::InterfaceDefinition.Define(NodeCategory::Decl);

  InterfaceDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// `impl`...`as` declarations and definitions
// ------------------------------------------

// `impl`
using ImplIntroducer = LeafNode<NodeKind::ImplIntroducer>;

// `forall [...]`
struct ImplForall {
  static constexpr auto Kind = NodeKind::ImplForall.Define();

  ImplicitParamListId params;
};

// `as` with no type before it
using DefaultSelfImplAs =
    LeafNode<NodeKind::DefaultSelfImplAs, NodeCategory::ImplAs>;

// `<type> as`
struct TypeImplAs {
  static constexpr auto Kind =
      NodeKind::TypeImplAs.Define(NodeCategory::ImplAs);

  AnyExprId type_expr;
};

// `impl T as I`
template <const NodeKind& KindT, NodeCategory Category>
struct ImplSignature {
  static constexpr auto Kind = KindT.Define(Category);

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
  static constexpr auto Kind =
      NodeKind::ImplDefinition.Define(NodeCategory::Decl);

  ImplDefinitionStartId signature;
  llvm::SmallVector<AnyDeclId> members;
};

// Named constraint declarations and definitions
// ---------------------------------------------

// `constraint`
using NamedConstraintIntroducer = LeafNode<NodeKind::NamedConstraintIntroducer>;

// `constraint NC`
template <const NodeKind& KindT, NodeCategory Category>
struct NamedConstraintSignature {
  static constexpr auto Kind = KindT.Define(Category);

  NamedConstraintIntroducerId introducer;
  llvm::SmallVector<AnyModifierId> modifiers;
  AnyNameComponentId name;
  std::optional<ImplicitParamListId> implicit_params;
  std::optional<TuplePatternId> params;
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
  static constexpr auto Kind =
      NodeKind::NamedConstraintDefinition.Define(NodeCategory::Decl);

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
