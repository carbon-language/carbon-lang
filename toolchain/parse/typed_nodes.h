// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
#define CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_

#include "toolchain/parse/node_id.h"
#include "toolchain/parse/node_kind.h"

namespace Carbon::Parse {

// NodeId that matches any NodeKind whose `category()` overlaps with `Category`.
template <NodeCategory Category>
struct NodeIdInCategory : public NodeId {
  // An explicitly invalid instance.
  static const NodeIdInCategory<Category> Invalid;

  explicit NodeIdInCategory(NodeId node_id) : NodeId(node_id) {}
};

template <NodeCategory Category>
constexpr NodeIdInCategory<Category> NodeIdInCategory<Category>::Invalid =
    NodeIdInCategory<Category>(NodeId::InvalidIndex);

// Aliases for `NodeIdInCategory` to describe particular categories of nodes.
using AnyDecl = NodeIdInCategory<NodeCategory::Decl>;
using AnyExpr = NodeIdInCategory<NodeCategory::Expr>;
using AnyModifier = NodeIdInCategory<NodeCategory::Modifier>;
using AnyNameComponent = NodeIdInCategory<NodeCategory::NameComponent>;
using AnyPattern = NodeIdInCategory<NodeCategory::Pattern>;
using AnyStatement = NodeIdInCategory<NodeCategory::Statement>;

// NodeId with kind that matches either T::Kind or U::Kind.
template <typename T, typename U>
struct NodeIdOneOf : public NodeId {
  // An explicitly invalid instance.
  static const NodeIdOneOf<T, U> Invalid;

  explicit NodeIdOneOf(NodeId node_id) : NodeId(node_id) {}
};

template <typename T, typename U>
constexpr NodeIdOneOf<T, U> NodeIdOneOf<T, U>::Invalid =
    NodeIdOneOf<T, U>(NodeId::InvalidIndex);

// NodeId with kind that is anything but T::Kind.
template <typename T>
struct NodeIdNot : public NodeId {
  // An explicitly invalid instance.
  static const NodeIdNot<T> Invalid;

  explicit NodeIdNot(NodeId node_id) : NodeId(node_id) {}
};

template <typename T>
constexpr NodeIdNot<T> NodeIdNot<T>::Invalid =
    NodeIdNot<T>(NodeId::InvalidIndex);

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

// Each type defined below corresponds to a parse node kind, and describes the
// expected child structure of that parse node.

// Error nodes

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

// The start of the file.
using FileStart = LeafNode<NodeKind::FileStart>;

// The end of the file.
using FileEnd = LeafNode<NodeKind::FileEnd>;

// General-purpose nodes

// An empty declaration, such as `;`.
using EmptyDecl =
    LeafNode<NodeKind::EmptyDecl, NodeCategory::Decl | NodeCategory::Statement>;

// A name in a non-expression context, such as a declaration.
using IdentifierName =
    LeafNode<NodeKind::IdentifierName, NodeCategory::NameComponent>;

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
using BaseName = LeafNode<NodeKind::BaseName>;

// A qualified name: `A.B`.
//
// TODO: This is not a declaration. Rename this parse node.
struct QualifiedDecl {
  static constexpr auto Kind =
      NodeKind::QualifiedDecl.Define(NodeCategory::NameComponent);

  // For now, this is either an IdentifierName or a QualifiedDecl.
  AnyNameComponent lhs;

  // TODO: This will eventually need to support more general expressions, for
  // example `GenericType(type_args).ChildType(child_type_args).Name`.
  IdentifierNameId rhs;
};

// Library, package, import

// The `package` keyword in an expression.
using PackageExpr = LeafNode<NodeKind::PackageExpr, NodeCategory::Expr>;

// The name of a package or library for `package`, `import`, and `library`.
using PackageName = LeafNode<NodeKind::PackageName>;
using LibraryName = LeafNode<NodeKind::LibraryName>;
using DefaultLibrary = LeafNode<NodeKind::DefaultLibrary>;

using PackageIntroducer = LeafNode<NodeKind::PackageIntroducer>;
using PackageApi = LeafNode<NodeKind::PackageApi>;
using PackageImpl = LeafNode<NodeKind::PackageImpl>;

// `library` in `package` or `import`:
struct LibrarySpecifier {
  static constexpr auto Kind = NodeKind::LibrarySpecifier.Define();
  NodeIdOneOf<LibraryName, DefaultLibrary> name;
};

// First line of the file, such as:
//   `package MyPackage library "MyLibrary" impl;`
struct PackageDirective {
  static constexpr auto Kind = NodeKind::PackageDirective.Define();
  PackageIntroducerId introducer;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
  NodeIdOneOf<PackageApi, PackageImpl> api_or_impl;
};

// `import TheirPackage library "TheirLibrary";`
using ImportIntroducer = LeafNode<NodeKind::ImportIntroducer>;
struct ImportDirective {
  static constexpr auto Kind = NodeKind::ImportDirective.Define();
  ImportIntroducerId introducer;
  std::optional<PackageNameId> name;
  std::optional<LibrarySpecifierId> library;
};

// `library` as directive:
using LibraryIntroducer = LeafNode<NodeKind::LibraryIntroducer>;
struct LibraryDirective {
  static constexpr auto Kind = NodeKind::LibraryDirective.Define();
  LibraryIntroducerId introducer;
  NodeIdOneOf<LibraryName, DefaultLibrary> library_name;
  NodeIdOneOf<PackageApi, PackageImpl> api_or_impl;
};

// Namespace nodes

using NamespaceStart = LeafNode<NodeKind::NamespaceStart>;

// A namespace: `namespace N;`.
struct Namespace {
  static constexpr auto Kind = NodeKind::Namespace.Define(NodeCategory::Decl);
  NamespaceStartId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  NodeIdOneOf<IdentifierName, QualifiedDecl> name;
};

// Function nodes

using CodeBlockStart = LeafNode<NodeKind::CodeBlockStart>;

// A code block: `{ statement; statement; ... }`.
struct CodeBlock {
  static constexpr auto Kind = NodeKind::CodeBlock.Define();
  CodeBlockStartId left_brace;
  llvm::SmallVector<AnyStatement> statements;
};

using VariableIntroducer = LeafNode<NodeKind::VariableIntroducer>;

using TuplePatternStart = LeafNode<NodeKind::TuplePatternStart>;
using ImplicitParamListStart = LeafNode<NodeKind::ImplicitParamListStart>;
using PatternListComma = LeafNode<NodeKind::PatternListComma>;

// A parameter list: `(a: i32, b: i32)`.
struct TuplePattern {
  static constexpr auto Kind =
      NodeKind::TuplePattern.Define(NodeCategory::Pattern);
  TuplePatternStartId left_paren;
  CommaSeparatedList<AnyPattern, PatternListCommaId> params;
};

// An implicit parameter list: `[T:! type, self: Self]`.
struct ImplicitParamList {
  static constexpr auto Kind = NodeKind::ImplicitParamList.Define();
  ImplicitParamListStartId left_square;
  CommaSeparatedList<AnyPattern, PatternListCommaId> params;
};

using FunctionIntroducer = LeafNode<NodeKind::FunctionIntroducer>;

// A return type: `-> i32`.
struct ReturnType {
  static constexpr auto Kind = NodeKind::ReturnType.Define();
  AnyExpr type;
};

// A function signature: `fn F() -> i32`.
template <const NodeKind& KindT>
struct FunctionSignature {
  static constexpr auto Kind = KindT.Define(NodeCategory::Decl);
  FunctionIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  // For now, this is either an IdentifierName or a QualifiedDecl.
  AnyNameComponent name;
  std::optional<ImplicitParamListId> implicit_params;
  TuplePatternId params;
  std::optional<ReturnTypeId> return_type;
};

using FunctionDecl = FunctionSignature<NodeKind::FunctionDecl>;
using FunctionDefinitionStart =
    FunctionSignature<NodeKind::FunctionDefinitionStart>;

// A function definition: `fn F() -> i32 { ... }`.
struct FunctionDefinition {
  static constexpr auto Kind =
      NodeKind::FunctionDefinition.Define(NodeCategory::Decl);
  FunctionDefinitionStartId signature;
  llvm::SmallVector<AnyStatement> body;
};

// Pattern nodes

// A pattern binding, such as `name: Type`.
struct BindingPattern {
  static constexpr auto Kind =
      NodeKind::BindingPattern.Define(NodeCategory::Pattern);
  NodeIdOneOf<IdentifierName, SelfValueName> name;
  AnyExpr type;
};

// `name:! Type`
struct GenericBindingPattern {
  static constexpr auto Kind =
      NodeKind::GenericBindingPattern.Define(NodeCategory::Pattern);
  NodeIdOneOf<IdentifierName, SelfValueName> name;
  AnyExpr type;
};

// An address-of binding: `addr self: Self*`.
struct Address {
  static constexpr auto Kind = NodeKind::Address.Define(NodeCategory::Pattern);
  AnyPattern inner;
};

// A template binding: `template T:! type`.
struct Template {
  static constexpr auto Kind = NodeKind::Template.Define(NodeCategory::Pattern);
  // This is a GenericBindingPatternId in any valid program.
  // TODO: Should the parser enforce that?
  AnyPattern inner;
};

// `let` nodes

using LetIntroducer = LeafNode<NodeKind::LetIntroducer>;
using LetInitializer = LeafNode<NodeKind::LetInitializer>;

// A `let` declaration: `let a: i32 = 5;`.
struct LetDecl {
  static constexpr auto Kind =
      NodeKind::LetDecl.Define(NodeCategory::Decl | NodeCategory::Statement);
  LetIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  AnyPattern pattern;
  LetInitializerId equals;
  AnyExpr initializer;
};

// `var` nodes

using VariableIntroducer = LeafNode<NodeKind::VariableIntroducer>;
using ReturnedModifier = LeafNode<NodeKind::ReturnedModifier>;
using VariableInitializer = LeafNode<NodeKind::VariableInitializer>;

// A `var` declaration: `var a: i32;` or `var a: i32 = 5;`.
struct VariableDecl {
  static constexpr auto Kind = NodeKind::VariableDecl.Define(
      NodeCategory::Decl | NodeCategory::Statement);
  VariableIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  std::optional<ReturnedModifierId> returned;
  AnyPattern pattern;

  struct Initializer {
    VariableInitializerId equals;
    AnyExpr value;
  };
  std::optional<Initializer> initializer;
};

// Statement nodes

// An expression statement: `F(x);`.
struct ExprStatement {
  static constexpr auto Kind =
      NodeKind::ExprStatement.Define(NodeCategory::Statement);
  AnyExpr expr;
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
  std::optional<AnyExpr> expr;
  std::optional<ReturnVarModifierId> var;
};

using ForHeaderStart = LeafNode<NodeKind::ForHeaderStart>;

// The `var ... in` portion of a `for` statement.
struct ForIn {
  static constexpr auto Kind = NodeKind::ForIn.Define();
  VariableIntroducerId introducer;
  AnyPattern pattern;
};

// The `for (var ... in ...)` portion of a `for` statement.
struct ForHeader {
  static constexpr auto Kind = NodeKind::ForHeader.Define();
  ForHeaderStartId introducer;
  ForInId var;
  AnyExpr range;
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
  AnyExpr condition;
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
  AnyExpr condition;
};

// A `while` statement: `while (expr) { ... }`.
struct WhileStatement {
  static constexpr auto Kind =
      NodeKind::WhileStatement.Define(NodeCategory::Statement);
  WhileConditionId head;
  CodeBlockId body;
};

// Expression nodes

using ArrayExprStart = LeafNode<NodeKind::ArrayExprStart, NodeCategory::Expr>;

// The start of an array type, `[i32;`.
//
// TODO: Consider flattening this into `ArrayExpr`.
struct ArrayExprSemi {
  static constexpr auto Kind = NodeKind::ArrayExprSemi.Define();
  ArrayExprStartId left_square;
  AnyExpr type;
};

// An array type, such as  `[i32; 3]` or `[i32;]`.
struct ArrayExpr {
  static constexpr auto Kind = NodeKind::ArrayExpr.Define(NodeCategory::Expr);
  ArrayExprSemiId start;
  std::optional<AnyExpr> bound;
};

// The opening portion of an indexing expression: `a[`.
//
// TODO: Consider flattening this into `IndexExpr`.
struct IndexExprStart {
  static constexpr auto Kind = NodeKind::IndexExprStart.Define();
  AnyExpr sequence;
};

// An indexing expression, such as `a[1]`.
struct IndexExpr {
  static constexpr auto Kind = NodeKind::IndexExpr.Define(NodeCategory::Expr);
  IndexExprStartId start;
  AnyExpr index;
};

using ExprOpenParen = LeafNode<NodeKind::ExprOpenParen>;

// A parenthesized expression: `(a)`.
struct ParenExpr {
  static constexpr auto Kind = NodeKind::ParenExpr.Define(NodeCategory::Expr);
  ExprOpenParenId left_paren;
  AnyExpr expr;
};

using TupleLiteralComma = LeafNode<NodeKind::TupleLiteralComma>;

// A tuple literal: `()`, `(a, b, c)`, or `(a,)`.
struct TupleLiteral {
  static constexpr auto Kind =
      NodeKind::TupleLiteral.Define(NodeCategory::Expr);
  ExprOpenParenId left_paren;
  CommaSeparatedList<AnyExpr, TupleLiteralCommaId> elements;
};

// The opening portion of a call expression: `F(`.
//
// TODO: Consider flattening this into `CallExpr`.
struct CallExprStart {
  static constexpr auto Kind = NodeKind::CallExprStart.Define();
  AnyExpr callee;
};

using CallExprComma = LeafNode<NodeKind::CallExprComma>;

// A call expression: `F(a, b, c)`.
struct CallExpr {
  static constexpr auto Kind = NodeKind::CallExpr.Define(NodeCategory::Expr);
  CallExprStartId start;
  CommaSeparatedList<AnyExpr, CallExprCommaId> arguments;
};

// A simple member access expression: `a.b`.
struct MemberAccessExpr {
  static constexpr auto Kind =
      NodeKind::MemberAccessExpr.Define(NodeCategory::Expr);
  AnyExpr lhs;
  // TODO: Figure out which nodes can appear here
  NodeId rhs;
};

// A simple indirect member access expression: `a->b`.
struct PointerMemberAccessExpr {
  static constexpr auto Kind =
      NodeKind::PointerMemberAccessExpr.Define(NodeCategory::Expr);
  AnyExpr lhs;
  // TODO: Figure out which nodes can appear here
  NodeId rhs;
};

// A prefix operator expression.
template <const NodeKind& KindT>
struct PrefixOperator {
  static constexpr auto Kind = KindT.Define(NodeCategory::Expr);
  AnyExpr operand;
};

// An infix operator expression.
template <const NodeKind& KindT>
struct InfixOperator {
  static constexpr auto Kind = KindT.Define(NodeCategory::Expr);
  AnyExpr lhs;
  AnyExpr rhs;
};

// A postfix operator expression.
template <const NodeKind& KindT>
struct PostfixOperator {
  static constexpr auto Kind = KindT.Define(NodeCategory::Expr);
  AnyExpr operand;
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
// FIXME: should this be a template?
struct ShortCircuitOperandAnd {
  static constexpr auto Kind = NodeKind::ShortCircuitOperandAnd.Define();
  AnyExpr operand;
};

struct ShortCircuitOperandOr {
  static constexpr auto Kind = NodeKind::ShortCircuitOperandOr.Define();
  AnyExpr operand;
};

struct ShortCircuitOperatorAnd {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperatorAnd.Define(NodeCategory::Expr);
  ShortCircuitOperandAndId lhs;
  AnyExpr rhs;
};

struct ShortCircuitOperatorOr {
  static constexpr auto Kind =
      NodeKind::ShortCircuitOperatorOr.Define(NodeCategory::Expr);
  ShortCircuitOperandOrId lhs;
  AnyExpr rhs;
};

// The `if` portion of an `if` expression: `if expr`.
struct IfExprIf {
  static constexpr auto Kind = NodeKind::IfExprIf.Define();
  AnyExpr condition;
};

// The `then` portion of an `if` expression: `then expr`.
struct IfExprThen {
  static constexpr auto Kind = NodeKind::IfExprThen.Define();
  AnyExpr result;
};

// A full `if` expression: `if expr then expr else expr`.
struct IfExprElse {
  static constexpr auto Kind = NodeKind::IfExprElse.Define(NodeCategory::Expr);
  IfExprIfId start;
  IfExprThenId then;
  AnyExpr else_result;
};

// Struct literals and struct type literals

// `{`
using StructLiteralOrStructTypeLiteralStart =
    LeafNode<NodeKind::StructLiteralOrStructTypeLiteralStart>;
// `,`
using StructComma = LeafNode<NodeKind::StructComma>;

// `.a`
struct StructFieldDesignator {
  static constexpr auto Kind = NodeKind::StructFieldDesignator.Define();
  NodeIdOneOf<IdentifierName, BaseName> name;
};

// `.a = 0`
struct StructFieldValue {
  static constexpr auto Kind = NodeKind::StructFieldValue.Define();
  StructFieldDesignatorId designator;
  AnyExpr expr;
};

// `.a: i32`
struct StructFieldType {
  static constexpr auto Kind = NodeKind::StructFieldType.Define();
  StructFieldDesignatorId designator;
  AnyExpr type_expr;
};

// Struct literals, such as `{.a = 0}`:
struct StructLiteral {
  static constexpr auto Kind =
      NodeKind::StructLiteral.Define(NodeCategory::Expr);
  StructLiteralOrStructTypeLiteralStartId introducer;
  CommaSeparatedList<StructFieldValueId, StructCommaId> fields;
};

// Struct type literals, such as `{.a: i32}`:
struct StructTypeLiteral {
  static constexpr auto Kind =
      NodeKind::StructTypeLiteral.Define(NodeCategory::Expr);
  StructLiteralOrStructTypeLiteralStartId introducer;
  CommaSeparatedList<StructFieldTypeId, StructCommaId> fields;
};

// `class` declarations and definitions

// `class`
using ClassIntroducer = LeafNode<NodeKind::ClassIntroducer>;

// A class signature `class C`
template <const NodeKind& KindT, NodeCategory Category>
struct ClassSignature {
  static constexpr auto Kind = KindT.Define(Category);
  ClassIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  AnyNameComponent name;
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
  llvm::SmallVector<AnyDecl> members;
};

// Base class declaration

// `base`
using BaseIntroducer = LeafNode<NodeKind::BaseIntroducer>;
using BaseColon = LeafNode<NodeKind::BaseColon>;
// `extend base: BaseClass;`
struct BaseDecl {
  static constexpr auto Kind = NodeKind::BaseDecl.Define(NodeCategory::Decl);
  BaseIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  BaseColonId colon;
  AnyExpr base_class;
};

// Interface declarations and definitions

// `interface`
using InterfaceIntroducer = LeafNode<NodeKind::InterfaceIntroducer>;

// `interface I`
template <const NodeKind& KindT, NodeCategory Category>
struct InterfaceSignature {
  static constexpr auto Kind = KindT.Define(Category);
  InterfaceIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  AnyNameComponent name;
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
  llvm::SmallVector<AnyDecl> members;
};

// `impl`...`as` declarations and definitions

// `impl`
using ImplIntroducer = LeafNode<NodeKind::ImplIntroducer>;
// `as`
using ImplAs = LeafNode<NodeKind::ImplAs>;

// `forall [...]`
struct ImplForall {
  static constexpr auto Kind = NodeKind::ImplForall.Define();
  ImplicitParamListId params;
};

// `impl T as I`
template <const NodeKind& KindT, NodeCategory Category>
struct ImplSignature {
  static constexpr auto Kind = KindT.Define(Category);
  ImplIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  std::optional<ImplForallId> forall;
  std::optional<AnyExpr> type_expr;
  ImplAsId as;
  AnyExpr interface;
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
  llvm::SmallVector<AnyDecl> members;
};

// Named constraint declarations and definitions

// `constraint`
using NamedConstraintIntroducer = LeafNode<NodeKind::NamedConstraintIntroducer>;

// `constraint NC`
template <const NodeKind& KindT, NodeCategory Category>
struct NamedConstraintSignature {
  static constexpr auto Kind = KindT.Define(Category);
  NamedConstraintIntroducerId introducer;
  llvm::SmallVector<AnyModifier> modifiers;
  AnyNameComponent name;
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
  llvm::SmallVector<AnyDecl> members;
};

// Define `FooId` as the id for type `Foo`
#define CARBON_PARSE_NODE_KIND(KindName) \
  template <>                            \
  struct NodeForId<KindName##Id> {       \
    using Kind = KindName;               \
  };
#include "toolchain/parse/node_kind.def"

}  // namespace Carbon::Parse

#endif  // CARBON_TOOLCHAIN_PARSE_TYPED_NODES_H_
