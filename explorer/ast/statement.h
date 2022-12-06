// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_AST_STATEMENT_H_
#define CARBON_EXPLORER_AST_STATEMENT_H_

#include <utility>
#include <vector>

#include "common/ostream.h"
#include "explorer/ast/ast_node.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/return_term.h"
#include "explorer/ast/static_scope.h"
#include "explorer/ast/value_category.h"
#include "explorer/common/arena.h"
#include "explorer/common/source_location.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class CallableDeclaration;

class Statement : public AstNode {
 public:
  ~Statement() override = 0;

  void Print(llvm::raw_ostream& out) const override { PrintDepth(-1, out); }
  void PrintID(llvm::raw_ostream& out) const override { PrintDepth(1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;

  static auto classof(const AstNode* node) {
    return InheritsFromStatement(node->kind());
  }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> StatementKind {
    return static_cast<StatementKind>(root_kind());
  }

 protected:
  Statement(AstNodeKind kind, SourceLocation source_loc)
      : AstNode(kind, source_loc) {}
};

class Block : public Statement {
 public:
  Block(SourceLocation source_loc, std::vector<Nonnull<Statement*>> statements)
      : Statement(AstNodeKind::Block, source_loc),
        statements_(std::move(statements)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromBlock(node->kind());
  }

  auto statements() const -> llvm::ArrayRef<Nonnull<const Statement*>> {
    return statements_;
  }
  auto statements() -> llvm::MutableArrayRef<Nonnull<Statement*>> {
    return statements_;
  }

 private:
  std::vector<Nonnull<Statement*>> statements_;
};

class ExpressionStatement : public Statement {
 public:
  ExpressionStatement(SourceLocation source_loc,
                      Nonnull<Expression*> expression)
      : Statement(AstNodeKind::ExpressionStatement, source_loc),
        expression_(expression) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromExpressionStatement(node->kind());
  }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }

 private:
  Nonnull<Expression*> expression_;
};

class Assign : public Statement {
 public:
  Assign(SourceLocation source_loc, Nonnull<Expression*> lhs,
         Nonnull<Expression*> rhs)
      : Statement(AstNodeKind::Assign, source_loc), lhs_(lhs), rhs_(rhs) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAssign(node->kind());
  }

  auto lhs() const -> const Expression& { return *lhs_; }
  auto lhs() -> Expression& { return *lhs_; }
  auto rhs() const -> const Expression& { return *rhs_; }
  auto rhs() -> Expression& { return *rhs_; }

  // Can only be called by type-checking, if a conversion was required.
  void set_rhs(Nonnull<Expression*> rhs) { rhs_ = rhs; }

 private:
  Nonnull<Expression*> lhs_;
  Nonnull<Expression*> rhs_;
};

class VariableDefinition : public Statement {
 public:
  enum DefinitionType {
    Var,
    Returned,
  };

  VariableDefinition(SourceLocation source_loc, Nonnull<Pattern*> pattern,
                     std::optional<Nonnull<Expression*>> init,
                     ValueCategory value_category, DefinitionType def_type)
      : Statement(AstNodeKind::VariableDefinition, source_loc),
        pattern_(pattern),
        init_(init),
        value_category_(value_category),
        def_type_(def_type) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromVariableDefinition(node->kind());
  }

  auto pattern() const -> const Pattern& { return *pattern_; }
  auto pattern() -> Pattern& { return *pattern_; }

  auto init() const -> const Expression& {
    CARBON_CHECK(has_init());
    return **init_;
  }
  auto init() -> Expression& {
    CARBON_CHECK(has_init());
    return **init_;
  }

  auto has_init() const -> bool { return init_.has_value(); }

  // Can only be called by type-checking, if a conversion was required.
  void set_init(Nonnull<Expression*> init) {
    CARBON_CHECK(has_init()) << "should not add a new initializer";
    init_ = init;
  }

  auto value_category() const -> ValueCategory { return value_category_; }

  auto is_returned() const -> bool { return def_type_ == Returned; };

 private:
  Nonnull<Pattern*> pattern_;
  std::optional<Nonnull<Expression*>> init_;
  ValueCategory value_category_;
  const DefinitionType def_type_;
};

class If : public Statement {
 public:
  If(SourceLocation source_loc, Nonnull<Expression*> condition,
     Nonnull<Block*> then_block, std::optional<Nonnull<Block*>> else_block)
      : Statement(AstNodeKind::If, source_loc),
        condition_(condition),
        then_block_(then_block),
        else_block_(else_block) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromIf(node->kind());
  }

  auto condition() const -> const Expression& { return *condition_; }
  auto condition() -> Expression& { return *condition_; }
  auto then_block() const -> const Block& { return *then_block_; }
  auto then_block() -> Block& { return *then_block_; }
  auto else_block() const -> std::optional<Nonnull<const Block*>> {
    return else_block_;
  }
  auto else_block() -> std::optional<Nonnull<Block*>> { return else_block_; }

  // Can only be called by type-checking, if a conversion was required.
  void set_condition(Nonnull<Expression*> condition) { condition_ = condition; }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Block*> then_block_;
  std::optional<Nonnull<Block*>> else_block_;
};

class Return : public Statement {
 public:
  static auto classof(const AstNode* node) -> bool {
    return InheritsFromReturn(node->kind());
  }

  // The AST node representing the function body this statement returns from.
  // Can only be called after ResolveControlFlow has visited this node.
  //
  // Note that this function does not represent an edge in the tree
  // structure of the AST: the return value is not a child of this node,
  // but an ancestor.
  auto function() const -> const CallableDeclaration& { return **function_; }
  auto function() -> CallableDeclaration& { return **function_; }

  // Can only be called once, by ResolveControlFlow.
  void set_function(Nonnull<CallableDeclaration*> function) {
    CARBON_CHECK(!function_.has_value());
    function_ = function;
  }

 protected:
  Return(AstNodeKind node_kind, SourceLocation source_loc)
      : Statement(node_kind, source_loc) {}

 private:
  std::optional<Nonnull<CallableDeclaration*>> function_;
};

class ReturnVar : public Return {
 public:
  explicit ReturnVar(SourceLocation source_loc)
      : Return(AstNodeKind::ReturnVar, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromReturnVar(node->kind());
  }

  // Returns the value node of the BindingPattern of the returned var
  // definition. Cannot be called before name resolution.
  auto value_node() const -> const ValueNodeView& { return *value_node_; }

  // Can only be called once, by ResolveNames.
  void set_value_node(ValueNodeView value_node) {
    CARBON_CHECK(!value_node_.has_value());
    value_node_ = value_node;
  }

 private:
  // The value node of the BindingPattern of the returned var definition.
  std::optional<ValueNodeView> value_node_;
};

class ReturnExpression : public Return {
 public:
  ReturnExpression(Nonnull<Arena*> arena, SourceLocation source_loc)
      : ReturnExpression(source_loc, arena->New<TupleLiteral>(source_loc),
                         true) {}
  ReturnExpression(SourceLocation source_loc, Nonnull<Expression*> expression,
                   bool is_omitted_expression)
      : Return(AstNodeKind::ReturnExpression, source_loc),
        expression_(expression),
        is_omitted_expression_(is_omitted_expression) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromReturnExpression(node->kind());
  }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }
  auto is_omitted_expression() const -> bool { return is_omitted_expression_; }

  // Can only be called by type-checking, if a conversion was required.
  void set_expression(Nonnull<Expression*> expression) {
    expression_ = expression;
  }

 private:
  Nonnull<Expression*> expression_;
  bool is_omitted_expression_;
};

class While : public Statement {
 public:
  While(SourceLocation source_loc, Nonnull<Expression*> condition,
        Nonnull<Block*> body)
      : Statement(AstNodeKind::While, source_loc),
        condition_(condition),
        body_(body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromWhile(node->kind());
  }

  auto condition() const -> const Expression& { return *condition_; }
  auto condition() -> Expression& { return *condition_; }
  auto body() const -> const Block& { return *body_; }
  auto body() -> Block& { return *body_; }

  // Can only be called by type-checking, if a conversion was required.
  void set_condition(Nonnull<Expression*> condition) { condition_ = condition; }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Block*> body_;
};

class For : public Statement {
 public:
  For(SourceLocation source_loc, Nonnull<BindingPattern*> variable_declaration,
      Nonnull<Expression*> loop_target, Nonnull<Block*> body)
      : Statement(AstNodeKind::For, source_loc),
        variable_declaration_(variable_declaration),
        loop_target_(loop_target),
        body_(body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromFor(node->kind());
  }

  auto variable_declaration() const -> const BindingPattern& {
    return *variable_declaration_;
  }
  auto variable_declaration() -> BindingPattern& {
    return *variable_declaration_;
  }

  auto loop_target() const -> const Expression& { return *loop_target_; }
  auto loop_target() -> Expression& { return *loop_target_; }

  auto body() const -> const Block& { return *body_; }
  auto body() -> Block& { return *body_; }

 private:
  Nonnull<BindingPattern*> variable_declaration_;
  Nonnull<Expression*> loop_target_;
  Nonnull<Block*> body_;
};

class Break : public Statement {
 public:
  explicit Break(SourceLocation source_loc)
      : Statement(AstNodeKind::Break, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromBreak(node->kind());
  }

  // The AST node representing the loop this statement breaks out of.
  // Can only be called after ResolveControlFlow has visited this node.
  //
  // Note that this function does not represent an edge in the tree
  // structure of the AST: the return value is not a child of this node,
  // but an ancestor.
  auto loop() const -> const Statement& { return **loop_; }

  // Can only be called once, by ResolveControlFlow.
  void set_loop(Nonnull<const Statement*> loop) {
    CARBON_CHECK(!loop_.has_value());
    loop_ = loop;
  }

 private:
  std::optional<Nonnull<const Statement*>> loop_;
};

class Continue : public Statement {
 public:
  explicit Continue(SourceLocation source_loc)
      : Statement(AstNodeKind::Continue, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromContinue(node->kind());
  }

  // The AST node representing the loop this statement continues.
  // Can only be called after ResolveControlFlow has visited this node.
  //
  // Note that this function does not represent an edge in the tree
  // structure of the AST: the return value is not a child of this node,
  // but an ancestor.
  auto loop() const -> const Statement& { return **loop_; }

  // Can only be called once, by ResolveControlFlow.
  void set_loop(Nonnull<const Statement*> loop) {
    CARBON_CHECK(!loop_.has_value());
    loop_ = loop;
  }

 private:
  std::optional<Nonnull<const Statement*>> loop_;
};

class Match : public Statement {
 public:
  class Clause {
   public:
    Clause(Nonnull<Pattern*> pattern, Nonnull<Statement*> statement)
        : pattern_(pattern), statement_(statement) {}

    auto pattern() const -> const Pattern& { return *pattern_; }
    auto pattern() -> Pattern& { return *pattern_; }
    auto statement() const -> const Statement& { return *statement_; }
    auto statement() -> Statement& { return *statement_; }

   private:
    Nonnull<Pattern*> pattern_;
    Nonnull<Statement*> statement_;
  };

  Match(SourceLocation source_loc, Nonnull<Expression*> expression,
        std::vector<Clause> clauses)
      : Statement(AstNodeKind::Match, source_loc),
        expression_(expression),
        clauses_(std::move(clauses)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMatch(node->kind());
  }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }
  auto clauses() const -> llvm::ArrayRef<Clause> { return clauses_; }
  auto clauses() -> llvm::MutableArrayRef<Clause> { return clauses_; }

  // Can only be called by type-checking, if a conversion was required.
  void set_expression(Nonnull<Expression*> expression) {
    expression_ = expression;
  }

 private:
  Nonnull<Expression*> expression_;
  std::vector<Clause> clauses_;
};

// A continuation statement.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
class Continuation : public Statement {
 public:
  using ImplementsCarbonValueNode = void;

  Continuation(SourceLocation source_loc, std::string name,
               Nonnull<Block*> body)
      : Statement(AstNodeKind::Continuation, source_loc),
        name_(std::move(name)),
        body_(body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromContinuation(node->kind());
  }

  auto name() const -> const std::string& { return name_; }
  auto body() const -> const Block& { return *body_; }
  auto body() -> Block& { return *body_; }

  // The static type of the continuation. Cannot be called before typechecking.
  //
  // This will always be ContinuationType, but we must set it dynamically in
  // the typechecker because this code can't depend on ContinuationType.
  auto static_type() const -> const Value& { return **static_type_; }

  // Sets the static type of the continuation. Can only be called once,
  // during typechecking.
  void set_static_type(Nonnull<const Value*> type) {
    CARBON_CHECK(!static_type_.has_value());
    static_type_ = type;
  }

  auto value_category() const -> ValueCategory { return ValueCategory::Var; }
  auto constant_value() const -> std::optional<Nonnull<const Value*>> {
    return std::nullopt;
  }
  auto symbolic_identity() const -> std::optional<Nonnull<const Value*>> {
    return std::nullopt;
  }

 private:
  std::string name_;
  Nonnull<Block*> body_;
  std::optional<Nonnull<const Value*>> static_type_;
};

// A run statement.
//
//     __run <argument>;
class Run : public Statement {
 public:
  Run(SourceLocation source_loc, Nonnull<Expression*> argument)
      : Statement(AstNodeKind::Run, source_loc), argument_(argument) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromRun(node->kind());
  }

  auto argument() const -> const Expression& { return *argument_; }
  auto argument() -> Expression& { return *argument_; }

  // Can only be called by type-checking, if a conversion was required.
  void set_argument(Nonnull<Expression*> argument) { argument_ = argument; }

 private:
  Nonnull<Expression*> argument_;
};

// An await statement.
//
//    __await;
class Await : public Statement {
 public:
  explicit Await(SourceLocation source_loc)
      : Statement(AstNodeKind::Await, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAwait(node->kind());
  }
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_AST_STATEMENT_H_
