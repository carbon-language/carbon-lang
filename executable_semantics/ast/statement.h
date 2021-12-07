// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
#define EXECUTABLE_SEMANTICS_AST_STATEMENT_H_

#include <utility>
#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/ast/static_scope.h"
#include "executable_semantics/common/arena.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class FunctionDeclaration;

class Statement : public virtual AstNode {
 public:
  ~Statement() override = 0;

  void Print(llvm::raw_ostream& out) const override { PrintDepth(-1, out); }
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
  Statement() = default;
};

class Block : public Statement {
 public:
  Block(SourceLocation source_loc, std::vector<Nonnull<Statement*>> statements)
      : AstNode(AstNodeKind::Block, source_loc),
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
      : AstNode(AstNodeKind::ExpressionStatement, source_loc),
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
      : AstNode(AstNodeKind::Assign, source_loc), lhs_(lhs), rhs_(rhs) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAssign(node->kind());
  }

  auto lhs() const -> const Expression& { return *lhs_; }
  auto lhs() -> Expression& { return *lhs_; }
  auto rhs() const -> const Expression& { return *rhs_; }
  auto rhs() -> Expression& { return *rhs_; }

 private:
  Nonnull<Expression*> lhs_;
  Nonnull<Expression*> rhs_;
};

class VariableDefinition : public Statement {
 public:
  VariableDefinition(SourceLocation source_loc, Nonnull<Pattern*> pattern,
                     Nonnull<Expression*> init)
      : AstNode(AstNodeKind::VariableDefinition, source_loc),
        pattern_(pattern),
        init_(init) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromVariableDefinition(node->kind());
  }

  auto pattern() const -> const Pattern& { return *pattern_; }
  auto pattern() -> Pattern& { return *pattern_; }
  auto init() const -> const Expression& { return *init_; }
  auto init() -> Expression& { return *init_; }

 private:
  Nonnull<Pattern*> pattern_;
  Nonnull<Expression*> init_;
};

class If : public Statement {
 public:
  If(SourceLocation source_loc, Nonnull<Expression*> condition,
     Nonnull<Block*> then_block, std::optional<Nonnull<Block*>> else_block)
      : AstNode(AstNodeKind::If, source_loc),
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

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Block*> then_block_;
  std::optional<Nonnull<Block*>> else_block_;
};

class Return : public Statement {
 public:
  Return(Nonnull<Arena*> arena, SourceLocation source_loc)
      : Return(source_loc, arena->New<TupleLiteral>(source_loc), true) {}
  Return(SourceLocation source_loc, Nonnull<Expression*> expression,
         bool is_omitted_expression)
      : AstNode(AstNodeKind::Return, source_loc),
        expression_(expression),
        is_omitted_expression_(is_omitted_expression) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromReturn(node->kind());
  }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }
  auto is_omitted_expression() const -> bool { return is_omitted_expression_; }

  // The AST node representing the function body this statement returns from.
  // Can only be called after ResolveControlFlow has visited this node.
  //
  // Note that this function does not represent an edge in the tree
  // structure of the AST: the return value is not a child of this node,
  // but an ancestor.
  auto function() const -> const FunctionDeclaration& { return **function_; }
  auto function() -> FunctionDeclaration& { return **function_; }

  // Can only be called once, by ResolveControlFlow.
  void set_function(Nonnull<FunctionDeclaration*> function) {
    CHECK(!function_.has_value());
    function_ = function;
  }

 private:
  Nonnull<Expression*> expression_;
  bool is_omitted_expression_;
  std::optional<Nonnull<FunctionDeclaration*>> function_;
};

class While : public Statement {
 public:
  While(SourceLocation source_loc, Nonnull<Expression*> condition,
        Nonnull<Block*> body)
      : AstNode(AstNodeKind::While, source_loc),
        condition_(condition),
        body_(body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromWhile(node->kind());
  }

  auto condition() const -> const Expression& { return *condition_; }
  auto condition() -> Expression& { return *condition_; }
  auto body() const -> const Block& { return *body_; }
  auto body() -> Block& { return *body_; }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Block*> body_;
};

class Break : public Statement {
 public:
  explicit Break(SourceLocation source_loc)
      : AstNode(AstNodeKind::Break, source_loc) {}

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
    CHECK(!loop_.has_value());
    loop_ = loop;
  }

 private:
  std::optional<Nonnull<const Statement*>> loop_;
};

class Continue : public Statement {
 public:
  explicit Continue(SourceLocation source_loc)
      : AstNode(AstNodeKind::Continue, source_loc) {}

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
    CHECK(!loop_.has_value());
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
      : AstNode(AstNodeKind::Match, source_loc),
        expression_(expression),
        clauses_(std::move(clauses)) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromMatch(node->kind());
  }

  auto expression() const -> const Expression& { return *expression_; }
  auto expression() -> Expression& { return *expression_; }
  auto clauses() const -> llvm::ArrayRef<Clause> { return clauses_; }
  auto clauses() -> llvm::MutableArrayRef<Clause> { return clauses_; }

 private:
  Nonnull<Expression*> expression_;
  std::vector<Clause> clauses_;
};

// A continuation statement.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
class Continuation : public Statement, public NamedEntity {
 public:
  Continuation(SourceLocation source_loc, std::string continuation_variable,
               Nonnull<Block*> body)
      : AstNode(AstNodeKind::Continuation, source_loc),
        continuation_variable_(std::move(continuation_variable)),
        body_(body) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromContinuation(node->kind());
  }

  auto continuation_variable() const -> const std::string& {
    return continuation_variable_;
  }
  auto body() const -> const Block& { return *body_; }
  auto body() -> Block& { return *body_; }

 private:
  std::string continuation_variable_;
  Nonnull<Block*> body_;
};

// A run statement.
//
//     __run <argument>;
class Run : public Statement {
 public:
  Run(SourceLocation source_loc, Nonnull<Expression*> argument)
      : AstNode(AstNodeKind::Run, source_loc), argument_(argument) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromRun(node->kind());
  }

  auto argument() const -> const Expression& { return *argument_; }
  auto argument() -> Expression& { return *argument_; }

 private:
  Nonnull<Expression*> argument_;
};

// An await statement.
//
//    __await;
class Await : public Statement {
 public:
  explicit Await(SourceLocation source_loc)
      : AstNode(AstNodeKind::Await, source_loc) {}

  static auto classof(const AstNode* node) -> bool {
    return InheritsFromAwait(node->kind());
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
