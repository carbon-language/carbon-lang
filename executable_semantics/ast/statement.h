// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
#define EXECUTABLE_SEMANTICS_AST_STATEMENT_H_

#include <vector>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/source_location.h"
#include "executable_semantics/common/arena.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace Carbon {

class FunctionDeclaration;

class Statement {
 public:
  enum class Kind {
    ExpressionStatement,
    Assign,
    VariableDefinition,
    If,
    Return,
    Sequence,
    Block,
    While,
    Break,
    Continue,
    Match,
    Continuation,  // Create a first-class continuation.
    Run,           // Run a continuation to the next await or until it finishes.
    Await,         // Pause execution of the continuation.
  };

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto kind() const -> Kind { return kind_; }

  auto source_loc() const -> SourceLocation { return source_loc_; }

 protected:
  // Constructs an Statement representing syntax at the given line number.
  // `kind` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Statement(Kind kind, SourceLocation source_loc)
      : kind_(kind), source_loc_(source_loc) {}

 private:
  const Kind kind_;
  SourceLocation source_loc_;
};

class ExpressionStatement : public Statement {
 public:
  ExpressionStatement(SourceLocation source_loc,
                      Nonnull<Expression*> expression)
      : Statement(Kind::ExpressionStatement, source_loc),
        expression_(expression) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::ExpressionStatement;
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
      : Statement(Kind::Assign, source_loc), lhs_(lhs), rhs_(rhs) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Assign;
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
      : Statement(Kind::VariableDefinition, source_loc),
        pattern_(pattern),
        init_(init) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::VariableDefinition;
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
     Nonnull<Statement*> then_statement,
     std::optional<Nonnull<Statement*>> else_statement)
      : Statement(Kind::If, source_loc),
        condition_(condition),
        then_statement_(then_statement),
        else_statement_(else_statement) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::If;
  }

  auto condition() const -> const Expression& { return *condition_; }
  auto condition() -> Expression& { return *condition_; }
  auto then_statement() const -> const Statement& { return *then_statement_; }
  auto then_statement() -> Statement& { return *then_statement_; }
  auto else_statement() const -> std::optional<Nonnull<const Statement*>> {
    return else_statement_;
  }
  auto else_statement() -> std::optional<Nonnull<Statement*>> {
    return else_statement_;
  }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Statement*> then_statement_;
  std::optional<Nonnull<Statement*>> else_statement_;
};

class Return : public Statement {
 public:
  Return(Nonnull<Arena*> arena, SourceLocation source_loc)
      : Return(source_loc, arena->New<TupleLiteral>(source_loc), true) {}
  Return(SourceLocation source_loc, Nonnull<Expression*> expression,
         bool is_omitted_expression)
      : Statement(Kind::Return, source_loc),
        expression_(expression),
        is_omitted_expression_(is_omitted_expression) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Return;
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

  // Can only be called once, by ResolveControlFlow.
  void set_function(Nonnull<const FunctionDeclaration*> function) {
    CHECK(!function_.has_value());
    function_ = function;
  }

 private:
  Nonnull<Expression*> expression_;
  bool is_omitted_expression_;
  std::optional<Nonnull<const FunctionDeclaration*>> function_;
};

class Sequence : public Statement {
 public:
  Sequence(SourceLocation source_loc, Nonnull<Statement*> statement,
           std::optional<Nonnull<Statement*>> next)
      : Statement(Kind::Sequence, source_loc),
        statement_(statement),
        next_(next) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Sequence;
  }

  auto statement() const -> const Statement& { return *statement_; }
  auto statement() -> Statement& { return *statement_; }
  auto next() const -> std::optional<Nonnull<const Statement*>> {
    return next_;
  }
  auto next() -> std::optional<Nonnull<Statement*>> { return next_; }

 private:
  Nonnull<Statement*> statement_;
  std::optional<Nonnull<Statement*>> next_;
};

class Block : public Statement {
 public:
  Block(SourceLocation source_loc, std::optional<Nonnull<Statement*>> statement)
      : Statement(Kind::Block, source_loc), statement_(statement) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Block;
  }

  auto statement() const -> std::optional<Nonnull<const Statement*>> {
    return statement_;
  }
  auto statement() -> std::optional<Nonnull<Statement*>> { return statement_; }

 private:
  std::optional<Nonnull<Statement*>> statement_;
};

class While : public Statement {
 public:
  While(SourceLocation source_loc, Nonnull<Expression*> condition,
        Nonnull<Statement*> body)
      : Statement(Kind::While, source_loc),
        condition_(condition),
        body_(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::While;
  }

  auto condition() const -> const Expression& { return *condition_; }
  auto condition() -> Expression& { return *condition_; }
  auto body() const -> const Statement& { return *body_; }
  auto body() -> Statement& { return *body_; }

 private:
  Nonnull<Expression*> condition_;
  Nonnull<Statement*> body_;
};

class Break : public Statement {
 public:
  explicit Break(SourceLocation source_loc)
      : Statement(Kind::Break, source_loc) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Break;
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
      : Statement(Kind::Continue, source_loc) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Continue;
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
      : Statement(Kind::Match, source_loc),
        expression_(expression),
        clauses_(std::move(clauses)) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Match;
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
class Continuation : public Statement {
 public:
  Continuation(SourceLocation source_loc, std::string continuation_variable,
               Nonnull<Statement*> body)
      : Statement(Kind::Continuation, source_loc),
        continuation_variable_(std::move(continuation_variable)),
        body_(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Continuation;
  }

  auto continuation_variable() const -> const std::string& {
    return continuation_variable_;
  }
  auto body() const -> const Statement& { return *body_; }
  auto body() -> Statement& { return *body_; }

 private:
  std::string continuation_variable_;
  Nonnull<Statement*> body_;
};

// A run statement.
//
//     __run <argument>;
class Run : public Statement {
 public:
  Run(SourceLocation source_loc, Nonnull<Expression*> argument)
      : Statement(Kind::Run, source_loc), argument_(argument) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Run;
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
      : Statement(Kind::Await, source_loc) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->kind() == Kind::Await;
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
