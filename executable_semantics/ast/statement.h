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

  // Returns the enumerator corresponding to the most-derived type of this
  // object.
  auto Tag() const -> Kind { return kind; }

  auto SourceLoc() const -> SourceLocation { return loc; }

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs an Statement representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Statement(Kind kind, SourceLocation loc) : kind(kind), loc(loc) {}

 private:
  const Kind kind;
  SourceLocation loc;
};

class ExpressionStatement : public Statement {
 public:
  ExpressionStatement(SourceLocation loc, Nonnull<Expression*> exp)
      : Statement(Kind::ExpressionStatement, loc), exp(exp) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::ExpressionStatement;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }
  auto Exp() -> Nonnull<Expression*> { return exp; }

 private:
  Nonnull<Expression*> exp;
};

class Assign : public Statement {
 public:
  Assign(SourceLocation loc, Nonnull<Expression*> lhs, Nonnull<Expression*> rhs)
      : Statement(Kind::Assign, loc), lhs(lhs), rhs(rhs) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Assign;
  }

  auto Lhs() const -> Nonnull<const Expression*> { return lhs; }
  auto Lhs() -> Nonnull<Expression*> { return lhs; }
  auto Rhs() const -> Nonnull<const Expression*> { return rhs; }
  auto Rhs() -> Nonnull<Expression*> { return rhs; }

 private:
  Nonnull<Expression*> lhs;
  Nonnull<Expression*> rhs;
};

class VariableDefinition : public Statement {
 public:
  VariableDefinition(SourceLocation loc, Nonnull<Pattern*> pat,
                     Nonnull<Expression*> init)
      : Statement(Kind::VariableDefinition, loc), pat(pat), init(init) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::VariableDefinition;
  }

  auto Pat() const -> Nonnull<const Pattern*> { return pat; }
  auto Pat() -> Nonnull<Pattern*> { return pat; }
  auto Init() const -> Nonnull<const Expression*> { return init; }
  auto Init() -> Nonnull<Expression*> { return init; }

 private:
  Nonnull<Pattern*> pat;
  Nonnull<Expression*> init;
};

class If : public Statement {
 public:
  If(SourceLocation loc, Nonnull<Expression*> cond,
     Nonnull<Statement*> then_stmt,
     std::optional<Nonnull<Statement*>> else_stmt)
      : Statement(Kind::If, loc),
        cond(cond),
        then_stmt(then_stmt),
        else_stmt(else_stmt) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::If;
  }

  auto Cond() const -> Nonnull<const Expression*> { return cond; }
  auto Cond() -> Nonnull<Expression*> { return cond; }
  auto ThenStmt() const -> Nonnull<const Statement*> { return then_stmt; }
  auto ThenStmt() -> Nonnull<Statement*> { return then_stmt; }
  auto ElseStmt() const -> std::optional<Nonnull<const Statement*>> {
    return else_stmt;
  }
  auto ElseStmt() -> std::optional<Nonnull<Statement*>> { return else_stmt; }

 private:
  Nonnull<Expression*> cond;
  Nonnull<Statement*> then_stmt;
  std::optional<Nonnull<Statement*>> else_stmt;
};

class Return : public Statement {
 public:
  Return(Nonnull<Arena*> arena, SourceLocation loc)
      : Return(loc, arena->New<TupleLiteral>(loc), true) {}
  Return(SourceLocation loc, Nonnull<Expression*> exp, bool is_omitted_exp)
      : Statement(Kind::Return, loc),
        exp(exp),
        is_omitted_exp(is_omitted_exp) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Return;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }
  auto Exp() -> Nonnull<Expression*> { return exp; }
  auto IsOmittedExp() const -> bool { return is_omitted_exp; }

 private:
  Nonnull<Expression*> exp;
  bool is_omitted_exp;
};

class Sequence : public Statement {
 public:
  Sequence(SourceLocation loc, Nonnull<Statement*> stmt,
           std::optional<Nonnull<Statement*>> next)
      : Statement(Kind::Sequence, loc), stmt(stmt), next(next) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Sequence;
  }

  auto Stmt() const -> Nonnull<const Statement*> { return stmt; }
  auto Stmt() -> Nonnull<Statement*> { return stmt; }
  auto Next() const -> std::optional<Nonnull<const Statement*>> { return next; }
  auto Next() -> std::optional<Nonnull<Statement*>> { return next; }

 private:
  Nonnull<Statement*> stmt;
  std::optional<Nonnull<Statement*>> next;
};

class Block : public Statement {
 public:
  Block(SourceLocation loc, std::optional<Nonnull<Statement*>> stmt)
      : Statement(Kind::Block, loc), stmt(stmt) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Block;
  }

  auto Stmt() const -> std::optional<Nonnull<const Statement*>> { return stmt; }
  auto Stmt() -> std::optional<Nonnull<Statement*>> { return stmt; }

 private:
  std::optional<Nonnull<Statement*>> stmt;
};

class While : public Statement {
 public:
  While(SourceLocation loc, Nonnull<Expression*> cond, Nonnull<Statement*> body)
      : Statement(Kind::While, loc), cond(cond), body(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::While;
  }

  auto Cond() const -> Nonnull<const Expression*> { return cond; }
  auto Cond() -> Nonnull<Expression*> { return cond; }
  auto Body() const -> Nonnull<const Statement*> { return body; }
  auto Body() -> Nonnull<Statement*> { return body; }

 private:
  Nonnull<Expression*> cond;
  Nonnull<Statement*> body;
};

class Break : public Statement {
 public:
  explicit Break(SourceLocation loc) : Statement(Kind::Break, loc) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Break;
  }
};

class Continue : public Statement {
 public:
  explicit Continue(SourceLocation loc) : Statement(Kind::Continue, loc) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Continue;
  }
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

  Match(SourceLocation loc, Nonnull<Expression*> expression,
        std::vector<Clause> clauses)
      : Statement(Kind::Match, loc),
        expression_(expression),
        clauses_(std::move(clauses)) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Match;
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
  Continuation(SourceLocation loc, std::string continuation_variable,
               Nonnull<Statement*> body)
      : Statement(Kind::Continuation, loc),
        continuation_variable(std::move(continuation_variable)),
        body(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Continuation;
  }

  auto ContinuationVariable() const -> const std::string& {
    return continuation_variable;
  }
  auto Body() const -> Nonnull<const Statement*> { return body; }
  auto Body() -> Nonnull<Statement*> { return body; }

 private:
  std::string continuation_variable;
  Nonnull<Statement*> body;
};

// A run statement.
//
//     __run <argument>;
class Run : public Statement {
 public:
  Run(SourceLocation loc, Nonnull<Expression*> argument)
      : Statement(Kind::Run, loc), argument(argument) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Run;
  }

  auto Argument() const -> Nonnull<const Expression*> { return argument; }
  auto Argument() -> Nonnull<Expression*> { return argument; }

 private:
  Nonnull<Expression*> argument;
};

// An await statement.
//
//    __await;
class Await : public Statement {
 public:
  explicit Await(SourceLocation loc) : Statement(Kind::Await, loc) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Await;
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
