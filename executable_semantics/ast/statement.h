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
  auto Tag() const -> Kind { return tag; }

  auto SourceLoc() const -> SourceLocation { return loc; }

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs an Statement representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Statement(Kind tag, SourceLocation loc) : tag(tag), loc(loc) {}

 private:
  const Kind tag;
  SourceLocation loc;
};

class ExpressionStatement : public Statement {
 public:
  ExpressionStatement(SourceLocation loc, Nonnull<const Expression*> exp)
      : Statement(Kind::ExpressionStatement, loc), exp(exp) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::ExpressionStatement;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }

 private:
  Nonnull<const Expression*> exp;
};

class Assign : public Statement {
 public:
  Assign(SourceLocation loc, Nonnull<const Expression*> lhs,
         Nonnull<const Expression*> rhs)
      : Statement(Kind::Assign, loc), lhs(lhs), rhs(rhs) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Assign;
  }

  auto Lhs() const -> Nonnull<const Expression*> { return lhs; }
  auto Rhs() const -> Nonnull<const Expression*> { return rhs; }

 private:
  Nonnull<const Expression*> lhs;
  Nonnull<const Expression*> rhs;
};

class VariableDefinition : public Statement {
 public:
  VariableDefinition(SourceLocation loc, Nonnull<const Pattern*> pat,
                     Nonnull<const Expression*> init)
      : Statement(Kind::VariableDefinition, loc), pat(pat), init(init) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::VariableDefinition;
  }

  auto Pat() const -> Nonnull<const Pattern*> { return pat; }
  auto Init() const -> Nonnull<const Expression*> { return init; }

 private:
  Nonnull<const Pattern*> pat;
  Nonnull<const Expression*> init;
};

class If : public Statement {
 public:
  If(SourceLocation loc, Nonnull<const Expression*> cond,
     Nonnull<const Statement*> then_stmt,
     std::optional<Nonnull<const Statement*>> else_stmt)
      : Statement(Kind::If, loc),
        cond(cond),
        then_stmt(then_stmt),
        else_stmt(else_stmt) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::If;
  }

  auto Cond() const -> Nonnull<const Expression*> { return cond; }
  auto ThenStmt() const -> Nonnull<const Statement*> { return then_stmt; }
  auto ElseStmt() const -> std::optional<Nonnull<const Statement*>> {
    return else_stmt;
  }

 private:
  Nonnull<const Expression*> cond;
  Nonnull<const Statement*> then_stmt;
  std::optional<Nonnull<const Statement*>> else_stmt;
};

class Return : public Statement {
 public:
  Return(Nonnull<Arena*> arena, SourceLocation loc)
      : Return(loc, arena->New<TupleLiteral>(loc), true) {}
  Return(SourceLocation loc, Nonnull<const Expression*> exp,
         bool is_omitted_exp)
      : Statement(Kind::Return, loc),
        exp(exp),
        is_omitted_exp(is_omitted_exp) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Return;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }
  auto IsOmittedExp() const -> bool { return is_omitted_exp; }

 private:
  Nonnull<const Expression*> exp;
  bool is_omitted_exp;
};

class Sequence : public Statement {
 public:
  Sequence(SourceLocation loc, Nonnull<const Statement*> stmt,
           std::optional<Nonnull<const Statement*>> next)
      : Statement(Kind::Sequence, loc), stmt(stmt), next(next) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Sequence;
  }

  auto Stmt() const -> Nonnull<const Statement*> { return stmt; }
  auto Next() const -> std::optional<Nonnull<const Statement*>> { return next; }

 private:
  Nonnull<const Statement*> stmt;
  std::optional<Nonnull<const Statement*>> next;
};

class Block : public Statement {
 public:
  Block(SourceLocation loc, std::optional<Nonnull<const Statement*>> stmt)
      : Statement(Kind::Block, loc), stmt(stmt) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Block;
  }

  auto Stmt() const -> std::optional<Nonnull<const Statement*>> { return stmt; }

 private:
  std::optional<Nonnull<const Statement*>> stmt;
};

class While : public Statement {
 public:
  While(SourceLocation loc, Nonnull<const Expression*> cond,
        Nonnull<const Statement*> body)
      : Statement(Kind::While, loc), cond(cond), body(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::While;
  }

  auto Cond() const -> Nonnull<const Expression*> { return cond; }
  auto Body() const -> Nonnull<const Statement*> { return body; }

 private:
  Nonnull<const Expression*> cond;
  Nonnull<const Statement*> body;
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
  Match(
      SourceLocation loc, Nonnull<const Expression*> exp,
      std::vector<std::pair<Nonnull<const Pattern*>, Nonnull<const Statement*>>>
          clauses)
      : Statement(Kind::Match, loc), exp(exp), clauses(std::move(clauses)) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Match;
  }

  auto Exp() const -> Nonnull<const Expression*> { return exp; }
  auto Clauses() const -> const std::vector<
      std::pair<Nonnull<const Pattern*>, Nonnull<const Statement*>>>& {
    return clauses;
  }

 private:
  Nonnull<const Expression*> exp;
  std::vector<std::pair<Nonnull<const Pattern*>, Nonnull<const Statement*>>>
      clauses;
};

// A continuation statement.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
class Continuation : public Statement {
 public:
  Continuation(SourceLocation loc, std::string continuation_variable,
               Nonnull<const Statement*> body)
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

 private:
  std::string continuation_variable;
  Nonnull<const Statement*> body;
};

// A run statement.
//
//     __run <argument>;
class Run : public Statement {
 public:
  Run(SourceLocation loc, Nonnull<const Expression*> argument)
      : Statement(Kind::Run, loc), argument(argument) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Run;
  }

  auto Argument() const -> Nonnull<const Expression*> { return argument; }

 private:
  Nonnull<const Expression*> argument;
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
