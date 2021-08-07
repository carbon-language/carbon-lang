// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
#define EXECUTABLE_SEMANTICS_AST_STATEMENT_H_

#include <list>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/pattern.h"
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

  auto LineNumber() const -> int { return line_num; }

  void Print(llvm::raw_ostream& out) const { PrintDepth(-1, out); }
  void PrintDepth(int depth, llvm::raw_ostream& out) const;
  LLVM_DUMP_METHOD void Dump() const { Print(llvm::errs()); }

 protected:
  // Constructs an Statement representing syntax at the given line number.
  // `tag` must be the enumerator corresponding to the most-derived type being
  // constructed.
  Statement(Kind tag, int line_num) : tag(tag), line_num(line_num) {}

 private:
  const Kind tag;
  int line_num;
};

class ExpressionStatement : public Statement {
 public:
  ExpressionStatement(int line_num, const Expression* exp)
      : Statement(Kind::ExpressionStatement, line_num), exp(exp) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::ExpressionStatement;
  }

  auto Exp() const -> const Expression* { return exp; }

 private:
  const Expression* exp;
};

class Assign : public Statement {
 public:
  Assign(int line_num, const Expression* lhs, const Expression* rhs)
      : Statement(Kind::Assign, line_num), lhs(lhs), rhs(rhs) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Assign;
  }

  auto Lhs() const -> const Expression* { return lhs; }
  auto Rhs() const -> const Expression* { return rhs; }

 private:
  const Expression* lhs;
  const Expression* rhs;
};

class VariableDefinition : public Statement {
 public:
  VariableDefinition(int line_num, const Pattern* pat, const Expression* init)
      : Statement(Kind::VariableDefinition, line_num), pat(pat), init(init) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::VariableDefinition;
  }

  auto Pat() const -> const Pattern* { return pat; }
  auto Init() const -> const Expression* { return init; }

 private:
  const Pattern* pat;
  const Expression* init;
};

class If : public Statement {
 public:
  If(int line_num, const Expression* cond, const Statement* then_stmt,
     const Statement* else_stmt)
      : Statement(Kind::If, line_num),
        cond(cond),
        then_stmt(then_stmt),
        else_stmt(else_stmt) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::If;
  }

  auto Cond() const -> const Expression* { return cond; }
  auto ThenStmt() const -> const Statement* { return then_stmt; }
  auto ElseStmt() const -> const Statement* { return else_stmt; }

 private:
  const Expression* cond;
  const Statement* then_stmt;
  const Statement* else_stmt;
};

class Return : public Statement {
 public:
  Return(int line_num, const Expression* exp, bool is_omitted_exp);

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Return;
  }

  auto Exp() const -> const Expression* { return exp; }
  auto IsOmittedExp() const -> bool { return is_omitted_exp; }

 private:
  const Expression* exp;
  bool is_omitted_exp;
};

class Sequence : public Statement {
 public:
  Sequence(int line_num, const Statement* stmt, const Statement* next)
      : Statement(Kind::Sequence, line_num), stmt(stmt), next(next) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Sequence;
  }

  auto Stmt() const -> const Statement* { return stmt; }
  auto Next() const -> const Statement* { return next; }

 private:
  const Statement* stmt;
  const Statement* next;
};

class Block : public Statement {
 public:
  Block(int line_num, const Statement* stmt)
      : Statement(Kind::Block, line_num), stmt(stmt) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Block;
  }

  auto Stmt() const -> const Statement* { return stmt; }

 private:
  const Statement* stmt;
};

class While : public Statement {
 public:
  While(int line_num, const Expression* cond, const Statement* body)
      : Statement(Kind::While, line_num), cond(cond), body(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::While;
  }

  auto Cond() const -> const Expression* { return cond; }
  auto Body() const -> const Statement* { return body; }

 private:
  const Expression* cond;
  const Statement* body;
};

class Break : public Statement {
 public:
  explicit Break(int line_num) : Statement(Kind::Break, line_num) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Break;
  }
};

class Continue : public Statement {
 public:
  explicit Continue(int line_num) : Statement(Kind::Continue, line_num) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Continue;
  }
};

class Match : public Statement {
 public:
  Match(int line_num, const Expression* exp,
        std::list<std::pair<const Pattern*, const Statement*>>* clauses)
      : Statement(Kind::Match, line_num), exp(exp), clauses(clauses) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Match;
  }

  auto Exp() const -> const Expression* { return exp; }
  auto Clauses() const
      -> const std::list<std::pair<const Pattern*, const Statement*>>* {
    return clauses;
  }

 private:
  const Expression* exp;
  std::list<std::pair<const Pattern*, const Statement*>>* clauses;
};

// A continuation statement.
//
//     __continuation <continuation_variable> {
//       <body>
//     }
class Continuation : public Statement {
 public:
  Continuation(int line_num, std::string continuation_variable,
               const Statement* body)
      : Statement(Kind::Continuation, line_num),
        continuation_variable(std::move(continuation_variable)),
        body(body) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Continuation;
  }

  auto ContinuationVariable() const -> const std::string& {
    return continuation_variable;
  }
  auto Body() const -> const Statement* { return body; }

 private:
  std::string continuation_variable;
  const Statement* body;
};

// A run statement.
//
//     __run <argument>;
class Run : public Statement {
 public:
  Run(int line_num, const Expression* argument)
      : Statement(Kind::Run, line_num), argument(argument) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Run;
  }

  auto Argument() const -> const Expression* { return argument; }

 private:
  const Expression* argument;
};

// An await statement.
//
//    __await;
class Await : public Statement {
 public:
  explicit Await(int line_num) : Statement(Kind::Await, line_num) {}

  static auto classof(const Statement* stmt) -> bool {
    return stmt->Tag() == Kind::Await;
  }
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_AST_STATEMENT_H_
