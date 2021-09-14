// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_

#include <set>

#include "common/ostream.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

using TypeEnv = Dictionary<std::string, const Value*>;

class TypeChecker {
 public:
  explicit TypeChecker(Arena* arena) : arena(arena), interpreter(arena) {}

  struct TypeCheckContext {
    TypeCheckContext(Arena* arena) : types(arena), values(arena) {}

    // Symbol table mapping names of runtime entities to their type.
    TypeEnv types;
    // Symbol table mapping names of compile time entities to their value.
    Env values;
  };

  auto MakeTypeChecked(const Declaration* d, const TypeEnv& types,
                       const Env& values) -> const Declaration*;

  auto TopLevel(const std::vector<const Declaration*>& fs) -> TypeCheckContext;

 private:
  struct TCExpression {
    TCExpression(const Expression* e, const Value* t, TypeEnv types)
        : exp(e), type(t), types(types) {}

    const Expression* exp;
    const Value* type;
    TypeEnv types;
  };

  struct TCPattern {
    const Pattern* pattern;
    const Value* type;
    TypeEnv types;
  };

  struct TCStatement {
    TCStatement(const Statement* s, TypeEnv types) : stmt(s), types(types) {}

    const Statement* stmt;
    TypeEnv types;
  };

  // TypeCheckExp performs semantic analysis on an expression.  It returns a new
  // version of the expression, its type, and an updated environment which are
  // bundled into a TCResult object.  The purpose of the updated environment is
  // to bring pattern variables into scope, for example, in a match case.  The
  // new version of the expression may include more information, for example,
  // the type arguments deduced for the type parameters of a generic.
  //
  // e is the expression to be analyzed.
  // types maps variable names to the type of their run-time value.
  // values maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterExp.
  auto TypeCheckExp(const Expression* e, TypeEnv types, Env values)
      -> TCExpression;

  // Equivalent to TypeCheckExp, but operates on Patterns instead of
  // Expressions. `expected` is the type that this pattern is expected to have,
  // if the surrounding context gives us that information. Otherwise, it is
  // nullopt.
  auto TypeCheckPattern(const Pattern* p, TypeEnv types, Env values,
                        std::optional<const Value*> expected) -> TCPattern;

  // TypeCheckStmt performs semantic analysis on a statement.  It returns a new
  // version of the statement and a new type environment.
  //
  // The ret_type parameter is used for analyzing return statements.  It is the
  // declared return type of the enclosing function definition.  If the return
  // type is "auto", then the return type is inferred from the first return
  // statement.
  auto TypeCheckStmt(const Statement* s, TypeEnv types, Env values,
                     const Value*& ret_type, bool is_omitted_ret_type)
      -> TCStatement;

  auto TypeCheckFunDef(const FunctionDefinition* f, TypeEnv types, Env values)
      -> const FunctionDefinition*;

  auto TypeCheckCase(const Value* expected, const Pattern* pat,
                     const Statement* body, TypeEnv types, Env values,
                     const Value*& ret_type, bool is_omitted_ret_type)
      -> std::pair<const Pattern*, const Statement*>;

  auto TypeOfFunDef(TypeEnv types, Env values,
                    const FunctionDefinition* fun_def) -> const Value*;
  auto TypeOfClassDef(const ClassDefinition* sd, TypeEnv /*types*/, Env ct_top)
      -> const Value*;

  void TopLevel(const Declaration& d, TypeCheckContext* tops);

  auto CheckOrEnsureReturn(std::optional<const Statement*> opt_stmt,
                           bool omitted_ret_type, SourceLocation loc)
      -> const Statement*;

  // Reify type to type expression.
  auto ReifyType(const Value* t, SourceLocation loc) -> const Expression*;

  auto Substitute(TypeEnv dict, const Value* type) -> const Value*;

  Arena* arena;
  Interpreter interpreter;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
