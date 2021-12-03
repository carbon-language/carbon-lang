// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_

#include <set>

#include "common/ostream.h"
#include "executable_semantics/ast/ast.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/common/nonnull.h"
#include "executable_semantics/interpreter/dictionary.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

class TypeChecker {
 public:
  explicit TypeChecker(Nonnull<Arena*> arena, bool trace)
      : arena_(arena), interpreter_(arena, trace), trace_(trace) {}

  void TypeCheck(AST& ast);

 private:
  using TypeEnv = Dictionary<std::string, Nonnull<const Value*>>;

  struct TypeCheckContext {
    explicit TypeCheckContext(Nonnull<Arena*> arena)
        : types(arena), values(arena) {}

    // Symbol table mapping names of runtime entities to their type.
    TypeEnv types;
    // Symbol table mapping names of compile time entities to their value.
    Env values;
  };

  struct TCResult {
    explicit TCResult(TypeEnv types) : types(types) {}

    TypeEnv types;
  };

  static void PrintTypeEnv(TypeEnv types, llvm::raw_ostream& out);

  // Perform type argument deduction, matching the parameter type `param`
  // against the argument type `arg`. Whenever there is an VariableType
  // in the parameter type, it is deduced to be the corresponding type
  // inside the argument type.
  // The `deduced` parameter is an accumulator, that is, it holds the
  // results so-far.
  static auto ArgumentDeduction(SourceLocation source_loc, TypeEnv deduced,
                                Nonnull<const Value*> param,
                                Nonnull<const Value*> arg) -> TypeEnv;

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `types` maps variable names to the type of their run-time value.
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterExp.
  auto TypeCheckExp(Nonnull<Expression*> e, TypeEnv types, Env values)
      -> TCResult;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `p`.
  //
  // `expected` is the type that this pattern is expected to have, if the
  // surrounding context gives us that information. Otherwise, it is
  // nullopt.
  auto TypeCheckPattern(Nonnull<Pattern*> p, TypeEnv types, Env values,
                        std::optional<Nonnull<const Value*>> expected)
      -> TCResult;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `d`.
  void TypeCheckDeclaration(Nonnull<Declaration*> d, const TypeEnv& types,
                            const Env& values);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `s`.
  //
  // REQUIRES: f.return_term().has_static_type() || f.return_term().is_auto(),
  // where `f` is nearest enclosing FunctionDeclaration of `s`.
  auto TypeCheckStmt(Nonnull<Statement*> s, TypeEnv types, Env values)
      -> TCResult;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `f`,
  // and may not traverse f->body() if `check_body` is false.
  auto TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                    TypeEnv types, Env values, bool check_body)
      -> TCResult;

  auto TypeCheckCase(Nonnull<const Value*> expected, Nonnull<Pattern*> pat,
                     Nonnull<Statement*> body, TypeEnv types, Env values)
      -> Match::Clause;

  auto TypeOfClassDecl(ClassDeclaration& class_decl, TypeEnv /*types*/,
                       Env ct_top) -> Nonnull<const Value*>;

  auto TopLevel(std::vector<Nonnull<Declaration*>>* fs) -> TypeCheckContext;
  void TopLevel(Nonnull<Declaration*> d, TypeCheckContext* tops);

  // Verifies that opt_stmt holds a statement, and it is structurally impossible
  // for control flow to leave that statement except via a `return`.
  void ExpectReturnOnAllPaths(std::optional<Nonnull<Statement*>> opt_stmt,
                              SourceLocation source_loc);

  // Verifies that *value represents a concrete type, as opposed to a
  // type pattern or a non-type value.
  void ExpectIsConcreteType(SourceLocation source_loc,
                            Nonnull<const Value*> value);

  auto Substitute(TypeEnv dict, Nonnull<const Value*> type)
      -> Nonnull<const Value*>;

  Nonnull<Arena*> arena_;
  Interpreter interpreter_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
