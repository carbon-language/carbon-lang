// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_

#include <map>
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
  // Perform type argument deduction, matching the parameter type `param`
  // against the argument type `arg`. Whenever there is an VariableType
  // in the parameter type, it is deduced to be the corresponding type
  // inside the argument type.
  // The `deduced` parameter is an accumulator, that is, it holds the
  // results so-far.
  static void ArgumentDeduction(
      SourceLocation source_loc,
      std::map<Nonnull<const GenericBinding*>, Nonnull<const Value*>>& deduced,
      Nonnull<const Value*> param, Nonnull<const Value*> arg);

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterExp.
  void TypeCheckExp(Nonnull<Expression*> e, Env values);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `p`.
  //
  // `expected` is the type that this pattern is expected to have, if the
  // surrounding context gives us that information. Otherwise, it is
  // nullopt.
  void TypeCheckPattern(Nonnull<Pattern*> p, Env values,
                        std::optional<Nonnull<const Value*>> expected);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `d`.
  void TypeCheckDeclaration(Nonnull<Declaration*> d, const Env& values);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `s`.
  //
  // REQUIRES: f.return_term().has_static_type() || f.return_term().is_auto(),
  // where `f` is nearest enclosing FunctionDeclaration of `s`.
  void TypeCheckStmt(Nonnull<Statement*> s, Env values);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `f`,
  // and may not traverse f->body() if `check_body` is false.
  void TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f, Env values,
                                    bool check_body);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at class_decl.
  void TypeCheckClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                 Env ct_top);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at choice_decl.
  void TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                  Env ct_top);

  void TopLevel(Nonnull<Declaration*> d, Nonnull<Env*> values);

  // Verifies that opt_stmt holds a statement, and it is structurally impossible
  // for control flow to leave that statement except via a `return`.
  void ExpectReturnOnAllPaths(std::optional<Nonnull<Statement*>> opt_stmt,
                              SourceLocation source_loc);

  // Verifies that *value represents a concrete type, as opposed to a
  // type pattern or a non-type value.
  void ExpectIsConcreteType(SourceLocation source_loc,
                            Nonnull<const Value*> value);

  auto Substitute(const std::map<Nonnull<const GenericBinding*>,
                                 Nonnull<const Value*>>& dict,
                  Nonnull<const Value*> type) -> Nonnull<const Value*>;

  Nonnull<Arena*> arena_;
  Interpreter interpreter_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
