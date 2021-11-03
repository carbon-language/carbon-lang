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

  // Context about the return type, which may be updated during type checking.
  class ReturnTypeContext {
   public:
    // If orig_return_type is auto, deduced_return_type_ will be nullopt;
    // otherwise, it's orig_return_type. is_auto_ is set accordingly.
    ReturnTypeContext(Nonnull<const Value*> orig_return_type, bool is_omitted);

    auto is_auto() const -> bool { return is_auto_; }

    auto deduced_return_type() const -> std::optional<Nonnull<const Value*>> {
      return deduced_return_type_;
    }
    void set_deduced_return_type(Nonnull<const Value*> type) {
      deduced_return_type_ = type;
    }

    auto is_omitted() const -> bool { return is_omitted_; }

   private:
    // Indicates an `auto` return type, as in `fn Foo() -> auto { return 0; }`.
    const bool is_auto_;

    // The actual return type. May be nullopt for an `auto` return type that has
    // yet to be determined.
    std::optional<Nonnull<const Value*>> deduced_return_type_;

    // Indicates the return type was omitted and is implicitly the empty tuple,
    // as in `fn Foo() {}`.
    const bool is_omitted_;
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
  auto TypeCheckExp(Nonnull<Expression*> e, TypeEnv types, Env values)
      -> TCResult;

  // Equivalent to TypeCheckExp, but operates on Patterns instead of
  // Expressions. `expected` is the type that this pattern is expected to have,
  // if the surrounding context gives us that information. Otherwise, it is
  // nullopt.
  auto TypeCheckPattern(Nonnull<Pattern*> p, TypeEnv types, Env values,
                        std::optional<Nonnull<const Value*>> expected)
      -> TCResult;

  void TypeCheckDeclaration(Nonnull<Declaration*> d, const TypeEnv& types,
                            const Env& values);

  // TypeCheckStmt performs semantic analysis on a statement.  It returns a new
  // version of the statement and a new type environment.
  //
  // The ret_type parameter is used for analyzing return statements.  It is the
  // declared return type of the enclosing function definition.  If the return
  // type is "auto", then the return type is inferred from the first return
  // statement.
  auto TypeCheckStmt(Nonnull<Statement*> s, TypeEnv types, Env values,
                     Nonnull<ReturnTypeContext*> return_type_context)
      -> TCResult;

  auto TypeCheckFunDef(FunctionDeclaration* f, TypeEnv types, Env values)
      -> TCResult;

  auto TypeCheckCase(Nonnull<const Value*> expected, Nonnull<Pattern*> pat,
                     Nonnull<Statement*> body, TypeEnv types, Env values,
                     Nonnull<ReturnTypeContext*> return_type_context)
      -> Match::Clause;

  auto TypeOfFunDef(TypeEnv types, Env values, FunctionDeclaration* fun_def)
      -> Nonnull<const Value*>;
  auto TypeOfClassDef(const ClassDefinition* sd, TypeEnv /*types*/, Env ct_top)
      -> Nonnull<const Value*>;

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
