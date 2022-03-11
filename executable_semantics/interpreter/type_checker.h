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
#include "executable_semantics/interpreter/impl_scope.h"
#include "executable_semantics/interpreter/interpreter.h"

namespace Carbon {

class TypeChecker {
 public:
  explicit TypeChecker(Nonnull<Arena*> arena, bool trace)
      : arena_(arena), trace_(trace) {}

  // Type-checks `ast` and sets properties such as `static_type`, as documented
  // on the individual nodes.
  // On failure, `ast` is left in a partial state and should not be further
  // processed.
  auto TypeCheck(AST& ast) -> llvm::Error;

 private:
  // Perform type argument deduction, matching the parameter type `param`
  // against the argument type `arg`. Whenever there is an VariableType
  // in the parameter type, it is deduced to be the corresponding type
  // inside the argument type.
  // The `deduced` parameter is an accumulator, that is, it holds the
  // results so-far.
  static auto ArgumentDeduction(SourceLocation source_loc, BindingMap& deduced,
                                Nonnull<const Value*> param,
                                Nonnull<const Value*> arg) -> llvm::Error;

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterExp.
  auto TypeCheckExp(Nonnull<Expression*> e, const ImplScope& impl_scope)
      -> llvm::Error;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `p`.
  //
  // `expected` is the type that this pattern is expected to have, if the
  // surrounding context gives us that information. Otherwise, it is
  // nullopt.
  auto TypeCheckPattern(Nonnull<Pattern*> p,
                        std::optional<Nonnull<const Value*>> expected,
                        const ImplScope& impl_scope) -> llvm::Error;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `s`.
  //
  // REQUIRES: f.return_term().has_static_type() || f.return_term().is_auto(),
  // where `f` is nearest enclosing FunctionDeclaration of `s`.
  auto TypeCheckStmt(Nonnull<Statement*> s, const ImplScope& impl_scope)
      -> llvm::Error;

  // Establish the `static_type` and `constant_value` of the
  // declaration and all of its nested declarations. This involves the
  // compile-time interpretation of any type expressions in the
  // declaration. It does not involve type checking statements and
  // (runtime) expressions, as in the body of a function or a method.
  // Dispatches to one of the following functions.
  auto DeclareDeclaration(Nonnull<Declaration*> d, ImplScope& enclosing_scope)
      -> llvm::Error;

  auto DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                  const ImplScope& enclosing_scope)
      -> llvm::Error;

  auto DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                               ImplScope& enclosing_scope) -> llvm::Error;

  auto DeclareInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                   ImplScope& enclosing_scope) -> llvm::Error;

  auto DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                              ImplScope& enclosing_scope) -> llvm::Error;

  auto DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                const ImplScope& enclosing_scope)
      -> llvm::Error;

  // Checks the statements and (runtime) expressions within the
  // declaration, such as the body of a function.
  // Dispatches to one of the following functions.
  // Assumes that DeclareDeclaration has already been invoked on `d`.
  auto TypeCheckDeclaration(Nonnull<Declaration*> d,
                            const ImplScope& impl_scope) -> llvm::Error;

  // Type check the body of the function.
  auto TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                    const ImplScope& impl_scope) -> llvm::Error;

  // Type check all the members of the class.
  auto TypeCheckClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                 const ImplScope& impl_scope) -> llvm::Error;

  // Type check all the members of the interface.
  auto TypeCheckInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                     const ImplScope& impl_scope)
      -> llvm::Error;

  // Type check all the members of the implementation.
  auto TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                const ImplScope& impl_scope) -> llvm::Error;

  // This currently does nothing, but perhaps that will change in the future.
  auto TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                  const ImplScope& impl_scope) -> llvm::Error;

  // Verifies that opt_stmt holds a statement, and it is structurally impossible
  // for control flow to leave that statement except via a `return`.
  auto ExpectReturnOnAllPaths(std::optional<Nonnull<Statement*>> opt_stmt,
                              SourceLocation source_loc) -> llvm::Error;

  // Verifies that *value represents a concrete type, as opposed to a
  // type pattern or a non-type value.
  auto ExpectIsConcreteType(SourceLocation source_loc,
                            Nonnull<const Value*> value) -> llvm::Error;

  auto Substitute(const std::map<Nonnull<const GenericBinding*>,
                                 Nonnull<const Value*>>& dict,
                  Nonnull<const Value*> type) -> Nonnull<const Value*>;

  // Sets named_entity.constant_value() to `value`. Can be called multiple
  // times on the same named_entity, so long as it is always called with
  // the same value.
  template <typename T>
  void SetConstantValue(Nonnull<T*> named_entity, Nonnull<const Value*> value);

  void PrintConstants(llvm::raw_ostream& out);

  Nonnull<Arena*> arena_;
  std::set<ValueNodeView> constants_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_TYPE_CHECKER_H_
