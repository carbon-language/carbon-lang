// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPLORER_INTERPRETER_TYPE_CHECKER_H_
#define EXPLORER_INTERPRETER_TYPE_CHECKER_H_

#include <set>

#include "common/ostream.h"
#include "explorer/ast/ast.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/statement.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/dictionary.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/interpreter.h"

namespace Carbon {

class TypeChecker {
 public:
  explicit TypeChecker(Nonnull<Arena*> arena, bool trace)
      : arena_(arena), trace_(trace) {}

  // Type-checks `ast` and sets properties such as `static_type`, as documented
  // on the individual nodes.
  // On failure, `ast` is left in a partial state and should not be further
  // processed.
  auto TypeCheck(AST& ast) -> ErrorOr<Success>;

 private:
  // Perform type argument deduction, matching the parameter type `param`
  // against the argument type `arg`. Whenever there is an VariableType
  // in the parameter type, it is deduced to be the corresponding type
  // inside the argument type.
  // The `deduced` parameter is an accumulator, that is, it holds the
  // results so-far.
  auto ArgumentDeduction(SourceLocation source_loc, BindingMap& deduced,
                         Nonnull<const Value*> param_type,
                         Nonnull<const Value*> arg_type) -> ErrorOr<Success>;

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterExp.
  auto TypeCheckExp(Nonnull<Expression*> e, const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `p`.
  //
  // `expected` is the type that this pattern is expected to have, if the
  // surrounding context gives us that information. Otherwise, it is
  // nullopt.
  auto TypeCheckPattern(Nonnull<Pattern*> p,
                        std::optional<Nonnull<const Value*>> expected,
                        const ImplScope& impl_scope,
                        ValueCategory enclosing_value_category)
      -> ErrorOr<Success>;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `s`.
  //
  // REQUIRES: f.return_term().has_static_type() || f.return_term().is_auto(),
  // where `f` is nearest enclosing FunctionDeclaration of `s`.
  auto TypeCheckStmt(Nonnull<Statement*> s, const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Establish the `static_type` and `constant_value` of the
  // declaration and all of its nested declarations. This involves the
  // compile-time interpretation of any type expressions in the
  // declaration. It does not involve type checking statements and
  // (runtime) expressions, as in the body of a function or a method.
  // Dispatches to one of the following functions.
  auto DeclareDeclaration(Nonnull<Declaration*> d, ImplScope& enclosing_scope)
      -> ErrorOr<Success>;

  auto DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                  const ImplScope& enclosing_scope)
      -> ErrorOr<Success>;

  auto DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                               ImplScope& enclosing_scope) -> ErrorOr<Success>;

  auto DeclareInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                   ImplScope& enclosing_scope)
      -> ErrorOr<Success>;

  auto DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                              ImplScope& enclosing_scope) -> ErrorOr<Success>;

  auto DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                const ImplScope& enclosing_scope)
      -> ErrorOr<Success>;

  // Add the impls from the pattern into the given `impl_scope`.
  void AddPatternImpls(Nonnull<Pattern*> p, ImplScope& impl_scope);

  // Checks the statements and (runtime) expressions within the
  // declaration, such as the body of a function.
  // Dispatches to one of the following functions.
  // Assumes that DeclareDeclaration has already been invoked on `d`.
  auto TypeCheckDeclaration(Nonnull<Declaration*> d,
                            const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Type check the body of the function.
  auto TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                    const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Type check all the members of the class.
  auto TypeCheckClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                 const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Type check all the members of the interface.
  auto TypeCheckInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                     const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Type check all the members of the implementation.
  auto TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // This currently does nothing, but perhaps that will change in the future.
  auto TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                  const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Verifies that opt_stmt holds a statement, and it is structurally impossible
  // for control flow to leave that statement except via a `return`.
  auto ExpectReturnOnAllPaths(std::optional<Nonnull<Statement*>> opt_stmt,
                              SourceLocation source_loc) -> ErrorOr<Success>;

  // Verifies that *value represents a concrete type, as opposed to a
  // type pattern or a non-type value.
  auto ExpectIsConcreteType(SourceLocation source_loc,
                            Nonnull<const Value*> value) -> ErrorOr<Success>;

  // Returns the field names of the class together with their types.
  auto FieldTypes(const NominalClassType& class_type)
      -> std::vector<NamedValue>;

  // Returns true if source_fields and destination_fields contain the same set
  // of names, and each value in source_fields is implicitly convertible to
  // the corresponding value in destination_fields. All values in both arguments
  // must be types.
  auto FieldTypesImplicitlyConvertible(
      llvm::ArrayRef<NamedValue> source_fields,
      llvm::ArrayRef<NamedValue> destination_fields);

  // Returns true if *source is implicitly convertible to *destination. *source
  // and *destination must be concrete types.
  auto IsImplicitlyConvertible(Nonnull<const Value*> source,
                               Nonnull<const Value*> destination) -> bool;

  // Check whether `actual` is implicitly convertible to `expected`
  // and halt with a fatal compilation error if it is not.
  auto ExpectType(SourceLocation source_loc, const std::string& context,
                  Nonnull<const Value*> expected, Nonnull<const Value*> actual)
      -> ErrorOr<Success>;

  auto Substitute(const std::map<Nonnull<const GenericBinding*>,
                                 Nonnull<const Value*>>& dict,
                  Nonnull<const Value*> type) -> Nonnull<const Value*>;

  // Sets value_node.constant_value() to `value`. Can be called multiple
  // times on the same value_node, so long as it is always called with
  // the same value.
  template <typename T>
  void SetConstantValue(Nonnull<T*> value_node, Nonnull<const Value*> value);

  void PrintConstants(llvm::raw_ostream& out);

  Nonnull<Arena*> arena_;
  std::set<ValueNodeView> constants_;

  bool trace_;
};

}  // namespace Carbon

#endif  // EXPLORER_INTERPRETER_TYPE_CHECKER_H_
