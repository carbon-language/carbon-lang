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

  void TypeCheck(AST& ast);

 private:
  // Perform type argument deduction, matching the parameter type `param`
  // against the argument type `arg`. Whenever there is an VariableType
  // in the parameter type, it is deduced to be the corresponding type
  // inside the argument type.
  // The `deduced` parameter is an accumulator, that is, it holds the
  // results so-far.
  void ArgumentDeduction(SourceLocation source_loc, BindingMap& deduced,
                         Nonnull<const Value*> param_type,
                         Nonnull<const Value*> arg_type);

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterExp.
  void TypeCheckExp(Nonnull<Expression*> e, const ImplScope& impl_scope);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `p`.
  //
  // `expected` is the type that this pattern is expected to have, if the
  // surrounding context gives us that information. Otherwise, it is
  // nullopt.
  void TypeCheckPattern(Nonnull<Pattern*> p,
                        std::optional<Nonnull<const Value*>> expected,
                        const ImplScope& impl_scope);

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `s`.
  //
  // REQUIRES: f.return_term().has_static_type() || f.return_term().is_auto(),
  // where `f` is nearest enclosing FunctionDeclaration of `s`.
  void TypeCheckStmt(Nonnull<Statement*> s, const ImplScope& impl_scope);

  // Establish the `static_type` and `constant_value` of the
  // declaration and all of its nested declarations. This involves the
  // compile-time interpretation of any type expressions in the
  // declaration. It does not involve type checking statements and
  // (runtime) expressions, as in the body of a function or a method.
  // Dispatches to one of the following functions.
  void DeclareDeclaration(Nonnull<Declaration*> d, ImplScope& enclosing_scope);

  void DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                  const ImplScope& enclosing_scope);

  void DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                               ImplScope& enclosing_scope);

  void DeclareInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                   ImplScope& enclosing_scope);

  void DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                              ImplScope& enclosing_scope);

  void DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                ImplScope& enclosing_scope);

  // Add the impls from the pattern into the given `impl_scope`.
  void AddPatternImpls(Nonnull<Pattern*> p, ImplScope& impl_scope);

  // Checks the statements and (runtime) expressions within the
  // declaration, such as the body of a function.
  // Dispatches to one of the following functions.
  // Assumes that DeclareDeclaration has already been invoked on `d`.
  void TypeCheckDeclaration(Nonnull<Declaration*> d, ImplScope& impl_scope);

  // Type check the body of the function.
  void TypeCheckFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                    const ImplScope& impl_scope);

  // Type check all the members of the class.
  void TypeCheckClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                 ImplScope& impl_scope);

  // Type check all the members of the interface.
  void TypeCheckInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                     ImplScope& impl_scope);

  // Type check all the members of the implementation.
  void TypeCheckImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                                ImplScope& impl_scope);

  // This currently does nothing, but perhaps that will change in the future.
  void TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                  ImplScope& impl_scope);

  // Verifies that opt_stmt holds a statement, and it is structurally impossible
  // for control flow to leave that statement except via a `return`.
  void ExpectReturnOnAllPaths(std::optional<Nonnull<Statement*>> opt_stmt,
                              SourceLocation source_loc);

  // Verifies that *value represents a concrete type, as opposed to a
  // type pattern or a non-type value.
  void ExpectIsConcreteType(SourceLocation source_loc,
                            Nonnull<const Value*> value);

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
  void ExpectType(SourceLocation source_loc, const std::string& context,
                  Nonnull<const Value*> expected, Nonnull<const Value*> actual);

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
