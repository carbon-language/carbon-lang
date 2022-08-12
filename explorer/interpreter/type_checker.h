// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_H_
#define CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_H_

#include <set>

#include "common/ostream.h"
#include "explorer/ast/ast.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/statement.h"
#include "explorer/common/nonnull.h"
#include "explorer/interpreter/builtins.h"
#include "explorer/interpreter/dictionary.h"
#include "explorer/interpreter/impl_scope.h"
#include "explorer/interpreter/interpreter.h"

namespace Carbon {

class TypeChecker {
 public:
  explicit TypeChecker(Nonnull<Arena*> arena,
                       std::optional<Nonnull<llvm::raw_ostream*>> trace_stream)
      : arena_(arena), trace_stream_(trace_stream) {}

  // Type-checks `ast` and sets properties such as `static_type`, as documented
  // on the individual nodes.
  // On failure, `ast` is left in a partial state and should not be further
  // processed.
  auto TypeCheck(AST& ast) -> ErrorOr<Success>;

  // Perform type argument deduction, matching the parameter value `param`
  // against the argument value `arg`. Whenever there is an VariableType in the
  // parameter, it is deduced to be the corresponding type inside the argument
  // type. The argument and parameter will typically be types, but can be
  // non-type values when deduction recurses into the arguments of a
  // parameterized type.
  // The `deduced` parameter is an accumulator, that is, it holds the
  // results so-far.
  // `allow_implicit_conversion` specifies whether implicit conversions are
  // permitted from the argument to the parameter type. If so, an `impl_scope`
  // must be provided.
  auto ArgumentDeduction(
      SourceLocation source_loc, const std::string& context,
      llvm::ArrayRef<Nonnull<const GenericBinding*>> bindings_to_deduce,
      BindingMap& deduced, Nonnull<const Value*> param,
      Nonnull<const Value*> arg, bool allow_implicit_conversion,
      const ImplScope& impl_scope) const -> ErrorOr<Success>;

  // Construct a type that is the same as `type` except that occurrences
  // of type variables (aka. `GenericBinding`) are replaced by their
  // corresponding type in `dict`.
  auto Substitute(const std::map<Nonnull<const GenericBinding*>,
                                 Nonnull<const Value*>>& dict,
                  Nonnull<const Value*> type) const -> Nonnull<const Value*>;

  // If `impl` can be an implementation of interface `iface` for the
  // given `type`, then return an expression that will produce the witness
  // for this `impl` (at runtime). Otherwise return std::nullopt.
  auto MatchImpl(const InterfaceType& iface, Nonnull<const Value*> type,
                 const ImplScope::Impl& impl, const ImplScope& impl_scope,
                 SourceLocation source_loc) const
      -> std::optional<Nonnull<Expression*>>;

  // Given the witnesses for the components of a constraint, form a witness for
  // the constraint.
  auto MakeConstraintWitness(
      const ConstraintType& constraint,
      std::vector<Nonnull<Expression*>> impl_constraint_witnesses,
      SourceLocation source_loc) const -> Nonnull<Expression*>;

  // Given the witnesses for the components of a constraint, form a witness for
  // the constraint.
  auto MakeConstraintWitnessAccess(Nonnull<Expression*> witness,
                                   size_t impl_offset) const
      -> Nonnull<Expression*>;

 private:
  struct SingleStepEqualityContext;

  // Information about the currently enclosing scopes.
  struct ScopeInfo {
    static auto ForNonClassScope(Nonnull<ImplScope*> impl_scope) -> ScopeInfo {
      return {.innermost_scope = impl_scope,
              .innermost_non_class_scope = impl_scope,
              .bindings = {}};
    }

    static auto ForClassScope(
        const ScopeInfo& outer, Nonnull<ImplScope*> class_impl_scope,
        std::vector<Nonnull<const GenericBinding*>> class_bindings)
        -> ScopeInfo {
      return {.innermost_scope = class_impl_scope,
              .innermost_non_class_scope = outer.innermost_non_class_scope,
              .bindings = std::move(class_bindings)};
    }

    // The innermost enclosing impl scope, within which impls should be looked
    // up.
    Nonnull<ImplScope*> innermost_scope;
    // The innermost enclosing non-class impl scope, where impl declarations
    // should introduce new impls.
    Nonnull<ImplScope*> innermost_non_class_scope;
    // The enclosing generic bindings, if any.
    std::vector<Nonnull<const GenericBinding*>> bindings;
  };

  // Traverses the AST rooted at `e`, populating the static_type() of all nodes
  // and ensuring they follow Carbon's typing rules.
  //
  // `values` maps variable names to their compile-time values. It is not
  //    directly used in this function but is passed to InterpExp.
  auto TypeCheckExp(Nonnull<Expression*> e, const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Type checks and interprets `type_expression`, and validates it represents a
  // [concrete] type.
  auto TypeCheckTypeExp(Nonnull<Expression*> type_expression,
                        const ImplScope& impl_scope, bool concrete = true)
      -> ErrorOr<Nonnull<const Value*>>;

  // Type checks and interprets `clause`, and validates it represents a valid
  // `where` clause.
  auto TypeCheckWhereClause(Nonnull<WhereClause*> clause,
                            const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `p`.
  //
  // `expected` is the type that this pattern is expected to have, if the
  // surrounding context gives us that information. Otherwise, it is nullopt.
  // Implicit conversions from `expected` to the pattern's type are permitted.
  //
  // `impl_scope` is extended with all impls implied by the pattern.
  auto TypeCheckPattern(Nonnull<Pattern*> p,
                        std::optional<Nonnull<const Value*>> expected,
                        ImplScope& impl_scope,
                        ValueCategory enclosing_value_category)
      -> ErrorOr<Success>;

  // Equivalent to TypeCheckExp, but operates on the AST rooted at `s`.
  //
  // REQUIRES: f.return_term().has_static_type() || f.return_term().is_auto(),
  // where `f` is nearest enclosing FunctionDeclaration of `s`.
  auto TypeCheckStmt(Nonnull<Statement*> s, const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Perform deduction for the deduced bindings in a function call, and set its
  // lists of generic bindings and impls.
  //
  // -   `params` is the list of parameters.
  // -   `generic_params` indicates which parameters are generic parameters,
  //     which require a constant argument.
  // -   `deduced_bindings` is a list of the bindings that are expected to be
  //     deduced by the call.
  // -   `impl_bindings` is a list of the impl bindings that are expected to be
  //     satisfied by the call.
  auto DeduceCallBindings(
      CallExpression& call, Nonnull<const Value*> params,
      llvm::ArrayRef<FunctionType::GenericParameter> generic_params,
      llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
      llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
      const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Establish the `static_type` and `constant_value` of the
  // declaration and all of its nested declarations. This involves the
  // compile-time interpretation of any type expressions in the
  // declaration. It does not involve type checking statements and
  // (runtime) expressions, as in the body of a function or a method.
  // Dispatches to one of the following functions.
  auto DeclareDeclaration(Nonnull<Declaration*> d, const ScopeInfo& scope_info)
      -> ErrorOr<Success>;

  auto DeclareFunctionDeclaration(Nonnull<FunctionDeclaration*> f,
                                  const ScopeInfo& scope_info)
      -> ErrorOr<Success>;

  auto DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                               const ScopeInfo& scope_info) -> ErrorOr<Success>;

  auto DeclareInterfaceDeclaration(Nonnull<InterfaceDeclaration*> iface_decl,
                                   const ScopeInfo& scope_info)
      -> ErrorOr<Success>;

  // Check that the deduced parameters of an impl are actually deducible from
  // the form of the interface, for a declaration of the form
  // `impl forall [deduced_bindings] impl_type as impl_iface`.
  auto CheckImplIsDeducible(
      SourceLocation source_loc, Nonnull<const Value*> impl_type,
      Nonnull<const InterfaceType*> impl_iface,
      llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
      const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Check that each required declaration in an implementation of the given
  // interface is present in the given `impl`.
  auto CheckImplIsComplete(Nonnull<const InterfaceType*> iface_type,
                           Nonnull<const ImplDeclaration*> impl_decl,
                           Nonnull<const Value*> self_type,
                           const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Check that an `impl` declaration satisfies its constraints and add the
  // corresponding `ImplBinding`s to the impl scope.
  auto CheckAndAddImplBindings(Nonnull<const ImplDeclaration*> impl_decl,
                               Nonnull<const Value*> impl_type,
                               const ScopeInfo& scope_info) -> ErrorOr<Success>;

  auto DeclareImplDeclaration(Nonnull<ImplDeclaration*> impl_decl,
                              const ScopeInfo& scope_info) -> ErrorOr<Success>;

  auto DeclareChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                const ScopeInfo& scope_info)
      -> ErrorOr<Success>;
  auto DeclareAliasDeclaration(Nonnull<AliasDeclaration*> alias,
                               const ScopeInfo& scope_info) -> ErrorOr<Success>;

  // Find all of the GenericBindings in the given pattern.
  void CollectGenericBindingsInPattern(
      Nonnull<const Pattern*> p,
      std::vector<Nonnull<const GenericBinding*>>& generic_bindings);

  // Find all of the ImplBindings in the given pattern. The pattern is required
  // to have already been type-checked.
  void CollectImplBindingsInPattern(
      Nonnull<const Pattern*> p,
      std::vector<Nonnull<const ImplBinding*>>& impl_bindings);

  // Add the impls from the pattern into the given `impl_scope`.
  void BringPatternImplsIntoScope(Nonnull<const Pattern*> p,
                                  ImplScope& impl_scope);

  // Create a reference to the given `impl` binding.
  auto CreateImplReference(Nonnull<const ImplBinding*> impl_binding)
      -> Nonnull<Expression*>;

  // Add the given ImplBinding to the given `impl_scope`.
  void BringImplIntoScope(Nonnull<const ImplBinding*> impl_binding,
                          ImplScope& impl_scope);

  // Add all of the `impl_bindings` into the `scope`.
  void BringImplsIntoScope(
      llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
      ImplScope& scope);

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

  // Bring the associated constants in `constraint` that constrain the
  // implementation of `interface` for `self` into `scope`.
  void BringAssociatedConstantsIntoScope(
      Nonnull<const ConstraintType*> constraint, Nonnull<const Value*> self,
      Nonnull<const InterfaceType*> interface, ImplScope& scope);

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

  // Verifies that *value represents the result of a type expression,
  // as opposed to a non-type value.
  auto ExpectIsType(SourceLocation source_loc, Nonnull<const Value*> value)
      -> ErrorOr<Success>;

  // Verifies that *value represents a concrete type, as opposed to a
  // type pattern or a non-type value.
  auto ExpectIsConcreteType(SourceLocation source_loc,
                            Nonnull<const Value*> value) -> ErrorOr<Success>;

  // Returns the field names of the class together with their types.
  auto FieldTypes(const NominalClassType& class_type) const
      -> std::vector<NamedValue>;

  // Returns true if source_fields and destination_fields contain the same set
  // of names, and each value in source_fields is implicitly convertible to
  // the corresponding value in destination_fields. All values in both arguments
  // must be types.
  auto FieldTypesImplicitlyConvertible(
      llvm::ArrayRef<NamedValue> source_fields,
      llvm::ArrayRef<NamedValue> destination_fields,
      const ImplScope& impl_scope) const -> bool;

  // Returns true if *source is implicitly convertible to *destination. *source
  // and *destination must be concrete types.
  //
  // If allow_user_defined_conversions, conversions requiring a user-defined
  // `ImplicitAs` implementation are not considered, and only builtin
  // conversions will be allowed.
  auto IsImplicitlyConvertible(Nonnull<const Value*> source,
                               Nonnull<const Value*> destination,
                               const ImplScope& impl_scope,
                               bool allow_user_defined_conversions) const
      -> bool;

  // Attempt to implicitly convert type-checked expression `source` to the type
  // `destination`.
  auto ImplicitlyConvert(const std::string& context,
                         const ImplScope& impl_scope,
                         Nonnull<Expression*> source,
                         Nonnull<const Value*> destination)
      -> ErrorOr<Nonnull<Expression*>>;

  // Determine whether `type1` and `type2` are considered to be the same type
  // in the given scope. This is true if they're structurally identical or if
  // there is an equality relation in scope that specifies that they are the
  // same.
  auto IsSameType(Nonnull<const Value*> type1, Nonnull<const Value*> type2,
                  const ImplScope& impl_scope) const -> bool;

  // Check whether `actual` is implicitly convertible to `expected`
  // and halt with a fatal compilation error if it is not.
  //
  // TODO: Does not actually perform the conversion if a user-defined
  // conversion is needed. Should be used very rarely for that reason.
  auto ExpectType(SourceLocation source_loc, const std::string& context,
                  Nonnull<const Value*> expected, Nonnull<const Value*> actual,
                  const ImplScope& impl_scope) const -> ErrorOr<Success>;

  // Check whether `actual` is the same type as `expected` and halt with a
  // fatal compilation error if it is not.
  auto ExpectExactType(SourceLocation source_loc, const std::string& context,
                       Nonnull<const Value*> expected,
                       Nonnull<const Value*> actual,
                       const ImplScope& impl_scope) const -> ErrorOr<Success>;

  // The name of a builtin interface, with any arguments.
  struct BuiltinInterfaceName {
    Builtins::Builtin builtin;
    llvm::ArrayRef<Nonnull<const Value*>> arguments = {};
  };
  // The name of a method on a builtin interface, with any arguments.
  struct BuiltinMethodCall {
    const std::string& name;
    llvm::ArrayRef<Nonnull<Expression*>> arguments = {};
  };

  // Form a builtin method call. Ensures that the type of `source` implements
  // the interface `interface`, which should be defined in the prelude, and
  // forms a call to the method `method` on that interface.
  auto BuildBuiltinMethodCall(const ImplScope& impl_scope,
                              Nonnull<Expression*> source,
                              BuiltinInterfaceName interface,
                              BuiltinMethodCall method)
      -> ErrorOr<Nonnull<Expression*>>;

  // Get a type for a builtin interface.
  auto GetBuiltinInterfaceType(SourceLocation source_loc,
                               BuiltinInterfaceName interface) const
      -> ErrorOr<Nonnull<const InterfaceType*>>;

  // Find impls that satisfy all of the `impl_bindings`, but with the
  // type variables in the `impl_bindings` replaced by the argument
  // type in `deduced_type_args`.  The results are placed in the
  // `impls` map.
  auto SatisfyImpls(llvm::ArrayRef<Nonnull<const ImplBinding*>> impl_bindings,
                    const ImplScope& impl_scope, SourceLocation source_loc,
                    const BindingMap& deduced_type_args,
                    ImplExpMap& impls) const -> ErrorOr<Success>;

  // Given an interface type, form a corresponding constraint type.
  auto MakeConstraintForInterface(SourceLocation source_loc,
                                  Nonnull<const InterfaceType*> iface_type)
      -> Nonnull<const ConstraintType*>;

  // Given a list of constraint types, form the combined constraint.
  auto CombineConstraints(
      SourceLocation source_loc,
      llvm::ArrayRef<Nonnull<const ConstraintType*>> constraints)
      -> Nonnull<const ConstraintType*>;

  // Sets value_node.constant_value() to `value`. Can be called multiple
  // times on the same value_node, so long as it is always called with
  // the same value.
  template <typename T>
  void SetConstantValue(Nonnull<T*> value_node, Nonnull<const Value*> value);

  void PrintConstants(llvm::raw_ostream& out);

  Nonnull<Arena*> arena_;
  std::set<ValueNodeView> constants_;
  Builtins builtins_;

  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_H_
