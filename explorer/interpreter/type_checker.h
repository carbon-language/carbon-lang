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
#include "explorer/interpreter/value.h"

namespace Carbon {

using CollectedMembersMap =
    std::unordered_map<std::string_view, Nonnull<const Declaration*>>;

using GlobalMembersMap =
    std::unordered_map<Nonnull<const Declaration*>, CollectedMembersMap>;

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

  // Construct a type that is the same as `type` except that occurrences
  // of type variables (aka. `GenericBinding` and references to `ImplBinding`)
  // are replaced by their corresponding type or witness in `dict`.
  auto Substitute(const Bindings& bindings, Nonnull<const Value*> type) const
      -> Nonnull<const Value*>;

  // Attempts to refine a witness that might be symbolic into an impl witness,
  // using `impl` declarations that have been declared and type-checked so far.
  // If a more precise witness cannot be found, returns `witness`.
  auto RefineWitness(Nonnull<const Witness*> witness,
                     Nonnull<const Value*> type,
                     Nonnull<const Value*> constraint) const
      -> Nonnull<const Witness*>;

  // If `impl` can be an implementation of interface `iface` for the given
  // `type`, then return the witness for this `impl`. Otherwise return
  // std::nullopt.
  auto MatchImpl(const InterfaceType& iface, Nonnull<const Value*> type,
                 const ImplScope::Impl& impl, const ImplScope& impl_scope,
                 SourceLocation source_loc) const
      -> std::optional<Nonnull<const Witness*>>;

  // Return the declaration of the member with the given name, from the class
  // and its parents
  auto FindMemberWithParents(std::string_view name,
                             Nonnull<const NominalClassType*> enclosing_type)
      -> ErrorOr<std::optional<
          std::pair<Nonnull<const Value*>, Nonnull<const Declaration*>>>>;

  // Finds the direct or indirect member of a class or mixin by its name and
  // returns the member's declaration and type. Indirect members are members of
  // mixins that are mixed by member mix declarations. If the member is an
  // indirect member from a mix declaration, then the Self type variable within
  // the member's type is substituted with the type of the enclosing declaration
  // containing the mix declaration.
  auto FindMixedMemberAndType(const std::string_view& name,
                              llvm::ArrayRef<Nonnull<Declaration*>> members,
                              Nonnull<const Value*> enclosing_type)
      -> ErrorOr<std::optional<
          std::pair<Nonnull<const Value*>, Nonnull<const Declaration*>>>>;

  // Given the witnesses for the components of a constraint, form a witness for
  // the constraint.
  auto MakeConstraintWitness(
      std::vector<Nonnull<const Witness*>> impl_constraint_witnesses) const
      -> Nonnull<const Witness*>;

  // Given the witnesses for the components of a constraint, form a witness for
  // the constraint.
  auto MakeConstraintWitnessAccess(Nonnull<const Witness*> witness,
                                   int impl_offset) const
      -> Nonnull<const Witness*>;

 private:
  class ConstraintTypeBuilder;
  class SubstitutedGenericBindings;
  class ArgumentDeduction;

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

  // Result from a lookup in a constraint.
  struct ConstraintLookupResult {
    Nonnull<const InterfaceType*> interface;
    Nonnull<const Declaration*> member;
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

  // Type checks a generic binding. `symbolic_value` is the symbolic name by
  // which this generic binding is known in its scope. `impl_scope` is updated
  // with the impl implied by the binding, if any.
  auto TypeCheckGenericBinding(GenericBinding& binding,
                               std::string_view context, ImplScope& impl_scope)
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
      const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Establish the `static_type` and `constant_value` of the
  // declaration and all of its nested declarations. This involves the
  // compile-time interpretation of any type expressions in the
  // declaration. It does not involve type checking statements and
  // (runtime) expressions, as in the body of a function or a method.
  // Dispatches to one of the following functions.
  auto DeclareDeclaration(Nonnull<Declaration*> d, const ScopeInfo& scope_info)
      -> ErrorOr<Success>;

  auto DeclareCallableDeclaration(Nonnull<CallableDeclaration*> f,
                                  const ScopeInfo& scope_info)
      -> ErrorOr<Success>;

  auto DeclareClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                               const ScopeInfo& scope_info) -> ErrorOr<Success>;

  auto DeclareMixinDeclaration(Nonnull<MixinDeclaration*> mixin_decl,
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
                           Nonnull<const Witness*> self_witness,
                           Nonnull<const Witness*> iface_witness,
                           const ImplScope& impl_scope) -> ErrorOr<Success>;

  // Check that an `impl` declaration satisfies its constraints and add the
  // corresponding `ImplBinding`s to the impl scope.
  auto CheckAndAddImplBindings(
      Nonnull<const ImplDeclaration*> impl_decl,
      Nonnull<const Value*> impl_type, Nonnull<const Witness*> self_witness,
      Nonnull<const Witness*> impl_witness,
      llvm::ArrayRef<Nonnull<const GenericBinding*>> deduced_bindings,
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

  // Create a witness for the given `impl` binding.
  auto CreateImplBindingWitness(Nonnull<const ImplBinding*> impl_binding)
      -> Nonnull<const Witness*>;

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
  auto TypeCheckDeclaration(
      Nonnull<Declaration*> d, const ImplScope& impl_scope,
      std::optional<Nonnull<const Declaration*>> enclosing_decl)
      -> ErrorOr<Success>;

  // Type check the body of the function.
  auto TypeCheckCallableDeclaration(Nonnull<CallableDeclaration*> f,
                                    const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Type check all the members of the class.
  auto TypeCheckClassDeclaration(Nonnull<ClassDeclaration*> class_decl,
                                 const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Type check all the members of the mixin.
  auto TypeCheckMixinDeclaration(Nonnull<const MixinDeclaration*> mixin_decl,
                                 const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  auto TypeCheckMixDeclaration(
      Nonnull<MixDeclaration*> mix_decl, const ImplScope& impl_scope,
      std::optional<Nonnull<const Declaration*>> enclosing_decl)
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
                                const ImplScope& enclosing_scope)
      -> ErrorOr<Success>;

  // This currently does nothing, but perhaps that will change in the future.
  auto TypeCheckChoiceDeclaration(Nonnull<ChoiceDeclaration*> choice,
                                  const ImplScope& impl_scope)
      -> ErrorOr<Success>;

  // Verifies that opt_stmt holds a statement, and it is structurally impossible
  // for control flow to leave that statement except via a `return`.
  auto ExpectReturnOnAllPaths(std::optional<Nonnull<Statement*>> opt_stmt,
                              SourceLocation source_loc) -> ErrorOr<Success>;

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
  auto ImplicitlyConvert(std::string_view context, const ImplScope& impl_scope,
                         Nonnull<Expression*> source,
                         Nonnull<const Value*> destination)
      -> ErrorOr<Nonnull<Expression*>>;

  // Checks that the given type is not a placeholder type. Diagnoses otherwise.
  auto ExpectNonPlaceholderType(SourceLocation source_loc,
                                Nonnull<const Value*> type) -> ErrorOr<Success>;

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
  auto ExpectType(SourceLocation source_loc, std::string_view context,
                  Nonnull<const Value*> expected, Nonnull<const Value*> actual,
                  const ImplScope& impl_scope) const -> ErrorOr<Success>;

  // Check whether `actual` is the same type as `expected` and halt with a
  // fatal compilation error if it is not.
  auto ExpectExactType(SourceLocation source_loc, std::string_view context,
                       Nonnull<const Value*> expected,
                       Nonnull<const Value*> actual,
                       const ImplScope& impl_scope) const -> ErrorOr<Success>;

  // Rebuild a value in the current type-checking context. Applies any rewrites
  // that are in scope and attempts to resolve associated constants using impls
  // that have been declared since the value was formed.
  auto RebuildValue(Nonnull<const Value*> value) const -> Nonnull<const Value*>;

  // Implementation of Substitute and RebuildValue. Does not check that
  // bindings are nonempty, nor does it trace its progress.
  auto SubstituteImpl(const Bindings& bindings,
                      Nonnull<const Value*> type) const
      -> Nonnull<const Value*>;

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

  // Given an interface type, form a corresponding constraint type. The
  // interface must be a complete type.
  auto MakeConstraintForInterface(
      SourceLocation source_loc, Nonnull<const InterfaceType*> iface_type) const
      -> ErrorOr<Nonnull<const ConstraintType*>>;

  // Convert a value that is expected to represent a constraint into a
  // `ConstraintType`.
  auto ConvertToConstraintType(SourceLocation source_loc,
                               std::string_view context,
                               Nonnull<const Value*> constraint) const
      -> ErrorOr<Nonnull<const ConstraintType*>>;

  // Given a list of constraint types, form the combined constraint.
  auto CombineConstraints(
      SourceLocation source_loc,
      llvm::ArrayRef<Nonnull<const ConstraintType*>> constraints)
      -> ErrorOr<Nonnull<const ConstraintType*>>;

  // Gets the type for the given associated constant.
  auto GetTypeForAssociatedConstant(
      Nonnull<const AssociatedConstant*> assoc) const -> Nonnull<const Value*>;

  // Look up a member name in a constraint, which might be a single interface or
  // a compound constraint.
  auto LookupInConstraint(SourceLocation source_loc,
                          std::string_view lookup_kind,
                          Nonnull<const Value*> type,
                          std::string_view member_name)
      -> ErrorOr<ConstraintLookupResult>;

  // Given `type.(interface.member)`, look for a rewrite in the declared type
  // of `type`.
  auto LookupRewriteInTypeOf(Nonnull<const Value*> type,
                             Nonnull<const InterfaceType*> interface,
                             Nonnull<const Declaration*> member) const
      -> std::optional<const RewriteConstraint*>;

  // Given a witness value, look for a rewrite for the given associated
  // constant.
  auto LookupRewriteInWitness(Nonnull<const Witness*> witness,
                              Nonnull<const InterfaceType*> interface,
                              Nonnull<const Declaration*> member) const
      -> std::optional<const RewriteConstraint*>;

  // Adds a member of a declaration to collected_members_
  auto CollectMember(Nonnull<const Declaration*> enclosing_decl,
                     Nonnull<const Declaration*> member_decl)
      -> ErrorOr<Success>;

  // Fetches all direct and indirect members of a class or mixin declaration
  // stored within collected_members_
  auto FindCollectedMembers(Nonnull<const Declaration*> decl)
      -> CollectedMembersMap&;

  Nonnull<Arena*> arena_;
  Builtins builtins_;

  // Maps a mixin/class declaration to all of its direct and indirect members.
  GlobalMembersMap collected_members_;

  std::optional<Nonnull<llvm::raw_ostream*>> trace_stream_;

  // The top-level ImplScope, containing `impl` declarations that should be
  // usable from any context. This is used when we want to try to refine a
  // symbolic witness into an impl witness during substitution.
  std::optional<const ImplScope*> top_level_impl_scope_;

  // Constraint types that are currently being resolved. These may have
  // rewrites that are not yet visible in any type.
  std::vector<ConstraintTypeBuilder*> partial_constraint_types_;
};

}  // namespace Carbon

#endif  // CARBON_EXPLORER_INTERPRETER_TYPE_CHECKER_H_
