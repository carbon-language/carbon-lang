// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "explorer/interpreter/resolve_names.h"

#include <set>

#include "explorer/ast/declaration.h"
#include "explorer/ast/expression.h"
#include "explorer/ast/pattern.h"
#include "explorer/ast/statement.h"
#include "explorer/ast/static_scope.h"
#include "explorer/base/print_as_id.h"
#include "explorer/interpreter/stack_space.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

namespace Carbon {
namespace {

// The name resolver implements a pass that traverses the AST, builds scope
// objects for each scope encountered, and updates all name references to point
// at the value node referenced by the corresponding name.
//
// In scopes where names are only visible below their point of declaration
// (such as block scopes in C++), this is implemented as a single pass,
// recursively calling ResolveNames on the elements of the scope in order. In
// scopes where names are also visible above their point of declaration (such
// as class scopes in C++), this is done in three passes: first calling
// AddExposedNames on each element of the scope to populate a StaticScope, and
// then calling ResolveNames on each element, passing it the already-populated
// StaticScope but skipping member function bodies, and finally calling
// ResolvedNames again on each element, and this time resolving member function
// bodies.
class NameResolver {
 public:
  explicit NameResolver(Nonnull<TraceStream*> trace_stream)
      : trace_stream_(trace_stream) {}
  enum class ResolveFunctionBodies {
    // Do not resolve names in function bodies.
    Skip,
    // Resolve all names. When visiting a declaration with members, resolve
    // names in member function bodies after resolving the names in all member
    // declarations, as if the bodies appeared after all the declarations.
    AfterDeclarations,
    // Resolve names in function bodies immediately. This is appropriate when
    // the declarations of all members of enclosing classes, interfaces, and
    // similar have already been resolved.
    Immediately,
  };

  // Resolve the qualifier of the given declared name to a scope.
  auto ResolveQualifier(DeclaredName name, StaticScope& enclosing_scope,
                        bool allow_undeclared = false)
      -> ErrorOr<Nonnull<StaticScope*>>;

  // Add the given name to enclosing_scope. Returns the scope in which the name
  // was declared.
  auto AddExposedName(DeclaredName name, ValueNodeView value,
                      StaticScope& enclosing_scope, bool allow_qualified_names)
      -> ErrorOr<Nonnull<StaticScope*>>;

  // Add the names exposed by the given AST node to enclosing_scope.
  auto AddExposedNames(const Declaration& declaration,
                       StaticScope& enclosing_scope,
                       bool allow_qualified_names = false) -> ErrorOr<Success>;

  // Resolve all names within the given expression by looking them up in the
  // enclosing scope. The value returned is the value of the expression, if it
  // is an expression within which we can immediately do further name lookup,
  // such as a namespace.
  auto ResolveNames(Expression& expression, const StaticScope& enclosing_scope)
      -> ErrorOr<std::optional<ValueNodeView>>;
  // For RunWithExtraStack.
  auto ResolveNamesImpl(Expression& expression,
                        const StaticScope& enclosing_scope)
      -> ErrorOr<std::optional<ValueNodeView>>;

  // Resolve all names within the given where clause by looking them up in the
  // enclosing scope.
  auto ResolveNames(WhereClause& clause, const StaticScope& enclosing_scope)
      -> ErrorOr<Success>;
  // For RunWithExtraStack.
  auto ResolveNamesImpl(WhereClause& clause, const StaticScope& enclosing_scope)
      -> ErrorOr<Success>;

  // Resolve all names within the given pattern, extending the given scope with
  // any introduced names.
  auto ResolveNames(Pattern& pattern, StaticScope& enclosing_scope)
      -> ErrorOr<Success>;
  // For RunWithExtraStack.
  auto ResolveNamesImpl(Pattern& pattern, StaticScope& enclosing_scope)
      -> ErrorOr<Success>;

  // Resolve all names within the given statement, extending the given scope
  // with any names introduced by declaration statements.
  auto ResolveNames(Statement& statement, StaticScope& enclosing_scope)
      -> ErrorOr<Success>;
  // For RunWithExtraStack.
  auto ResolveNamesImpl(Statement& statement, StaticScope& enclosing_scope)
      -> ErrorOr<Success>;

  // Resolve all names within the given declaration, extending the given scope
  // with the any names introduced by the declaration if they're not already
  // present.
  auto ResolveNames(Declaration& declaration, StaticScope& enclosing_scope,
                    ResolveFunctionBodies bodies) -> ErrorOr<Success>;
  // For RunWithExtraStack.
  auto ResolveNamesImpl(Declaration& declaration, StaticScope& enclosing_scope,
                        ResolveFunctionBodies bodies) -> ErrorOr<Success>;

  auto ResolveMemberNames(llvm::ArrayRef<Nonnull<Declaration*>> members,
                          StaticScope& scope, ResolveFunctionBodies bodies)
      -> ErrorOr<Success>;

 private:
  // Mapping from namespaces to their scopes.
  llvm::DenseMap<const NamespaceDeclaration*, StaticScope> namespace_scopes_;

  // Mapping from declarations to the scope in which they expose a name.
  llvm::DenseMap<const Declaration*, StaticScope*> exposed_name_scopes_;

  Nonnull<TraceStream*> trace_stream_;
};

}  // namespace

auto NameResolver::ResolveQualifier(DeclaredName name,
                                    StaticScope& enclosing_scope,
                                    bool allow_undeclared)
    -> ErrorOr<Nonnull<StaticScope*>> {
  Nonnull<StaticScope*> scope = &enclosing_scope;
  std::optional<ValueNodeView> scope_node;

  for (const auto& [loc, qualifier] : name.qualifiers()) {
    // TODO: If we permit qualified names anywhere other than the top level, we
    // will need to decide whether the first name in the qualifier is looked up
    // only in the innermost enclosing scope or in all enclosing scopes.
    CARBON_ASSIGN_OR_RETURN(
        ValueNodeView node,
        scope->ResolveHere(scope_node, qualifier, loc, allow_undeclared));

    scope_node = node;
    if (const auto* namespace_decl =
            dyn_cast<NamespaceDeclaration>(&node.base())) {
      scope = &namespace_scopes_[namespace_decl];
    } else {
      return ProgramError(name.source_loc())
             << PrintAsID(node.base()) << " cannot be used as a name qualifier";
    }
  }
  return scope;
}

auto NameResolver::AddExposedName(DeclaredName name, ValueNodeView value,
                                  StaticScope& enclosing_scope,
                                  bool allow_qualified_names)
    -> ErrorOr<Nonnull<StaticScope*>> {
  if (name.is_qualified() && !allow_qualified_names) {
    return ProgramError(name.source_loc())
           << "qualified declaration names are not permitted in this context";
  }

  // We are just collecting names at this stage, so nothing is marked as
  // declared yet. Therefore we don't complain if the qualifier contains a
  // known but not declared namespace name.
  CARBON_ASSIGN_OR_RETURN(Nonnull<StaticScope*> scope,
                          ResolveQualifier(name, enclosing_scope,
                                           /*allow_undeclared=*/true));
  CARBON_RETURN_IF_ERROR(scope->Add(
      name.inner_name(), value, StaticScope::NameStatus::KnownButNotDeclared));
  return scope;
}

auto NameResolver::AddExposedNames(const Declaration& declaration,
                                   StaticScope& enclosing_scope,
                                   bool allow_qualified_names)
    -> ErrorOr<Success> {
  switch (declaration.kind()) {
    case DeclarationKind::NamespaceDeclaration: {
      const auto& namespace_decl = cast<NamespaceDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<StaticScope*> scope,
          AddExposedName(namespace_decl.name(), &namespace_decl,
                         enclosing_scope, allow_qualified_names));
      namespace_scopes_.try_emplace(&namespace_decl, scope, &namespace_decl);
      break;
    }
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      const auto& iface_decl = cast<ConstraintTypeDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(AddExposedName(iface_decl.name(), &iface_decl,
                                            enclosing_scope,
                                            allow_qualified_names));
      break;
    }
    case DeclarationKind::DestructorDeclaration: {
      // TODO: It should not be possible to name the destructor by unqualified
      // name.
      const auto& func = cast<DestructorDeclaration>(declaration);
      // TODO: Add support for qualified destructor declarations. Currently the
      // syntax for this is
      //   destructor Class [self: Self] { ... }
      // but see #2567.
      CARBON_RETURN_IF_ERROR(enclosing_scope.Add(
          "destructor", &func, StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      const auto& func = cast<FunctionDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(AddExposedName(func.name(), &func, enclosing_scope,
                                            allow_qualified_names));
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(AddExposedName(class_decl.name(), &class_decl,
                                            enclosing_scope,
                                            allow_qualified_names));
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      const auto& mixin_decl = cast<MixinDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(AddExposedName(mixin_decl.name(), &mixin_decl,
                                            enclosing_scope,
                                            allow_qualified_names));
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(AddExposedName(
          choice.name(), &choice, enclosing_scope, allow_qualified_names));
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      const auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name() != AnonymousName) {
        CARBON_RETURN_IF_ERROR(
            enclosing_scope.Add(var.binding().name(), &var.binding(),
                                StaticScope::NameStatus::KnownButNotDeclared));
      }
      break;
    }
    case DeclarationKind::AssociatedConstantDeclaration: {
      const auto& let = cast<AssociatedConstantDeclaration>(declaration);
      if (let.binding().name() != AnonymousName) {
        CARBON_RETURN_IF_ERROR(
            enclosing_scope.Add(let.binding().name(), &let,
                                StaticScope::NameStatus::KnownButNotDeclared));
      }
      break;
    }
    case DeclarationKind::SelfDeclaration: {
      const auto& self = cast<SelfDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(enclosing_scope.Add("Self", &self));
      break;
    }
    case DeclarationKind::AliasDeclaration: {
      const auto& alias = cast<AliasDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(AddExposedName(
          alias.name(), &alias, enclosing_scope, allow_qualified_names));
      break;
    }
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::MatchFirstDeclaration:
    case DeclarationKind::MixDeclaration:
    case DeclarationKind::InterfaceExtendDeclaration:
    case DeclarationKind::InterfaceRequireDeclaration:
    case DeclarationKind::ExtendBaseDeclaration: {
      // These declarations don't have a name to expose.
      break;
    }
  }
  return Success();
}

auto NameResolver::ResolveNames(Expression& expression,
                                const StaticScope& enclosing_scope)
    -> ErrorOr<std::optional<ValueNodeView>> {
  return RunWithExtraStack(
      [&]() { return ResolveNamesImpl(expression, enclosing_scope); });
}

auto NameResolver::ResolveNamesImpl(Expression& expression,
                                    const StaticScope& enclosing_scope)
    -> ErrorOr<std::optional<ValueNodeView>> {
  switch (expression.kind()) {
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(expression);
      CARBON_RETURN_IF_ERROR(ResolveNames(call.function(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(call.argument(), enclosing_scope));
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fun_type = cast<FunctionTypeLiteral>(expression);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(fun_type.parameter(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(fun_type.return_type(), enclosing_scope));
      break;
    }
    case ExpressionKind::SimpleMemberAccessExpression: {
      // If the left-hand side of the `.` is a namespace or alias to namespace,
      // resolve the name.
      auto& access = cast<SimpleMemberAccessExpression>(expression);
      CARBON_ASSIGN_OR_RETURN(std::optional<ValueNodeView> scope,
                              ResolveNames(access.object(), enclosing_scope));
      if (!scope) {
        break;
      }

      Nonnull<const AstNode*> base = &scope->base();
      // recursively resolve aliases.
      while (const auto* alias = dyn_cast<AliasDeclaration>(base)) {
        if (auto resolved = alias->resolved_declaration()) {
          base = *resolved;
        } else {
          break;
        }
      }
      if (const auto* namespace_decl = dyn_cast<NamespaceDeclaration>(base)) {
        auto ns_it = namespace_scopes_.find(namespace_decl);
        CARBON_CHECK(ns_it != namespace_scopes_.end())
            << "name resolved to undeclared namespace";
        CARBON_ASSIGN_OR_RETURN(
            const auto value_node,
            ns_it->second.ResolveHere(scope, access.member_name(),
                                      access.source_loc(),
                                      /*allow_undeclared=*/false));
        access.set_value_node(value_node);
        return {value_node};
      }
      break;
    }
    case ExpressionKind::CompoundMemberAccessExpression: {
      auto& access = cast<CompoundMemberAccessExpression>(expression);
      CARBON_RETURN_IF_ERROR(ResolveNames(access.object(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(access.path(), enclosing_scope));
      break;
    }
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(expression);
      CARBON_RETURN_IF_ERROR(ResolveNames(index.object(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(index.offset(), enclosing_scope));
      break;
    }
    case ExpressionKind::OperatorExpression:
      for (Nonnull<Expression*> operand :
           cast<OperatorExpression>(expression).arguments()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*operand, enclosing_scope));
      }
      break;
    case ExpressionKind::TupleLiteral:
      for (Nonnull<Expression*> field :
           cast<TupleLiteral>(expression).fields()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*field, enclosing_scope));
      }
      break;
    case ExpressionKind::StructLiteral: {
      std::set<std::string_view> member_names;
      for (FieldInitializer& init : cast<StructLiteral>(expression).fields()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(init.expression(), enclosing_scope));
        if (!member_names.insert(init.name()).second) {
          return ProgramError(init.expression().source_loc())
                 << "Duplicate name `" << init.name() << "` in struct literal";
        }
      }
      break;
    }
    case ExpressionKind::StructTypeLiteral: {
      std::set<std::string_view> member_names;
      for (FieldInitializer& init :
           cast<StructTypeLiteral>(expression).fields()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(init.expression(), enclosing_scope));
        if (!member_names.insert(init.name()).second) {
          return ProgramError(init.expression().source_loc())
                 << "Duplicate name `" << init.name()
                 << "` in struct type literal";
        }
      }
      break;
    }
    case ExpressionKind::IdentifierExpression: {
      auto& identifier = cast<IdentifierExpression>(expression);
      CARBON_ASSIGN_OR_RETURN(
          const auto value_node,
          enclosing_scope.Resolve(identifier.name(), identifier.source_loc()));
      identifier.set_value_node(value_node);
      return {value_node};
    }
    case ExpressionKind::DotSelfExpression: {
      auto& dot_self = cast<DotSelfExpression>(expression);
      CARBON_ASSIGN_OR_RETURN(
          const auto value_node,
          enclosing_scope.Resolve(".Self", dot_self.source_loc()));
      dot_self.set_self_binding(const_cast<GenericBinding*>(
          &cast<GenericBinding>(value_node.base())));
      break;
    }
    case ExpressionKind::IntrinsicExpression:
      CARBON_RETURN_IF_ERROR(ResolveNames(
          cast<IntrinsicExpression>(expression).args(), enclosing_scope));
      break;
    case ExpressionKind::IfExpression: {
      auto& if_expr = cast<IfExpression>(expression);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(if_expr.condition(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(if_expr.then_expression(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(if_expr.else_expression(), enclosing_scope));
      break;
    }
    case ExpressionKind::WhereExpression: {
      auto& where = cast<WhereExpression>(expression);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(where.self_binding().type(), enclosing_scope));
      // If we're already in a `.Self` context, remember it so that we can
      // reuse its value for the inner `.Self`.
      if (auto enclosing_dot_self =
              enclosing_scope.Resolve(".Self", where.source_loc());
          enclosing_dot_self.ok()) {
        where.set_enclosing_dot_self(
            &cast<GenericBinding>(enclosing_dot_self->base()));
      }
      // Introduce `.Self` into scope on the right of the `where` keyword.
      StaticScope where_scope(&enclosing_scope, &where);
      CARBON_RETURN_IF_ERROR(where_scope.Add(".Self", &where.self_binding()));
      for (Nonnull<WhereClause*> clause : where.clauses()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*clause, where_scope));
      }
      break;
    }
    case ExpressionKind::ArrayTypeLiteral: {
      auto& array_literal = cast<ArrayTypeLiteral>(expression);
      CARBON_RETURN_IF_ERROR(ResolveNames(
          array_literal.element_type_expression(), enclosing_scope));
      if (array_literal.has_size_expression()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(array_literal.size_expression(), enclosing_scope));
      }
      break;
    }
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::IntLiteral:
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
      break;
    case ExpressionKind::ValueLiteral:
    case ExpressionKind::BuiltinConvertExpression:
    case ExpressionKind::BaseAccessExpression:
      CARBON_FATAL() << "should not exist before type checking";
    case ExpressionKind::UnimplementedExpression:
      return ProgramError(expression.source_loc()) << "Unimplemented";
  }

  return {std::nullopt};
}

auto NameResolver::ResolveNames(WhereClause& clause,
                                const StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  return RunWithExtraStack(
      [&]() { return ResolveNamesImpl(clause, enclosing_scope); });
}

auto NameResolver::ResolveNamesImpl(WhereClause& clause,
                                    const StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (clause.kind()) {
    case WhereClauseKind::ImplsWhereClause: {
      auto& impls_clause = cast<ImplsWhereClause>(clause);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(impls_clause.type(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(impls_clause.constraint(), enclosing_scope));
      break;
    }
    case WhereClauseKind::EqualsWhereClause: {
      auto& equals_clause = cast<EqualsWhereClause>(clause);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(equals_clause.lhs(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(equals_clause.rhs(), enclosing_scope));
      break;
    }
    case WhereClauseKind::RewriteWhereClause: {
      auto& rewrite_clause = cast<RewriteWhereClause>(clause);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(rewrite_clause.replacement(), enclosing_scope));
      break;
    }
  }

  return Success();
}

auto NameResolver::ResolveNames(Pattern& pattern, StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  return RunWithExtraStack(
      [&]() { return ResolveNamesImpl(pattern, enclosing_scope); });
}

auto NameResolver::ResolveNamesImpl(Pattern& pattern,
                                    StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (pattern.kind()) {
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(pattern);
      CARBON_RETURN_IF_ERROR(ResolveNames(binding.type(), enclosing_scope));
      if (binding.name() != AnonymousName) {
        CARBON_RETURN_IF_ERROR(enclosing_scope.Add(binding.name(), &binding));
      }
      break;
    }
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(pattern);
      // `.Self` is in scope in the context of the type.
      StaticScope self_scope(&enclosing_scope, &binding);
      CARBON_RETURN_IF_ERROR(self_scope.Add(".Self", &binding));
      CARBON_RETURN_IF_ERROR(ResolveNames(binding.type(), self_scope));
      if (binding.name() != AnonymousName) {
        CARBON_RETURN_IF_ERROR(enclosing_scope.Add(binding.name(), &binding));
      }
      break;
    }
    case PatternKind::TuplePattern:
      for (Nonnull<Pattern*> field : cast<TuplePattern>(pattern).fields()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*field, enclosing_scope));
      }
      break;
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(pattern);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(alternative.choice_type(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(alternative.arguments(), enclosing_scope));
      break;
    }
    case PatternKind::ExpressionPattern:
      CARBON_RETURN_IF_ERROR(ResolveNames(
          cast<ExpressionPattern>(pattern).expression(), enclosing_scope));
      break;
    case PatternKind::AutoPattern:
      break;
    case PatternKind::VarPattern:
      CARBON_RETURN_IF_ERROR(
          ResolveNames(cast<VarPattern>(pattern).pattern(), enclosing_scope));
      break;
    case PatternKind::AddrPattern:
      CARBON_RETURN_IF_ERROR(
          ResolveNames(cast<AddrPattern>(pattern).binding(), enclosing_scope));
      break;
  }

  return Success();
}

auto NameResolver::ResolveNames(Statement& statement,
                                StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  return RunWithExtraStack(
      [&]() { return ResolveNamesImpl(statement, enclosing_scope); });
}

auto NameResolver::ResolveNamesImpl(Statement& statement,
                                    StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  if (trace_stream_->is_enabled()) {
    trace_stream_->Start() << "resolving stmt `" << PrintAsID(statement)
                           << "` (" << statement.source_loc() << ")\n";
  }
  switch (statement.kind()) {
    case StatementKind::ExpressionStatement:
      CARBON_RETURN_IF_ERROR(ResolveNames(
          cast<ExpressionStatement>(statement).expression(), enclosing_scope));
      break;
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(statement);
      CARBON_RETURN_IF_ERROR(ResolveNames(assign.lhs(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(assign.rhs(), enclosing_scope));
      break;
    }
    case StatementKind::IncrementDecrement: {
      auto& inc_dec = cast<IncrementDecrement>(statement);
      CARBON_RETURN_IF_ERROR(ResolveNames(inc_dec.argument(), enclosing_scope));
      break;
    }
    case StatementKind::VariableDefinition: {
      auto& def = cast<VariableDefinition>(statement);
      if (def.has_init()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(def.init(), enclosing_scope));
      }
      CARBON_RETURN_IF_ERROR(ResolveNames(def.pattern(), enclosing_scope));
      if (def.is_returned()) {
        CARBON_CHECK(def.pattern().kind() == PatternKind::BindingPattern)
            << def.pattern().source_loc()
            << "returned var definition can only be a binding pattern";
        CARBON_RETURN_IF_ERROR(enclosing_scope.AddReturnedVar(
            ValueNodeView(&cast<BindingPattern>(def.pattern()))));
      }
      break;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(statement);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(if_stmt.condition(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(if_stmt.then_block(), enclosing_scope));
      if (auto else_block = if_stmt.else_block()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**else_block, enclosing_scope));
      }
      break;
    }
    case StatementKind::ReturnVar: {
      auto& ret_var_stmt = cast<ReturnVar>(statement);
      std::optional<ValueNodeView> returned_var_def_view =
          enclosing_scope.ResolveReturned();
      if (!returned_var_def_view.has_value()) {
        return ProgramError(ret_var_stmt.source_loc())
               << "`return var` is not allowed without a returned var defined "
                  "in scope.";
      }
      ret_var_stmt.set_value_node(*returned_var_def_view);
      break;
    }
    case StatementKind::ReturnExpression: {
      auto& ret_exp_stmt = cast<ReturnExpression>(statement);
      std::optional<ValueNodeView> returned_var_def_view =
          enclosing_scope.ResolveReturned();
      if (returned_var_def_view.has_value()) {
        return ProgramError(ret_exp_stmt.source_loc())
               << "`return <expression>` is not allowed with a returned var "
                  "defined in scope: "
               << returned_var_def_view->base().source_loc();
      }
      CARBON_RETURN_IF_ERROR(
          ResolveNames(ret_exp_stmt.expression(), enclosing_scope));
      break;
    }
    case StatementKind::Block: {
      auto& block = cast<Block>(statement);
      StaticScope block_scope(&enclosing_scope, &block);
      for (Nonnull<Statement*> sub_statement : block.statements()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*sub_statement, block_scope));
      }
      break;
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(statement);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(while_stmt.condition(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(while_stmt.body(), enclosing_scope));
      break;
    }
    case StatementKind::For: {
      auto& for_stmt = cast<For>(statement);
      StaticScope statement_scope(&enclosing_scope, &for_stmt);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(for_stmt.loop_target(), statement_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(for_stmt.variable_declaration(), statement_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(for_stmt.body(), statement_scope));

      break;
    }
    case StatementKind::Match: {
      auto& match = cast<Match>(statement);
      CARBON_RETURN_IF_ERROR(ResolveNames(match.expression(), enclosing_scope));
      for (Match::Clause& clause : match.clauses()) {
        StaticScope clause_scope(&enclosing_scope, &clause.statement());
        CARBON_RETURN_IF_ERROR(ResolveNames(clause.pattern(), clause_scope));
        CARBON_RETURN_IF_ERROR(ResolveNames(clause.statement(), clause_scope));
      }
      break;
    }
    case StatementKind::Break:
    case StatementKind::Continue:
      break;
  }

  if (trace_stream_->is_enabled()) {
    trace_stream_->End() << "finished resolving stmt `" << PrintAsID(statement)
                         << "` (" << statement.source_loc() << ")\n";
  }

  return Success();
}

auto NameResolver::ResolveMemberNames(
    llvm::ArrayRef<Nonnull<Declaration*>> members, StaticScope& scope,
    ResolveFunctionBodies bodies) -> ErrorOr<Success> {
  for (Nonnull<Declaration*> member : members) {
    CARBON_RETURN_IF_ERROR(AddExposedNames(*member, scope));
  }
  if (bodies != ResolveFunctionBodies::Immediately) {
    for (Nonnull<Declaration*> member : members) {
      CARBON_RETURN_IF_ERROR(
          ResolveNames(*member, scope, ResolveFunctionBodies::Skip));
    }
  }
  if (bodies != ResolveFunctionBodies::Skip) {
    for (Nonnull<Declaration*> member : members) {
      CARBON_RETURN_IF_ERROR(
          ResolveNames(*member, scope, ResolveFunctionBodies::Immediately));
    }
  }
  return Success();
}

auto NameResolver::ResolveNames(Declaration& declaration,
                                StaticScope& enclosing_scope,
                                ResolveFunctionBodies bodies)
    -> ErrorOr<Success> {
  return RunWithExtraStack(
      [&]() { return ResolveNamesImpl(declaration, enclosing_scope, bodies); });
}

auto NameResolver::ResolveNamesImpl(Declaration& declaration,
                                    StaticScope& enclosing_scope,
                                    ResolveFunctionBodies bodies)
    -> ErrorOr<Success> {
  if (trace_stream_->is_enabled()) {
    trace_stream_->Start() << "resolving decl `" << PrintAsID(declaration)
                           << "` (" << declaration.source_loc() << ")\n";
  }

  switch (declaration.kind()) {
    case DeclarationKind::NamespaceDeclaration: {
      auto& namespace_decl = cast<NamespaceDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<StaticScope*> scope,
          ResolveQualifier(namespace_decl.name(), enclosing_scope));
      scope->MarkUsable(namespace_decl.name().inner_name());
      break;
    }
    case DeclarationKind::InterfaceDeclaration:
    case DeclarationKind::ConstraintDeclaration: {
      auto& iface = cast<ConstraintTypeDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(Nonnull<StaticScope*> scope,
                              ResolveQualifier(iface.name(), enclosing_scope));
      StaticScope iface_scope(scope, &iface);
      scope->MarkDeclared(iface.name().inner_name());
      if (auto params = iface.params()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**params, iface_scope));
      }
      scope->MarkUsable(iface.name().inner_name());
      // Don't resolve names in the type of the self binding. The
      // ConstraintTypeDeclaration constructor already did that.
      CARBON_RETURN_IF_ERROR(iface_scope.Add("Self", iface.self()));
      CARBON_RETURN_IF_ERROR(
          ResolveMemberNames(iface.members(), iface_scope, bodies));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl = cast<ImplDeclaration>(declaration);
      StaticScope impl_scope(&enclosing_scope, &impl);
      for (Nonnull<GenericBinding*> binding : impl.deduced_parameters()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(binding->type(), impl_scope));
        CARBON_RETURN_IF_ERROR(impl_scope.Add(binding->name(), binding));
      }
      CARBON_RETURN_IF_ERROR(ResolveNames(*impl.impl_type(), impl_scope));
      // Only add `Self` to the impl_scope if it is not already in the enclosing
      // scope. Add `Self` after we resolve names for the impl_type, so you
      // can't write something like `impl Vector(Self) as ...`. Add `Self`
      // before resolving names in the interface, so you can write something
      // like `impl VeryLongTypeName as AddWith(Self)`
      if (!enclosing_scope.Resolve("Self", impl.source_loc()).ok()) {
        CARBON_RETURN_IF_ERROR(AddExposedNames(*impl.self(), impl_scope));
      }
      CARBON_RETURN_IF_ERROR(ResolveNames(impl.interface(), impl_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveMemberNames(impl.members(), impl_scope, bodies));
      break;
    }
    case DeclarationKind::MatchFirstDeclaration: {
      // A `match_first` declaration does not introduce a scope.
      for (auto* impl :
           cast<MatchFirstDeclaration>(declaration).impl_declarations()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*impl, enclosing_scope, bodies));
      }
      break;
    }
    case DeclarationKind::DestructorDeclaration:
    case DeclarationKind::FunctionDeclaration: {
      auto& function = cast<CallableDeclaration>(declaration);
      // TODO: Destructors should track their qualified name.
      const DeclaredName& name =
          isa<FunctionDeclaration>(declaration)
              ? cast<FunctionDeclaration>(declaration).name()
              : DeclaredName(function.source_loc(), "destructor");
      CARBON_ASSIGN_OR_RETURN(Nonnull<StaticScope*> scope,
                              ResolveQualifier(name, enclosing_scope));
      StaticScope function_scope(scope, &function);
      scope->MarkDeclared(name.inner_name());
      for (Nonnull<GenericBinding*> binding : function.deduced_parameters()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*binding, function_scope));
      }
      if (function.is_method()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(function.self_pattern(), function_scope));
      }
      CARBON_RETURN_IF_ERROR(
          ResolveNames(function.param_pattern(), function_scope));
      if (auto return_type_expr = function.return_term().type_expression()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(**return_type_expr, function_scope));
      }
      scope->MarkUsable(name.inner_name());
      if (auto body = function.body();
          body.has_value() && bodies != ResolveFunctionBodies::Skip) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**body, function_scope));
      }
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<StaticScope*> scope,
          ResolveQualifier(class_decl.name(), enclosing_scope));
      StaticScope class_scope(scope, &class_decl);
      scope->MarkDeclared(class_decl.name().inner_name());
      if (auto type_params = class_decl.type_params()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**type_params, class_scope));
      }
      scope->MarkUsable(class_decl.name().inner_name());
      CARBON_RETURN_IF_ERROR(AddExposedNames(*class_decl.self(), class_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveMemberNames(class_decl.members(), class_scope, bodies));
      break;
    }
    case DeclarationKind::ExtendBaseDeclaration: {
      auto& extend_base_decl = cast<ExtendBaseDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(*extend_base_decl.base_class(), enclosing_scope));
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      auto& mixin_decl = cast<MixinDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(
          Nonnull<StaticScope*> scope,
          ResolveQualifier(mixin_decl.name(), enclosing_scope));
      StaticScope mixin_scope(scope, &mixin_decl);
      scope->MarkDeclared(mixin_decl.name().inner_name());
      if (auto params = mixin_decl.params()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**params, mixin_scope));
      }
      scope->MarkUsable(mixin_decl.name().inner_name());
      CARBON_RETURN_IF_ERROR(mixin_scope.Add("Self", mixin_decl.self()));
      CARBON_RETURN_IF_ERROR(
          ResolveMemberNames(mixin_decl.members(), mixin_scope, bodies));
      break;
    }
    case DeclarationKind::MixDeclaration: {
      auto& mix_decl = cast<MixDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(ResolveNames(mix_decl.mixin(), enclosing_scope));
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(Nonnull<StaticScope*> scope,
                              ResolveQualifier(choice.name(), enclosing_scope));
      StaticScope choice_scope(scope, &choice);
      scope->MarkDeclared(choice.name().inner_name());
      if (auto type_params = choice.type_params()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**type_params, choice_scope));
      }
      // Alternative names are never used unqualified, so we don't need to
      // add the alternatives to a scope, or introduce a new scope; we only
      // need to check for duplicates.
      std::set<std::string_view> alternative_names;
      for (Nonnull<AlternativeSignature*> alternative : choice.alternatives()) {
        if (auto params = alternative->parameters()) {
          CARBON_RETURN_IF_ERROR(ResolveNames(**params, choice_scope));
        }
        if (!alternative_names.insert(alternative->name()).second) {
          return ProgramError(alternative->source_loc())
                 << "Duplicate name `" << alternative->name()
                 << "` in choice type";
        }
      }
      scope->MarkUsable(choice.name().inner_name());
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(ResolveNames(var.binding(), enclosing_scope));
      if (var.has_initializer()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(var.initializer(), enclosing_scope));
      }
      break;
    }
    case DeclarationKind::InterfaceExtendDeclaration: {
      auto& extends = cast<InterfaceExtendDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(ResolveNames(*extends.base(), enclosing_scope));
      break;
    }
    case DeclarationKind::InterfaceRequireDeclaration: {
      auto& require = cast<InterfaceRequireDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(
          ResolveNames(*require.impl_type(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(*require.constraint(), enclosing_scope));
      break;
    }
    case DeclarationKind::AssociatedConstantDeclaration: {
      auto& let = cast<AssociatedConstantDeclaration>(declaration);
      StaticScope constant_scope(&enclosing_scope, &let);
      enclosing_scope.MarkDeclared(let.binding().name());
      CARBON_RETURN_IF_ERROR(ResolveNames(let.binding(), constant_scope));
      enclosing_scope.MarkUsable(let.binding().name());
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      CARBON_FATAL() << "Unreachable: resolving names for `Self` declaration";
    }

    case DeclarationKind::AliasDeclaration: {
      auto& alias = cast<AliasDeclaration>(declaration);
      CARBON_ASSIGN_OR_RETURN(Nonnull<StaticScope*> scope,
                              ResolveQualifier(alias.name(), enclosing_scope));
      scope->MarkDeclared(alias.name().inner_name());
      CARBON_ASSIGN_OR_RETURN(auto target,
                              ResolveNames(alias.target(), *scope));
      if (target && isa<Declaration>(target->base())) {
        if (auto resolved_declaration = alias.resolved_declaration()) {
          // Skip if the declaration is already resolved in a previous name
          // resolution phase.
          CARBON_CHECK(*resolved_declaration == &target->base());
        } else {
          alias.set_resolved_declaration(&cast<Declaration>(target->base()));
        }
      }
      scope->MarkUsable(alias.name().inner_name());
      break;
    }
  }

  if (trace_stream_->is_enabled()) {
    trace_stream_->End() << "finished resolving decl `"
                         << PrintAsID(declaration) << "` ("
                         << declaration.source_loc() << ")\n";
  }

  return Success();
}

auto ResolveNames(AST& ast, Nonnull<TraceStream*> trace_stream)
    -> ErrorOr<Success> {
  return RunWithExtraStack([&]() -> ErrorOr<Success> {
    NameResolver resolver(trace_stream);
    SetFileContext set_file_ctx(*trace_stream, std::nullopt);
    StaticScope file_scope(trace_stream);

    for (auto* declaration : ast.declarations) {
      set_file_ctx.update_source_loc(declaration->source_loc());
      CARBON_RETURN_IF_ERROR(resolver.AddExposedNames(
          *declaration, file_scope, /*allow_qualified_names=*/true));
    }

    for (auto* declaration : ast.declarations) {
      set_file_ctx.update_source_loc(declaration->source_loc());
      CARBON_RETURN_IF_ERROR(resolver.ResolveNames(
          *declaration, file_scope,
          NameResolver::ResolveFunctionBodies::AfterDeclarations));
    }
    CARBON_RETURN_IF_ERROR(resolver.ResolveNames(**ast.main_call, file_scope));
    return Success();
  });
}

}  // namespace Carbon
