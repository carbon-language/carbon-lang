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
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

using llvm::cast;

namespace Carbon {

// Adds the names exposed by the given AST node to enclosing_scope.
static auto AddExposedNames(const Declaration& declaration,
                            StaticScope& enclosing_scope) -> ErrorOr<Success> {
  switch (declaration.kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      const auto& iface_decl = cast<InterfaceDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(
          enclosing_scope.Add(iface_decl.name(), &iface_decl,
                              StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::DestructorDeclaration: {
      // TODO: Remove this code. With this code, it is possible to create not
      // useful carbon code.
      //       Without this code, a Segfault is generated
      const auto& func = cast<DestructorDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(enclosing_scope.Add(
          "destructor", &func, StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      const auto& func = cast<FunctionDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(enclosing_scope.Add(
          func.name(), &func, StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      const auto& class_decl = cast<ClassDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(
          enclosing_scope.Add(class_decl.name(), &class_decl,
                              StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      const auto& mixin_decl = cast<MixinDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(
          enclosing_scope.Add(mixin_decl.name(), &mixin_decl,
                              StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      const auto& choice = cast<ChoiceDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(
          enclosing_scope.Add(choice.name(), &choice,
                              StaticScope::NameStatus::KnownButNotDeclared));
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
        CARBON_RETURN_IF_ERROR(enclosing_scope.Add(let.binding().name(), &let));
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
      CARBON_RETURN_IF_ERROR(enclosing_scope.Add(
          alias.name(), &alias, StaticScope::NameStatus::KnownButNotDeclared));
      break;
    }
    case DeclarationKind::ImplDeclaration:
    case DeclarationKind::MixDeclaration:
    case DeclarationKind::InterfaceExtendsDeclaration:
    case DeclarationKind::InterfaceImplDeclaration: {
      // These declarations don't have a name to expose.
      break;
    }
  }
  return Success();
}

namespace {
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
}  // namespace

// Traverses the sub-AST rooted at the given node, resolving all names within
// it using enclosing_scope, and updating enclosing_scope to add names to
// it as they become available. In scopes where names are only visible below
// their point of declaration (such as block scopes in C++), this is implemented
// as a single pass, recursively calling ResolveNames on the elements of the
// scope in order. In scopes where names are also visible above their point of
// declaration (such as class scopes in C++), this requires three passes: first
// calling AddExposedNames on each element of the scope to populate a
// StaticScope, and then calling ResolveNames on each element, passing it the
// already-populated StaticScope but skipping member function bodies, and
// finally calling ResolvedNames again on each element, and this time resolving
// member function bodies.
static auto ResolveNames(Expression& expression,
                         const StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(WhereClause& clause,
                         const StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(Pattern& pattern, StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(Statement& statement, StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(Declaration& declaration, StaticScope& enclosing_scope,
                         ResolveFunctionBodies bodies) -> ErrorOr<Success>;

static auto ResolveNames(Expression& expression,
                         const StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
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
    case ExpressionKind::SimpleMemberAccessExpression:
      CARBON_RETURN_IF_ERROR(
          ResolveNames(cast<SimpleMemberAccessExpression>(expression).object(),
                       enclosing_scope));
      break;
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
    case ExpressionKind::StructLiteral:
      for (FieldInitializer& init : cast<StructLiteral>(expression).fields()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(init.expression(), enclosing_scope));
      }
      break;
    case ExpressionKind::StructTypeLiteral:
      for (FieldInitializer& init :
           cast<StructTypeLiteral>(expression).fields()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(init.expression(), enclosing_scope));
      }
      break;
    case ExpressionKind::IdentifierExpression: {
      auto& identifier = cast<IdentifierExpression>(expression);
      CARBON_ASSIGN_OR_RETURN(
          const auto value_node,
          enclosing_scope.Resolve(identifier.name(), identifier.source_loc()));
      identifier.set_value_node(value_node);
      break;
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
      StaticScope where_scope;
      where_scope.AddParent(&enclosing_scope);
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
      CARBON_RETURN_IF_ERROR(
          ResolveNames(array_literal.size_expression(), enclosing_scope));
      break;
    }
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
    case ExpressionKind::IntLiteral:
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
      break;
    case ExpressionKind::ValueLiteral:
    case ExpressionKind::BuiltinConvertExpression:
      CARBON_FATAL() << "should not exist before type checking";
    case ExpressionKind::UnimplementedExpression:
      return ProgramError(expression.source_loc()) << "Unimplemented";
  }
  return Success();
}

static auto ResolveNames(WhereClause& clause,
                         const StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (clause.kind()) {
    case WhereClauseKind::IsWhereClause: {
      auto& is_clause = cast<IsWhereClause>(clause);
      CARBON_RETURN_IF_ERROR(ResolveNames(is_clause.type(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveNames(is_clause.constraint(), enclosing_scope));
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

static auto ResolveNames(Pattern& pattern, StaticScope& enclosing_scope)
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
      StaticScope self_scope;
      self_scope.AddParent(&enclosing_scope);
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

static auto ResolveNames(Statement& statement, StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
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
      if (if_stmt.else_block().has_value()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(**if_stmt.else_block(), enclosing_scope));
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
      StaticScope block_scope;
      block_scope.AddParent(&enclosing_scope);
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
      StaticScope statement_scope;
      statement_scope.AddParent(&enclosing_scope);
      auto& for_stmt = cast<For>(statement);
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
        StaticScope clause_scope;
        clause_scope.AddParent(&enclosing_scope);
        CARBON_RETURN_IF_ERROR(ResolveNames(clause.pattern(), clause_scope));
        CARBON_RETURN_IF_ERROR(ResolveNames(clause.statement(), clause_scope));
      }
      break;
    }
    case StatementKind::Continuation: {
      auto& continuation = cast<Continuation>(statement);
      CARBON_RETURN_IF_ERROR(
          enclosing_scope.Add(continuation.name(), &continuation,
                              StaticScope::NameStatus::DeclaredButNotUsable));
      StaticScope continuation_scope;
      continuation_scope.AddParent(&enclosing_scope);
      CARBON_RETURN_IF_ERROR(ResolveNames(cast<Continuation>(statement).body(),
                                          continuation_scope));
      enclosing_scope.MarkUsable(continuation.name());
      break;
    }
    case StatementKind::Run:
      CARBON_RETURN_IF_ERROR(
          ResolveNames(cast<Run>(statement).argument(), enclosing_scope));
      break;
    case StatementKind::Await:
    case StatementKind::Break:
    case StatementKind::Continue:
      break;
  }
  return Success();
}

static auto ResolveMemberNames(llvm::ArrayRef<Nonnull<Declaration*>> members,
                               StaticScope& scope, ResolveFunctionBodies bodies)
    -> ErrorOr<Success> {
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

static auto ResolveNames(Declaration& declaration, StaticScope& enclosing_scope,
                         ResolveFunctionBodies bodies) -> ErrorOr<Success> {
  switch (declaration.kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface = cast<InterfaceDeclaration>(declaration);
      StaticScope iface_scope;
      iface_scope.AddParent(&enclosing_scope);
      enclosing_scope.MarkDeclared(iface.name());
      if (iface.params().has_value()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**iface.params(), iface_scope));
      }
      enclosing_scope.MarkUsable(iface.name());
      // Don't resolve names in the type of the self binding. The
      // InterfaceDeclaration constructor already did that.
      CARBON_RETURN_IF_ERROR(iface_scope.Add("Self", iface.self()));
      CARBON_RETURN_IF_ERROR(
          ResolveMemberNames(iface.members(), iface_scope, bodies));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl = cast<ImplDeclaration>(declaration);
      StaticScope impl_scope;
      impl_scope.AddParent(&enclosing_scope);
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
    case DeclarationKind::DestructorDeclaration:
    case DeclarationKind::FunctionDeclaration: {
      auto& function = cast<CallableDeclaration>(declaration);
      StaticScope function_scope;
      function_scope.AddParent(&enclosing_scope);
      enclosing_scope.MarkDeclared(function.name());
      for (Nonnull<GenericBinding*> binding : function.deduced_parameters()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(*binding, function_scope));
      }
      if (function.is_method()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(function.me_pattern(), function_scope));
      }
      CARBON_RETURN_IF_ERROR(
          ResolveNames(function.param_pattern(), function_scope));
      if (function.return_term().type_expression().has_value()) {
        CARBON_RETURN_IF_ERROR(ResolveNames(
            **function.return_term().type_expression(), function_scope));
      }
      enclosing_scope.MarkUsable(function.name());
      if (function.body().has_value() &&
          bodies != ResolveFunctionBodies::Skip) {
        CARBON_RETURN_IF_ERROR(ResolveNames(**function.body(), function_scope));
      }
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      StaticScope class_scope;
      class_scope.AddParent(&enclosing_scope);
      enclosing_scope.MarkDeclared(class_decl.name());
      if (class_decl.base_expr().has_value()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(**class_decl.base_expr(), class_scope));
      }
      if (class_decl.type_params().has_value()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(**class_decl.type_params(), class_scope));
      }
      enclosing_scope.MarkUsable(class_decl.name());
      CARBON_RETURN_IF_ERROR(AddExposedNames(*class_decl.self(), class_scope));
      CARBON_RETURN_IF_ERROR(
          ResolveMemberNames(class_decl.members(), class_scope, bodies));
      break;
    }
    case DeclarationKind::MixinDeclaration: {
      auto& mixin_decl = cast<MixinDeclaration>(declaration);
      StaticScope mixin_scope;
      mixin_scope.AddParent(&enclosing_scope);
      enclosing_scope.MarkDeclared(mixin_decl.name());
      if (mixin_decl.params().has_value()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(**mixin_decl.params(), mixin_scope));
      }
      enclosing_scope.MarkUsable(mixin_decl.name());
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
      StaticScope choice_scope;
      choice_scope.AddParent(&enclosing_scope);
      enclosing_scope.MarkDeclared(choice.name());
      if (choice.type_params().has_value()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(**choice.type_params(), choice_scope));
      }
      // Alternative names are never used unqualified, so we don't need to
      // add the alternatives to a scope, or introduce a new scope; we only
      // need to check for duplicates.
      std::set<std::string_view> alternative_names;
      for (Nonnull<AlternativeSignature*> alternative : choice.alternatives()) {
        CARBON_RETURN_IF_ERROR(
            ResolveNames(alternative->signature(), choice_scope));
        if (!alternative_names.insert(alternative->name()).second) {
          return ProgramError(alternative->source_loc())
                 << "Duplicate name `" << alternative->name()
                 << "` in choice type";
        }
      }
      enclosing_scope.MarkUsable(choice.name());
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
    case DeclarationKind::InterfaceExtendsDeclaration: {
      auto& extends = cast<InterfaceExtendsDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(ResolveNames(*extends.base(), enclosing_scope));
      break;
    }
    case DeclarationKind::InterfaceImplDeclaration: {
      auto& impl = cast<InterfaceImplDeclaration>(declaration);
      CARBON_RETURN_IF_ERROR(ResolveNames(*impl.impl_type(), enclosing_scope));
      CARBON_RETURN_IF_ERROR(ResolveNames(*impl.constraint(), enclosing_scope));
      break;
    }
    case DeclarationKind::AssociatedConstantDeclaration: {
      auto& let = cast<AssociatedConstantDeclaration>(declaration);
      StaticScope constant_scope;
      constant_scope.AddParent(&enclosing_scope);
      CARBON_RETURN_IF_ERROR(ResolveNames(let.binding(), constant_scope));
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      CARBON_FATAL() << "Unreachable: resolving names for `Self` declaration";
    }

    case DeclarationKind::AliasDeclaration: {
      auto& alias = cast<AliasDeclaration>(declaration);
      enclosing_scope.MarkDeclared(alias.name());
      CARBON_RETURN_IF_ERROR(ResolveNames(alias.target(), enclosing_scope));
      enclosing_scope.MarkUsable(alias.name());
      break;
    }
  }
  return Success();
}

auto ResolveNames(AST& ast) -> ErrorOr<Success> {
  StaticScope file_scope;
  for (auto* declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(AddExposedNames(*declaration, file_scope));
  }
  for (auto* declaration : ast.declarations) {
    CARBON_RETURN_IF_ERROR(ResolveNames(
        *declaration, file_scope, ResolveFunctionBodies::AfterDeclarations));
  }
  return ResolveNames(**ast.main_call, file_scope);
}

}  // namespace Carbon
