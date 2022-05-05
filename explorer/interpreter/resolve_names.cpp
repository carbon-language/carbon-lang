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
                            StaticScope& enclosing_scope) -> ErrorOr<Success>;

static auto AddExposedNames(const Declaration& declaration,
                            StaticScope& enclosing_scope) -> ErrorOr<Success> {
  switch (declaration.kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface_decl = cast<InterfaceDeclaration>(declaration);
      RETURN_IF_ERROR(enclosing_scope.Add(iface_decl.name(), &iface_decl));
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      // Nothing to do here
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      auto& func = cast<FunctionDeclaration>(declaration);
      RETURN_IF_ERROR(enclosing_scope.Add(func.name(), &func));
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      RETURN_IF_ERROR(enclosing_scope.Add(class_decl.name(), &class_decl));
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      RETURN_IF_ERROR(enclosing_scope.Add(choice.name(), &choice));
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name() != AnonymousName) {
        RETURN_IF_ERROR(
            enclosing_scope.Add(var.binding().name(), &var.binding()));
      }
      break;
    }
    case DeclarationKind::SelfDeclaration: {
      FATAL() << "Unreachable AddExposedNames() on a `Self` declaration.";
      break;
    }
  }
  return Success();
}

// Traverses the sub-AST rooted at the given node, resolving all names within
// it using enclosing_scope, and updating enclosing_scope to add names to
// it as they become available. In scopes where names are only visible below
// their point of declaration (such as block scopes in C++), this is implemented
// as a single pass, recursively calling ResolveNames on the elements of the
// scope in order. In scopes where names are also visible above their point of
// declaration (such as class scopes in C++), this requires two passes: first
// calling AddExposedNames on each element of the scope to populate a
// StaticScope, and then calling ResolveNames on each element, passing it the
// already-populated StaticScope.
static auto ResolveNames(Expression& expression,
                         const StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(Pattern& pattern, StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(Statement& statement, StaticScope& enclosing_scope)
    -> ErrorOr<Success>;
static auto ResolveNames(Declaration& declaration, StaticScope& enclosing_scope)
    -> ErrorOr<Success>;

static auto ResolveNames(Expression& expression,
                         const StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (expression.kind()) {
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(expression);
      RETURN_IF_ERROR(ResolveNames(call.function(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(call.argument(), enclosing_scope));
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fun_type = cast<FunctionTypeLiteral>(expression);
      RETURN_IF_ERROR(ResolveNames(fun_type.parameter(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(fun_type.return_type(), enclosing_scope));
      break;
    }
    case ExpressionKind::FieldAccessExpression:
      RETURN_IF_ERROR(
          ResolveNames(cast<FieldAccessExpression>(expression).aggregate(),
                       enclosing_scope));
      break;
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(expression);
      RETURN_IF_ERROR(ResolveNames(index.aggregate(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(index.offset(), enclosing_scope));
      break;
    }
    case ExpressionKind::PrimitiveOperatorExpression:
      for (Nonnull<Expression*> operand :
           cast<PrimitiveOperatorExpression>(expression).arguments()) {
        RETURN_IF_ERROR(ResolveNames(*operand, enclosing_scope));
      }
      break;
    case ExpressionKind::TupleLiteral:
      for (Nonnull<Expression*> field :
           cast<TupleLiteral>(expression).fields()) {
        RETURN_IF_ERROR(ResolveNames(*field, enclosing_scope));
      }
      break;
    case ExpressionKind::StructLiteral:
      for (FieldInitializer& init : cast<StructLiteral>(expression).fields()) {
        RETURN_IF_ERROR(ResolveNames(init.expression(), enclosing_scope));
      }
      break;
    case ExpressionKind::StructTypeLiteral:
      for (FieldInitializer& init :
           cast<StructTypeLiteral>(expression).fields()) {
        RETURN_IF_ERROR(ResolveNames(init.expression(), enclosing_scope));
      }
      break;
    case ExpressionKind::IdentifierExpression: {
      auto& identifier = cast<IdentifierExpression>(expression);
      ASSIGN_OR_RETURN(
          const auto value_node,
          enclosing_scope.Resolve(identifier.name(), identifier.source_loc()));
      identifier.set_value_node(value_node);
      break;
    }
    case ExpressionKind::IntrinsicExpression:
      RETURN_IF_ERROR(ResolveNames(cast<IntrinsicExpression>(expression).args(),
                                   enclosing_scope));
      break;
    case ExpressionKind::IfExpression: {
      auto& if_expr = cast<IfExpression>(expression);
      RETURN_IF_ERROR(ResolveNames(if_expr.condition(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(if_expr.then_expression(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(if_expr.else_expression(), enclosing_scope));
      break;
    }
    case ExpressionKind::ArrayTypeLiteral: {
      auto& array_literal = cast<ArrayTypeLiteral>(expression);
      RETURN_IF_ERROR(ResolveNames(array_literal.element_type_expression(),
                                   enclosing_scope));
      RETURN_IF_ERROR(
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
    case ExpressionKind::InstantiateImpl:  // created after name resolution
    case ExpressionKind::UnimplementedExpression:
      return CompilationError(expression.source_loc()) << "Unimplemented";
  }
  return Success();
}

static auto ResolveNames(Pattern& pattern, StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (pattern.kind()) {
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(pattern);
      RETURN_IF_ERROR(ResolveNames(binding.type(), enclosing_scope));
      if (binding.name() != AnonymousName) {
        RETURN_IF_ERROR(enclosing_scope.Add(binding.name(), &binding));
      }
      break;
    }
    case PatternKind::GenericBinding: {
      auto& binding = cast<GenericBinding>(pattern);
      RETURN_IF_ERROR(ResolveNames(binding.type(), enclosing_scope));
      if (binding.name() != AnonymousName) {
        RETURN_IF_ERROR(enclosing_scope.Add(binding.name(), &binding));
      }
      break;
    }
    case PatternKind::TuplePattern:
      for (Nonnull<Pattern*> field : cast<TuplePattern>(pattern).fields()) {
        RETURN_IF_ERROR(ResolveNames(*field, enclosing_scope));
      }
      break;
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(pattern);
      RETURN_IF_ERROR(ResolveNames(alternative.choice_type(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(alternative.arguments(), enclosing_scope));
      break;
    }
    case PatternKind::ExpressionPattern:
      RETURN_IF_ERROR(ResolveNames(
          cast<ExpressionPattern>(pattern).expression(), enclosing_scope));
      break;
    case PatternKind::AutoPattern:
      break;
    case PatternKind::VarPattern:
      RETURN_IF_ERROR(
          ResolveNames(cast<VarPattern>(pattern).pattern(), enclosing_scope));
      break;
  }
  return Success();
}

static auto ResolveNames(Statement& statement, StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (statement.kind()) {
    case StatementKind::ExpressionStatement:
      RETURN_IF_ERROR(ResolveNames(
          cast<ExpressionStatement>(statement).expression(), enclosing_scope));
      break;
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(statement);
      RETURN_IF_ERROR(ResolveNames(assign.lhs(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(assign.rhs(), enclosing_scope));
      break;
    }
    case StatementKind::VariableDefinition: {
      auto& def = cast<VariableDefinition>(statement);
      RETURN_IF_ERROR(ResolveNames(def.init(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(def.pattern(), enclosing_scope));
      break;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(statement);
      RETURN_IF_ERROR(ResolveNames(if_stmt.condition(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(if_stmt.then_block(), enclosing_scope));
      if (if_stmt.else_block().has_value()) {
        RETURN_IF_ERROR(ResolveNames(**if_stmt.else_block(), enclosing_scope));
      }
      break;
    }
    case StatementKind::Return:
      RETURN_IF_ERROR(
          ResolveNames(cast<Return>(statement).expression(), enclosing_scope));
      break;
    case StatementKind::Block: {
      auto& block = cast<Block>(statement);
      StaticScope block_scope;
      block_scope.AddParent(&enclosing_scope);
      for (Nonnull<Statement*> sub_statement : block.statements()) {
        RETURN_IF_ERROR(ResolveNames(*sub_statement, block_scope));
      }
      break;
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(statement);
      RETURN_IF_ERROR(ResolveNames(while_stmt.condition(), enclosing_scope));
      RETURN_IF_ERROR(ResolveNames(while_stmt.body(), enclosing_scope));
      break;
    }
    case StatementKind::Match: {
      auto& match = cast<Match>(statement);
      RETURN_IF_ERROR(ResolveNames(match.expression(), enclosing_scope));
      for (Match::Clause& clause : match.clauses()) {
        StaticScope clause_scope;
        clause_scope.AddParent(&enclosing_scope);
        RETURN_IF_ERROR(ResolveNames(clause.pattern(), clause_scope));
        RETURN_IF_ERROR(ResolveNames(clause.statement(), clause_scope));
      }
      break;
    }
    case StatementKind::Continuation: {
      auto& continuation = cast<Continuation>(statement);
      RETURN_IF_ERROR(enclosing_scope.Add(continuation.name(), &continuation));
      StaticScope continuation_scope;
      continuation_scope.AddParent(&enclosing_scope);
      RETURN_IF_ERROR(ResolveNames(cast<Continuation>(statement).body(),
                                   continuation_scope));
      break;
    }
    case StatementKind::Run:
      RETURN_IF_ERROR(
          ResolveNames(cast<Run>(statement).argument(), enclosing_scope));
      break;
    case StatementKind::Await:
    case StatementKind::Break:
    case StatementKind::Continue:
      break;
  }
  return Success();
}

static auto ResolveNames(Declaration& declaration, StaticScope& enclosing_scope)
    -> ErrorOr<Success> {
  switch (declaration.kind()) {
    case DeclarationKind::InterfaceDeclaration: {
      auto& iface = cast<InterfaceDeclaration>(declaration);
      StaticScope iface_scope;
      iface_scope.AddParent(&enclosing_scope);
      if (iface.params().has_value()) {
        RETURN_IF_ERROR(ResolveNames(**iface.params(), iface_scope));
      }
      RETURN_IF_ERROR(iface_scope.Add("Self", iface.self()));
      for (Nonnull<Declaration*> member : iface.members()) {
        RETURN_IF_ERROR(AddExposedNames(*member, iface_scope));
      }
      for (Nonnull<Declaration*> member : iface.members()) {
        RETURN_IF_ERROR(ResolveNames(*member, iface_scope));
      }
      break;
    }
    case DeclarationKind::ImplDeclaration: {
      auto& impl = cast<ImplDeclaration>(declaration);
      StaticScope impl_scope;
      impl_scope.AddParent(&enclosing_scope);
      for (Nonnull<GenericBinding*> binding : impl.deduced_parameters()) {
        RETURN_IF_ERROR(ResolveNames(binding->type(), impl_scope));
        RETURN_IF_ERROR(impl_scope.Add(binding->name(), binding));
      }
      RETURN_IF_ERROR(ResolveNames(*impl.impl_type(), impl_scope));
      // Only add `Self` to the impl_scope if it is not already in the enclosing
      // scope. Add `Self` after we resolve names for the impl_type, so you
      // can't write something like `impl Vector(Self) as ...`. Add `Self`
      // before resolving names in the interface, so you can write something
      // like `impl VeryLongTypeName as AddWith(Self)`
      if (!enclosing_scope.Resolve("Self", impl.source_loc()).ok()) {
        // FIXME: Should this instead be
        // RETURN_IF_ERROR(AddExposedNames(impl.self(), impl_scope));?
        RETURN_IF_ERROR(impl_scope.Add("Self", impl.self()));
      }
      RETURN_IF_ERROR(ResolveNames(impl.interface(), impl_scope));
      for (Nonnull<Declaration*> member : impl.members()) {
        RETURN_IF_ERROR(AddExposedNames(*member, impl_scope));
      }
      for (Nonnull<Declaration*> member : impl.members()) {
        RETURN_IF_ERROR(ResolveNames(*member, impl_scope));
      }
      break;
    }
    case DeclarationKind::FunctionDeclaration: {
      auto& function = cast<FunctionDeclaration>(declaration);
      StaticScope function_scope;
      function_scope.AddParent(&enclosing_scope);
      for (Nonnull<GenericBinding*> binding : function.deduced_parameters()) {
        RETURN_IF_ERROR(ResolveNames(binding->type(), function_scope));
        RETURN_IF_ERROR(function_scope.Add(binding->name(), binding));
      }
      if (function.is_method()) {
        RETURN_IF_ERROR(ResolveNames(function.me_pattern(), function_scope));
      }
      RETURN_IF_ERROR(ResolveNames(function.param_pattern(), function_scope));
      if (function.return_term().type_expression().has_value()) {
        RETURN_IF_ERROR(ResolveNames(**function.return_term().type_expression(),
                                     function_scope));
      }
      if (function.body().has_value()) {
        RETURN_IF_ERROR(ResolveNames(**function.body(), function_scope));
      }
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      StaticScope class_scope;
      class_scope.AddParent(&enclosing_scope);
      RETURN_IF_ERROR(class_scope.Add(class_decl.name(), &class_decl));
      // FIXME: Should this instead be
      // RETURN_IF_ERROR(AddExposedNames(class_decl.self(), class_scope));?
      RETURN_IF_ERROR(class_scope.Add("Self", class_decl.self()));
      if (class_decl.type_params().has_value()) {
        RETURN_IF_ERROR(ResolveNames(**class_decl.type_params(), class_scope));
      }

      // TODO: Disable unqualified access of members by other members for now.
      // Put it back later, but in a way that turns unqualified accesses
      // into qualified ones, so that generic classes and impls
      // behave the in the right way. -Jeremy
      // for (Nonnull<Declaration*> member : class_decl.members()) {
      //   AddExposedNames(*member, class_scope);
      // }
      for (Nonnull<Declaration*> member : class_decl.members()) {
        RETURN_IF_ERROR(ResolveNames(*member, class_scope));
      }
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      // Alternative names are never used unqualified, so we don't need to
      // add the alternatives to a scope, or introduce a new scope; we only
      // need to check for duplicates.
      std::set<std::string_view> alternative_names;
      for (Nonnull<AlternativeSignature*> alternative : choice.alternatives()) {
        RETURN_IF_ERROR(
            ResolveNames(alternative->signature(), enclosing_scope));
        if (!alternative_names.insert(alternative->name()).second) {
          return CompilationError(alternative->source_loc())
                 << "Duplicate name `" << alternative->name()
                 << "` in choice type";
        }
      }
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(declaration);
      RETURN_IF_ERROR(ResolveNames(var.binding(), enclosing_scope));
      if (var.has_initializer()) {
        RETURN_IF_ERROR(ResolveNames(var.initializer(), enclosing_scope));
      }
      break;
    }

    case DeclarationKind::SelfDeclaration: {
      FATAL() << "Unreachable: resolving names for `Self` declaration";
    }
  }
  return Success();
}

auto ResolveNames(AST& ast) -> ErrorOr<Success> {
  StaticScope file_scope;
  for (auto declaration : ast.declarations) {
    RETURN_IF_ERROR(AddExposedNames(*declaration, file_scope));
  }
  for (auto declaration : ast.declarations) {
    RETURN_IF_ERROR(ResolveNames(*declaration, file_scope));
  }
  return ResolveNames(**ast.main_call, file_scope);
}

}  // namespace Carbon
