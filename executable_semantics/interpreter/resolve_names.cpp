// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_names.h"

#include <set>

#include "executable_semantics/ast/declaration.h"
#include "executable_semantics/ast/expression.h"
#include "executable_semantics/ast/member.h"
#include "executable_semantics/ast/pattern.h"
#include "executable_semantics/ast/statement.h"
#include "executable_semantics/ast/static_scope.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

namespace {

// Adds the names exposed by the given AST node to enclosing_scope.
void AddExposedNames(const Declaration& declaration,
                     StaticScope& enclosing_scope);
void AddExposedNames(const Member& member, StaticScope& enclosing_scope);

void AddExposedNames(const Member& member, StaticScope& enclosing_scope) {
  switch (member.kind()) {
    case MemberKind::FieldMember: {
      const auto& field = cast<FieldMember>(member);
      if (field.binding().name().has_value()) {
        enclosing_scope.Add(*field.binding().name(), &field.binding());
      }
      break;
    }
  }
}

void AddExposedNames(const Declaration& declaration,
                     StaticScope& enclosing_scope) {
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration: {
      auto& func = cast<FunctionDeclaration>(declaration);
      enclosing_scope.Add(func.name(), &func);
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      enclosing_scope.Add(class_decl.name(), &class_decl);
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      enclosing_scope.Add(choice.name(), &choice);
      break;
    }
    case DeclarationKind::VariableDeclaration:
      auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name().has_value()) {
        enclosing_scope.Add(*(var.binding().name()), &var.binding());
      }
      return;
  }
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
static void ResolveNames(Expression& expression,
                         const StaticScope& enclosing_scope);
static void ResolveNames(Pattern& pattern, StaticScope& enclosing_scope);
static void ResolveNames(Statement& statement, StaticScope& enclosing_scope);
void ResolveNames(Member& member, StaticScope& enclosing_scope);
void ResolveNames(Declaration& declaration, StaticScope& enclosing_scope);

static void ResolveNames(Expression& expression,
                         const StaticScope& enclosing_scope) {
  switch (expression.kind()) {
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(expression);
      ResolveNames(call.function(), enclosing_scope);
      ResolveNames(call.argument(), enclosing_scope);
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fun_type = cast<FunctionTypeLiteral>(expression);
      ResolveNames(fun_type.parameter(), enclosing_scope);
      ResolveNames(fun_type.return_type(), enclosing_scope);
      break;
    }
    case ExpressionKind::FieldAccessExpression:
      ResolveNames(cast<FieldAccessExpression>(expression).aggregate(),
                   enclosing_scope);
      break;
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(expression);
      ResolveNames(index.aggregate(), enclosing_scope);
      ResolveNames(index.offset(), enclosing_scope);
      break;
    }
    case ExpressionKind::PrimitiveOperatorExpression:
      for (Nonnull<Expression*> operand :
           cast<PrimitiveOperatorExpression>(expression).arguments()) {
        ResolveNames(*operand, enclosing_scope);
      }
      break;
    case ExpressionKind::TupleLiteral:
      for (Nonnull<Expression*> field :
           cast<TupleLiteral>(expression).fields()) {
        ResolveNames(*field, enclosing_scope);
      }
      break;
    case ExpressionKind::StructLiteral:
      for (FieldInitializer& init : cast<StructLiteral>(expression).fields()) {
        ResolveNames(init.expression(), enclosing_scope);
      }
      break;
    case ExpressionKind::StructTypeLiteral:
      for (FieldInitializer& init :
           cast<StructTypeLiteral>(expression).fields()) {
        ResolveNames(init.expression(), enclosing_scope);
      }
      break;
    case ExpressionKind::IdentifierExpression: {
      auto& identifier = cast<IdentifierExpression>(expression);
      identifier.set_named_entity(
          enclosing_scope.Resolve(identifier.name(), identifier.source_loc()));
      break;
    }
    case ExpressionKind::IntrinsicExpression:
      ResolveNames(cast<IntrinsicExpression>(expression).args(),
                   enclosing_scope);
      break;
    case ExpressionKind::BoolTypeLiteral:
    case ExpressionKind::BoolLiteral:
    case ExpressionKind::IntTypeLiteral:
    case ExpressionKind::ContinuationTypeLiteral:
    case ExpressionKind::IntLiteral:
    case ExpressionKind::StringLiteral:
    case ExpressionKind::StringTypeLiteral:
    case ExpressionKind::TypeTypeLiteral:
      break;
    case ExpressionKind::UnimplementedExpression:
      FATAL() << "Unimplemented";
  }
}

static void ResolveNames(Pattern& pattern, StaticScope& enclosing_scope) {
  switch (pattern.kind()) {
    case PatternKind::BindingPattern: {
      auto& binding = cast<BindingPattern>(pattern);
      ResolveNames(binding.type(), enclosing_scope);
      if (binding.name().has_value()) {
        enclosing_scope.Add(*binding.name(), &binding);
      }
      break;
    }
    case PatternKind::TuplePattern:
      for (Nonnull<Pattern*> field : cast<TuplePattern>(pattern).fields()) {
        ResolveNames(*field, enclosing_scope);
      }
      break;
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(pattern);
      ResolveNames(alternative.choice_type(), enclosing_scope);
      ResolveNames(alternative.arguments(), enclosing_scope);
      break;
    }
    case PatternKind::ExpressionPattern:
      ResolveNames(cast<ExpressionPattern>(pattern).expression(),
                   enclosing_scope);
      break;
    case PatternKind::AutoPattern:
      break;
  }
}

static void ResolveNames(Statement& statement, StaticScope& enclosing_scope) {
  switch (statement.kind()) {
    case StatementKind::ExpressionStatement:
      ResolveNames(cast<ExpressionStatement>(statement).expression(),
                   enclosing_scope);
      break;
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(statement);
      ResolveNames(assign.lhs(), enclosing_scope);
      ResolveNames(assign.rhs(), enclosing_scope);
      break;
    }
    case StatementKind::VariableDefinition: {
      auto& def = cast<VariableDefinition>(statement);
      ResolveNames(def.init(), enclosing_scope);
      ResolveNames(def.pattern(), enclosing_scope);
      break;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(statement);
      ResolveNames(if_stmt.condition(), enclosing_scope);
      ResolveNames(if_stmt.then_block(), enclosing_scope);
      if (if_stmt.else_block().has_value()) {
        ResolveNames(**if_stmt.else_block(), enclosing_scope);
      }
      break;
    }
    case StatementKind::Return:
      ResolveNames(cast<Return>(statement).expression(), enclosing_scope);
      break;
    case StatementKind::Block: {
      auto& block = cast<Block>(statement);
      StaticScope block_scope;
      block_scope.AddParent(&enclosing_scope);
      for (Nonnull<Statement*> sub_statement : block.statements()) {
        ResolveNames(*sub_statement, block_scope);
      }
      break;
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(statement);
      ResolveNames(while_stmt.condition(), enclosing_scope);
      ResolveNames(while_stmt.body(), enclosing_scope);
      break;
    }
    case StatementKind::Match: {
      auto& match = cast<Match>(statement);
      ResolveNames(match.expression(), enclosing_scope);
      for (Match::Clause& clause : match.clauses()) {
        StaticScope clause_scope;
        clause_scope.AddParent(&enclosing_scope);
        ResolveNames(clause.pattern(), clause_scope);
        ResolveNames(clause.statement(), clause_scope);
      }
      break;
    }
    case StatementKind::Continuation: {
      auto& continuation = cast<Continuation>(statement);
      enclosing_scope.Add(continuation.continuation_variable(), &continuation);
      StaticScope continuation_scope;
      continuation_scope.AddParent(&enclosing_scope);
      ResolveNames(cast<Continuation>(statement).body(), continuation_scope);
      break;
    }
    case StatementKind::Run:
      ResolveNames(cast<Run>(statement).argument(), enclosing_scope);
      break;
    case StatementKind::Await:
    case StatementKind::Break:
    case StatementKind::Continue:
      break;
  }
}

void ResolveNames(Member& member, StaticScope& enclosing_scope) {
  switch (member.kind()) {
    case MemberKind::FieldMember:
      ResolveNames(cast<FieldMember>(member).binding(), enclosing_scope);
  }
}

void ResolveNames(Declaration& declaration, StaticScope& enclosing_scope) {
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration: {
      auto& function = cast<FunctionDeclaration>(declaration);
      StaticScope function_scope;
      function_scope.AddParent(&enclosing_scope);
      for (Nonnull<GenericBinding*> binding : function.deduced_parameters()) {
        function_scope.Add(binding->name(), binding);
        ResolveNames(binding->type(), function_scope);
      }
      ResolveNames(function.param_pattern(), function_scope);
      if (function.return_term().type_expression().has_value()) {
        ResolveNames(**function.return_term().type_expression(),
                     function_scope);
      }
      if (function.body().has_value()) {
        ResolveNames(**function.body(), function_scope);
      }
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      StaticScope class_scope;
      class_scope.AddParent(&enclosing_scope);
      for (Nonnull<Member*> member : class_decl.members()) {
        AddExposedNames(*member, class_scope);
      }
      for (Nonnull<Member*> member : class_decl.members()) {
        ResolveNames(*member, class_scope);
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
        ResolveNames(alternative->signature(), enclosing_scope);
        if (!alternative_names.insert(alternative->name()).second) {
          FATAL_COMPILATION_ERROR(alternative->source_loc())
              << "Duplicate name `" << alternative->name()
              << "` in choice type";
        }
      }
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(declaration);
      ResolveNames(var.binding(), enclosing_scope);
      ResolveNames(var.initializer(), enclosing_scope);
      break;
    }
  }
}

}  // namespace

void ResolveNames(AST& ast) {
  StaticScope file_scope;
  for (auto declaration : ast.declarations) {
    AddExposedNames(*declaration, file_scope);
  }
  for (auto declaration : ast.declarations) {
    ResolveNames(*declaration, file_scope);
  }
  if (ast.main_call.has_value()) {
    ResolveNames(**ast.main_call, file_scope);
  }
}

}  // namespace Carbon
