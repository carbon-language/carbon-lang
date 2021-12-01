// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_names.h"

#include "executable_semantics/ast/declaration.h"
#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

namespace {

// Populates names for a pattern. See PopulateNamesInDeclaration for overall
// flow.
void PopulateNamesInPattern(const Pattern& pattern, StaticScope& static_scope) {
  switch (pattern.kind()) {
    case PatternKind::AlternativePattern: {
      const auto& alt = cast<AlternativePattern>(pattern);
      PopulateNamesInPattern(alt.arguments(), static_scope);
      break;
    }
    case PatternKind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      if (binding.name().has_value()) {
        static_scope.Add(*binding.name(), &binding);
      }
      break;
    }
    case PatternKind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(pattern);
      for (auto* field : tuple.fields()) {
        PopulateNamesInPattern(*field, static_scope);
      }
      break;
    }
    case PatternKind::AutoPattern:
    case PatternKind::ExpressionPattern:
      // These don't add names.
      break;
  }
}

// Populates names for a statement. See PopulateNamesInDeclaration for overall
// flow.
void PopulateNamesInStatement(Arena* arena,
                              std::optional<Nonnull<Statement*>> opt_statement,
                              StaticScope& static_scope) {
  if (!opt_statement.has_value()) {
    return;
  }
  Statement& statement = **opt_statement;
  switch (statement.kind()) {
    case StatementKind::Block: {
      // Defines a new scope for names.
      auto& block = cast<Block>(statement);
      block.static_scope().AddParent(&static_scope);
      for (const auto& statement : block.statements()) {
        PopulateNamesInStatement(arena, statement, block.static_scope());
      }
      break;
    }
    case StatementKind::Continuation: {
      // Defines a new name and contains a block.
      auto& cont = cast<Continuation>(statement);
      static_scope.Add(cont.continuation_variable(), &cont);
      PopulateNamesInStatement(arena, &cont.body(), static_scope);
      break;
    }
    case StatementKind::VariableDefinition: {
      // Defines a new name.
      const auto& var = cast<VariableDefinition>(statement);
      PopulateNamesInPattern(var.pattern(), static_scope);
      break;
    }
    case StatementKind::If: {
      // Contains blocks.
      auto& if_stmt = cast<If>(statement);
      PopulateNamesInStatement(arena, &if_stmt.then_block(), static_scope);
      PopulateNamesInStatement(arena, if_stmt.else_block(), static_scope);
      break;
    }
    case StatementKind::While: {
      // Contains a block.
      auto& while_stmt = cast<While>(statement);
      PopulateNamesInStatement(arena, &while_stmt.body(), static_scope);
      break;
    }
    case StatementKind::Match: {
      // Contains blocks.
      auto& match = cast<Match>(statement);
      for (auto& clause : match.clauses()) {
        clause.static_scope().AddParent(&static_scope);
        PopulateNamesInPattern(clause.pattern(), clause.static_scope());
        PopulateNamesInStatement(arena, &clause.statement(),
                                 clause.static_scope());
      }
      break;
    }
    case StatementKind::Assign:
    case StatementKind::Await:
    case StatementKind::Break:
    case StatementKind::Continue:
    case StatementKind::ExpressionStatement:
    case StatementKind::Return:
    case StatementKind::Run:
      // Neither contains names nor a scope.
      break;
  }
}

// Populates names for a member. See PopulateNamesInDeclaration for overall
// flow.
void PopulateNamesInMember(Arena* arena, const Member& member,
                           StaticScope& static_scope) {
  switch (member.kind()) {
    case MemberKind::FieldMember: {
      const auto& field = cast<FieldMember>(member);
      if (field.binding().name().has_value()) {
        static_scope.Add(*field.binding().name(), &member);
      }
      break;
    }
  }
}

// Populates declared names at scoped boundaries, such as file-level or
// function bodies. This doesn't currently recurse into expressions, but
// likely will in the future in order to resolve names in lambdas.
void PopulateNamesInDeclaration(Arena* arena, Declaration& declaration,
                                StaticScope& static_scope) {
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration: {
      auto& func = cast<FunctionDeclaration>(declaration);
      func.static_scope().AddParent(&static_scope);
      static_scope.Add(func.name(), &declaration);
      for (Nonnull<const GenericBinding*> param : func.deduced_parameters()) {
        func.static_scope().Add(param->name(), param);
      }
      PopulateNamesInPattern(func.param_pattern(), func.static_scope());
      PopulateNamesInStatement(arena, func.body(), func.static_scope());
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      class_decl.static_scope().AddParent(&static_scope);
      static_scope.Add(class_decl.name(), &declaration);
      for (auto* member : class_decl.members()) {
        PopulateNamesInMember(arena, *member, class_decl.static_scope());
      }
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      choice.static_scope().AddParent(&static_scope);
      static_scope.Add(choice.name(), &declaration);
      for (Nonnull<const AlternativeSignature*> alt : choice.alternatives()) {
        choice.static_scope().Add(alt->name(), alt);
      }
      // Populate name into declared_names.
      // Init the choice's declared_names, and populate it with the
      // alternatives.
      break;
    }
    case DeclarationKind::VariableDeclaration:
      auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name().has_value()) {
        static_scope.Add(*(var.binding().name()), &var.binding());
      }
      return;
  }
}

// Populates the named_entity member of all IdentifierExpressions in the
// subtree rooted at `expression`, whose nearest enclosing scope is
// `enclosing_scope`.
static void ResolveNamesInExpression(Expression& expression,
                                     const StaticScope& enclosing_scope) {
  switch (expression.kind()) {
    case ExpressionKind::CallExpression: {
      auto& call = cast<CallExpression>(expression);
      ResolveNamesInExpression(call.function(), enclosing_scope);
      ResolveNamesInExpression(call.argument(), enclosing_scope);
      break;
    }
    case ExpressionKind::FunctionTypeLiteral: {
      auto& fun_type = cast<FunctionTypeLiteral>(expression);
      ResolveNamesInExpression(fun_type.parameter(), enclosing_scope);
      ResolveNamesInExpression(fun_type.return_type(), enclosing_scope);
      break;
    }
    case ExpressionKind::FieldAccessExpression:
      ResolveNamesInExpression(
          cast<FieldAccessExpression>(expression).aggregate(), enclosing_scope);
      break;
    case ExpressionKind::IndexExpression: {
      auto& index = cast<IndexExpression>(expression);
      ResolveNamesInExpression(index.aggregate(), enclosing_scope);
      ResolveNamesInExpression(index.offset(), enclosing_scope);
      break;
    }
    case ExpressionKind::PrimitiveOperatorExpression:
      for (Nonnull<Expression*> operand :
           cast<PrimitiveOperatorExpression>(expression).arguments()) {
        ResolveNamesInExpression(*operand, enclosing_scope);
      }
      break;
    case ExpressionKind::TupleLiteral:
      for (Nonnull<Expression*> field :
           cast<TupleLiteral>(expression).fields()) {
        ResolveNamesInExpression(*field, enclosing_scope);
      }
      break;
    case ExpressionKind::StructLiteral:
      for (FieldInitializer& init : cast<StructLiteral>(expression).fields()) {
        ResolveNamesInExpression(init.expression(), enclosing_scope);
      }
      break;
    case ExpressionKind::StructTypeLiteral:
      for (FieldInitializer& init :
           cast<StructTypeLiteral>(expression).fields()) {
        ResolveNamesInExpression(init.expression(), enclosing_scope);
      }
      break;
    case ExpressionKind::IdentifierExpression: {
      auto& identifier = cast<IdentifierExpression>(expression);
      identifier.set_named_entity(
          enclosing_scope.Resolve(identifier.name(), identifier.source_loc()));
      break;
    }
    case ExpressionKind::IntrinsicExpression:
      ResolveNamesInExpression(cast<IntrinsicExpression>(expression).args(),
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

// Equivalent to ResolveNamesInExpression, but operates on patterns.
static void ResolveNamesInPattern(Pattern& pattern,
                                  const StaticScope& enclosing_scope) {
  switch (pattern.kind()) {
    case PatternKind::BindingPattern:
      ResolveNamesInPattern(cast<BindingPattern>(pattern).type(),
                            enclosing_scope);
      break;
    case PatternKind::TuplePattern:
      for (Nonnull<Pattern*> field : cast<TuplePattern>(pattern).fields()) {
        ResolveNamesInPattern(*field, enclosing_scope);
      }
      break;
    case PatternKind::AlternativePattern: {
      auto& alternative = cast<AlternativePattern>(pattern);
      ResolveNamesInExpression(alternative.choice_type(), enclosing_scope);
      ResolveNamesInPattern(alternative.arguments(), enclosing_scope);
      break;
    }
    case PatternKind::ExpressionPattern:
      ResolveNamesInExpression(cast<ExpressionPattern>(pattern).expression(),
                               enclosing_scope);
      break;
    case PatternKind::AutoPattern:
      break;
  }
}

// Equivalent to ResolveNamesInExpression, but operates on statements.
static void ResolveNamesInStatement(Statement& statement,
                                    const StaticScope& enclosing_scope) {
  switch (statement.kind()) {
    case StatementKind::ExpressionStatement:
      ResolveNamesInExpression(
          cast<ExpressionStatement>(statement).expression(), enclosing_scope);
      break;
    case StatementKind::Assign: {
      auto& assign = cast<Assign>(statement);
      ResolveNamesInExpression(assign.lhs(), enclosing_scope);
      ResolveNamesInExpression(assign.rhs(), enclosing_scope);
      break;
    }
    case StatementKind::VariableDefinition: {
      auto& def = cast<VariableDefinition>(statement);
      ResolveNamesInPattern(def.pattern(), enclosing_scope);
      ResolveNamesInExpression(def.init(), enclosing_scope);
      break;
    }
    case StatementKind::If: {
      auto& if_stmt = cast<If>(statement);
      ResolveNamesInExpression(if_stmt.condition(), enclosing_scope);
      ResolveNamesInStatement(if_stmt.then_block(), enclosing_scope);
      if (if_stmt.else_block().has_value()) {
        ResolveNamesInStatement(**if_stmt.else_block(), enclosing_scope);
      }
      break;
    }
    case StatementKind::Return:
      ResolveNamesInExpression(cast<Return>(statement).expression(),
                               enclosing_scope);
      break;
    case StatementKind::Block: {
      // TODO: this will resolve usages to declarations that occur later in
      //   the block. Figure out how to avoid that.
      auto& block = cast<Block>(statement);
      for (Nonnull<Statement*> sub_statement : block.statements()) {
        ResolveNamesInStatement(*sub_statement, block.static_scope());
      }
      break;
    }
    case StatementKind::While: {
      auto& while_stmt = cast<While>(statement);
      ResolveNamesInExpression(while_stmt.condition(), enclosing_scope);
      ResolveNamesInStatement(while_stmt.body(), enclosing_scope);
      break;
    }
    case StatementKind::Match: {
      auto& match = cast<Match>(statement);
      ResolveNamesInExpression(match.expression(), enclosing_scope);
      for (Match::Clause& clause : match.clauses()) {
        ResolveNamesInPattern(clause.pattern(), clause.static_scope());
        ResolveNamesInStatement(clause.statement(), clause.static_scope());
      }
      break;
    }
    case StatementKind::Continuation:
      ResolveNamesInStatement(cast<Continuation>(statement).body(),
                              enclosing_scope);
      break;
    case StatementKind::Run:
      ResolveNamesInExpression(cast<Run>(statement).argument(),
                               enclosing_scope);
      break;
    case StatementKind::Await:
    case StatementKind::Break:
    case StatementKind::Continue:
      break;
  }
}

// Recurses through a declaration to find and resolve IdentifierExpressions
// using declared_names.
void ResolveNamesInDeclaration(Declaration& declaration,
                               const StaticScope& enclosing_scope) {
  switch (declaration.kind()) {
    case DeclarationKind::FunctionDeclaration: {
      auto& function = cast<FunctionDeclaration>(declaration);
      for (Nonnull<GenericBinding*> binding : function.deduced_parameters()) {
        ResolveNamesInExpression(binding->type(), function.static_scope());
      }
      ResolveNamesInPattern(function.param_pattern(), function.static_scope());
      if (function.return_term().type_expression().has_value()) {
        ResolveNamesInExpression(**function.return_term().type_expression(),
                                 function.static_scope());
      }
      if (function.body().has_value()) {
        ResolveNamesInStatement(**function.body(), function.static_scope());
      }
      break;
    }
    case DeclarationKind::ClassDeclaration: {
      auto& class_decl = cast<ClassDeclaration>(declaration);
      for (Nonnull<Member*> member : class_decl.members()) {
        switch (member->kind()) {
          case MemberKind::FieldMember:
            ResolveNamesInPattern(cast<FieldMember>(member)->binding(),
                                  class_decl.static_scope());
        }
      }
      break;
    }
    case DeclarationKind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      for (Nonnull<AlternativeSignature*> alternative : choice.alternatives()) {
        ResolveNamesInExpression(alternative->signature(),
                                 choice.static_scope());
      }
      break;
    }
    case DeclarationKind::VariableDeclaration: {
      auto& var = cast<VariableDeclaration>(declaration);
      ResolveNamesInPattern(var.binding(), enclosing_scope);
      ResolveNamesInExpression(var.initializer(), enclosing_scope);
      break;
    }
  }
}

}  // namespace

void ResolveNames(Arena* arena, AST& ast) {
  for (auto declaration : ast.declarations) {
    PopulateNamesInDeclaration(arena, *declaration, ast.static_scope);
  }
  for (auto declaration : ast.declarations) {
    ResolveNamesInDeclaration(*declaration, ast.static_scope);
  }
  if (ast.main_call.has_value()) {
    ResolveNamesInExpression(**ast.main_call, ast.static_scope);
  }
}

}  // namespace Carbon
