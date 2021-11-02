// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_names.h"

#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

namespace {

// Populates names for a pattern. See PopulateNamesInDeclaration for overall
// flow.
void PopulateNamesInPattern(const Pattern& pattern, StaticScope& static_scope) {
  switch (pattern.kind()) {
    case Pattern::Kind::AlternativePattern: {
      const auto& alt = cast<AlternativePattern>(pattern);
      PopulateNamesInPattern(alt.arguments(), static_scope);
      break;
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      if (binding.name().has_value()) {
        static_scope.Add(*binding.name(), &binding);
      }
      break;
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(pattern);
      for (auto* field : tuple.fields()) {
        PopulateNamesInPattern(*field, static_scope);
      }
      break;
    }
    case Pattern::Kind::AutoPattern:
    case Pattern::Kind::ExpressionPattern:
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
    case Statement::Kind::Block: {
      // Defines a new scope for names.
      auto& block = cast<Block>(statement);
      block.set_static_scope(arena->New<StaticScope>());
      for (const auto& statement : block.statements()) {
        PopulateNamesInStatement(arena, statement, block.static_scope());
      }
      break;
    }
    case Statement::Kind::Continuation: {
      // Defines a new name and contains a block.
      auto& cont = cast<Continuation>(statement);
      static_scope.Add(cont.continuation_variable(), &cont);
      PopulateNamesInStatement(arena, &cont.body(), static_scope);
      break;
    }
    case Statement::Kind::VariableDefinition: {
      // Defines a new name.
      const auto& var = cast<VariableDefinition>(statement);
      PopulateNamesInPattern(var.pattern(), static_scope);
      break;
    }
    case Statement::Kind::If: {
      // Contains blocks.
      auto& if_stmt = cast<If>(statement);
      PopulateNamesInStatement(arena, &if_stmt.then_block(), static_scope);
      PopulateNamesInStatement(arena, if_stmt.else_block(), static_scope);
      break;
    }
    case Statement::Kind::While: {
      // Contains a block.
      auto& while_stmt = cast<While>(statement);
      PopulateNamesInStatement(arena, &while_stmt.body(), static_scope);
      break;
    }
    case Statement::Kind::Match: {
      // Contains blocks.
      auto& match = cast<Match>(statement);
      for (auto& clause : match.clauses()) {
        clause.set_static_scope(arena->New<StaticScope>());
        PopulateNamesInPattern(clause.pattern(), clause.static_scope());
        PopulateNamesInStatement(arena, &clause.statement(),
                                 clause.static_scope());
      }
      break;
    }
    case Statement::Kind::Assign:
    case Statement::Kind::Await:
    case Statement::Kind::Break:
    case Statement::Kind::Continue:
    case Statement::Kind::ExpressionStatement:
    case Statement::Kind::Return:
    case Statement::Kind::Run:
      // Neither contains names nor a scope.
      break;
  }
}

// Populates names for a member. See PopulateNamesInDeclaration for overall
// flow.
void PopulateNamesInMember(Arena* arena, const Member& member,
                           StaticScope& static_scope) {
  switch (member.kind()) {
    case Member::Kind::FieldMember: {
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
    case Declaration::Kind::FunctionDeclaration: {
      auto& func = cast<FunctionDeclaration>(declaration);
      static_scope.Add(func.name(), &declaration);
      func.set_static_scope(arena->New<StaticScope>());
      for (const auto& param : func.deduced_parameters()) {
        func.static_scope().Add(param.name(), &param);
      }
      PopulateNamesInPattern(func.param_pattern(), func.static_scope());
      PopulateNamesInStatement(arena, func.body(), static_scope);
      break;
    }
    case Declaration::Kind::ClassDeclaration: {
      auto& class_def = cast<ClassDeclaration>(declaration).definition();
      static_scope.Add(class_def.name(), &declaration);
      class_def.set_static_scope(arena->New<StaticScope>());
      for (auto* member : class_def.members()) {
        PopulateNamesInMember(arena, *member, class_def.static_scope());
      }
      break;
    }
    case Declaration::Kind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      static_scope.Add(choice.name(), &declaration);
      choice.set_static_scope(arena->New<StaticScope>());
      for (auto& alt : choice.alternatives()) {
        choice.static_scope().Add(alt.name(), &alt);
      }
      // Populate name into declared_names.
      // Init the choice's declared_names, and populate it with the
      // alternatives.
      break;
    }
    case Declaration::Kind::VariableDeclaration:
      auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name().has_value()) {
        static_scope.Add(*(var.binding().name()), &var.binding());
      }
      return;
  }
}

// TODO: ResolveNames for Expression, Member, Pattern, and Statement will be
// needed for recursion.

// Recurses through a declaration to find and resolve IdentifierExpressions
// using declared_names.
void ResolveNamesInDeclaration(Declaration& declaration,
                               const StaticScope& static_scope) {
  switch (declaration.kind()) {
    case Declaration::Kind::FunctionDeclaration:
    case Declaration::Kind::ClassDeclaration:
    case Declaration::Kind::ChoiceDeclaration:
    case Declaration::Kind::VariableDeclaration:
      break;
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
}

}  // namespace Carbon
