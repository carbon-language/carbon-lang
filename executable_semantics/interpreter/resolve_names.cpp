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
void PopulateNamesInPattern(const Pattern& pattern, ScopedNames& scoped_names) {
  switch (pattern.kind()) {
    case Pattern::Kind::AlternativePattern: {
      const auto& alt = cast<AlternativePattern>(pattern);
      PopulateNamesInPattern(alt.arguments(), scoped_names);
      break;
    }
    case Pattern::Kind::BindingPattern: {
      const auto& binding = cast<BindingPattern>(pattern);
      if (binding.name().has_value()) {
        scoped_names.Add(*binding.name(), &binding);
      }
      break;
    }
    case Pattern::Kind::TuplePattern: {
      const auto& tuple = cast<TuplePattern>(pattern);
      for (auto* field : tuple.fields()) {
        PopulateNamesInPattern(*field, scoped_names);
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
                              ScopedNames& scoped_names) {
  if (!opt_statement.has_value()) {
    return;
  }
  Statement& statement = **opt_statement;
  switch (statement.kind()) {
    case Statement::Kind::Block: {
      // Defines a new scope for names.
      auto& block = cast<Block>(statement);
      block.set_scoped_names(arena->New<ScopedNames>());
      PopulateNamesInStatement(arena, block.sequence(), block.scoped_names());
      break;
    }
    case Statement::Kind::Sequence: {
      // Hopefully collapse into Block.
      auto& seq = cast<Sequence>(statement);
      PopulateNamesInStatement(arena, &seq.statement(), scoped_names);
      PopulateNamesInStatement(arena, seq.next(), scoped_names);
      break;
    }
    case Statement::Kind::Continuation: {
      // Defines a new name and contains a block.
      auto& cont = cast<Continuation>(statement);
      scoped_names.Add(cont.continuation_variable(), &cont);
      PopulateNamesInStatement(arena, &cont.body(), scoped_names);
      break;
    }
    case Statement::Kind::VariableDefinition: {
      // Defines a new name.
      const auto& var = cast<VariableDefinition>(statement);
      PopulateNamesInPattern(var.pattern(), scoped_names);
      break;
    }
    case Statement::Kind::If: {
      // Contains blocks.
      auto& if_stmt = cast<If>(statement);
      PopulateNamesInStatement(arena, &if_stmt.then_block(), scoped_names);
      PopulateNamesInStatement(arena, if_stmt.else_block(), scoped_names);
      break;
    }
    case Statement::Kind::While: {
      // Contains a block.
      auto& while_stmt = cast<While>(statement);
      PopulateNamesInStatement(arena, &while_stmt.body(), scoped_names);
      break;
    }
    case Statement::Kind::Match: {
      // Contains blocks.
      auto& match = cast<Match>(statement);
      for (auto& clause : match.clauses()) {
        clause.set_scoped_names(arena->New<ScopedNames>());
        PopulateNamesInPattern(clause.pattern(), clause.scoped_names());
        PopulateNamesInStatement(arena, &clause.statement(),
                                 clause.scoped_names());
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
                           ScopedNames& scoped_names) {
  switch (member.kind()) {
    case Member::Kind::FieldMember: {
      const auto& field = cast<FieldMember>(member);
      if (field.binding().name().has_value()) {
        scoped_names.Add(*field.binding().name(), &member);
      }
      break;
    }
  }
}

// Populates declared names at scoped boundaries, such as file-level or
// function bodies. This doesn't currently recurse into expressions, but
// likely will in the future in order to resolve names in lambdas.
void PopulateNamesInDeclaration(Arena* arena, Declaration& declaration,
                                ScopedNames& scoped_names) {
  switch (declaration.kind()) {
    case Declaration::Kind::FunctionDeclaration: {
      auto& func = cast<FunctionDeclaration>(declaration);
      scoped_names.Add(func.name(), &declaration);
      // TODO: Add function parameters to func.scoped_names.
      PopulateNamesInStatement(arena, func.body(), scoped_names);
      break;
    }
    case Declaration::Kind::ClassDeclaration: {
      auto& class_def = cast<ClassDeclaration>(declaration).definition();
      scoped_names.Add(class_def.name(), &declaration);
      class_def.set_scoped_names(arena->New<ScopedNames>());
      for (auto* member : class_def.members()) {
        PopulateNamesInMember(arena, *member, class_def.scoped_names());
      }
      break;
    }
    case Declaration::Kind::ChoiceDeclaration: {
      auto& choice = cast<ChoiceDeclaration>(declaration);
      scoped_names.Add(choice.name(), &declaration);
      choice.set_scoped_names(arena->New<ScopedNames>());
      for (auto& alt : choice.alternatives()) {
        choice.scoped_names().Add(alt.name(), &alt);
      }
      // Populate name into declared_names.
      // Init the choice's declared_names, and populate it with the
      // alternatives.
      break;
    }
    case Declaration::Kind::VariableDeclaration:
      auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name().has_value()) {
        scoped_names.Add(*(var.binding().name()), &var.binding());
      }
      return;
  }
}

// TODO: ResolveNames for Expression, Member, Pattern, and Statement will be
// needed for recursion.

// Recurses through a declaration to find and resolve IdentifierExpressions
// using declared_names.
void ResolveNamesInDeclaration(Declaration& declaration,
                               const ScopedNames& scoped_names) {
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
    PopulateNamesInDeclaration(arena, *declaration, ast.scoped_names);
  }
  for (auto declaration : ast.declarations) {
    ResolveNamesInDeclaration(*declaration, ast.scoped_names);
  }
}

}  // namespace Carbon
