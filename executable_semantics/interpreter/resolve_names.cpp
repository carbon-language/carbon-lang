// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_names.h"

#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

namespace {

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
      // Defines a new name.
      auto& cont = cast<Continuation>(statement);
      scoped_names.Add(cont.continuation_variable(), &cont);
      break;
    }
    case Statement::Kind::VariableDefinition: {
      // Defines a new name.
      auto& var = cast<VariableDefinition>(statement);
      scoped_names.Add(var.pattern()
      break;
    }
    case Statement::Kind::If: {
      // Contains statements that need to be populated.
      auto& if_stmt = cast<If>(statement);
      PopulateNamesInStatement(arena, &if_stmt.then_block(), scoped_names);
      PopulateNamesInStatement(arena, if_stmt.else_block(), scoped_names);
      break;
    }
    case Statement::Kind::While: {
      // Contains statements that need to be populated.
      auto& while_stmt = cast<While>(statement);
      PopulateNamesInStatement(arena, &while_stmt.body(), scoped_names);
      break;
    }
    case Statement::Kind::Match: {
      // Contains statements that need to be populated.
      // TODO
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

// Populates declared names at scoped boundaries, such as file-level or function
// bodies.
// This doesn't currently recurse into expressions, but likely will in the
// future in order to resolve names in lambdas.
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
    case Declaration::Kind::ClassDeclaration:
      // Populate name into declared_names.
      // Init the class's declared_names, and populate it with members.
      // Recurse into Members with a switch, but because there are only fields
      // at present, there's nothing more to do. Eventually functions and
      // similar will provide their own scopes to recurse into.
      break;
    case Declaration::Kind::ChoiceDeclaration:
      // Populate name into declared_names.
      // Init the choice's declared_names, and populate it with the
      // alternatives.
      break;
    case Declaration::Kind::VariableDeclaration:
      auto& var = cast<VariableDeclaration>(declaration);
      if (var.binding().name().has_value()) {
        scoped_names.Add(*(var.binding().name()), &declaration);
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
