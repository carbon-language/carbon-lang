// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/interpreter/resolve_names.h"

#include "llvm/Support/Casting.h"

using llvm::cast;

namespace Carbon {

// TODO: PopulateNames for Statement will be needed for recursion.

// Populates declared names at scoped boundaries, such as file-level or function
// bodies.
// This doesn't currently recurse into expressions, but likely will in the
// future in order to resolve names in lambdas.
void PopulateNamesInDeclaration(Declaration& declaration,
                                ScopedNames& scoped_names) {
  switch (declaration.kind()) {
    case Declaration::Kind::FunctionDeclaration:
      // Populate name into declared_names.
      // params
      break;
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
      // Populate name into declared_names.
      // No need to recurse (maybe in the future when expressions may contain
      // lambdas, but not right now).
      break;
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

void ResolveNames(AST& ast) {
  for (auto declaration : ast.declarations) {
    PopulateNamesInDeclaration(*declaration, ast.scoped_names);
  }
  for (auto declaration : ast.declarations) {
    ResolveNamesInDeclaration(*declaration, ast.scoped_names);
  }
}

}  // namespace Carbon
