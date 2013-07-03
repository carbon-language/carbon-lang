//===-- ReplaceAutoPtrMatchers.cpp - std::auto_ptr replacement -*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the definitions for matcher-generating functions
/// and names for bound nodes found by AST matchers.
///
//===----------------------------------------------------------------------===//

#include "ReplaceAutoPtrMatchers.h"

const char *AutoPtrTokenId = "AutoPtrTokenId";
const char *AutoPtrOwnershipTransferId = "AutoPtrOwnershipTransferId";

namespace clang {
namespace ast_matchers {

/// \brief Matches expressions that are lvalues.
///
/// In the following example, a[0] matches expr(isLValue()):
/// \code
///     std::string a[2];
///     std::string b;
///     b = a[0];
///     b = "this string won't match";
/// \endcode
AST_MATCHER(Expr, isLValue) {
  return Node.getValueKind() == VK_LValue;
}

/// \brief Matches declarations whose declaration context is the C++ standard
/// library namespace \c std.
///
/// Note that inline namespaces are silently ignored during the lookup since
/// both libstdc++ and libc++ are known to use them for versioning purposes.
///
/// Given
/// \code
///   namespace ns {
///     struct my_type {};
///     using namespace std;
///   }
///
///   using std::vector;
///   using ns::my_type;
///   using ns::list;
/// \endcode
/// usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(isFromStdNamespace())))
///   matches "using std::vector" and "using ns::list".
AST_MATCHER(Decl, isFromStdNamespace) {
  const DeclContext *D = Node.getDeclContext();

  while (D->isInlineNamespace())
    D = D->getParent();

  if (!D->isNamespace() || !D->getParent()->isTranslationUnit())
    return false;

  const IdentifierInfo *Info = cast<NamespaceDecl>(D)->getIdentifier();

  return Info && Info->isStr("std");
}

} // end namespace ast_matchers
} // end namespace clang

using namespace clang;
using namespace clang::ast_matchers;

// shared matchers
static DeclarationMatcher AutoPtrDecl =
    recordDecl(hasName("auto_ptr"), isFromStdNamespace());

static TypeMatcher AutoPtrType = qualType(hasDeclaration(AutoPtrDecl));

// Matcher that finds expressions that are candidates to be wrapped with
// 'std::move()'.
//
// Binds the id \c AutoPtrOwnershipTransferId to the expression.
static StatementMatcher MovableArgumentMatcher = expr(
    allOf(isLValue(), hasType(AutoPtrType))).bind(AutoPtrOwnershipTransferId);

TypeLocMatcher makeAutoPtrTypeLocMatcher() {
  // skip elaboratedType() as the named type will match soon thereafter.
  return typeLoc(loc(qualType(AutoPtrType, unless(elaboratedType()))))
      .bind(AutoPtrTokenId);
}

DeclarationMatcher makeAutoPtrUsingDeclMatcher() {
  return usingDecl(hasAnyUsingShadowDecl(hasTargetDecl(
      allOf(hasName("auto_ptr"), isFromStdNamespace())))).bind(AutoPtrTokenId);
}

StatementMatcher makeTransferOwnershipExprMatcher() {
  StatementMatcher assignOperator =
    operatorCallExpr(allOf(
      hasOverloadedOperatorName("="),
      callee(methodDecl(ofClass(AutoPtrDecl))),
      hasArgument(1, MovableArgumentMatcher)));

  StatementMatcher copyCtor =
    constructExpr(allOf(hasType(AutoPtrType),
                        argumentCountIs(1),
                        hasArgument(0, MovableArgumentMatcher)));

  return anyOf(assignOperator, copyCtor);
}
