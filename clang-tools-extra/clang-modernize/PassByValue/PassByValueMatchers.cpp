//===-- PassByValueMatchers.cpp -------------------------------------------===//
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

#include "PassByValueMatchers.h"

const char *PassByValueCtorId = "Ctor";
const char *PassByValueParamId = "Param";
const char *PassByValueInitializerId = "Initializer";

namespace clang {
namespace ast_matchers {

/// \brief Matches move constructible classes.
///
/// Given
/// \code
///   // POD types are trivially move constructible
///   struct Foo { int a; };
///
///   struct Bar {
///     Bar(Bar &&) = deleted;
///     int a;
///   };
/// \endcode
/// recordDecl(isMoveConstructible())
///   matches "Foo".
AST_MATCHER(CXXRecordDecl, isMoveConstructible) {
  for (CXXRecordDecl::ctor_iterator I = Node.ctor_begin(), E = Node.ctor_end(); I != E; ++I) {
    const CXXConstructorDecl *Ctor = *I;
    if (Ctor->isMoveConstructor() && !Ctor->isDeleted())
      return true;
  }
  return false;
}

/// \brief Matches non-deleted copy constructors.
///
/// Given
/// \code
///   struct Foo { Foo(const Foo &) = default; };
///   struct Bar { Bar(const Bar &) = deleted; };
/// \endcode
/// constructorDecl(isNonDeletedCopyConstructor())
///   matches "Foo(const Foo &)".
AST_MATCHER(CXXConstructorDecl, isNonDeletedCopyConstructor) {
  return Node.isCopyConstructor() && !Node.isDeleted();
}
} // namespace ast_matchers
} // namespace clang

using namespace clang;
using namespace clang::ast_matchers;

DeclarationMatcher makePassByValueCtorParamMatcher() {
  return constructorDecl(
      forEachConstructorInitializer(ctorInitializer(
          // Clang builds a CXXConstructExpr only when it knowns which
          // constructor will be called. In dependent contexts a ParenListExpr
          // is generated instead of a CXXConstructExpr, filtering out templates
          // automatically for us.
          withInitializer(constructExpr(
              has(declRefExpr(to(parmVarDecl().bind(PassByValueParamId)))),
              hasDeclaration(constructorDecl(
                  isNonDeletedCopyConstructor(),
                  hasDeclContext(recordDecl(isMoveConstructible())))))))
                                        .bind(PassByValueInitializerId)))
      .bind(PassByValueCtorId);
}
