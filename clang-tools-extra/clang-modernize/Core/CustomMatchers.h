//===-- Core/CustomMatchers.h - Perf measurement helpers -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides custom matchers to be used by different
/// transforms that requier the same matchers.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_CUSTOMMATCHERS_H
#define CLANG_MODERNIZE_CUSTOMMATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

namespace clang {
namespace ast_matchers {

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
} // namespace ast_matchers
} // namespace clang

#endif // CLANG_MODERNIZE_CUSTOMMATCHERS_H
