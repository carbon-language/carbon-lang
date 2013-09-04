//===-- ReplaceAutoPtrMatchers.h ---- std::auto_ptr replacement -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the declarations for matcher-generating functions
/// and names for bound nodes found by AST matchers.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_REPLACE_AUTO_PTR_MATCHERS_H
#define CPP11_MIGRATE_REPLACE_AUTO_PTR_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

/// Names to bind with matched expressions.
extern const char *AutoPtrTokenId;
extern const char *AutoPtrOwnershipTransferId;

/// \brief Creates a matcher that finds the locations of types referring to the
/// \c std::auto_ptr() type.
///
/// \code
///   std::auto_ptr<int> a;
///        ^~~~~~~~~~~~~
///
///   typedef std::auto_ptr<int> int_ptr_t;
///                ^~~~~~~~~~~~~
///
///   std::auto_ptr<int> fn(std::auto_ptr<int>);
///        ^~~~~~~~~~~~~         ^~~~~~~~~~~~~
///
///   <etc...>
/// \endcode
clang::ast_matchers::TypeLocMatcher makeAutoPtrTypeLocMatcher();

/// \brief Creates a matcher that finds the using declarations referring to
/// \c std::auto_ptr.
///
/// \code
///   using std::auto_ptr;
///   ^~~~~~~~~~~~~~~~~~~
/// \endcode
clang::ast_matchers::DeclarationMatcher makeAutoPtrUsingDeclMatcher();

/// \brief Creates a matcher that finds the \c std::auto_ptr copy-ctor and
/// assign-operator expressions.
///
/// \c AutoPtrOwnershipTransferId is assigned to the argument of the expression,
/// this is the part that has to be wrapped by \c std::move().
///
/// \code
///   std::auto_ptr<int> i, j;
///   i = j;
///   ~~~~^
/// \endcode
clang::ast_matchers::StatementMatcher makeTransferOwnershipExprMatcher();

#endif // CPP11_MIGRATE_REPLACE_AUTO_PTR_MATCHERS_H
