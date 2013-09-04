//===-- PassByValueMatchers.h -----------------------------------*- C++ -*-===//
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

#ifndef CLANG_MODERNIZE_REPLACE_AUTO_PTR_MATCHERS_H
#define CLANG_MODERNIZE_REPLACE_AUTO_PTR_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

/// \name Names to bind with matched expressions
/// @{
extern const char *PassByValueCtorId;
extern const char *PassByValueParamId;
extern const char *PassByValueInitializerId;
/// @}

/// \brief Creates a matcher that finds class field initializations that can
/// benefit from using the move constructor.
///
/// \code
///   class A {
///   public:
///    A(const std::string &S) : S(S) {}
///    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PassByValueCtorId
///      ~~~~~~~~~~~~~~~~~~~~ PassByValueParamId
///                                ~ PassByValueInitializerId
///   private:
///    std::string S;
///  };
/// \endcode
clang::ast_matchers::DeclarationMatcher makePassByValueCtorParamMatcher();

#endif // CLANG_MODERNIZE_REPLACE_AUTO_PTR_MATCHERS_H
