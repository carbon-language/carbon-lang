//===-- nullptr-convert/Matchers.h - Matchers for null casts ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
///  \file
///  \brief This file contains the declarations for matcher-generating functions
///  and names for bound nodes found by AST matchers.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_MATCHERS_H
#define LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_MATCHERS_H

#include "clang/ASTMatchers/ASTMatchers.h"

// Names to bind with matched expressions.
extern const char *ImplicitCastNode;
extern const char *CastSequence;

/// \brief Create a matcher to find the implicit casts clang inserts
/// when writing null to a pointer.
///
/// However, don't match those implicit casts that have explicit casts as
/// an ancestor. Explicit casts are handled by makeCastSequenceMatcher().
clang::ast_matchers::StatementMatcher makeImplicitCastMatcher();

/// \brief Create a matcher that finds the head of a sequence of nested explicit
/// casts that have an implicit cast to null within.
///
/// This matcher is necessary so that an entire sequence of explicit casts can
/// be replaced instead of just the inner-most implicit cast.
clang::ast_matchers::StatementMatcher makeCastSequenceMatcher();

#endif // LLVM_TOOLS_CLANG_TOOLS_EXTRA_CPP11_MIGRATE_USE_NULLPTR_MATCHERS_H
