//===--- Statistics.h - Helpers for Clang AST Statistics --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides helper classes, functions, and macros for tracking
//  various statistics about the Clang AST and its usage.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_STATISTICS_H
#define LLVM_CLANG_AST_STATISTICS_H

#ifndef NDEBUG
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Casting.h"

/** \brief Tracks the number of time the \c isa() function template is
 * used to try to cast to the given \c Type, by bumping the \c Counter.
 *
 * Note that this macro must be expanded in the global scope, and that
 * both the type and the counter will be assumed to reside within the
 * \c clang namespace.
 */
#define CLANG_ISA_STATISTIC(Type,Counter)       \
namespace llvm {                                \
template <typename From>                        \
struct isa_impl<clang::Type, From> {            \
  static inline bool doit(const From &Val) {    \
    ++clang::Counter;                           \
    return clang::Type::classof(&Val);          \
  }                                             \
};                                              \
}

#else
#define CLANG_ISA_STATISTIC(Type,Counter)
#endif

#endif // LLVM_CLANG_AST_STATISTICS_H
