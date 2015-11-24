//===--- DataRecursiveASTVisitor.h - Data-Recursive AST Visitor -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides a legacy name for the RecursiveASTVisitor.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_AST_DATARECURSIVEASTVISITOR_H
#define LLVM_CLANG_AST_DATARECURSIVEASTVISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
template<typename T> struct DataRecursiveASTVisitor : RecursiveASTVisitor<T> {};
} // end namespace clang

#endif // LLVM_CLANG_LIBCLANG_RECURSIVEASTVISITOR_H
