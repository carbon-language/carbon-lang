//===- unittests/AST/DeclTest.cpp --- Declaration tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for the ASTVector container.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"
#include "clang/AST/ASTVector.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "gtest/gtest.h"

using namespace clang;

LLVM_ATTRIBUTE_UNUSED void CompileTest() {
  ASTContext *C = 0;
  ASTVector<int> V;
  V.insert(*C, V.begin(), 0);
}
