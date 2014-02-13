//===- unittests/libclang/LibclangTest.cpp --- libclang tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "gtest/gtest.h"

TEST(libclang, TestInvalidArgs) {
  EXPECT_EQ(CXError_InvalidArguments,
            clang_parseTranslationUnit2(0, 0, 0, 0, 0, 0, 0, 0));
}

