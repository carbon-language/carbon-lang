//===- llvm/unittest/IR/ValueTest.cpp - Value unit tests ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(ValueTest, UsedInBasicBlock) {
  LLVMContext C;

  const char *ModuleString = "define void @f(i32 %x, i32 %y) {\n"
                             "bb0:\n"
                             "  %y1 = add i32 %y, 1\n"
                             "  %y2 = add i32 %y, 1\n"
                             "  %y3 = add i32 %y, 1\n"
                             "  %y4 = add i32 %y, 1\n"
                             "  %y5 = add i32 %y, 1\n"
                             "  %y6 = add i32 %y, 1\n"
                             "  %y7 = add i32 %y, 1\n"
                             "  %y8 = add i32 %x, 1\n"
                             "  ret void\n"
                             "}\n";
  SMDiagnostic Err;
  Module *M = ParseAssemblyString(ModuleString, NULL, Err, C);

  Function *F = M->getFunction("f");

  EXPECT_FALSE(F->isUsedInBasicBlock(F->begin()));
  EXPECT_TRUE((++F->arg_begin())->isUsedInBasicBlock(F->begin()));
  EXPECT_TRUE(F->arg_begin()->isUsedInBasicBlock(F->begin()));
}

} // end anonymous namespace
