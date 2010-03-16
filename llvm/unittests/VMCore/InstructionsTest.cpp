//===- llvm/unittest/VMCore/InstructionsTest.cpp - Instructions unit tests ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(InstructionsTest, ReturnInst_0) {
  LLVMContext &C(getGlobalContext());

  // reproduction recipe for PR6589
  const ReturnInst* r0 = ReturnInst::Create(C);
  EXPECT_NE(r0->op_begin(), r0->op_end());
}

}  // end anonymous namespace
}  // end namespace llvm
