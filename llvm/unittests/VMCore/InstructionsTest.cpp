//===- llvm/unittest/VMCore/InstructionsTest.cpp - Instructions unit tests ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(InstructionsTest, ReturnInst) {
  LLVMContext &C(getGlobalContext());

  // test for PR6589
  const ReturnInst* r0 = ReturnInst::Create(C);
  EXPECT_EQ(r0->op_begin(), r0->op_end());

  const IntegerType* Int1 = IntegerType::get(C, 1);
  Constant* One = ConstantInt::get(Int1, 1, true);
  const ReturnInst* r1 = ReturnInst::Create(C, One);
  User::const_op_iterator b(r1->op_begin());
  EXPECT_NE(b, r1->op_end());
  EXPECT_EQ(*b, One);
  EXPECT_EQ(r1->getOperand(0), One);
  ++b;
  EXPECT_EQ(b, r1->op_end());
}

}  // end anonymous namespace
}  // end namespace llvm
