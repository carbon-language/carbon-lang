//===- llvm/unittest/VMCore/InstructionsTest.cpp - Instructions unit tests ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/BasicBlock.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(InstructionsTest, ReturnInst) {
  LLVMContext &C(getGlobalContext());

  // test for PR6589
  const ReturnInst* r0 = ReturnInst::Create(C);
  EXPECT_EQ(r0->getNumOperands(), 0U);
  EXPECT_EQ(r0->op_begin(), r0->op_end());

  const IntegerType* Int1 = IntegerType::get(C, 1);
  Constant* One = ConstantInt::get(Int1, 1, true);
  const ReturnInst* r1 = ReturnInst::Create(C, One);
  EXPECT_EQ(r1->getNumOperands(), 1U);
  User::const_op_iterator b(r1->op_begin());
  EXPECT_NE(b, r1->op_end());
  EXPECT_EQ(*b, One);
  EXPECT_EQ(r1->getOperand(0), One);
  ++b;
  EXPECT_EQ(b, r1->op_end());

  // clean up
  delete r0;
  delete r1;
}

TEST(InstructionsTest, BranchInst) {
  LLVMContext &C(getGlobalContext());

  // Make a BasicBlocks
  BasicBlock* bb0 = BasicBlock::Create(C);
  BasicBlock* bb1 = BasicBlock::Create(C);

  // Mandatory BranchInst
  const BranchInst* b0 = BranchInst::Create(bb0);

  EXPECT_TRUE(b0->isUnconditional());
  EXPECT_FALSE(b0->isConditional());
  EXPECT_EQ(b0->getNumSuccessors(), 1U);

  // check num operands
  EXPECT_EQ(b0->getNumOperands(), 1U);

  EXPECT_NE(b0->op_begin(), b0->op_end());
  EXPECT_EQ(next(b0->op_begin()), b0->op_end());

  EXPECT_EQ(next(b0->op_begin()), b0->op_end());

  const IntegerType* Int1 = IntegerType::get(C, 1);
  Constant* One = ConstantInt::get(Int1, 1, true);

  // Conditional BranchInst
  BranchInst* b1 = BranchInst::Create(bb0, bb1, One);

  EXPECT_FALSE(b1->isUnconditional());
  EXPECT_TRUE(b1->isConditional());
  EXPECT_EQ(b1->getNumSuccessors(), 2U);

  // check num operands
  EXPECT_EQ(b1->getNumOperands(), 3U);

  User::const_op_iterator b(b1->op_begin());

  // check COND
  EXPECT_NE(b, b1->op_end());
  EXPECT_EQ(*b, One);
  EXPECT_EQ(b1->getOperand(0), One);
  EXPECT_EQ(b1->getCondition(), One);
  ++b;

  // check ELSE
  EXPECT_EQ(*b, bb1);
  EXPECT_EQ(b1->getOperand(1), bb1);
  EXPECT_EQ(b1->getSuccessor(1), bb1);
  ++b;

  // check THEN
  EXPECT_EQ(*b, bb0);
  EXPECT_EQ(b1->getOperand(2), bb0);
  EXPECT_EQ(b1->getSuccessor(0), bb0);
  ++b;

  EXPECT_EQ(b, b1->op_end());

  // shrink it
  b1->setUnconditionalDest(bb1);

  // check num operands
  EXPECT_EQ(b1->getNumOperands(), 1U);

  User::const_op_iterator c(b1->op_begin());
  EXPECT_NE(c, b1->op_end());

  // check THEN
  EXPECT_EQ(*c, bb1);
  EXPECT_EQ(b1->getOperand(0), bb1);
  EXPECT_EQ(b1->getSuccessor(0), bb1);
  ++c;

  EXPECT_EQ(c, b1->op_end());

  // clean up
  delete b0;
  delete b1;

  delete bb0;
  delete bb1;
}

}  // end anonymous namespace
}  // end namespace llvm
