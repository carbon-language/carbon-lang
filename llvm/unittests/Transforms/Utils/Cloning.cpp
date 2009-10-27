//===- Cloning.cpp - Unit tests for the Cloner ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Argument.h"
#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"

using namespace llvm;

TEST(CloneInstruction, OverflowBits) {
  LLVMContext context;
  Value *V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *Add = BinaryOperator::Create(Instruction::Add, V, V);
  BinaryOperator *Sub = BinaryOperator::Create(Instruction::Sub, V, V);
  BinaryOperator *Mul = BinaryOperator::Create(Instruction::Mul, V, V);

  EXPECT_FALSE(Add->clone()->hasNoUnsignedWrap());
  EXPECT_FALSE(Add->clone()->hasNoSignedWrap());
  EXPECT_FALSE(Sub->clone()->hasNoUnsignedWrap());
  EXPECT_FALSE(Sub->clone()->hasNoSignedWrap());
  EXPECT_FALSE(Mul->clone()->hasNoUnsignedWrap());
  EXPECT_FALSE(Mul->clone()->hasNoSignedWrap());

  Add->setHasNoUnsignedWrap();
  Sub->setHasNoUnsignedWrap();
  Mul->setHasNoUnsignedWrap();

  EXPECT_TRUE(Add->clone()->hasNoUnsignedWrap());
  EXPECT_FALSE(Add->clone()->hasNoSignedWrap());
  EXPECT_TRUE(Sub->clone()->hasNoUnsignedWrap());
  EXPECT_FALSE(Sub->clone()->hasNoSignedWrap());
  EXPECT_TRUE(Mul->clone()->hasNoUnsignedWrap());
  EXPECT_FALSE(Mul->clone()->hasNoSignedWrap());

  Add->setHasNoSignedWrap();
  Sub->setHasNoSignedWrap();
  Mul->setHasNoSignedWrap();

  EXPECT_TRUE(Add->clone()->hasNoUnsignedWrap());
  EXPECT_TRUE(Add->clone()->hasNoSignedWrap());
  EXPECT_TRUE(Sub->clone()->hasNoUnsignedWrap());
  EXPECT_TRUE(Sub->clone()->hasNoSignedWrap());
  EXPECT_TRUE(Mul->clone()->hasNoUnsignedWrap());
  EXPECT_TRUE(Mul->clone()->hasNoSignedWrap());

  Add->setHasNoUnsignedWrap(false);
  Sub->setHasNoUnsignedWrap(false);
  Mul->setHasNoUnsignedWrap(false);

  EXPECT_FALSE(Add->clone()->hasNoUnsignedWrap());
  EXPECT_TRUE(Add->clone()->hasNoSignedWrap());
  EXPECT_FALSE(Sub->clone()->hasNoUnsignedWrap());
  EXPECT_TRUE(Sub->clone()->hasNoSignedWrap());
  EXPECT_FALSE(Mul->clone()->hasNoUnsignedWrap());
  EXPECT_TRUE(Mul->clone()->hasNoSignedWrap());
}

TEST(CloneInstruction, Inbounds) {
  LLVMContext context;
  Value *V = new Argument(Type::getInt32PtrTy(context));
  Constant *Z = Constant::getNullValue(Type::getInt32Ty(context));
  std::vector<Value *> ops;
  ops.push_back(Z);
  GetElementPtrInst *GEP = GetElementPtrInst::Create(V, ops.begin(), ops.end());
  EXPECT_FALSE(GEP->clone()->isInBounds());

  GEP->setIsInBounds();
  EXPECT_TRUE(GEP->clone()->isInBounds());
}

TEST(CloneInstruction, Exact) {
  LLVMContext context;
  Value *V = new Argument(Type::getInt32Ty(context));

  BinaryOperator *SDiv = BinaryOperator::Create(Instruction::SDiv, V, V);
  EXPECT_FALSE(SDiv->clone()->isExact());

  SDiv->setIsExact(true);
  EXPECT_TRUE(SDiv->clone()->isExact());
}
