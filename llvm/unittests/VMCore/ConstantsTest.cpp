//===- llvm/unittest/VMCore/ConstantsTest.cpp - Constants unit tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/LLVMContext.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(ConstantsTest, Integer_i1) {
  const IntegerType* Int1 = IntegerType::get(1);
  Constant* One = ConstantInt::get(Int1, 1, true);
  Constant* Zero = ConstantInt::get(Int1, 0);
  Constant* NegOne = ConstantInt::get(Int1, static_cast<uint64_t>(-1), true);
  EXPECT_EQ(NegOne, ConstantInt::getSigned(Int1, -1));
  Constant* Undef = getGlobalContext().getUndef(Int1);

  // Input:  @b = constant i1 add(i1 1 , i1 1)
  // Output: @b = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprAdd(One, One));

  // @c = constant i1 add(i1 -1, i1 1)
  // @c = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprAdd(NegOne, One));

  // @d = constant i1 add(i1 -1, i1 -1)
  // @d = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprAdd(NegOne, NegOne));

  // @e = constant i1 sub(i1 -1, i1 1)
  // @e = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprSub(NegOne, One));

  // @f = constant i1 sub(i1 1 , i1 -1)
  // @f = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprSub(One, NegOne));

  // @g = constant i1 sub(i1 1 , i1 1)
  // @g = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprSub(One, One));

  // @h = constant i1 shl(i1 1 , i1 1)  ; undefined
  // @h = constant i1 undef
  EXPECT_EQ(Undef, getGlobalContext().getConstantExprShl(One, One));

  // @i = constant i1 shl(i1 1 , i1 0)
  // @i = constant i1 true
  EXPECT_EQ(One, getGlobalContext().getConstantExprShl(One, Zero));

  // @j = constant i1 lshr(i1 1, i1 1)  ; undefined
  // @j = constant i1 undef
  EXPECT_EQ(Undef, getGlobalContext().getConstantExprLShr(One, One));

  // @m = constant i1 ashr(i1 1, i1 1)  ; undefined
  // @m = constant i1 undef
  EXPECT_EQ(Undef, getGlobalContext().getConstantExprAShr(One, One));

  // @n = constant i1 mul(i1 -1, i1 1)
  // @n = constant i1 true
  EXPECT_EQ(One, getGlobalContext().getConstantExprMul(NegOne, One));

  // @o = constant i1 sdiv(i1 -1, i1 1) ; overflow
  // @o = constant i1 true
  EXPECT_EQ(One, getGlobalContext().getConstantExprSDiv(NegOne, One));

  // @p = constant i1 sdiv(i1 1 , i1 -1); overflow
  // @p = constant i1 true
  EXPECT_EQ(One, getGlobalContext().getConstantExprSDiv(One, NegOne));

  // @q = constant i1 udiv(i1 -1, i1 1)
  // @q = constant i1 true
  EXPECT_EQ(One, getGlobalContext().getConstantExprUDiv(NegOne, One));

  // @r = constant i1 udiv(i1 1, i1 -1)
  // @r = constant i1 true
  EXPECT_EQ(One, getGlobalContext().getConstantExprUDiv(One, NegOne));

  // @s = constant i1 srem(i1 -1, i1 1) ; overflow
  // @s = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprSRem(NegOne, One));

  // @t = constant i1 urem(i1 -1, i1 1)
  // @t = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprURem(NegOne, One));

  // @u = constant i1 srem(i1  1, i1 -1) ; overflow
  // @u = constant i1 false
  EXPECT_EQ(Zero, getGlobalContext().getConstantExprSRem(One, NegOne));
}

TEST(ConstantsTest, IntSigns) {
  const IntegerType* Int8Ty = Type::Int8Ty;
  EXPECT_EQ(100, ConstantInt::get(Int8Ty, 100, false)->getSExtValue());
  EXPECT_EQ(100, ConstantInt::get(Int8Ty, 100, true)->getSExtValue());
  EXPECT_EQ(100, ConstantInt::getSigned(Int8Ty, 100)->getSExtValue());
  EXPECT_EQ(-50, ConstantInt::get(Int8Ty, 206)->getSExtValue());
  EXPECT_EQ(-50, ConstantInt::getSigned(Int8Ty, -50)->getSExtValue());
  EXPECT_EQ(206U, ConstantInt::getSigned(Int8Ty, -50)->getZExtValue());

  // Overflow is handled by truncation.
  EXPECT_EQ(0x3b, ConstantInt::get(Int8Ty, 0x13b)->getSExtValue());
}

}  // end anonymous namespace
}  // end namespace llvm
