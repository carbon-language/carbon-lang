//===- llvm/unittest/Support/ConstantRangeTest.cpp - ConstantRange tests --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ConstantRange.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {

class ConstantRangeTest : public ::testing::Test {
protected:
  static ConstantRange Full;
  static ConstantRange Empty;
  static ConstantRange One;
  static ConstantRange Some;
  static ConstantRange Wrap;
};

ConstantRange ConstantRangeTest::Full(16);
ConstantRange ConstantRangeTest::Empty(16, false);
ConstantRange ConstantRangeTest::One(APInt(16, 0xa));
ConstantRange ConstantRangeTest::Some(APInt(16, 0xa), APInt(16, 0xaaa));
ConstantRange ConstantRangeTest::Wrap(APInt(16, 0xaaa), APInt(16, 0xa));

TEST_F(ConstantRangeTest, Basics) {
  EXPECT_TRUE(Full.isFullSet());
  EXPECT_FALSE(Full.isEmptySet());
  EXPECT_FALSE(Full.isWrappedSet());
  EXPECT_TRUE(Full.contains(APInt(16, 0x0)));
  EXPECT_TRUE(Full.contains(APInt(16, 0x9)));
  EXPECT_TRUE(Full.contains(APInt(16, 0xa)));
  EXPECT_TRUE(Full.contains(APInt(16, 0xaa9)));
  EXPECT_TRUE(Full.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(Empty.isFullSet());
  EXPECT_TRUE(Empty.isEmptySet());
  EXPECT_FALSE(Empty.isWrappedSet());
  EXPECT_FALSE(Empty.contains(APInt(16, 0x0)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0x9)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0xa)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0xaa9)));
  EXPECT_FALSE(Empty.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(One.isFullSet());
  EXPECT_FALSE(One.isEmptySet());
  EXPECT_FALSE(One.isWrappedSet());
  EXPECT_FALSE(One.contains(APInt(16, 0x0)));
  EXPECT_FALSE(One.contains(APInt(16, 0x9)));
  EXPECT_TRUE(One.contains(APInt(16, 0xa)));
  EXPECT_FALSE(One.contains(APInt(16, 0xaa9)));
  EXPECT_FALSE(One.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(Some.isFullSet());
  EXPECT_FALSE(Some.isEmptySet());
  EXPECT_FALSE(Some.isWrappedSet());
  EXPECT_FALSE(Some.contains(APInt(16, 0x0)));
  EXPECT_FALSE(Some.contains(APInt(16, 0x9)));
  EXPECT_TRUE(Some.contains(APInt(16, 0xa)));
  EXPECT_TRUE(Some.contains(APInt(16, 0xaa9)));
  EXPECT_FALSE(Some.contains(APInt(16, 0xaaa)));

  EXPECT_FALSE(Wrap.isFullSet());
  EXPECT_FALSE(Wrap.isEmptySet());
  EXPECT_TRUE(Wrap.isWrappedSet());
  EXPECT_TRUE(Wrap.contains(APInt(16, 0x0)));
  EXPECT_TRUE(Wrap.contains(APInt(16, 0x9)));
  EXPECT_FALSE(Wrap.contains(APInt(16, 0xa)));
  EXPECT_FALSE(Wrap.contains(APInt(16, 0xaa9)));
  EXPECT_TRUE(Wrap.contains(APInt(16, 0xaaa)));
}

TEST_F(ConstantRangeTest, Equality) {
  EXPECT_EQ(Full, Full);
  EXPECT_EQ(Empty, Empty);
  EXPECT_EQ(One, One);
  EXPECT_EQ(Some, Some);
  EXPECT_EQ(Wrap, Wrap);
  EXPECT_NE(Full, Empty);
  EXPECT_NE(Full, One);
  EXPECT_NE(Full, Some);
  EXPECT_NE(Full, Wrap);
  EXPECT_NE(Empty, One);
  EXPECT_NE(Empty, Some);
  EXPECT_NE(Empty, Wrap);
  EXPECT_NE(One, Some);
  EXPECT_NE(One, Wrap);
  EXPECT_NE(Some, Wrap);
}

TEST_F(ConstantRangeTest, SingleElement) {
  EXPECT_EQ(Full.getSingleElement(), static_cast<APInt *>(NULL));
  EXPECT_EQ(Empty.getSingleElement(), static_cast<APInt *>(NULL));
  EXPECT_EQ(*One.getSingleElement(), APInt(16, 0xa));
  EXPECT_EQ(Some.getSingleElement(), static_cast<APInt *>(NULL));
  EXPECT_EQ(Wrap.getSingleElement(), static_cast<APInt *>(NULL));

  EXPECT_FALSE(Full.isSingleElement());
  EXPECT_FALSE(Empty.isSingleElement());
  EXPECT_TRUE(One.isSingleElement());
  EXPECT_FALSE(Some.isSingleElement());
  EXPECT_FALSE(Wrap.isSingleElement());
}

TEST_F(ConstantRangeTest, GetSetSize) {
  EXPECT_EQ(Full.getSetSize(), APInt(16, 0));
  EXPECT_EQ(Empty.getSetSize(), APInt(16, 0));
  EXPECT_EQ(One.getSetSize(), APInt(16, 1));
  EXPECT_EQ(Some.getSetSize(), APInt(16, 0xaa0));
  EXPECT_EQ(Wrap.getSetSize(), APInt(16, 0x10000 - 0xaa0));
}

TEST_F(ConstantRangeTest, GetMinsAndMaxes) {
  EXPECT_EQ(Full.getUnsignedMax(), APInt(16, UINT16_MAX));
  EXPECT_EQ(One.getUnsignedMax(), APInt(16, 0xa));
  EXPECT_EQ(Some.getUnsignedMax(), APInt(16, 0xaa9));
  EXPECT_EQ(Wrap.getUnsignedMax(), APInt(16, UINT16_MAX));

  EXPECT_EQ(Full.getUnsignedMin(), APInt(16, 0));
  EXPECT_EQ(One.getUnsignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Some.getUnsignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Wrap.getUnsignedMin(), APInt(16, 0));

  EXPECT_EQ(Full.getSignedMax(), APInt(16, INT16_MAX));
  EXPECT_EQ(One.getSignedMax(), APInt(16, 0xa));
  EXPECT_EQ(Some.getSignedMax(), APInt(16, 0xaa9));
  EXPECT_EQ(Wrap.getSignedMax(), APInt(16, INT16_MAX));

  EXPECT_EQ(Full.getSignedMin(), APInt(16, INT16_MIN));
  EXPECT_EQ(One.getSignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Some.getSignedMin(), APInt(16, 0xa));
  EXPECT_EQ(Wrap.getSignedMin(), APInt(16, INT16_MIN));
}

TEST_F(ConstantRangeTest, Trunc) {
  ConstantRange TFull = Full.truncate(10);
  ConstantRange TEmpty = Empty.truncate(10);
  ConstantRange TOne = One.truncate(10);
  ConstantRange TSome = Some.truncate(10);
  ConstantRange TWrap = Wrap.truncate(10);
  EXPECT_TRUE(TFull.isFullSet());
  EXPECT_TRUE(TEmpty.isEmptySet());
  EXPECT_EQ(TOne, ConstantRange(APInt(One.getLower()).trunc(10),
                                APInt(One.getUpper()).trunc(10)));
  EXPECT_TRUE(TSome.isFullSet());
}

TEST_F(ConstantRangeTest, ZExt) {
  ConstantRange ZFull = Full.zeroExtend(20);
  ConstantRange ZEmpty = Empty.zeroExtend(20);
  ConstantRange ZOne = One.zeroExtend(20);
  ConstantRange ZSome = Some.zeroExtend(20);
  ConstantRange ZWrap = Wrap.zeroExtend(20);
  EXPECT_EQ(ZFull, ConstantRange(APInt(20, 0), APInt(20, 0x10000)));
  EXPECT_TRUE(ZEmpty.isEmptySet());
  EXPECT_EQ(ZOne, ConstantRange(APInt(One.getLower()).zext(20),
                                APInt(One.getUpper()).zext(20)));
  EXPECT_EQ(ZSome, ConstantRange(APInt(Some.getLower()).zext(20),
                                 APInt(Some.getUpper()).zext(20)));
  EXPECT_EQ(ZWrap, ConstantRange(APInt(Wrap.getLower()).zext(20),
                                 APInt(Wrap.getUpper()).zext(20)));
}

TEST_F(ConstantRangeTest, SExt) {
  ConstantRange SFull = Full.signExtend(20);
  ConstantRange SEmpty = Empty.signExtend(20);
  ConstantRange SOne = One.signExtend(20);
  ConstantRange SSome = Some.signExtend(20);
  ConstantRange SWrap = Wrap.signExtend(20);
  EXPECT_EQ(SFull, ConstantRange(APInt(20, INT16_MIN, true),
                                 APInt(20, INT16_MAX + 1, true)));
  EXPECT_TRUE(SEmpty.isEmptySet());
  EXPECT_EQ(SOne, ConstantRange(APInt(One.getLower()).sext(20),
                                APInt(One.getUpper()).sext(20)));
  EXPECT_EQ(SSome, ConstantRange(APInt(Some.getLower()).sext(20),
                                 APInt(Some.getUpper()).sext(20)));
  EXPECT_EQ(SWrap, ConstantRange(APInt(Wrap.getLower()).sext(20),
                                 APInt(Wrap.getUpper()).sext(20)));
}

TEST_F(ConstantRangeTest, IntersectWith) {
  EXPECT_TRUE(Empty.intersectWith(Full).isEmptySet());
  EXPECT_TRUE(Empty.intersectWith(Empty).isEmptySet());
  EXPECT_TRUE(Empty.intersectWith(One).isEmptySet());
  EXPECT_TRUE(Empty.intersectWith(Some).isEmptySet());
  EXPECT_TRUE(Empty.intersectWith(Wrap).isEmptySet());
  EXPECT_TRUE(Full.intersectWith(Full).isFullSet());
  EXPECT_TRUE(Some.intersectWith(Some) == Some);
  EXPECT_TRUE(Some.intersectWith(One) == One);
  EXPECT_TRUE(Full.intersectWith(One) == One);
  EXPECT_TRUE(Full.intersectWith(Some) == Some);
  EXPECT_TRUE(Some.intersectWith(Wrap).isEmptySet());
  EXPECT_TRUE(One.intersectWith(Wrap).isEmptySet());
  EXPECT_EQ(One.intersectWith(Wrap), Wrap.intersectWith(One));
}

TEST_F(ConstantRangeTest, MaximalIntersectWith) {
  EXPECT_TRUE(Empty.maximalIntersectWith(Full).isEmptySet());
  EXPECT_TRUE(Empty.maximalIntersectWith(Empty).isEmptySet());
  EXPECT_TRUE(Empty.maximalIntersectWith(One).isEmptySet());
  EXPECT_TRUE(Empty.maximalIntersectWith(Some).isEmptySet());
  EXPECT_TRUE(Empty.maximalIntersectWith(Wrap).isEmptySet());
  EXPECT_TRUE(Full.maximalIntersectWith(Full).isFullSet());
  EXPECT_TRUE(Some.maximalIntersectWith(Some) == Some);
  EXPECT_TRUE(Some.maximalIntersectWith(One) == One);
  EXPECT_TRUE(Full.maximalIntersectWith(One) == One);
  EXPECT_TRUE(Full.maximalIntersectWith(Some) == Some);
  EXPECT_TRUE(Some.maximalIntersectWith(Wrap).isEmptySet());
  EXPECT_TRUE(One.maximalIntersectWith(Wrap).isEmptySet());
  EXPECT_EQ(One.maximalIntersectWith(Wrap), Wrap.maximalIntersectWith(One));
}

TEST_F(ConstantRangeTest, UnionWith) {
  EXPECT_EQ(Wrap.unionWith(One),
            ConstantRange(APInt(16, 0xaaa), APInt(16, 0xb)));
  EXPECT_EQ(One.unionWith(Wrap), Wrap.unionWith(One));
  EXPECT_TRUE(Empty.unionWith(Empty).isEmptySet());
  EXPECT_TRUE(Full.unionWith(Full).isFullSet());
  EXPECT_TRUE(Some.unionWith(Wrap).isFullSet());
}

TEST_F(ConstantRangeTest, SubtractAPInt) {
  EXPECT_TRUE(Full.subtract(APInt(16, 4)).isFullSet());
  EXPECT_TRUE(Empty.subtract(APInt(16, 4)).isEmptySet());
  EXPECT_EQ(Some.subtract(APInt(16, 4)),
            ConstantRange(APInt(16, 0x6), APInt(16, 0xaa6)));
  EXPECT_EQ(Wrap.subtract(APInt(16, 4)),
            ConstantRange(APInt(16, 0xaa6), APInt(16, 0x6)));
  EXPECT_EQ(One.subtract(APInt(16, 4)),
            ConstantRange(APInt(16, 0x6)));
}

TEST_F(ConstantRangeTest, Add) {
  EXPECT_TRUE(Full.add(APInt(16, 4)).isFullSet());
  EXPECT_EQ(Full.add(Full), Full);
  EXPECT_EQ(Full.add(Empty), Empty);
  EXPECT_EQ(Full.add(One), Full);
  EXPECT_EQ(Full.add(Some), Full);
  EXPECT_EQ(Full.add(Wrap), Full);
  EXPECT_EQ(Empty.add(Empty), Empty);
  EXPECT_EQ(Empty.add(One), Empty);
  EXPECT_EQ(Empty.add(Some), Empty);
  EXPECT_EQ(Empty.add(Wrap), Empty);
  EXPECT_TRUE(Empty.add(APInt(16, 4)).isEmptySet());
  EXPECT_EQ(Some.add(APInt(16, 4)),
            ConstantRange(APInt(16, 0xe), APInt(16, 0xaae)));
  EXPECT_EQ(Wrap.add(APInt(16, 4)),
            ConstantRange(APInt(16, 0xaae), APInt(16, 0xe)));
  EXPECT_EQ(One.add(APInt(16, 4)),
            ConstantRange(APInt(16, 0xe)));
}

TEST_F(ConstantRangeTest, Multiply) {
  EXPECT_EQ(Full.multiply(Full), Full);
  EXPECT_EQ(Full.multiply(Empty), Empty);
  EXPECT_EQ(Full.multiply(One), Full);
  EXPECT_EQ(Full.multiply(Some), Full);
  EXPECT_EQ(Full.multiply(Wrap), Full);
  EXPECT_EQ(Empty.multiply(Empty), Empty);
  EXPECT_EQ(Empty.multiply(One), Empty);
  EXPECT_EQ(Empty.multiply(Some), Empty);
  EXPECT_EQ(Empty.multiply(Wrap), Empty);
  EXPECT_EQ(One.multiply(One), ConstantRange(APInt(16, 0xa*0xa),
                                             APInt(16, 0xa*0xa + 1)));
  EXPECT_EQ(One.multiply(Some), ConstantRange(APInt(16, 0xa*0xa),
                                              APInt(16, 0xa*0xaa9 + 1)));
  EXPECT_TRUE(One.multiply(Wrap).isFullSet());
  EXPECT_TRUE(Some.multiply(Some).isFullSet());
  EXPECT_EQ(Some.multiply(Wrap), Full);
  EXPECT_EQ(Wrap.multiply(Wrap), Full);

  // http://llvm.org/PR4545
  EXPECT_EQ(ConstantRange(APInt(4, 1), APInt(4, 6)).multiply(
              ConstantRange(APInt(4, 6), APInt(4, 2))),
            ConstantRange(4, /*isFullSet=*/true));
}

TEST_F(ConstantRangeTest, UMax) {
  EXPECT_TRUE(Full.umax(Full).isFullSet());
  EXPECT_TRUE(Full.umax(Empty).isEmptySet());
  EXPECT_EQ(Full.umax(Some), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_TRUE(Full.umax(Wrap).isFullSet());
  EXPECT_EQ(Full.umax(Some), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(Empty.umax(Empty), Empty);
  EXPECT_EQ(Empty.umax(Some), Empty);
  EXPECT_EQ(Empty.umax(Wrap), Empty);
  EXPECT_EQ(Empty.umax(One), Empty);
  EXPECT_EQ(Some.umax(Some), Some);
  EXPECT_EQ(Some.umax(Wrap), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(Some.umax(One), Some);
  // TODO: ConstantRange is currently over-conservative here.
  EXPECT_EQ(Wrap.umax(Wrap), Full);
  EXPECT_EQ(Wrap.umax(One), ConstantRange(APInt(16, 0xa), APInt(16, 0)));
  EXPECT_EQ(One.umax(One), One);
}

TEST_F(ConstantRangeTest, SMax) {
  EXPECT_TRUE(Full.smax(Full).isFullSet());
  EXPECT_TRUE(Full.smax(Empty).isEmptySet());
  EXPECT_EQ(Full.smax(Some), ConstantRange(APInt(16, 0xa),
                             APInt::getSignedMinValue(16)));
  EXPECT_TRUE(Full.smax(Wrap).isFullSet());
  EXPECT_EQ(Full.smax(One), ConstantRange(APInt(16, 0xa),
                            APInt::getSignedMinValue(16)));
  EXPECT_EQ(Empty.smax(Empty), Empty);
  EXPECT_EQ(Empty.smax(Some), Empty);
  EXPECT_EQ(Empty.smax(Wrap), Empty);
  EXPECT_EQ(Empty.smax(One), Empty);
  EXPECT_EQ(Some.smax(Some), Some);
  EXPECT_EQ(Some.smax(Wrap), ConstantRange(APInt(16, 0xa),
                                           APInt(16, INT16_MIN)));
  EXPECT_EQ(Some.smax(One), Some);
  EXPECT_EQ(Wrap.smax(One), ConstantRange(APInt(16, 0xa),
                                          APInt(16, INT16_MIN)));
  EXPECT_EQ(One.smax(One), One);
}

TEST_F(ConstantRangeTest, UDiv) {
  EXPECT_EQ(Full.udiv(Full), Full);
  EXPECT_EQ(Full.udiv(Empty), Empty);
  EXPECT_EQ(Full.udiv(One), ConstantRange(APInt(16, 0),
                                          APInt(16, 0xffff / 0xa + 1)));
  EXPECT_EQ(Full.udiv(Some), ConstantRange(APInt(16, 0),
                                           APInt(16, 0xffff / 0xa + 1)));
  EXPECT_EQ(Full.udiv(Wrap), Full);
  EXPECT_EQ(Empty.udiv(Empty), Empty);
  EXPECT_EQ(Empty.udiv(One), Empty);
  EXPECT_EQ(Empty.udiv(Some), Empty);
  EXPECT_EQ(Empty.udiv(Wrap), Empty);
  EXPECT_EQ(One.udiv(One), ConstantRange(APInt(16, 1)));
  EXPECT_EQ(One.udiv(Some), ConstantRange(APInt(16, 0), APInt(16, 2)));
  EXPECT_EQ(One.udiv(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xb)));
  EXPECT_EQ(Some.udiv(Some), ConstantRange(APInt(16, 0), APInt(16, 0x111)));
  EXPECT_EQ(Some.udiv(Wrap), ConstantRange(APInt(16, 0), APInt(16, 0xaaa)));
  EXPECT_EQ(Wrap.udiv(Wrap), Full);
}

}  // anonymous namespace
