//===- IslTest.cpp ----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/GICHelper.h"
#include "gtest/gtest.h"
#include "isl/val.h"

using namespace llvm;
using namespace polly;

namespace {

TEST(Isl, APIntToIslVal) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    APInt APZero(1, 0, true);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_zero(IslZero));
    isl_val_free(IslZero);
  }

  {
    APInt APNOne(1, -1, true);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_negone(IslNOne));
    isl_val_free(IslNOne);
  }

  {
    APInt APZero(1, 0, false);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_zero(IslZero));
    isl_val_free(IslZero);
  }

  {
    APInt APOne(1, 1, false);
    auto *IslOne = isl_valFromAPInt(IslCtx, APOne, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_one(IslOne));
    isl_val_free(IslOne);
  }

  {
    APInt APNTwo(2, -2, true);
    auto *IslNTwo = isl_valFromAPInt(IslCtx, APNTwo, true);
    auto *IslNTwoCmp = isl_val_int_from_si(IslCtx, -2);
    EXPECT_EQ(isl_bool_true, isl_val_eq(IslNTwo, IslNTwoCmp));
    isl_val_free(IslNTwo);
    isl_val_free(IslNTwoCmp);
  }

  {
    APInt APNOne(32, -1, true);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, true);
    EXPECT_EQ(isl_bool_true, isl_val_is_negone(IslNOne));
    isl_val_free(IslNOne);
  }

  {
    APInt APZero(32, 0, false);
    auto *IslZero = isl_valFromAPInt(IslCtx, APZero, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_zero(IslZero));
    isl_val_free(IslZero);
  }

  {
    APInt APOne(32, 1, false);
    auto *IslOne = isl_valFromAPInt(IslCtx, APOne, false);
    EXPECT_EQ(isl_bool_true, isl_val_is_one(IslOne));
    isl_val_free(IslOne);
  }

  {
    APInt APTwo(32, 2, false);
    auto *IslTwo = isl_valFromAPInt(IslCtx, APTwo, false);
    EXPECT_EQ(0, isl_val_cmp_si(IslTwo, 2));
    isl_val_free(IslTwo);
  }

  {
    APInt APNOne(32, (1ull << 32) - 1, false);
    auto *IslNOne = isl_valFromAPInt(IslCtx, APNOne, false);
    auto *IslRef = isl_val_int_from_ui(IslCtx, (1ull << 32) - 1);
    EXPECT_EQ(isl_bool_true, isl_val_eq(IslNOne, IslRef));
    isl_val_free(IslNOne);
    isl_val_free(IslRef);
  }

  {
    APInt APLarge(130, 2, false);
    APLarge = APLarge.shl(70);
    auto *IslLarge = isl_valFromAPInt(IslCtx, APLarge, false);
    auto *IslRef = isl_val_int_from_ui(IslCtx, 71);
    IslRef = isl_val_2exp(IslRef);
    EXPECT_EQ(isl_bool_true, isl_val_eq(IslLarge, IslRef));
    isl_val_free(IslLarge);
    isl_val_free(IslRef);
  }

  isl_ctx_free(IslCtx);
}

TEST(Isl, IslValToAPInt) {
  isl_ctx *IslCtx = isl_ctx_alloc();

  {
    auto *IslNOne = isl_val_int_from_si(IslCtx, -1);
    auto APNOne = APIntFromVal(IslNOne);
    // Compare with the two's complement of -1 in a 1-bit integer.
    EXPECT_EQ(APNOne, 1);
    EXPECT_EQ(APNOne.getBitWidth(), 1u);
  }

  {
    auto *IslNTwo = isl_val_int_from_si(IslCtx, -2);
    auto APNTwo = APIntFromVal(IslNTwo);
    // Compare with the two's complement of -2 in a 2-bit integer.
    EXPECT_EQ(APNTwo, 2);
    EXPECT_EQ(APNTwo.getBitWidth(), 2u);
  }

  {
    auto *IslNThree = isl_val_int_from_si(IslCtx, -3);
    auto APNThree = APIntFromVal(IslNThree);
    // Compare with the two's complement of -3 in a 3-bit integer.
    EXPECT_EQ(APNThree, 5);
    EXPECT_EQ(APNThree.getBitWidth(), 3u);
  }

  {
    auto *IslNFour = isl_val_int_from_si(IslCtx, -4);
    auto APNFour = APIntFromVal(IslNFour);
    // Compare with the two's complement of -4 in a 3-bit integer.
    EXPECT_EQ(APNFour, 4);
    EXPECT_EQ(APNFour.getBitWidth(), 3u);
  }

  {
    auto *IslZero = isl_val_int_from_ui(IslCtx, 0);
    auto APZero = APIntFromVal(IslZero);
    EXPECT_EQ(APZero, 0);
    EXPECT_EQ(APZero.getBitWidth(), 1u);
  }

  {
    auto *IslOne = isl_val_int_from_ui(IslCtx, 1);
    auto APOne = APIntFromVal(IslOne);
    EXPECT_EQ(APOne, 1);
    EXPECT_EQ(APOne.getBitWidth(), 2u);
  }

  {
    auto *IslTwo = isl_val_int_from_ui(IslCtx, 2);
    auto APTwo = APIntFromVal(IslTwo);
    EXPECT_EQ(APTwo, 2);
    EXPECT_EQ(APTwo.getBitWidth(), 3u);
  }

  {
    auto *IslThree = isl_val_int_from_ui(IslCtx, 3);
    auto APThree = APIntFromVal(IslThree);
    EXPECT_EQ(APThree, 3);

    EXPECT_EQ(APThree.getBitWidth(), 3u);
  }

  {
    auto *IslFour = isl_val_int_from_ui(IslCtx, 4);
    auto APFour = APIntFromVal(IslFour);
    EXPECT_EQ(APFour, 4);
    EXPECT_EQ(APFour.getBitWidth(), 4u);
  }

  {
    auto *IslNOne = isl_val_int_from_ui(IslCtx, (1ull << 32) - 1);
    auto APNOne = APIntFromVal(IslNOne);
    EXPECT_EQ(APNOne, (1ull << 32) - 1);
    EXPECT_EQ(APNOne.getBitWidth(), 33u);
  }

  {
    auto *IslLargeNum = isl_val_int_from_ui(IslCtx, (1ull << 60) - 1);
    auto APLargeNum = APIntFromVal(IslLargeNum);
    EXPECT_EQ(APLargeNum, (1ull << 60) - 1);
    EXPECT_EQ(APLargeNum.getBitWidth(), 61u);
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 500);
    auto *IslLargePow2 = isl_val_2exp(IslExp);
    auto APLargePow2 = APIntFromVal(IslLargePow2);
    EXPECT_TRUE(APLargePow2.isPowerOf2());
    EXPECT_EQ(APLargePow2.getBitWidth(), 502u);
    EXPECT_EQ(APLargePow2.getMinSignedBits(), 502u);
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 500);
    auto *IslLargeNPow2 = isl_val_neg(isl_val_2exp(IslExp));
    auto APLargeNPow2 = APIntFromVal(IslLargeNPow2);
    EXPECT_EQ(APLargeNPow2.getBitWidth(), 501u);
    EXPECT_EQ(APLargeNPow2.getMinSignedBits(), 501u);
    EXPECT_EQ((-APLargeNPow2).exactLogBase2(), 500);
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 512);
    auto *IslLargePow2 = isl_val_2exp(IslExp);
    auto APLargePow2 = APIntFromVal(IslLargePow2);
    EXPECT_TRUE(APLargePow2.isPowerOf2());
    EXPECT_EQ(APLargePow2.getBitWidth(), 514u);
    EXPECT_EQ(APLargePow2.getMinSignedBits(), 514u);
  }

  {
    auto *IslExp = isl_val_int_from_ui(IslCtx, 512);
    auto *IslLargeNPow2 = isl_val_neg(isl_val_2exp(IslExp));
    auto APLargeNPow2 = APIntFromVal(IslLargeNPow2);
    EXPECT_EQ(APLargeNPow2.getBitWidth(), 513u);
    EXPECT_EQ(APLargeNPow2.getMinSignedBits(), 513u);
    EXPECT_EQ((-APLargeNPow2).exactLogBase2(), 512);
  }

  isl_ctx_free(IslCtx);
}

} // anonymous namespace
