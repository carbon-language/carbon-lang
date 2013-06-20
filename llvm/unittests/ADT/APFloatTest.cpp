//===- llvm/unittest/ADT/APFloat.cpp - APFloat unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <ostream>
#include <string>

using namespace llvm;

static double convertToDoubleFromString(const char *Str) {
  llvm::APFloat F(0.0);
  F.convertFromString(Str, llvm::APFloat::rmNearestTiesToEven);
  return F.convertToDouble();
}

static std::string convertToString(double d, unsigned Prec, unsigned Pad) {
  llvm::SmallVector<char, 100> Buffer;
  llvm::APFloat F(d);
  F.toString(Buffer, Prec, Pad);
  return std::string(Buffer.data(), Buffer.size());
}

namespace {

TEST(APFloatTest, isSignaling) {
  // We test qNaN, -qNaN, +sNaN, -sNaN with and without payloads. *NOTE* The
  // positive/negative distinction is included only since the getQNaN/getSNaN
  // API provides the option.
  APInt payload = APInt::getOneBitSet(4, 2);
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, false).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, true).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, false, &payload).isSignaling());
  EXPECT_FALSE(APFloat::getQNaN(APFloat::IEEEsingle, true, &payload).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, false).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, true).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, false, &payload).isSignaling());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, true, &payload).isSignaling());
}

TEST(APFloatTest, next) {

  APFloat test(APFloat::IEEEquad, APFloat::uninitialized);
  APFloat expected(APFloat::IEEEquad, APFloat::uninitialized);

  // 1. Test Special Cases Values.
  //
  // Test all special values for nextUp and nextDown perscribed by IEEE-754R
  // 2008. These are:
  //   1.  +inf
  //   2.  -inf
  //   3.  getLargest()
  //   4.  -getLargest()
  //   5.  getSmallest()
  //   6.  -getSmallest()
  //   7.  qNaN
  //   8.  sNaN
  //   9.  +0
  //   10. -0

  // nextUp(+inf) = +inf.
  test = APFloat::getInf(APFloat::IEEEquad, false);
  expected = APFloat::getInf(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+inf) = -nextUp(-inf) = -(-getLargest()) = getLargest()
  test = APFloat::getInf(APFloat::IEEEquad, false);
  expected = APFloat::getLargest(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-inf) = -getLargest()
  test = APFloat::getInf(APFloat::IEEEquad, true);
  expected = APFloat::getLargest(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-inf) = -nextUp(+inf) = -(+inf) = -inf.
  test = APFloat::getInf(APFloat::IEEEquad, true);
  expected = APFloat::getInf(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(getLargest()) = +inf
  test = APFloat::getLargest(APFloat::IEEEquad, false);
  expected = APFloat::getInf(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(getLargest()) = -nextUp(-getLargest())
  //                        = -(-getLargest() + inc)
  //                        = getLargest() - inc.
  test = APFloat::getLargest(APFloat::IEEEquad, false);
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.fffffffffffffffffffffffffffep+16383");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isInfinity() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-getLargest()) = -getLargest() + inc.
  test = APFloat::getLargest(APFloat::IEEEquad, true);
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.fffffffffffffffffffffffffffep+16383");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-getLargest()) = -nextUp(getLargest()) = -(inf) = -inf.
  test = APFloat::getLargest(APFloat::IEEEquad, true);
  expected = APFloat::getInf(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isInfinity() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(getSmallest()) = getSmallest() + inc.
  test = APFloat(APFloat::IEEEquad, "0x0.0000000000000000000000000001p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x0.0000000000000000000000000002p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(getSmallest()) = -nextUp(-getSmallest()) = -(-0) = +0.
  test = APFloat(APFloat::IEEEquad, "0x0.0000000000000000000000000001p-16382");
  expected = APFloat::getZero(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isZero() && !test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-getSmallest()) = -0.
  test = APFloat(APFloat::IEEEquad, "-0x0.0000000000000000000000000001p-16382");
  expected = APFloat::getZero(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isZero() && test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-getSmallest()) = -nextUp(getSmallest()) = -getSmallest() - inc.
  test = APFloat(APFloat::IEEEquad, "-0x0.0000000000000000000000000001p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x0.0000000000000000000000000002p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(qNaN) = qNaN
  test = APFloat::getQNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(qNaN) = qNaN
  test = APFloat::getQNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(sNaN) = qNaN
  test = APFloat::getSNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(sNaN) = qNaN
  test = APFloat::getSNaN(APFloat::IEEEquad, false);
  expected = APFloat::getQNaN(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(true), APFloat::opInvalidOp);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+0) = +getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, false);
  expected = APFloat::getSmallest(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+0) = -nextUp(-0) = -getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, false);
  expected = APFloat::getSmallest(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-0) = +getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, true);
  expected = APFloat::getSmallest(APFloat::IEEEquad, false);
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-0) = -nextUp(0) = -getSmallest()
  test = APFloat::getZero(APFloat::IEEEquad, true);
  expected = APFloat::getSmallest(APFloat::IEEEquad, true);
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2. Binade Boundary Tests.

  // 2a. Test denormal <-> normal binade boundaries.
  //     * nextUp(+Largest Denormal) -> +Smallest Normal.
  //     * nextDown(-Largest Denormal) -> -Smallest Normal.
  //     * nextUp(-Smallest Normal) -> -Largest Denormal.
  //     * nextDown(+Smallest Normal) -> +Largest Denormal.

  // nextUp(+Largest Denormal) -> +Smallest Normal.
  test = APFloat(APFloat::IEEEquad, "0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.0000000000000000000000000000p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Largest Denormal) -> -Smallest Normal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.0000000000000000000000000000p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Smallest Normal) -> -LargestDenormal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x0.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Smallest Normal) -> +Largest Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "+0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "+0x0.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2b. Test normal <-> normal binade boundaries.
  //     * nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
  //     * nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
  //     * nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
  //     * nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.

  // nextUp(-Normal Binade Boundary) -> -Normal Binade Boundary + 1.
  test = APFloat(APFloat::IEEEquad, "-0x1p+1");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.ffffffffffffffffffffffffffffp+0");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Normal Binade Boundary) -> +Normal Binade Boundary - 1.
  test = APFloat(APFloat::IEEEquad, "0x1p+1");
  expected = APFloat(APFloat::IEEEquad, "0x1.ffffffffffffffffffffffffffffp+0");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Normal Binade Boundary - 1) -> +Normal Binade Boundary.
  test = APFloat(APFloat::IEEEquad, "0x1.ffffffffffffffffffffffffffffp+0");
  expected = APFloat(APFloat::IEEEquad, "0x1p+1");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Normal Binade Boundary + 1) -> -Normal Binade Boundary.
  test = APFloat(APFloat::IEEEquad, "-0x1.ffffffffffffffffffffffffffffp+0");
  expected = APFloat(APFloat::IEEEquad, "-0x1p+1");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2c. Test using next at binade boundaries with a direction away from the
  // binade boundary. Away from denormal <-> normal boundaries.
  //
  // This is to make sure that even though we are at a binade boundary, since
  // we are rounding away, we do not trigger the binade boundary code. Thus we
  // test:
  //   * nextUp(-Largest Denormal) -> -Largest Denormal + inc.
  //   * nextDown(+Largest Denormal) -> +Largest Denormal - inc.
  //   * nextUp(+Smallest Normal) -> +Smallest Normal + inc.
  //   * nextDown(-Smallest Normal) -> -Smallest Normal - inc.

  // nextUp(-Largest Denormal) -> -Largest Denormal + inc.
  test = APFloat(APFloat::IEEEquad, "-0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x0.fffffffffffffffffffffffffffep-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Largest Denormal) -> +Largest Denormal - inc.
  test = APFloat(APFloat::IEEEquad, "0x0.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x0.fffffffffffffffffffffffffffep-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Smallest Normal) -> +Smallest Normal + inc.
  test = APFloat(APFloat::IEEEquad, "0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.0000000000000000000000000001p-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Smallest Normal) -> -Smallest Normal - inc.
  test = APFloat(APFloat::IEEEquad, "-0x1.0000000000000000000000000000p-16382");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.0000000000000000000000000001p-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 2d. Test values which cause our exponent to go to min exponent. This
  // is to ensure that guards in the code to check for min exponent
  // trigger properly.
  //     * nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
  //     * nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
  //         -0x1p-16381
  //     * nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16382
  //     * nextDown(0x1p-16382) -> 0x1.ffffffffffffffffffffffffffffp-16382

  // nextUp(-0x1p-16381) -> -0x1.ffffffffffffffffffffffffffffp-16382
  test = APFloat(APFloat::IEEEquad, "-0x1p-16381");
  expected = APFloat(APFloat::IEEEquad,
                     "-0x1.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-0x1.ffffffffffffffffffffffffffffp-16382) ->
  //         -0x1p-16381
  test = APFloat(APFloat::IEEEquad, "-0x1.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad, "-0x1p-16381");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(0x1.ffffffffffffffffffffffffffffp-16382) -> 0x1p-16381
  test = APFloat(APFloat::IEEEquad, "0x1.ffffffffffffffffffffffffffffp-16382");
  expected = APFloat(APFloat::IEEEquad, "0x1p-16381");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(0x1p-16381) -> 0x1.ffffffffffffffffffffffffffffp-16382
  test = APFloat(APFloat::IEEEquad, "0x1p-16381");
  expected = APFloat(APFloat::IEEEquad,
                     "0x1.ffffffffffffffffffffffffffffp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // 3. Now we test both denormal/normal computation which will not cause us
  // to go across binade boundaries. Specifically we test:
  //   * nextUp(+Denormal) -> +Denormal.
  //   * nextDown(+Denormal) -> +Denormal.
  //   * nextUp(-Denormal) -> -Denormal.
  //   * nextDown(-Denormal) -> -Denormal.
  //   * nextUp(+Normal) -> +Normal.
  //   * nextDown(+Normal) -> +Normal.
  //   * nextUp(-Normal) -> -Normal.
  //   * nextDown(-Normal) -> -Normal.

  // nextUp(+Denormal) -> +Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000dp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Denormal) -> +Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "0x0.ffffffffffffffffffffffff000bp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Denormal) -> -Denormal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000bp-16382");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Denormal) -> -Denormal
  test = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000cp-16382");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x0.ffffffffffffffffffffffff000dp-16382");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(+Normal) -> +Normal.
  test = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000dp-16000");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(+Normal) -> +Normal.
  test = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "0x1.ffffffffffffffffffffffff000bp-16000");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(!test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextUp(-Normal) -> -Normal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000bp-16000");
  EXPECT_EQ(test.next(false), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  // nextDown(-Normal) -> -Normal.
  test = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000cp-16000");
  expected = APFloat(APFloat::IEEEquad,
                 "-0x1.ffffffffffffffffffffffff000dp-16000");
  EXPECT_EQ(test.next(true), APFloat::opOK);
  EXPECT_TRUE(!test.isDenormal());
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));
}

TEST(APFloatTest, FMA) {
  APFloat::roundingMode rdmd = APFloat::rmNearestTiesToEven;

  {
    APFloat f1(14.5f);
    APFloat f2(-14.5f);
    APFloat f3(225.0f);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_EQ(14.75f, f1.convertToFloat());
  }

  {
    APFloat Val2(2.0f);
    APFloat f1((float)1.17549435e-38F);
    APFloat f2((float)1.17549435e-38F);
    f1.divide(Val2, rdmd);
    f2.divide(Val2, rdmd);
    APFloat f3(12.0f);
    f1.fusedMultiplyAdd(f2, f3, APFloat::rmNearestTiesToEven);
    EXPECT_EQ(12.0f, f1.convertToFloat());
  }
}

TEST(APFloatTest, Denormal) {
  APFloat::roundingMode rdmd = APFloat::rmNearestTiesToEven;

  // Test single precision
  {
    const char *MinNormalStr = "1.17549435082228750797e-38";
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEsingle, 0.0).isDenormal());

    APFloat Val2(APFloat::IEEEsingle, 2.0e0);
    APFloat T(APFloat::IEEEsingle, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }

  // Test double precision
  {
    const char *MinNormalStr = "2.22507385850720138309e-308";
    EXPECT_FALSE(APFloat(APFloat::IEEEdouble, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEdouble, 0.0).isDenormal());

    APFloat Val2(APFloat::IEEEdouble, 2.0e0);
    APFloat T(APFloat::IEEEdouble, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }

  // Test Intel double-ext
  {
    const char *MinNormalStr = "3.36210314311209350626e-4932";
    EXPECT_FALSE(APFloat(APFloat::x87DoubleExtended, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::x87DoubleExtended, 0.0).isDenormal());

    APFloat Val2(APFloat::x87DoubleExtended, 2.0e0);
    APFloat T(APFloat::x87DoubleExtended, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }

  // Test quadruple precision
  {
    const char *MinNormalStr = "3.36210314311209350626267781732175260e-4932";
    EXPECT_FALSE(APFloat(APFloat::IEEEquad, MinNormalStr).isDenormal());
    EXPECT_FALSE(APFloat(APFloat::IEEEquad, 0.0).isDenormal());

    APFloat Val2(APFloat::IEEEquad, 2.0e0);
    APFloat T(APFloat::IEEEquad, MinNormalStr);
    T.divide(Val2, rdmd);
    EXPECT_TRUE(T.isDenormal());
  }
}

TEST(APFloatTest, Zero) {
  EXPECT_EQ(0.0f,  APFloat(0.0f).convertToFloat());
  EXPECT_EQ(-0.0f, APFloat(-0.0f).convertToFloat());
  EXPECT_TRUE(APFloat(-0.0f).isNegative());

  EXPECT_EQ(0.0,  APFloat(0.0).convertToDouble());
  EXPECT_EQ(-0.0, APFloat(-0.0).convertToDouble());
  EXPECT_TRUE(APFloat(-0.0).isNegative());
}

TEST(APFloatTest, fromZeroDecimalString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  ".0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.0").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "00000.").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+00000.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-00000.").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, ".00000").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.00000").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.00000").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0000.00000").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0000.00000").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0000.00000").convertToDouble());
}

TEST(APFloatTest, fromZeroDecimalSingleExponentString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   "0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   "0.e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+0.e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-0.e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.e-1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   ".0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+.0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-.0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  ".0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  ".0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+.0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,   "0.0e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble,  "+0.0e1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble,  "-0.0e1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.0e+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.0e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0e+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0.0e-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0.0e-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0e-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "000.0000e1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+000.0000e+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-000.0000e+1").convertToDouble());
}

TEST(APFloatTest, fromZeroDecimalLargeExponentString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e1234").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e+1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e+1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e+1234").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0e-1234").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0e-1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, StringRef("0e1234\02", 6)).convertToDouble());
}

TEST(APFloatTest, fromZeroHexadecimalString) {
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x.0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x.0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x.0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x.0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x.0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x.0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x.0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x.0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x.0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.0p1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.0p1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.0p+1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.0p+1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.0p+1").convertToDouble());

  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble,  "0x0.0p-1").convertToDouble());
  EXPECT_EQ(+0.0, APFloat(APFloat::IEEEdouble, "+0x0.0p-1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0.0p-1").convertToDouble());


  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x00000.p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0000.00000p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x.00000p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0.p1").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0p1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x00000.p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0000.00000p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x.00000p1234").convertToDouble());
  EXPECT_EQ( 0.0, APFloat(APFloat::IEEEdouble, "0x0.p1234").convertToDouble());
}

TEST(APFloatTest, fromDecimalString) {
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble, "1").convertToDouble());
  EXPECT_EQ(2.0,      APFloat(APFloat::IEEEdouble, "2.").convertToDouble());
  EXPECT_EQ(0.5,      APFloat(APFloat::IEEEdouble, ".5").convertToDouble());
  EXPECT_EQ(1.0,      APFloat(APFloat::IEEEdouble, "1.0").convertToDouble());
  EXPECT_EQ(-2.0,     APFloat(APFloat::IEEEdouble, "-2").convertToDouble());
  EXPECT_EQ(-4.0,     APFloat(APFloat::IEEEdouble, "-4.").convertToDouble());
  EXPECT_EQ(-0.5,     APFloat(APFloat::IEEEdouble, "-.5").convertToDouble());
  EXPECT_EQ(-1.5,     APFloat(APFloat::IEEEdouble, "-1.5").convertToDouble());
  EXPECT_EQ(1.25e12,  APFloat(APFloat::IEEEdouble, "1.25e12").convertToDouble());
  EXPECT_EQ(1.25e+12, APFloat(APFloat::IEEEdouble, "1.25e+12").convertToDouble());
  EXPECT_EQ(1.25e-12, APFloat(APFloat::IEEEdouble, "1.25e-12").convertToDouble());
  EXPECT_EQ(1024.0,   APFloat(APFloat::IEEEdouble, "1024.").convertToDouble());
  EXPECT_EQ(1024.05,  APFloat(APFloat::IEEEdouble, "1024.05000").convertToDouble());
  EXPECT_EQ(0.05,     APFloat(APFloat::IEEEdouble, ".05000").convertToDouble());
  EXPECT_EQ(2.0,      APFloat(APFloat::IEEEdouble, "2.").convertToDouble());
  EXPECT_EQ(2.0e2,    APFloat(APFloat::IEEEdouble, "2.e2").convertToDouble());
  EXPECT_EQ(2.0e+2,   APFloat(APFloat::IEEEdouble, "2.e+2").convertToDouble());
  EXPECT_EQ(2.0e-2,   APFloat(APFloat::IEEEdouble, "2.e-2").convertToDouble());
  EXPECT_EQ(2.05e2,    APFloat(APFloat::IEEEdouble, "002.05000e2").convertToDouble());
  EXPECT_EQ(2.05e+2,   APFloat(APFloat::IEEEdouble, "002.05000e+2").convertToDouble());
  EXPECT_EQ(2.05e-2,   APFloat(APFloat::IEEEdouble, "002.05000e-2").convertToDouble());
  EXPECT_EQ(2.05e12,   APFloat(APFloat::IEEEdouble, "002.05000e12").convertToDouble());
  EXPECT_EQ(2.05e+12,  APFloat(APFloat::IEEEdouble, "002.05000e+12").convertToDouble());
  EXPECT_EQ(2.05e-12,  APFloat(APFloat::IEEEdouble, "002.05000e-12").convertToDouble());

  // These are "carefully selected" to overflow the fast log-base
  // calculations in APFloat.cpp
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "99e99999").isInfinity());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "-99e99999").isInfinity());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "1e-99999").isPosZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEdouble, "-1e-99999").isNegZero());
}

TEST(APFloatTest, fromHexadecimalString) {
  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble,  "0x1p0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble, "+0x1p0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble, "-0x1p0").convertToDouble());

  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble,  "0x1p+0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble, "+0x1p+0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble, "-0x1p+0").convertToDouble());

  EXPECT_EQ( 1.0, APFloat(APFloat::IEEEdouble,  "0x1p-0").convertToDouble());
  EXPECT_EQ(+1.0, APFloat(APFloat::IEEEdouble, "+0x1p-0").convertToDouble());
  EXPECT_EQ(-1.0, APFloat(APFloat::IEEEdouble, "-0x1p-0").convertToDouble());


  EXPECT_EQ( 2.0, APFloat(APFloat::IEEEdouble,  "0x1p1").convertToDouble());
  EXPECT_EQ(+2.0, APFloat(APFloat::IEEEdouble, "+0x1p1").convertToDouble());
  EXPECT_EQ(-2.0, APFloat(APFloat::IEEEdouble, "-0x1p1").convertToDouble());

  EXPECT_EQ( 2.0, APFloat(APFloat::IEEEdouble,  "0x1p+1").convertToDouble());
  EXPECT_EQ(+2.0, APFloat(APFloat::IEEEdouble, "+0x1p+1").convertToDouble());
  EXPECT_EQ(-2.0, APFloat(APFloat::IEEEdouble, "-0x1p+1").convertToDouble());

  EXPECT_EQ( 0.5, APFloat(APFloat::IEEEdouble,  "0x1p-1").convertToDouble());
  EXPECT_EQ(+0.5, APFloat(APFloat::IEEEdouble, "+0x1p-1").convertToDouble());
  EXPECT_EQ(-0.5, APFloat(APFloat::IEEEdouble, "-0x1p-1").convertToDouble());


  EXPECT_EQ( 3.0, APFloat(APFloat::IEEEdouble,  "0x1.8p1").convertToDouble());
  EXPECT_EQ(+3.0, APFloat(APFloat::IEEEdouble, "+0x1.8p1").convertToDouble());
  EXPECT_EQ(-3.0, APFloat(APFloat::IEEEdouble, "-0x1.8p1").convertToDouble());

  EXPECT_EQ( 3.0, APFloat(APFloat::IEEEdouble,  "0x1.8p+1").convertToDouble());
  EXPECT_EQ(+3.0, APFloat(APFloat::IEEEdouble, "+0x1.8p+1").convertToDouble());
  EXPECT_EQ(-3.0, APFloat(APFloat::IEEEdouble, "-0x1.8p+1").convertToDouble());

  EXPECT_EQ( 0.75, APFloat(APFloat::IEEEdouble,  "0x1.8p-1").convertToDouble());
  EXPECT_EQ(+0.75, APFloat(APFloat::IEEEdouble, "+0x1.8p-1").convertToDouble());
  EXPECT_EQ(-0.75, APFloat(APFloat::IEEEdouble, "-0x1.8p-1").convertToDouble());


  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000.000p1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000.000p1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000.000p1").convertToDouble());

  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000.000p+1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000.000p+1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000.000p+1").convertToDouble());

  EXPECT_EQ( 2048.0, APFloat(APFloat::IEEEdouble,  "0x1000.000p-1").convertToDouble());
  EXPECT_EQ(+2048.0, APFloat(APFloat::IEEEdouble, "+0x1000.000p-1").convertToDouble());
  EXPECT_EQ(-2048.0, APFloat(APFloat::IEEEdouble, "-0x1000.000p-1").convertToDouble());


  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000p1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000p1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000p1").convertToDouble());

  EXPECT_EQ( 8192.0, APFloat(APFloat::IEEEdouble,  "0x1000p+1").convertToDouble());
  EXPECT_EQ(+8192.0, APFloat(APFloat::IEEEdouble, "+0x1000p+1").convertToDouble());
  EXPECT_EQ(-8192.0, APFloat(APFloat::IEEEdouble, "-0x1000p+1").convertToDouble());

  EXPECT_EQ( 2048.0, APFloat(APFloat::IEEEdouble,  "0x1000p-1").convertToDouble());
  EXPECT_EQ(+2048.0, APFloat(APFloat::IEEEdouble, "+0x1000p-1").convertToDouble());
  EXPECT_EQ(-2048.0, APFloat(APFloat::IEEEdouble, "-0x1000p-1").convertToDouble());


  EXPECT_EQ( 16384.0, APFloat(APFloat::IEEEdouble,  "0x10p10").convertToDouble());
  EXPECT_EQ(+16384.0, APFloat(APFloat::IEEEdouble, "+0x10p10").convertToDouble());
  EXPECT_EQ(-16384.0, APFloat(APFloat::IEEEdouble, "-0x10p10").convertToDouble());

  EXPECT_EQ( 16384.0, APFloat(APFloat::IEEEdouble,  "0x10p+10").convertToDouble());
  EXPECT_EQ(+16384.0, APFloat(APFloat::IEEEdouble, "+0x10p+10").convertToDouble());
  EXPECT_EQ(-16384.0, APFloat(APFloat::IEEEdouble, "-0x10p+10").convertToDouble());

  EXPECT_EQ( 0.015625, APFloat(APFloat::IEEEdouble,  "0x10p-10").convertToDouble());
  EXPECT_EQ(+0.015625, APFloat(APFloat::IEEEdouble, "+0x10p-10").convertToDouble());
  EXPECT_EQ(-0.015625, APFloat(APFloat::IEEEdouble, "-0x10p-10").convertToDouble());

  EXPECT_EQ(1.0625, APFloat(APFloat::IEEEdouble, "0x1.1p0").convertToDouble());
  EXPECT_EQ(1.0, APFloat(APFloat::IEEEdouble, "0x1p0").convertToDouble());

  EXPECT_EQ(2.71828, convertToDoubleFromString("2.71828"));
}

TEST(APFloatTest, toString) {
  ASSERT_EQ("10", convertToString(10.0, 6, 3));
  ASSERT_EQ("1.0E+1", convertToString(10.0, 6, 0));
  ASSERT_EQ("10100", convertToString(1.01E+4, 5, 2));
  ASSERT_EQ("1.01E+4", convertToString(1.01E+4, 4, 2));
  ASSERT_EQ("1.01E+4", convertToString(1.01E+4, 5, 1));
  ASSERT_EQ("0.0101", convertToString(1.01E-2, 5, 2));
  ASSERT_EQ("0.0101", convertToString(1.01E-2, 4, 2));
  ASSERT_EQ("1.01E-2", convertToString(1.01E-2, 5, 1));
  ASSERT_EQ("0.7853981633974483", convertToString(0.78539816339744830961, 0, 3));
  ASSERT_EQ("4.940656458412465E-324", convertToString(4.9406564584124654e-324, 0, 3));
  ASSERT_EQ("873.1834", convertToString(873.1834, 0, 1));
  ASSERT_EQ("8.731834E+2", convertToString(873.1834, 0, 0));
}

TEST(APFloatTest, toInteger) {
  bool isExact = false;
  APSInt result(5, /*isUnsigned=*/true);

  EXPECT_EQ(APFloat::opOK,
            APFloat(APFloat::IEEEdouble, "10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_TRUE(isExact);
  EXPECT_EQ(APSInt(APInt(5, 10), true), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "-10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMinValue(5, true), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "32")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMaxValue(5, true), result);

  EXPECT_EQ(APFloat::opInexact,
            APFloat(APFloat::IEEEdouble, "7.9")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt(APInt(5, 7), true), result);

  result.setIsUnsigned(false);
  EXPECT_EQ(APFloat::opOK,
            APFloat(APFloat::IEEEdouble, "-10")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_TRUE(isExact);
  EXPECT_EQ(APSInt(APInt(5, -10, true), false), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "-17")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMinValue(5, false), result);

  EXPECT_EQ(APFloat::opInvalidOp,
            APFloat(APFloat::IEEEdouble, "16")
            .convertToInteger(result, APFloat::rmTowardZero, &isExact));
  EXPECT_FALSE(isExact);
  EXPECT_EQ(APSInt::getMaxValue(5, false), result);
}

static APInt nanbits(const fltSemantics &Sem,
                     bool SNaN, bool Negative, uint64_t fill) {
  APInt apfill(64, fill);
  if (SNaN)
    return APFloat::getSNaN(Sem, Negative, &apfill).bitcastToAPInt();
  else
    return APFloat::getQNaN(Sem, Negative, &apfill).bitcastToAPInt();
}

TEST(APFloatTest, makeNaN) {
  ASSERT_EQ(0x7fc00000, nanbits(APFloat::IEEEsingle, false, false, 0));
  ASSERT_EQ(0xffc00000, nanbits(APFloat::IEEEsingle, false, true, 0));
  ASSERT_EQ(0x7fc0ae72, nanbits(APFloat::IEEEsingle, false, false, 0xae72));
  ASSERT_EQ(0x7fffae72, nanbits(APFloat::IEEEsingle, false, false, 0xffffae72));
  ASSERT_EQ(0x7fa00000, nanbits(APFloat::IEEEsingle, true, false, 0));
  ASSERT_EQ(0xffa00000, nanbits(APFloat::IEEEsingle, true, true, 0));
  ASSERT_EQ(0x7f80ae72, nanbits(APFloat::IEEEsingle, true, false, 0xae72));
  ASSERT_EQ(0x7fbfae72, nanbits(APFloat::IEEEsingle, true, false, 0xffffae72));

  ASSERT_EQ(0x7ff8000000000000ULL, nanbits(APFloat::IEEEdouble, false, false, 0));
  ASSERT_EQ(0xfff8000000000000ULL, nanbits(APFloat::IEEEdouble, false, true, 0));
  ASSERT_EQ(0x7ff800000000ae72ULL, nanbits(APFloat::IEEEdouble, false, false, 0xae72));
  ASSERT_EQ(0x7fffffffffffae72ULL, nanbits(APFloat::IEEEdouble, false, false, 0xffffffffffffae72ULL));
  ASSERT_EQ(0x7ff4000000000000ULL, nanbits(APFloat::IEEEdouble, true, false, 0));
  ASSERT_EQ(0xfff4000000000000ULL, nanbits(APFloat::IEEEdouble, true, true, 0));
  ASSERT_EQ(0x7ff000000000ae72ULL, nanbits(APFloat::IEEEdouble, true, false, 0xae72));
  ASSERT_EQ(0x7ff7ffffffffae72ULL, nanbits(APFloat::IEEEdouble, true, false, 0xffffffffffffae72ULL));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(APFloatTest, SemanticsDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEsingle, 0.0f).convertToDouble(), "Float semantics are not IEEEdouble");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, 0.0 ).convertToFloat(),  "Float semantics are not IEEEsingle");
}

TEST(APFloatTest, StringDecimalDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ""), "Invalid string length");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+"), "String has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-"), "String has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("\0", 1)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\0", 2)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\02", 3)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\02e1", 5)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e\0", 3)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e1\0", 4)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e1\02", 5)), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0f"), "Invalid character in significand");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ".."), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "..0"), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0.0"), "String contains multiple dots");
}

TEST(APFloatTest, StringDecimalSignificandDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-."), "Significand has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-e"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-e1"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ".e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+.e1"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-.e1"), "Significand has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ".e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+.e"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-.e"), "Significand has no digits");
}

TEST(APFloatTest, StringDecimalExponentDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   "1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-1e"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   "1.e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+1.e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-1.e"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   ".1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+.1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-.1e"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,   "1.1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "+1.1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "-1.1e"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1e-"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  ".1e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ".1e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ".1e-"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "1.0e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0e-"), "Exponent has no digits");
}

TEST(APFloatTest, StringHexadecimalDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x"), "Invalid string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x"), "Invalid string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x"), "Invalid string");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x0"), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x0."), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x0."), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x0."), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.0"), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x0.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x0.0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x0.0"), "Hex strings require an exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x\0", 3)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\0", 4)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\02", 5)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\02p1", 7)), "Invalid character in significand");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p\0", 5)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p1\0", 6)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p1\02", 7)), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x1p0f"), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x..p1"), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x..0p1"), "String contains multiple dots");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x1.0.0p1"), "String contains multiple dots");
}

TEST(APFloatTest, StringHexadecimalSignificandDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x."), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x."), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0xp"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0xp"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0xp"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0xp+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0xp+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0xp+"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0xp-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0xp-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0xp-"), "Significand has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.p"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.p"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.p"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.p+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.p+"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.p+"), "Significand has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.p-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.p-"), "Significand has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.p-"), "Significand has no digits");
}

TEST(APFloatTest, StringHexadecimalExponentDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1p-"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.p-"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.1p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.1p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x.1p-"), "Exponent has no digits");


  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.1p"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.1p"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.1p+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.1p+"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble,  "0x1.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "+0x1.1p-"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x1.1p-"), "Exponent has no digits");
}
#endif
#endif

TEST(APFloatTest, exactInverse) {
  APFloat inv(0.0f);

  // Trivial operation.
  EXPECT_TRUE(APFloat(2.0).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(0.5)));
  EXPECT_TRUE(APFloat(2.0f).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(0.5f)));
  EXPECT_TRUE(APFloat(APFloat::IEEEquad, "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::IEEEquad, "0.5")));
  EXPECT_TRUE(APFloat(APFloat::PPCDoubleDouble, "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::PPCDoubleDouble, "0.5")));
  EXPECT_TRUE(APFloat(APFloat::x87DoubleExtended, "2.0").getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(APFloat::x87DoubleExtended, "0.5")));

  // FLT_MIN
  EXPECT_TRUE(APFloat(1.17549435e-38f).getExactInverse(&inv));
  EXPECT_TRUE(inv.bitwiseIsEqual(APFloat(8.5070592e+37f)));

  // Large float, inverse is a denormal.
  EXPECT_FALSE(APFloat(1.7014118e38f).getExactInverse(0));
  // Zero
  EXPECT_FALSE(APFloat(0.0).getExactInverse(0));
  // Denormalized float
  EXPECT_FALSE(APFloat(1.40129846e-45f).getExactInverse(0));
}

TEST(APFloatTest, roundToIntegral) {
  APFloat T(-0.5), S(3.14), R(APFloat::getLargest(APFloat::IEEEdouble)), P(0.0);

  P = T;
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = T;
  P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(-1.0, P.convertToDouble());
  P = T;
  P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = T;
  P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(-0.0, P.convertToDouble());

  P = S;
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(3.0, P.convertToDouble());
  P = S;
  P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(3.0, P.convertToDouble());
  P = S;
  P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(4.0, P.convertToDouble());
  P = S;
  P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(3.0, P.convertToDouble());

  P = R;
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());
  P = R;
  P.roundToIntegral(APFloat::rmTowardNegative);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());
  P = R;
  P.roundToIntegral(APFloat::rmTowardPositive);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());
  P = R;
  P.roundToIntegral(APFloat::rmNearestTiesToEven);
  EXPECT_EQ(R.convertToDouble(), P.convertToDouble());

  P = APFloat::getZero(APFloat::IEEEdouble);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(0.0, P.convertToDouble());
  P = APFloat::getZero(APFloat::IEEEdouble, true);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_EQ(-0.0, P.convertToDouble());
  P = APFloat::getNaN(APFloat::IEEEdouble);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(IsNAN(P.convertToDouble()));
  P = APFloat::getInf(APFloat::IEEEdouble);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(IsInf(P.convertToDouble()) && P.convertToDouble() > 0.0);
  P = APFloat::getInf(APFloat::IEEEdouble, true);
  P.roundToIntegral(APFloat::rmTowardZero);
  EXPECT_TRUE(IsInf(P.convertToDouble()) && P.convertToDouble() < 0.0);

}

TEST(APFloatTest, getLargest) {
  EXPECT_EQ(3.402823466e+38f, APFloat::getLargest(APFloat::IEEEsingle).convertToFloat());
  EXPECT_EQ(1.7976931348623158e+308, APFloat::getLargest(APFloat::IEEEdouble).convertToDouble());
}

TEST(APFloatTest, getSmallest) {
  APFloat test = APFloat::getSmallest(APFloat::IEEEsingle, false);
  APFloat expected = APFloat(APFloat::IEEEsingle, "0x0.000002p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEsingle, true);
  expected = APFloat(APFloat::IEEEsingle, "-0x0.000002p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEquad, false);
  expected = APFloat(APFloat::IEEEquad, "0x0.0000000000000000000000000001p-16382");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallest(APFloat::IEEEquad, true);
  expected = APFloat(APFloat::IEEEquad, "-0x0.0000000000000000000000000001p-16382");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_TRUE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));  
}

TEST(APFloatTest, getSmallestNormalized) {
  APFloat test = APFloat::getSmallestNormalized(APFloat::IEEEsingle, false);
  APFloat expected = APFloat(APFloat::IEEEsingle, "0x1p-126");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallestNormalized(APFloat::IEEEsingle, true);
  expected = APFloat(APFloat::IEEEsingle, "-0x1p-126");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallestNormalized(APFloat::IEEEquad, false);
  expected = APFloat(APFloat::IEEEquad, "0x1p-16382");
  EXPECT_FALSE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));

  test = APFloat::getSmallestNormalized(APFloat::IEEEquad, true);
  expected = APFloat(APFloat::IEEEquad, "-0x1p-16382");
  EXPECT_TRUE(test.isNegative());
  EXPECT_TRUE(test.isFiniteNonZero());
  EXPECT_FALSE(test.isDenormal());
  EXPECT_TRUE(test.bitwiseIsEqual(expected));  
}

TEST(APFloatTest, getZero) {
  struct {
    const fltSemantics *semantics;
    const bool sign;
    const unsigned long long bitPattern[2];
    const unsigned bitPatternLength;
  } const GetZeroTest[] = {
    { &APFloat::IEEEhalf, false, {0, 0}, 1},
    { &APFloat::IEEEhalf, true, {0x8000ULL, 0}, 1},
    { &APFloat::IEEEsingle, false, {0, 0}, 1},
    { &APFloat::IEEEsingle, true, {0x80000000ULL, 0}, 1},
    { &APFloat::IEEEdouble, false, {0, 0}, 1},
    { &APFloat::IEEEdouble, true, {0x8000000000000000ULL, 0}, 1},
    { &APFloat::IEEEquad, false, {0, 0}, 2},
    { &APFloat::IEEEquad, true, {0, 0x8000000000000000ULL}, 2},
    { &APFloat::PPCDoubleDouble, false, {0, 0}, 2},
    { &APFloat::PPCDoubleDouble, true, {0x8000000000000000ULL, 0}, 2},
    { &APFloat::x87DoubleExtended, false, {0, 0}, 2},
    { &APFloat::x87DoubleExtended, true, {0, 0x8000ULL}, 2},
  };
  const unsigned NumGetZeroTests = 12;
  for (unsigned i = 0; i < NumGetZeroTests; ++i) {
    APFloat test = APFloat::getZero(*GetZeroTest[i].semantics,
				    GetZeroTest[i].sign);
    const char *pattern = GetZeroTest[i].sign? "-0x0p+0" : "0x0p+0";
    APFloat expected = APFloat(*GetZeroTest[i].semantics,
			       pattern);
    EXPECT_TRUE(test.isZero());
    EXPECT_TRUE(GetZeroTest[i].sign? test.isNegative() : !test.isNegative());
    EXPECT_TRUE(test.bitwiseIsEqual(expected));
    for (unsigned j = 0, je = GetZeroTest[i].bitPatternLength; j < je; ++j) {
      EXPECT_EQ(GetZeroTest[i].bitPattern[j],
		test.bitcastToAPInt().getRawData()[j]);
    }
  }
}

TEST(APFloatTest, convert) {
  bool losesInfo;
  APFloat test(APFloat::IEEEdouble, "1.0");
  test.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0f, test.convertToFloat());
  EXPECT_FALSE(losesInfo);

  test = APFloat(APFloat::x87DoubleExtended, "0x1p-53");
  test.add(APFloat(APFloat::x87DoubleExtended, "1.0"), APFloat::rmNearestTiesToEven);
  test.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0, test.convertToDouble());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::IEEEquad, "0x1p-53");
  test.add(APFloat(APFloat::IEEEquad, "1.0"), APFloat::rmNearestTiesToEven);
  test.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(1.0, test.convertToDouble());
  EXPECT_TRUE(losesInfo);

  test = APFloat(APFloat::x87DoubleExtended, "0xf.fffffffp+28");
  test.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
  EXPECT_EQ(4294967295.0, test.convertToDouble());
  EXPECT_FALSE(losesInfo);

  test = APFloat::getSNaN(APFloat::IEEEsingle);
  APFloat X87SNaN = APFloat::getSNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87SNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getQNaN(APFloat::IEEEsingle);
  APFloat X87QNaN = APFloat::getQNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87QNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getSNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87SNaN));
  EXPECT_FALSE(losesInfo);

  test = APFloat::getQNaN(APFloat::x87DoubleExtended);
  test.convert(APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven,
               &losesInfo);
  EXPECT_TRUE(test.bitwiseIsEqual(X87QNaN));
  EXPECT_FALSE(losesInfo);
}

TEST(APFloatTest, PPCDoubleDouble) {
  APFloat test(APFloat::PPCDoubleDouble, "1.0");
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  test.divide(APFloat(APFloat::PPCDoubleDouble, "3.0"), APFloat::rmNearestTiesToEven);
  EXPECT_EQ(0x3fd5555555555555ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x3c75555555555556ull, test.bitcastToAPInt().getRawData()[1]);

  // LDBL_MAX
  test = APFloat(APFloat::PPCDoubleDouble, "1.79769313486231580793728971405301e+308");
  EXPECT_EQ(0x7fefffffffffffffull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x7c8ffffffffffffeull, test.bitcastToAPInt().getRawData()[1]);

  // LDBL_MIN
  test = APFloat(APFloat::PPCDoubleDouble, "2.00416836000897277799610805135016e-292");
  EXPECT_EQ(0x0360000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  test = APFloat(APFloat::PPCDoubleDouble, "1.0");
  test.add(APFloat(APFloat::PPCDoubleDouble, "0x1p-105"), APFloat::rmNearestTiesToEven);
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
  EXPECT_EQ(0x3960000000000000ull, test.bitcastToAPInt().getRawData()[1]);

  test = APFloat(APFloat::PPCDoubleDouble, "1.0");
  test.add(APFloat(APFloat::PPCDoubleDouble, "0x1p-106"), APFloat::rmNearestTiesToEven);
  EXPECT_EQ(0x3ff0000000000000ull, test.bitcastToAPInt().getRawData()[0]);
#if 0 // XFAIL
  // This is what we would expect with a true double-double implementation
  EXPECT_EQ(0x3950000000000000ull, test.bitcastToAPInt().getRawData()[1]);
#else
  // This is what we get with our 106-bit mantissa approximation
  EXPECT_EQ(0x0000000000000000ull, test.bitcastToAPInt().getRawData()[1]);
#endif
}

TEST(APFloatTest, isNegative) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_FALSE(t.isNegative());
  t = APFloat(APFloat::IEEEsingle, "-0x1p+0");
  EXPECT_TRUE(t.isNegative());
  
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getInf(APFloat::IEEEsingle, true).isNegative());
  
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getZero(APFloat::IEEEsingle, true).isNegative());

  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getNaN(APFloat::IEEEsingle, true).isNegative());

  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isNegative());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, true).isNegative());
}

TEST(APFloatTest, isNormal) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_TRUE(t.isNormal());
  
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isNormal());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isNormal());  
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle, "0x1p-149").isNormal());
}

TEST(APFloatTest, isFinite) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_TRUE(t.isFinite());  
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isFinite());
  EXPECT_TRUE(APFloat::getZero(APFloat::IEEEsingle, false).isFinite());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isFinite());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isFinite());  
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "0x1p-149").isFinite());  
}

TEST(APFloatTest, isInfinity) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_FALSE(t.isInfinity());
  EXPECT_TRUE(APFloat::getInf(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isInfinity());
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isInfinity());  
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle, "0x1p-149").isInfinity());
}

TEST(APFloatTest, isNaN) {
  APFloat t(APFloat::IEEEsingle, "0x1p+0");
  EXPECT_FALSE(t.isNaN());  
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isNaN());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isNaN());
  EXPECT_TRUE(APFloat::getNaN(APFloat::IEEEsingle, false).isNaN());
  EXPECT_TRUE(APFloat::getSNaN(APFloat::IEEEsingle, false).isNaN());  
  EXPECT_FALSE(APFloat(APFloat::IEEEsingle, "0x1p-149").isNaN());
}

TEST(APFloatTest, isFiniteNonZero) {
  // Test positive/negative normal value.
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "0x1p+0").isFiniteNonZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "-0x1p+0").isFiniteNonZero());

  // Test positive/negative denormal value.
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "0x1p-149").isFiniteNonZero());
  EXPECT_TRUE(APFloat(APFloat::IEEEsingle, "-0x1p-149").isFiniteNonZero());

  // Test +/- Infinity.
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getInf(APFloat::IEEEsingle, true).isFiniteNonZero());

  // Test +/- Zero.
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getZero(APFloat::IEEEsingle, true).isFiniteNonZero());

  // Test +/- qNaN. +/- dont mean anything with qNaN but paranoia can't hurt in
  // this instance.
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, false).isFiniteNonZero());
  EXPECT_FALSE(APFloat::getNaN(APFloat::IEEEsingle, true).isFiniteNonZero());

  // Test +/- sNaN. +/- dont mean anything with sNaN but paranoia can't hurt in
  // this instance.
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, false).isFiniteNonZero()); 
  EXPECT_FALSE(APFloat::getSNaN(APFloat::IEEEsingle, true).isFiniteNonZero()); 
}

}
