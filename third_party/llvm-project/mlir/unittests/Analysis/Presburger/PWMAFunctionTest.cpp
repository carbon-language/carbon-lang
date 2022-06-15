//===- PWMAFunctionTest.cpp - Tests for PWMAFunction ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for PWMAFunction.
//
//===----------------------------------------------------------------------===//

#include "./Utils.h"

#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/MLIRContext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

using testing::ElementsAre;

TEST(PWAFunctionTest, isEqual) {
  // The output expressions are different but it doesn't matter because they are
  // equal in this domain.
  PWMAFunction idAtZeros = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (y == 0)", {{1, 0, 0}, {0, 1, 0}}},             // (x, y).
          {"(x, y) : (y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 1, 0}}}, // (x, y).
          {"(x, y) : (-y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 1, 0}}} // (x, y).
      });
  PWMAFunction idAtZeros2 = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (y == 0)", {{1, 0, 0}, {0, 20, 0}}}, // (x, 20y).
          {"(x, y) : (y - 1 >= 0, x == 0)", {{30, 0, 0}, {0, 1, 0}}}, //(30x, y)
          {"(x, y) : (-y - 1 > =0, x == 0)", {{30, 0, 0}, {0, 1, 0}}} //(30x, y)
      });
  EXPECT_TRUE(idAtZeros.isEqual(idAtZeros2));

  PWMAFunction notIdAtZeros = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (y == 0)", {{1, 0, 0}, {0, 1, 0}}},              // (x, y).
          {"(x, y) : (y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 2, 0}}},  // (x, 2y)
          {"(x, y) : (-y - 1 >= 0, x == 0)", {{1, 0, 0}, {0, 2, 0}}}, // (x, 2y)
      });
  EXPECT_FALSE(idAtZeros.isEqual(notIdAtZeros));

  // These match at their intersection but one has a bigger domain.
  PWMAFunction idNoNegNegQuadrant = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0)", {{1, 0, 0}, {0, 1, 0}}},             // (x, y).
          {"(x, y) : (-x - 1 >= 0, y >= 0)", {{1, 0, 0}, {0, 1, 0}}} // (x, y).
      });
  PWMAFunction idOnlyPosX =
      parsePWMAF(/*numInputs=*/2, /*numOutputs=*/2,
                 {
                     {"(x, y) : (x >= 0)", {{1, 0, 0}, {0, 1, 0}}}, // (x, y).
                 });
  EXPECT_FALSE(idNoNegNegQuadrant.isEqual(idOnlyPosX));

  // Different representations of the same domain.
  PWMAFunction sumPlusOne = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x >= 0)", {{1, 1, 1}}},                   // x + y + 1.
          {"(x, y) : (-x - 1 >= 0, -y - 1 >= 0)", {{1, 1, 1}}}, // x + y + 1.
          {"(x, y) : (-x - 1 >= 0, y >= 0)", {{1, 1, 1}}}       // x + y + 1.
      });
  PWMAFunction sumPlusOne2 =
      parsePWMAF(/*numInputs=*/2, /*numOutputs=*/1,
                 {
                     {"(x, y) : ()", {{1, 1, 1}}}, // x + y + 1.
                 });
  EXPECT_TRUE(sumPlusOne.isEqual(sumPlusOne2));

  // Functions with zero input dimensions.
  PWMAFunction noInputs1 = parsePWMAF(/*numInputs=*/0, /*numOutputs=*/1,
                                      {
                                          {"() : ()", {{1}}}, // 1.
                                      });
  PWMAFunction noInputs2 = parsePWMAF(/*numInputs=*/0, /*numOutputs=*/1,
                                      {
                                          {"() : ()", {{2}}}, // 1.
                                      });
  EXPECT_TRUE(noInputs1.isEqual(noInputs1));
  EXPECT_FALSE(noInputs1.isEqual(noInputs2));

  // Mismatched dimensionalities.
  EXPECT_FALSE(noInputs1.isEqual(sumPlusOne));
  EXPECT_FALSE(idOnlyPosX.isEqual(sumPlusOne));

  // Divisions.
  // Domain is only multiples of 6; x = 6k for some k.
  // x + 4(x/2) + 4(x/3) == 26k.
  PWMAFunction mul2AndMul3 = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 2*(x floordiv 2) == 0, x - 3*(x floordiv 3) == 0)",
           {{1, 4, 4, 0}}}, // x + 4(x/2) + 4(x/3).
      });
  PWMAFunction mul6 = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 6*(x floordiv 6) == 0)", {{0, 26, 0}}}, // 26(x/6).
      });
  EXPECT_TRUE(mul2AndMul3.isEqual(mul6));

  PWMAFunction mul6diff = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 5*(x floordiv 5) == 0)", {{0, 52, 0}}}, // 52(x/6).
      });
  EXPECT_FALSE(mul2AndMul3.isEqual(mul6diff));

  PWMAFunction mul5 = parsePWMAF(
      /*numInputs=*/1, /*numOutputs=*/1,
      {
          {"(x) : (x - 5*(x floordiv 5) == 0)", {{0, 26, 0}}}, // 26(x/5).
      });
  EXPECT_FALSE(mul2AndMul3.isEqual(mul5));
}

TEST(PWMAFunction, valueAt) {
  PWMAFunction nonNegPWMAF = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0)", {{1, 2, 3}, {3, 4, 5}}}, // (x, y).
          {"(x, y) : (y >= 0, -x - 1 >= 0)", {{-1, 2, 3}, {-3, 4, 5}}} // (x, y)
      });
  EXPECT_THAT(*nonNegPWMAF.valueAt({2, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*nonNegPWMAF.valueAt({-2, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*nonNegPWMAF.valueAt({2, -3}), ElementsAre(-1, -1));
  EXPECT_FALSE(nonNegPWMAF.valueAt({-2, -3}).hasValue());

  PWMAFunction divPWMAF = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/2,
      {
          {"(x, y) : (x >= 0, x - 2*(x floordiv 2) == 0)",
           {{0, 2, 1, 3}, {0, 4, 3, 5}}}, // (x, y).
          {"(x, y) : (y >= 0, -x - 1 >= 0)", {{-1, 2, 3}, {-3, 4, 5}}} // (x, y)
      });
  EXPECT_THAT(*divPWMAF.valueAt({4, 3}), ElementsAre(11, 23));
  EXPECT_THAT(*divPWMAF.valueAt({4, -3}), ElementsAre(-1, -1));
  EXPECT_FALSE(divPWMAF.valueAt({3, 3}).hasValue());
  EXPECT_FALSE(divPWMAF.valueAt({3, -3}).hasValue());

  EXPECT_THAT(*divPWMAF.valueAt({-2, 3}), ElementsAre(11, 23));
  EXPECT_FALSE(divPWMAF.valueAt({-2, -3}).hasValue());
}

TEST(PWMAFunction, removeIdRangeRegressionTest) {
  PWMAFunction pwmafA = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x == 0, y == 0, x - 2*(x floordiv 2) == 0, y - 2*(y "
           "floordiv 2) == 0)",
           {{0, 0, 0, 0, 0}}} // (0, 0)
      });
  PWMAFunction pwmafB = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x - 11*y == 0, 11*x - y == 0, x - 2*(x floordiv 2) == 0, "
           "y - 2*(y floordiv 2) == 0)",
           {{0, 0, 0, 0, 0}}} // (0, 0)
      });
  EXPECT_TRUE(pwmafA.isEqual(pwmafB));
}

TEST(PWMAFunction, eliminateRedundantLocalIdRegressionTest) {
  PWMAFunction pwmafA = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x - 2*(x floordiv 2) == 0, x - 2*y == 0)",
           {{0, 1, 0, 0}}} // (0, 0)
      });
  PWMAFunction pwmafB = parsePWMAF(
      /*numInputs=*/2, /*numOutputs=*/1,
      {
          {"(x, y) : (x - 2*(x floordiv 2) == 0, x - 2*y == 0)",
           {{1, -1, 0, 0}}} // (0, 0)
      });
  EXPECT_TRUE(pwmafA.isEqual(pwmafB));
}
