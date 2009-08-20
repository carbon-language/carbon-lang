//===- llvm/unittest/ADT/APFloat.cpp - APFloat unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ostream>
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"

using namespace llvm;

namespace {

TEST(APFloatTest, Zero) {
  EXPECT_EQ(0.0f,  APFloat(APFloat::IEEEsingle,  0.0f).convertToFloat());
  EXPECT_EQ(-0.0f, APFloat(APFloat::IEEEsingle, -0.0f).convertToFloat());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble,  0.0).convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, -0.0).convertToDouble());
}

TEST(APFloatTest, fromString) {
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0.").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, ".0").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-.0").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0.0").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0e1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0e1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "00000.").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0000.00000").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, ".00000").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0.").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0.e1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0.e+1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0.e-1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e0").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e-0").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "000.0000e-1234").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x0p1").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x00000.p1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x0000.00000p1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x.00000p1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x0.p1").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x0p1234").convertToDouble());
  EXPECT_EQ(-0.0, APFloat(APFloat::IEEEdouble, "-0x0p1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x00000.p1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x0000.00000p1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x.00000p1234").convertToDouble());
  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, "0x0.p1234").convertToDouble());
  EXPECT_EQ(1.0625,  APFloat(APFloat::IEEEdouble, "0x1.1p0").convertToDouble());
  EXPECT_EQ(1.0,  APFloat(APFloat::IEEEdouble, "0x1p0").convertToDouble());

  EXPECT_EQ(0.0,  APFloat(APFloat::IEEEdouble, StringRef("0e1\02", 3)).convertToDouble());
}

#ifdef GTEST_HAS_DEATH_TEST
TEST(APFloatTest, SemanticsDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEsingle, 0.0f).convertToDouble(), "Float semantics are not IEEEdouble");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, 0.0 ).convertToFloat(),  "Float semantics are not IEEEsingle");
}

TEST(APFloatTest, StringDeath) {
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, ""), "Invalid string length");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-"), "String is only a minus!");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x"), "Invalid string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "."), "String cannot be just a dot");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-."), "String cannot be just a dot");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x."), "String cannot be just a dot");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "-0x."),"String cannot be just a dot");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x0"), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0e"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0e+"), "Exponent has no digits");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0e-"), "Exponent has no digits");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("\0", 1)), "Invalid character in digit string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\0", 2)), "Invalid character in digit string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\02", 3)), "Invalid character in digit string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1\02e1", 5)), "Invalid character in digit string");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e\0", 3)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e1\0", 4)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("1e1\02", 5)), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "1.0f"), "Invalid character in digit string");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x\0", 3)), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\0", 4)), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\02", 5)), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1\02p1", 7)), "Hex strings require an exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p\0", 5)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p1\0", 6)), "Invalid character in exponent");
  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, StringRef("0x1p1\02", 7)), "Invalid character in exponent");

  EXPECT_DEATH(APFloat(APFloat::IEEEdouble, "0x1p0f"), "Invalid character in exponent");
}
#endif

}
