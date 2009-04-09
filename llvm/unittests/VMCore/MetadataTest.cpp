//===- llvm/unittest/VMCore/Metadata.cpp - Metadata unit tests ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/Constants.h"
#include <sstream>

using namespace llvm;

namespace {

// Test that construction of MDString with different value produces different
// MDString objects, even with the same string pointer and nulls in the string.
TEST(MDStringTest, CreateDifferent) {
  char x[3] = { 'f', 0, 'A' };
  MDString *s1 = MDString::get(&x[0], &x[3]);
  x[2] = 'B';
  MDString *s2 = MDString::get(&x[0], &x[3]);
  EXPECT_NE(s1, s2);
}

// Test that creation of MDStrings with the same string contents produces the
// same MDString object, even with different pointers.
TEST(MDStringTest, CreateSame) {
  char x[4] = { 'a', 'b', 'c', 'X' };
  char y[4] = { 'a', 'b', 'c', 'Y' };

  MDString *s1 = MDString::get(&x[0], &x[3]);
  MDString *s2 = MDString::get(&y[0], &y[3]);
  EXPECT_EQ(s1, s2);
}

// Test that MDString prints out the string we fed it.
TEST(MDStringTest, PrintingSimple) {
  char *str = new char[13];
  strncpy(str, "testing 1 2 3", 13);
  MDString *s = MDString::get(str, str+13);
  strncpy(str, "aaaaaaaaaaaaa", 13);
  delete[] str;

  std::ostringstream oss;
  s->print(oss);
  EXPECT_STREQ("{ } !\"testing 1 2 3\"", oss.str().c_str());
}

// Test printing of MDString with non-printable characters.
TEST(MDStringTest, PrintingComplex) {
  char str[5] = {0, '\n', '"', '\\', -1};
  MDString *s = MDString::get(str+0, str+5);
  std::ostringstream oss;
  s->print(oss);
  EXPECT_STREQ("{ } !\"\\00\\0A\\22\\5C\\FF\"", oss.str().c_str());
}

// Test the two constructors, and containing other Constants.
TEST(MDNodeTest, Everything) {
  char x[3] = { 'a', 'b', 'c' };
  char y[3] = { '1', '2', '3' };

  MDString *s1 = MDString::get(&x[0], &x[3]);
  MDString *s2 = MDString::get(&y[0], &y[3]);
  ConstantInt *CI = ConstantInt::get(APInt(8, 0));

  std::vector<Constant *> V;
  V.push_back(s1);
  V.push_back(CI);
  V.push_back(s2);

  MDNode *n1 = MDNode::get(&V[0], 3);
  Constant *const c1 = n1;
  MDNode *n2 = MDNode::get(&c1, 1);
  MDNode *n3 = MDNode::get(&V[0], 3);
  EXPECT_NE(n1, n2);
  EXPECT_EQ(n1, n3);

  EXPECT_EQ(3u, n1->getNumOperands());
  EXPECT_EQ(s1, n1->getOperand(0));
  EXPECT_EQ(CI, n1->getOperand(1));
  EXPECT_EQ(s2, n1->getOperand(2));

  EXPECT_EQ(1u, n2->getNumOperands());
  EXPECT_EQ(n1, n2->getOperand(0));

  std::ostringstream oss1, oss2;
  n1->print(oss1);
  n2->print(oss2);
  EXPECT_STREQ("{ } !{{ } !\"abc\", i8 0, { } !\"123\"}", oss1.str().c_str());
  EXPECT_STREQ("{ } !{{ } !{{ } !\"abc\", i8 0, { } !\"123\"}}",
               oss2.str().c_str());
}
}
