//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {

static void PrintTo(const StringRef &S, ::std::ostream *os) {
  *os << "(" << (const void *)S.data() << "," << S.size() << ") = '";
  for (auto C : S)
    if (C)
      *os << C;
    else
      *os << "\\00";
  *os << "'";
}
static void PrintTo(const DIHeaderFieldIterator &I, ::std::ostream *os) {
  PrintTo(I.getCurrent(), os);
  *os << " in ";
  PrintTo(I.getHeader(), os);
}

} // end namespace llvm

namespace {

#define MAKE_FIELD_ITERATOR(S)                                                 \
  DIHeaderFieldIterator(StringRef(S, sizeof(S) - 1))
TEST(DebugInfoTest, DIHeaderFieldIterator) {
  ASSERT_EQ(DIHeaderFieldIterator(), DIHeaderFieldIterator());

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR(""));
  ASSERT_EQ(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR(""));
  ASSERT_EQ("", *DIHeaderFieldIterator(""));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("stuff"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("stuff"));
  ASSERT_EQ("stuff", *DIHeaderFieldIterator("stuff"));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_NE(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++++MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_EQ("st", *MAKE_FIELD_ITERATOR("st\0uff"));
  ASSERT_EQ("uff", *++MAKE_FIELD_ITERATOR("st\0uff"));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_NE(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++++MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_EQ("stuff", *MAKE_FIELD_ITERATOR("stuff\0"));
  ASSERT_EQ("", *++MAKE_FIELD_ITERATOR("stuff\0"));

  ASSERT_NE(DIHeaderFieldIterator(), MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_NE(DIHeaderFieldIterator(), ++MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_EQ(DIHeaderFieldIterator(), ++++MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_EQ("", *MAKE_FIELD_ITERATOR("\0stuff"));
  ASSERT_EQ("stuff", *++MAKE_FIELD_ITERATOR("\0stuff"));
}

} // end namespace
