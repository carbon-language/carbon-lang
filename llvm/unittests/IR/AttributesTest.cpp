//===- llvm/unittest/IR/AttributesTest.cpp - Attributes unit tests --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Attributes.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"
using namespace llvm;

namespace {

TEST(Attributes, Uniquing) {
  LLVMContext C;

  Attribute AttrA = Attribute::get(C, Attribute::AlwaysInline);
  Attribute AttrB = Attribute::get(C, Attribute::AlwaysInline);
  EXPECT_EQ(AttrA, AttrB);

  AttributeSet ASs[] = {
    AttributeSet::get(C, 1, Attribute::ZExt),
    AttributeSet::get(C, 2, Attribute::SExt)
  };

  AttributeSet SetA = AttributeSet::get(C, ASs);
  AttributeSet SetB = AttributeSet::get(C, ASs);
  EXPECT_EQ(SetA, SetB);
}

} // end anonymous namespace
