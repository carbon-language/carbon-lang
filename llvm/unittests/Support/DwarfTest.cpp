//===- unittest/Support/DwarfTest.cpp - Dwarf support tests ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Dwarf.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dwarf;

namespace {

TEST(DwarfTest, TagStringOnInvalid) {
  // This is invalid, so it shouldn't be stringified.
  EXPECT_EQ(nullptr, TagString(DW_TAG_invalid));

  // These aren't really tags: they describe ranges within tags.  They
  // shouldn't be stringified either.
  EXPECT_EQ(nullptr, TagString(DW_TAG_lo_user));
  EXPECT_EQ(nullptr, TagString(DW_TAG_hi_user));
  EXPECT_EQ(nullptr, TagString(DW_TAG_user_base));
}

} // end namespace
