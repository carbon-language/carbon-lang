//===- unittests/Driver/UtilsTest.cpp --- Utils tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Unit tests for Driver/Util API.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Util.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Triple.h"
#include "gtest/gtest.h"

using namespace clang::driver;
using namespace clang;

TEST(UtilsTest, getARMCPUForMArch) {
  {
    llvm::Triple Triple("armv7s-apple-ios7");
    EXPECT_STREQ("swift", getARMCPUForMArch(Triple.getArchName(), Triple));
  }
  {
    llvm::Triple Triple("armv7-apple-ios7");
    EXPECT_STREQ("cortex-a8", getARMCPUForMArch(Triple.getArchName(), Triple));
  }
}
