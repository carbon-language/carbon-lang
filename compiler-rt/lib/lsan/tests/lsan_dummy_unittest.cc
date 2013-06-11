//===-- lsan_dummy_unittest.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"

TEST(LeakSanitizer, EmptyTest) {
  // Empty test to suppress LIT warnings about lack of tests.
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
