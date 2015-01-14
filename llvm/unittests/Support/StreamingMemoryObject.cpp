//===- llvm/unittest/Support/StreamingMemoryObject.cpp - unit tests -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StreamingMemoryObject.h"
#include "gtest/gtest.h"
#include <string.h>

using namespace llvm;

namespace {
class NullDataStreamer : public DataStreamer {
  size_t GetBytes(unsigned char *buf, size_t len) override {
    memset(buf, 0, len);
    return len;
  }
};
}

TEST(StreamingMemoryObject, Test) {
  auto *DS = new NullDataStreamer();
  StreamingMemoryObject O(DS);
  EXPECT_TRUE(O.isValidAddress(32 * 1024));
}
