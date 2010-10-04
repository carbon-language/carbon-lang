//===- llvm/unittest/Support/System.cpp - System tests --===//
#include "gtest/gtest.h"
#include "llvm/System/TimeValue.h"
#include <time.h>

using namespace llvm;
namespace {
class SystemTest : public ::testing::Test {
};

TEST_F(SystemTest, TimeValue) {
  sys::TimeValue now = sys::TimeValue::now();
  time_t now_t = time(NULL);
  EXPECT_TRUE(abs(static_cast<long>(now_t - now.toEpochTime())) < 2);
}
}
