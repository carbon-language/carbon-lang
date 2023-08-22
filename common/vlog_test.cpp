// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/vlog.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "testing/base/test_raw_ostream.h"

namespace Carbon::Testing {
namespace {

using ::testing::IsEmpty;
using ::testing::StrEq;

// Helper class with a vlog_stream_ member for CARBON_VLOG.
class VLogger {
 public:
  explicit VLogger(bool enable) {
    if (enable) {
      vlog_stream_ = &buffer_;
    }
  }

  void VLog() { CARBON_VLOG() << "Test\n"; }

  auto TakeStr() -> std::string { return buffer_.TakeStr(); }

 private:
  TestRawOstream buffer_;

  llvm::raw_ostream* vlog_stream_ = nullptr;
};

TEST(VLogTest, Enabled) {
  VLogger vlog(/*enable=*/true);
  vlog.VLog();
  EXPECT_THAT(vlog.TakeStr(), StrEq("Test\n"));
}

TEST(VLogTest, Disabled) {
  VLogger vlog(/*enable=*/false);
  vlog.VLog();
  EXPECT_THAT(vlog.TakeStr(), IsEmpty());
}

}  // namespace
}  // namespace Carbon::Testing
