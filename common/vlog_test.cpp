// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/vlog.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/raw_string_ostream.h"

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

  auto VLog() -> void { CARBON_VLOG("Test\n"); }
  auto VLogFormatArgs() -> void { CARBON_VLOG("Test {0} {1} {2}\n", 1, 2, 3); }

  auto TakeStr() -> std::string { return buffer_.TakeStr(); }

 private:
  RawStringOstream buffer_;

  llvm::raw_ostream* vlog_stream_ = nullptr;
};

TEST(VLogTest, Enabled) {
  VLogger vlog(/*enable=*/true);
  vlog.VLog();
  EXPECT_THAT(vlog.TakeStr(), StrEq("Test\n"));
  vlog.VLogFormatArgs();
  EXPECT_THAT(vlog.TakeStr(), StrEq("Test 1 2 3\n"));
}

TEST(VLogTest, Disabled) {
  VLogger vlog(/*enable=*/false);
  vlog.VLog();
  EXPECT_THAT(vlog.TakeStr(), IsEmpty());
}

TEST(VLogTest, To) {
  RawStringOstream buffer;
  CARBON_VLOG_TO(&buffer, "Test");
  EXPECT_THAT(buffer.TakeStr(), "Test");
}

TEST(VLogTest, ToNull) { CARBON_VLOG_TO(nullptr, "Unused"); }

}  // namespace
}  // namespace Carbon::Testing
