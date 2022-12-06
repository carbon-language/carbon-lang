// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/vlog.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

using ::testing::IsEmpty;
using ::testing::StrEq;

// Helper class with a vlog_stream_ member for CARBON_VLOG.
class VLogger {
 public:
  explicit VLogger(bool enable) : buffer_(buffer_str_) {
    if (enable) {
      vlog_stream_ = &buffer_;
    }
  }

  void VLog() { CARBON_VLOG() << "Test\n"; }

  auto buffer() -> llvm::StringRef { return buffer_str_; }

 private:
  std::string buffer_str_;
  llvm::raw_string_ostream buffer_;

  llvm::raw_ostream* vlog_stream_ = nullptr;
};

TEST(VLogTest, Enabled) {
  VLogger vlog(/*enable=*/true);
  vlog.VLog();
  EXPECT_THAT(vlog.buffer(), StrEq("Test\n"));
}

TEST(VLogTest, Disabled) {
  VLogger vlog(/*enable=*/false);
  vlog.VLog();
  EXPECT_THAT(vlog.buffer(), IsEmpty());
}

}  // namespace
}  // namespace Carbon::Testing
