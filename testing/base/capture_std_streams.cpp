// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing/base/capture_std_streams.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "common/ostream.h"

namespace Carbon::Testing::Internal {

// While these are marked as "internal" APIs, they seem to work and be pretty
// widely used for their exact documented behavior.
using ::testing::internal::CaptureStderr;
using ::testing::internal::CaptureStdout;
using ::testing::internal::GetCapturedStderr;
using ::testing::internal::GetCapturedStdout;

auto BeginStdStreamCapture() -> void {
  CaptureStderr();
  CaptureStdout();
}
auto EndStdStreamCapture(std::string& out, std::string& err) -> void {
  // No need to flush stderr.
  err = GetCapturedStderr();
  llvm::outs().flush();
  out = GetCapturedStdout();
}

}  // namespace Carbon::Testing::Internal
