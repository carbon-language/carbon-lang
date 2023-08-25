// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TESTING_BASE_TEST_RAW_OSTREAM_H_
#define CARBON_TESTING_BASE_TEST_RAW_OSTREAM_H_

#include <gtest/gtest.h>

#include <string>

#include "common/ostream.h"
#include "llvm/ADT/SmallString.h"

namespace Carbon::Testing {

// A raw_ostream that makes it easy to repeatedly check streamed output.
class TestRawOstream : public llvm::raw_svector_ostream {
 public:
  explicit TestRawOstream() : llvm::raw_svector_ostream(buffer_) {}

  ~TestRawOstream() override {
    if (!buffer_.empty()) {
      ADD_FAILURE() << "Unchecked output:\n" << buffer_;
    }
  }

  // Flushes the stream and returns the contents so far, clearing the stream
  // back to empty.
  auto TakeStr() -> std::string {
    std::string result = buffer_.str().str();
    buffer_.clear();
    return result;
  }

 private:
  llvm::SmallString<16> buffer_;
};

}  // namespace Carbon::Testing

#endif  // CARBON_TESTING_BASE_TEST_RAW_OSTREAM_H_
