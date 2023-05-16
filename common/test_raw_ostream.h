// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_TEST_RAW_OSTREAM_H_
#define CARBON_COMMON_TEST_RAW_OSTREAM_H_

#include <gtest/gtest.h>

#include <string>

#include "common/ostream.h"

namespace Carbon::Testing {

/// A raw_ostream that makes it easy to repeatedly check streamed output.
class TestRawOstream : public llvm::raw_ostream {
 public:
  ~TestRawOstream() override {
    flush();
    if (!buffer_.empty()) {
      ADD_FAILURE() << "Unchecked output:\n" << buffer_;
    }
  }

  /// Flushes the stream and returns the contents so far, clearing the stream
  /// back to empty.
  auto TakeStr() -> std::string {
    flush();
    std::string result = std::move(buffer_);
    buffer_.clear();
    return result;
  }

 private:
  void write_impl(const char* ptr, size_t size) override {
    buffer_.append(ptr, ptr + size);
  }

  [[nodiscard]] auto current_pos() const -> uint64_t override {
    return buffer_.size();
  }

  std::string buffer_;
};

}  // namespace Carbon::Testing

#endif  // CARBON_COMMON_TEST_RAW_OSTREAM_H_
