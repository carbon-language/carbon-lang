// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_STRING_OSTREAM_H_
#define CARBON_COMMON_RAW_STRING_OSTREAM_H_

#include "common/check.h"
#include "common/ostream.h"

namespace Carbon {

// Implements streaming output with an underlying string. The string must always
// be taken prior to destruction.
//
// Versus `llvm::raw_string_ostream`:
// - Owns the underlying string.
// - Supports `pwrite` for compatibility with more stream uses in tests.
class RawStringOstream : public llvm::raw_pwrite_stream {
 public:
  explicit RawStringOstream() : llvm::raw_pwrite_stream(/*Unbuffered=*/true) {}

  ~RawStringOstream() override {
    CARBON_CHECK(str_.empty(), "Expected to be emptied by TakeStr, have: {0}",
                 str_);
  }

  // Returns the streamed contents, clearing the stream back to empty.
  auto TakeStr() -> std::string {
    std::string result = std::move(str_);
    clear();
    return result;
  }

  // Clears the buffer, which can be helpful when destructing without
  // necessarily calling `TakeStr()`.
  auto clear() -> void { str_.clear(); }

  auto empty() -> bool { return str_.empty(); }
  auto size() -> size_t { return str_.size(); }

 private:
  auto current_pos() const -> uint64_t override { return str_.size(); }

  auto pwrite_impl(const char* ptr, size_t size, uint64_t offset)
      -> void override {
    str_.replace(offset, size, ptr, size);
  }

  auto write_impl(const char* ptr, size_t size) -> void override {
    str_.append(ptr, size);
  }

  auto reserveExtraSpace(uint64_t extra_size) -> void override {
    str_.reserve(str_.size() + extra_size);
  }

  // The actual buffer.
  std::string str_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_RAW_STRING_OSTREAM_H_
