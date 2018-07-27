// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "char-buffer.h"
#include "../common/idioms.h"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace Fortran::parser {

char *CharBuffer::FreeSpace(std::size_t *n) {
  int offset{LastBlockOffset()};
  if (blocks_.empty()) {
    blocks_.emplace_front();
    last_ = blocks_.begin();
    lastBlockEmpty_ = true;
  } else if (offset == 0 && !lastBlockEmpty_) {
    last_ = blocks_.emplace_after(last_);
    lastBlockEmpty_ = true;
  }
  *n = Block::capacity - offset;
  return last_->data + offset;
}

void CharBuffer::Claim(std::size_t n) {
  if (n > 0) {
    bytes_ += n;
    lastBlockEmpty_ = false;
  }
}

void CharBuffer::Put(const char *data, std::size_t n) {
  std::size_t chunk;
  for (std::size_t at{0}; at < n; at += chunk) {
    char *to{FreeSpace(&chunk)};
    chunk = std::min(n - at, chunk);
    Claim(chunk);
    std::memcpy(to, data + at, chunk);
  }
}

void CharBuffer::Put(const std::string &str) { Put(str.data(), str.size()); }

std::string CharBuffer::Marshal() const {
  std::string result;
  std::size_t bytes{bytes_};
  result.reserve(bytes);
  for (const Block &block : blocks_) {
    std::size_t chunk{std::min(bytes, Block::capacity)};
    for (std::size_t j{0}; j < chunk; ++j) {
      result += block.data[j];
    }
    bytes -= chunk;
  }
  result.shrink_to_fit();
  CHECK(result.size() == bytes_);
  return result;
}

std::string CharBuffer::MarshalNormalized() const {
  std::string result;
  std::size_t bytes{bytes_};
  result.reserve(bytes + 1 /* for terminal line feed */);
  char ch{'\0'};
  for (const Block &block : blocks_) {
    std::size_t chunk{std::min(bytes, Block::capacity)};
    for (std::size_t j{0}; j < chunk; ++j) {
      ch = block.data[j];
      if (ch != '\r') {
        result += ch;
      }
    }
    bytes -= chunk;
  }
  if (ch != '\n') {
    result += '\n';
  }
  result.shrink_to_fit();
  return result;
}

}  // namespace Fortran::parser
