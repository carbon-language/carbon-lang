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

#include "static-data.h"

namespace Fortran::evaluate {
std::ostream &StaticDataObject::Dump(std::ostream &o) const {
  o << "static data ";
  char sep{'{'};
  for (std::uint8_t byte : data_) {
    o << sep << "0x" << std::hex << byte;
    sep = ',';
  }
  if (sep == '{') {
    o << '{';
  }
  return o << '}';
}

StaticDataObject &StaticDataObject::Push(const std::string &string) {
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch));
  }
  return *this;
}

StaticDataObject &StaticDataObject::Push(const std::u16string &string) {
  // TODO here and below: big-endian targets
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch));
    data_.push_back(static_cast<std::uint8_t>(ch >> 8));
  }
  return *this;
}

StaticDataObject &StaticDataObject::Push(const std::u32string &string) {
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch));
    data_.push_back(static_cast<std::uint8_t>(ch >> 8));
    data_.push_back(static_cast<std::uint8_t>(ch >> 16));
    data_.push_back(static_cast<std::uint8_t>(ch >> 24));
  }
  return *this;
}

std::optional<std::string> StaticDataObject::AsString() const {
  if (itemBytes_ <= 1) {
    std::string result;
    for (std::uint8_t byte : data_) {
      result += static_cast<char>(byte);
    }
    return {std::move(result)};
  }
  return std::nullopt;
}
}
