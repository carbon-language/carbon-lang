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
#include "../parser/characters.h"

namespace Fortran::evaluate {

bool StaticDataObject::bigEndian{false};

std::ostream &StaticDataObject::AsFortran(std::ostream &o) const {
  if (auto string{AsString()}) {
    o << parser::QuoteCharacterLiteral(*string);
  } else if (auto string{AsU16String()}) {
    o << "2_" << parser::QuoteCharacterLiteral(*string);
  } else if (auto string{AsU32String()}) {
    o << "4_" << parser::QuoteCharacterLiteral(*string);
  } else {
    CRASH_NO_CASE;
  }
  return o;
}

StaticDataObject &StaticDataObject::Push(const std::string &string) {
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch));
  }
  return *this;
}

StaticDataObject &StaticDataObject::Push(const std::u16string &string) {
  int shift{bigEndian * 8};
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch >> shift));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 8)));
  }
  return *this;
}

StaticDataObject &StaticDataObject::Push(const std::u32string &string) {
  int shift{bigEndian * 24};
  for (auto ch : string) {
    data_.push_back(static_cast<std::uint8_t>(ch >> shift));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 8)));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 16)));
    data_.push_back(static_cast<std::uint8_t>(ch >> (shift ^ 24)));
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

std::optional<std::u16string> StaticDataObject::AsU16String() const {
  if (itemBytes_ == 2) {
    int shift{bigEndian * 8};
    std::u16string result;
    auto end{data_.cend()};
    for (auto byte{data_.cbegin()}; byte < end;) {
      result += static_cast<char16_t>(*byte++) << shift |
          static_cast<char16_t>(*byte++) << (shift ^ 8);
    }
    return {std::move(result)};
  }
  return std::nullopt;
}

std::optional<std::u32string> StaticDataObject::AsU32String() const {
  if (itemBytes_ == 4) {
    int shift{bigEndian * 24};
    std::u32string result;
    auto end{data_.cend()};
    for (auto byte{data_.cbegin()}; byte < end;) {
      result += static_cast<char32_t>(*byte++) << shift |
          static_cast<char32_t>(*byte++) << (shift ^ 8) |
          static_cast<char32_t>(*byte++) << (shift ^ 16) |
          static_cast<char32_t>(*byte++) << (shift ^ 24);
    }
    return {std::move(result)};
  }
  return std::nullopt;
}
}
