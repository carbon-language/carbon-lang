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

#ifndef FORTRAN_EVALUATE_STATIC_DATA_H_
#define FORTRAN_EVALUATE_STATIC_DATA_H_

// Represents constant static data objects

#include "type.h"
#include "../common/idioms.h"
#include <cinttypes>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace Fortran::evaluate {

class StaticDataObject {
public:
  using Pointer = std::shared_ptr<StaticDataObject>;

  StaticDataObject(const StaticDataObject &) = delete;
  StaticDataObject(StaticDataObject &&) = delete;
  StaticDataObject &operator=(const StaticDataObject &) = delete;
  StaticDataObject &operator=(StaticDataObject &&) = delete;

  static Pointer Create() { return Pointer{new StaticDataObject}; }

  const std::string &name() const { return name_; }
  StaticDataObject &set_name(std::string n) {
    name_ = n;
    return *this;
  }

  int alignment() const { return alignment_; }
  StaticDataObject &set_alignment(int a) {
    CHECK(a >= 0);
    alignment_ = a;
    return *this;
  }

  int itemBytes() const { return itemBytes_; }
  StaticDataObject &set_itemBytes(int b) {
    CHECK(b >= 1);
    itemBytes_ = b;
    return *this;
  }

  const std::vector<std::uint8_t> &data() const { return data_; }
  std::vector<std::uint8_t> &data() { return data_; }

  StaticDataObject &Push(const std::string &);
  StaticDataObject &Push(const std::u16string &);
  StaticDataObject &Push(const std::u32string &);
  std::optional<std::string> AsString() const;
  std::optional<std::u16string> AsU16String() const;
  std::optional<std::u32string> AsU32String() const;
  std::ostream &AsFortran(std::ostream &) const;

  static bool bigEndian;

private:
  StaticDataObject() {}

  std::string name_;
  int alignment_{1};
  int itemBytes_{1};
  std::vector<std::uint8_t> data_;
};
}
#endif  // FORTRAN_EVALUATE_STATIC_DATA_H_
