// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_INTERPRETER_ADDRESS_H_
#define EXECUTABLE_SEMANTICS_INTERPRETER_ADDRESS_H_

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "executable_semantics/interpreter/field_path.h"

namespace Carbon {

class Address {
 public:
  Address(const Address&) = default;
  Address(Address&&) = default;
  auto operator=(const Address&) -> Address& = default;
  auto operator=(Address&&) -> Address& = default;

  friend auto operator==(const Address& lhs, const Address& rhs) -> bool {
    return lhs.index == rhs.index;
  }

  friend auto operator!=(const Address& lhs, const Address& rhs) -> bool {
    return !(lhs == rhs);
  }

  friend auto operator<<(std::ostream& out, const Address& a) -> std::ostream& {
    out << a.index << a.field_path;
    return out;
  }

  auto SubobjectAddress(std::string field_name) const -> Address {
    Address result = *this;
    result.field_path.Append(std::move(field_name));
    return result;
  }

 private:
  friend class Heap;

  explicit Address(uint64_t index) : index(index) {}

  uint64_t index;
  FieldPath field_path;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_INTERPRETER_ADDRESS_H_
