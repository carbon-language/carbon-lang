// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_COMMON_INDEX_BASE_H_
#define CARBON_TOOLCHAIN_COMMON_INDEX_BASE_H_

#include <type_traits>

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Format.h"

namespace Carbon {

template <typename DataType>
class DataIterator;

// A lightweight handle to an item in a vector.
//
// DataIndex is designed to be passed by value, not reference or pointer. They
// are also designed to be small and efficient to store in data structures.
struct IndexBase {
  IndexBase() : index(-1) {}
  explicit IndexBase(int index) : index(index) {}

  auto Print(llvm::raw_ostream& output) const -> void { output << index; }

  int32_t index;
};

template <typename IndexType,
          typename std::enable_if_t<std::is_base_of_v<IndexBase, IndexType>>* =
              nullptr>
auto operator==(IndexType lhs, IndexType rhs) -> bool {
  return lhs.index == rhs.index;
}
template <typename IndexType,
          typename std::enable_if_t<std::is_base_of_v<IndexBase, IndexType>>* =
              nullptr>
auto operator!=(IndexType lhs, IndexType rhs) -> bool {
  return lhs.index != rhs.index;
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_COMMON_INDEX_BASE_H_
