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
  static constexpr int32_t InvalidIndex = -1;

  constexpr IndexBase() : index(InvalidIndex) {}
  constexpr explicit IndexBase(int index) : index(index) {}

  auto Print(llvm::raw_ostream& output) const -> void {
    if (is_valid()) {
      output << index;
    } else {
      output << "<invalid>";
    }
  }

  auto is_valid() const -> bool { return index != InvalidIndex; }

  int32_t index;
};

// Like IndexBase, but also provides < and > comparison operators.
struct ComparableIndexBase : public IndexBase {
  using IndexBase::IndexBase;
};

// Equality comparison for both IndexBase and ComparableIndexBase.
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

// The < and > comparisons for only ComparableIndexBase.
template <typename IndexType, typename std::enable_if_t<std::is_base_of_v<
                                  ComparableIndexBase, IndexType>>* = nullptr>
auto operator<(IndexType lhs, IndexType rhs) -> bool {
  return lhs.index < rhs.index;
}
template <typename IndexType, typename std::enable_if_t<std::is_base_of_v<
                                  ComparableIndexBase, IndexType>>* = nullptr>
auto operator<=(IndexType lhs, IndexType rhs) -> bool {
  return lhs.index <= rhs.index;
}
template <typename IndexType, typename std::enable_if_t<std::is_base_of_v<
                                  ComparableIndexBase, IndexType>>* = nullptr>
auto operator>(IndexType lhs, IndexType rhs) -> bool {
  return lhs.index > rhs.index;
}
template <typename IndexType, typename std::enable_if_t<std::is_base_of_v<
                                  ComparableIndexBase, IndexType>>* = nullptr>
auto operator>=(IndexType lhs, IndexType rhs) -> bool {
  return lhs.index >= rhs.index;
}

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_COMMON_INDEX_BASE_H_
