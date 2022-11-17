// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_COMMON_DATA_INDEX_H_
#define CARBON_TOOLCHAIN_COMMON_DATA_INDEX_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Format.h"

namespace Carbon {

template <typename DataType>
class DataIterator;

// A lightweight handle to an item in a SmallVector. These are constructed using
// `Append(data, el)` then used with `index.In(data)`.
//
// DataIndex is designed to be passed by value, not reference or pointer. They
// are also designed to be small and efficient to store in data structures.
//
// This templates VectorT on calls instead of using SmallVector explicitly
// mainly so that the DataType need not be fully declared before declaring
// DataIndex<DataType>.
template <typename DataType>
class DataIndex {
 public:
  // Appends an element to the data, and returns the added index.
  template <typename VectorT>
  static auto Append(VectorT& data, const DataType& el) -> DataIndex {
    DataIndex index(data.size());
    data.push_back(el);
    return index;
  }
  template <typename VectorT>
  static auto Append(VectorT& data, DataType&& el) -> DataIndex {
    DataIndex index(data.size());
    data.push_back(el);
    return index;
  }

  // Returns what the index will be if an element is manually added to the data,
  // after this call.
  template <typename VectorT>
  static auto BeforeManualAppend(const VectorT& data) -> DataIndex {
    return DataIndex(data.size());
  }

  DataIndex() : index_(-1) {}

  friend auto operator==(DataIndex lhs, DataIndex rhs) -> bool {
    return lhs.index_ == rhs.index_;
  }
  friend auto operator!=(DataIndex lhs, DataIndex rhs) -> bool {
    return lhs.index_ != rhs.index_;
  }
  friend auto operator<(DataIndex lhs, DataIndex rhs) -> bool {
    return lhs.index_ < rhs.index_;
  }
  friend auto operator<=(DataIndex lhs, DataIndex rhs) -> bool {
    return lhs.index_ <= rhs.index_;
  }
  friend auto operator>(DataIndex lhs, DataIndex rhs) -> bool {
    return lhs.index_ > rhs.index_;
  }
  friend auto operator>=(DataIndex lhs, DataIndex rhs) -> bool {
    return lhs.index_ >= rhs.index_;
  }

  // Returns the element at the index in data.
  template <typename VectorT>
  auto In(const VectorT& data) const -> const DataType& {
    return data[index_];
  }
  template <typename VectorT>
  auto In(VectorT& data) const -> DataType& {
    return data[index_];
  }

  auto Print(llvm::raw_ostream& output) const -> void { output << index_; }

  // Formats the index to a given width.
  auto Format(int width) const -> llvm::FormattedNumber {
    return llvm::format_decimal(index_, width);
  }

  // The raw index. This should only be used when the index value has a
  // semantic meaning, instead of just being a handle.
  auto raw_index() const -> int32_t { return index_; }

 private:
  friend DataIterator<DataType>;

  explicit DataIndex(int index) : index_(index) {}

  int32_t index_;
};

// Similar to DataIndex, but with some extra operations to support use as an
// iterator.
template <typename DataType>
class DataIterator
    : public llvm::iterator_facade_base<DataIterator<DataType>,
                                        std::random_access_iterator_tag,
                                        const DataIndex<DataType>, int> {
 public:
  template <typename VectorT>
  static auto MakeRange(const VectorT& data)
      -> llvm::iterator_range<DataIterator> {
    return llvm::make_range(DataIterator(DataIndex<DataType>(0)),
                            DataIterator(DataIndex<DataType>(data.size())));
  }

  DataIterator() = default;

  explicit DataIterator(DataIndex<DataType> index) : index_(index) {}

  auto operator==(const DataIterator& rhs) const -> bool {
    return index_ == rhs.index_;
  }
  auto operator<(const DataIterator& rhs) const -> bool {
    return index_ < rhs.index_;
  }

  auto operator*() const -> const DataIndex<DataType>& { return index_; }

  using llvm::iterator_facade_base<DataIterator<DataType>,
                                   std::random_access_iterator_tag,
                                   const DataIndex<DataType>, int>::operator-;
  auto operator-(const DataIterator& rhs) const -> int {
    return index_.index_ - rhs.index_.index_;
  }

  auto operator+=(int n) -> DataIterator& {
    index_.index_ += n;
    return *this;
  }
  auto operator-=(int n) -> DataIterator& {
    index_.index_ -= n;
    return *this;
  }

  // Prints the raw index index.
  auto Print(llvm::raw_ostream& output) const -> void {
    output << index_.index_;
  }

 private:
  DataIndex<DataType> index_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_COMMON_DATA_INDEX_H_
