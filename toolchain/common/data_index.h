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

// A lightweight handle to an item in a SmallVector.
//
// DataIndex is designed to be passed by value, not reference or pointer. They
// are also designed to be small and efficient to store in data structures.
template <typename DataType>
class DataIndex {
 public:
  template <typename VectorT>
  auto Append(VectorT& data, const DataType& el) -> DataIndex {
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

  template <typename VectorT>
  auto In(const VectorT& data) const -> const DataType& {
    return data[index_];
  }

  template <typename VectorT>
  auto In(VectorT& data) const -> DataType& {
    return data[index_];
  }

  auto Print(llvm::raw_ostream& output) const -> void { output << index_; }

  auto Format(int width) const -> llvm::FormattedNumber {
    return llvm::format_decimal(index_, width);
  }

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
