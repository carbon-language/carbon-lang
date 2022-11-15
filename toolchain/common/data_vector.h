// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_COMMON_DATA_VECTOR_H_
#define CARBON_TOOLCHAIN_COMMON_DATA_VECTOR_H_

#include "common/ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Format.h"

namespace Carbon {

template <typename DataType>
class DataIterator;
template <typename DataType>
class DataVector;

// A lightweight handle to an item in DataVector.
//
// DataIndex is designed to be passed by value, not reference or pointer. They
// are also designed to be small and efficient to store in data structures.
template <typename DataType>
class DataIndex {
 public:
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

  auto index() const -> int32_t { return index_; }

  auto Print(llvm::raw_ostream& output) const -> void { output << index_; }

  auto Format(int width) const -> llvm::FormattedNumber {
    return llvm::format_decimal(index_, width);
  }

 private:
  friend DataIterator<DataType>;
  friend DataVector<DataType>;

  explicit DataIndex(int index) : index_(index) {}

  int32_t index_;
};

template <typename DataType>
class DataIterator
    : public llvm::iterator_facade_base<DataIterator<DataType>,
                                        std::random_access_iterator_tag,
                                        const DataIndex<DataType>, int> {
 public:
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

template <typename DataType>
class DataVector {
 public:
  using Index = DataIndex<DataType>;
  using Iterator = DataIterator<DataType>;

  auto empty() const -> bool { return data_.empty(); }
  auto size() const -> size_t { return data_.size(); }

  auto back() -> DataType& { return data_.back(); }
  auto back() const -> const DataType& { return data_.back(); }

  auto begin() const -> typename llvm::SmallVector<DataType>::const_iterator {
    return data_.begin();
  }
  auto end() const -> typename llvm::SmallVector<DataType>::const_iterator {
    return data_.end();
  }

  auto next_index() -> Index { return Index(data_.size()); }

  auto push_back(const DataType& el) -> Index {
    Index index(data_.size());
    data_.push_back(el);
    return index;
  }

  auto push_back(DataType&& el) -> Index {
    Index index(data_.size());
    data_.push_back(el);
    return index;
  }

  auto operator[](Index index) -> DataType& { return data_[index.index_]; }
  auto operator[](Index index) const -> const DataType& {
    return data_[index.index_];
  }

  auto range() const -> llvm::iterator_range<Iterator> {
    return llvm::make_range(Iterator(Index(0)), Iterator(Index(data_.size())));
  }

 private:
  llvm::SmallVector<DataType> data_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_COMMON_DATA_VECTOR_H_
