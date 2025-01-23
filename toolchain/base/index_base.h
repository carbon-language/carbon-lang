// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_BASE_INDEX_BASE_H_
#define CARBON_TOOLCHAIN_BASE_INDEX_BASE_H_

#include <compare>
#include <concepts>
#include <iterator>
#include <type_traits>

#include "common/ostream.h"
#include "llvm/ADT/iterator.h"

namespace Carbon {

template <typename DataType>
class DataIterator;

// Non-templated portions of `IdBase`.
struct AnyIdBase {
  static constexpr int32_t NoneIndex = -1;

  AnyIdBase() = delete;
  constexpr explicit AnyIdBase(int index) : index(index) {}

  constexpr auto has_value() const -> bool { return index != NoneIndex; }

  int32_t index;
};

// A lightweight handle to an item identified by an opaque ID.
//
// This class is intended to be derived from by classes representing a specific
// kind of ID, whose meaning as an integer is an implementation detail of the
// type that vends the IDs. Typically this will be a vector index.
//
// Classes derived from IdBase are designed to be passed by value, not
// reference or pointer. They are also designed to be small and efficient to
// store in data structures.
//
// This uses CRTP for the `Print` function. Children should have:
//   static constexpr llvm::StringLiteral Label = "my_label";
// Children can also define their own `Print` function, removing the dependency
// on `Label`.
template <typename IdT>
struct IdBase : public AnyIdBase, public Printable<IdT> {
  using AnyIdBase::AnyIdBase;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << IdT::Label;
    if (has_value()) {
      out << index;
    } else {
      out << "<none>";
    }
  }

  // Support simple equality comparison for ID types.
  friend constexpr auto operator==(IdBase<IdT> lhs, IdBase<IdT> rhs) -> bool {
    return lhs.index == rhs.index;
  }
};

// A lightweight handle to an item that behaves like an index.
//
// Unlike IdBase, classes derived from IndexBase are not completely opaque, and
// provide at least an ordering between indexes that has meaning to an API
// user. Additional semantics may be specified by the derived class.
template <typename IdT>
struct IndexBase : public IdBase<IdT> {
  using IdBase<IdT>::IdBase;

  // Support relational comparisons for index types.
  friend auto operator<=>(IndexBase<IdT> lhs, IndexBase<IdT> rhs)
      -> std::strong_ordering {
    return lhs.index <=> rhs.index;
  }
};

// A random-access iterator for arrays using IndexBase-derived types.
template <typename IndexT>
class IndexIterator
    : public llvm::iterator_facade_base<IndexIterator<IndexT>,
                                        std::random_access_iterator_tag,
                                        const IndexT, int>,
      public Printable<IndexIterator<IndexT>> {
 public:
  IndexIterator() = delete;

  explicit IndexIterator(IndexT index) : index_(index) {}

  friend auto operator==(const IndexIterator& lhs, const IndexIterator& rhs)
      -> bool {
    return lhs.index_ == rhs.index_;
  }
  friend auto operator<=>(const IndexIterator& lhs, const IndexIterator& rhs)
      -> std::strong_ordering {
    return lhs.index_ <=> rhs.index_;
  }

  auto operator*() const -> const IndexT& { return index_; }

  using llvm::iterator_facade_base<IndexIterator,
                                   std::random_access_iterator_tag,
                                   const IndexT, int>::operator-;
  friend auto operator-(const IndexIterator& lhs, const IndexIterator& rhs)
      -> int {
    return lhs.index_.index - rhs.index_.index;
  }

  auto operator+=(int n) -> IndexIterator& {
    index_.index += n;
    return *this;
  }
  auto operator-=(int n) -> IndexIterator& {
    index_.index -= n;
    return *this;
  }

  // Prints the raw token index.
  auto Print(llvm::raw_ostream& output) const -> void {
    output << index_.index;
  }

 private:
  IndexT index_;
};

}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_BASE_INDEX_BASE_H_
