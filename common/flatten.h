// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_FLATTEN_H_
#define CARBON_COMMON_FLATTEN_H_

#include "llvm/ADT/STLExtras.h"

namespace Carbon {

template <typename RangeT>
class FlattenRange;

// Converts an range of ranges of `T` into a single range of `T`. The resulting
// range has iterators that will walk through each `T` in the group of ranges.
//
// `llvm::concat` can be used to combine a number of ranges of `T` into a single
// range, and then `Flatten` can be used to iterate the individual `T` values.
template <class Range>
  requires(
      !std::is_reference_v<Range> &&
      requires(Range&& r) {
        { llvm::adl_begin(r) };
        { llvm::adl_end(r) };
      } &&
      requires(decltype(llvm::adl_begin(std::declval<Range&&>()))& it) {
        { llvm::adl_begin(*it) };
        { llvm::adl_end(*it) };
      })
auto Flatten(Range&& range) -> FlattenRange<Range> {
  return FlattenRange<Range>(std::forward<Range>(range));
}

// Overload for Flatten() that works with a reference to a range. Uses
// lifetimebound to prevent mistakes, since it does not work correctly on a
// universal reference.
template <class Range>
  requires(
      std::is_reference_v<Range> &&
      requires(Range&& r) {
        { llvm::adl_begin(r) };
        { llvm::adl_end(r) };
      } &&
      requires(decltype(llvm::adl_begin(std::declval<Range&&>()))& it) {
        { llvm::adl_begin(*it) };
        { llvm::adl_end(*it) };
      })
auto Flatten(Range&& range [[clang::lifetimebound]]) -> FlattenRange<Range> {
  return FlattenRange<Range>(std::forward<Range>(range));
}

// Overload for Flatten() that works with an initializer list, for constructing
// a static range of ranges with curly braces.
template <class T>
  requires(requires(T& t) {
    { llvm::adl_begin(t) };
    { llvm::adl_end(t) };
  })
auto Flatten(std::initializer_list<T> range)
    -> FlattenRange<std::initializer_list<T>> {
  return FlattenRange<std::initializer_list<T>>(std::move(range));
}

// Flattening iterator. Given iterators of iterators of `T`, this iterates
// through each inner iterator in order, stopping at each `T` found within.
template <typename ValueT, typename IterT, typename IterEndT,
          typename InnerIterT, typename InnerIterEndT>
class FlattenIterator
    : public llvm::iterator_facade_base<
          FlattenIterator<ValueT, IterT, IterEndT, InnerIterT, InnerIterEndT>,
          std::forward_iterator_tag, ValueT> {
  using BaseT = typename FlattenIterator::iterator_facade_base;

  IterT outer_cur_;
  IterEndT outer_end_;
  // The inner iterators are intentionally left uninitialized if the outer
  // iterator is empty. They may not be used unless `outer_cur_ != outer_end_`.
  InnerIterT inner_cur_;
  InnerIterEndT inner_end_;

  // Increments the inner iterator, and if its at its end, increments the outer
  // iterator until it finds a non-empty inner iterator.
  void Increment() {
    if (outer_cur_ == outer_end_) {
      return;
    }
    if (inner_cur_ != inner_end_) {
      ++inner_cur_;
      if (inner_cur_ != inner_end_) {
        return;
      }
    }
    while (true) {
      ++outer_cur_;
      if (outer_cur_ == outer_end_) {
        return;
      }
      inner_cur_ = llvm::adl_begin(*outer_cur_);
      inner_end_ = llvm::adl_end(*outer_cur_);
      if (inner_cur_ != inner_end_) {
        return;
      }
    }
  }

 public:
  // Constructs an iterator over a range of iterators which returns the elements
  // from those iterators.
  FlattenIterator(IterT outer_begin, IterT outer_end)
      : outer_cur_(std::move(outer_begin)), outer_end_(std::move(outer_end)) {
    if (outer_cur_ != outer_end_) {
      inner_cur_ = llvm::adl_begin(*outer_cur_);
      inner_end_ = llvm::adl_end(*outer_cur_);
      // If the first iterator is empty, find the first non-empty iterator.
      if (inner_cur_ == inner_end_) {
        Increment();
      }
    }
  }

  using BaseT::operator++;

  auto operator++() -> FlattenIterator& {
    Increment();
    return *this;
  }

  auto operator*() const -> ValueT& {
    CARBON_CHECK(outer_cur_ != outer_end_);
    return *inner_cur_;
  }

  friend auto operator==(const FlattenIterator& lhs, const FlattenIterator& rhs)
      -> bool {
    bool lhs_ended = lhs.outer_cur_ == lhs.outer_end_;
    bool rhs_ended = rhs.outer_cur_ == rhs.outer_end_;
    if (lhs_ended && rhs_ended) {
      return true;
    }
    if (lhs_ended || rhs_ended) {
      return false;
    }
    return lhs.outer_cur_ == rhs.outer_cur_ && lhs.inner_cur_ == rhs.inner_cur_;
  }
};

template <typename RangeT>
class FlattenRange {
  using OuterIterT = decltype(llvm::adl_begin(std::declval<RangeT&>()));
  using OuterIterEndT = decltype(llvm::adl_end(std::declval<RangeT&>()));
  using InnerIterT = decltype(llvm::adl_begin(*std::declval<OuterIterT&>()));
  using InnerIterEndT = decltype(llvm::adl_end(*std::declval<OuterIterT&>()));
  using ValueT =
      std::remove_reference_t<decltype(*std::declval<InnerIterT&>())>;
  using ConstValueT =
      std::remove_reference_t<decltype(*std::declval<const InnerIterT&>())>;

 public:
  using Iterator = FlattenIterator<ValueT, OuterIterT, OuterIterEndT,
                                   InnerIterT, InnerIterEndT>;
  using ConstIterator = FlattenIterator<ConstValueT, OuterIterT, OuterIterEndT,
                                        InnerIterT, InnerIterEndT>;

  explicit FlattenRange(RangeT&& range) : range_(std::forward<RangeT>(range)) {}

  auto begin() const -> ConstIterator {
    return ConstIterator(llvm::adl_begin(range_), llvm::adl_end(range_));
  }
  auto begin() -> Iterator {
    return Iterator(llvm::adl_begin(range_), llvm::adl_end(range_));
  }
  auto end() const -> ConstIterator {
    return ConstIterator(llvm::adl_end(range_), llvm::adl_end(range_));
  }
  auto end() -> Iterator {
    return Iterator(llvm::adl_end(range_), llvm::adl_end(range_));
  }

 private:
  RangeT range_;
};

}  // namespace Carbon

#endif  // CARBON_COMMON_FLATTEN_H_
