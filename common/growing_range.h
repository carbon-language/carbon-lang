// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_GROWING_RANGE_H_
#define CARBON_COMMON_GROWING_RANGE_H_

#include <ranges>

namespace Carbon {

// A range adaptor for a random-access container such as `std::vector` or
// `llvm::SmallVector` that might have elements appended during the iteration.
// This adaptor avoids invalidation issues by tracking an index instead of an
// iterator, and by returning by value from `operator*` instead of by reference.
//
// This class is intended only for use as the range in a range-based for loop,
// and as such does not provide a complete range or iterator interface. Instead,
// it provides only the interface required by the range-based for loop.
template <typename ContainerT>
  requires std::ranges::sized_range<ContainerT> &&
           std::ranges::random_access_range<ContainerT>
class GrowingRange {
 public:
  // An end sentinel for the range.
  class End {};

  // An iterator into a potentially-growing range. Tracks the container and the
  // current index, and indexes the container on each dereference.
  class Iterator {
   public:
    // Dereferences the iterator. These intentionally don't return by reference,
    // to avoid handing out a reference that would be invalidated when the
    // container grows during the traversal.
    auto operator*() -> auto { return (*container_)[index_]; }
    auto operator*() const -> auto { return (*container_)[index_]; }

    friend auto operator!=(Iterator it, End /*end*/) -> bool {
      return it.index_ != it.container_->size();
    }

    auto operator++() -> void { ++index_; }

   private:
    friend class GrowingRange;

    explicit Iterator(const ContainerT* container)
        : container_(container), index_(0) {}

    const ContainerT* container_;
    size_t index_;
  };

  explicit GrowingRange(const ContainerT& container) : container_(&container) {}

  auto begin() const -> Iterator { return Iterator(container_); }
  auto end() const -> End { return {}; }

 private:
  const ContainerT* container_;
};

template <typename ContainerT>
GrowingRange(const ContainerT&) -> GrowingRange<ContainerT>;

}  // namespace Carbon

#endif  // CARBON_COMMON_GROWING_RANGE_H_
