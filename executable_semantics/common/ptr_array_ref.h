// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
#define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_

#include <optional>

#include "executable_semantics/common/ptr.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

// A wrapper similar to llvm::ArrayRef<Ptr<T>>, but which exposes the array
// elements as `Ptr<const T>`.
template <typename T>
class PtrArrayRef {
 public:
  class Iterator {
   public:
    using value_type = Ptr<const T>;
    using difference_type = std::ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = std::bidirectional_iterator_tag;

    Iterator(const Ptr<T>* x) : p(x) {}
    Iterator(const Iterator& iter) : p(iter.p) {}

    auto operator++() -> Iterator& {
      p = ++p;
      return *this;
    }

    auto operator++(int) -> Iterator {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }

    auto operator--() -> Iterator& {
      p = --p;
      return *this;
    }

    auto operator--(int) -> Iterator {
      Iterator tmp(*this);
      operator--();
      return tmp;
    }

    auto operator==(const Iterator& rhs) const -> bool { return p == rhs.p; }
    auto operator!=(const Iterator& rhs) const -> bool { return p != rhs.p; }
    auto operator*() -> const Ptr<const T> { return *p; }
    auto operator->() -> const Ptr<const T>* { return &*this; }

   private:
    const Ptr<T>* p;
  };

  using value_type = Ptr<const T>;
  using difference_type = std::ptrdiff_t;
  using pointer = const value_type*;
  using reference = const value_type&;
  using iterator = Iterator;
  using reverse_iterator = std::reverse_iterator<iterator>;

  // Mimic implicit constructors of llvm::ArrayRef, but based on what Carbon
  // uses.
  PtrArrayRef(const std::vector<Ptr<T>>& v) : ref(v) {}

  auto operator[](size_t i) const -> value_type { return ref[i]; }

  auto begin() const -> iterator { return iterator(ref.begin()); }
  auto end() const -> iterator { return iterator(ref.end()); }

  auto rbegin() const -> reverse_iterator { return reverse_iterator(end()); }
  auto rend() const -> reverse_iterator { return reverse_iterator(begin()); }

  auto empty() const -> bool { return ref.empty(); }
  auto size() const -> size_t { return ref.size(); }

 private:
  llvm::ArrayRef<Ptr<T>> ref;
};

}  // namespace Carbon

#endif  // #define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
