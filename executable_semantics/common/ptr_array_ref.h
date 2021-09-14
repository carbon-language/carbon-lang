// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
#define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_

#include <vector>

#include "executable_semantics/common/ptr.h"
#include "llvm/ADT/ArrayRef.h"

namespace Carbon {

// A wrapper similar to llvm::ArrayRef<Ptr<T>>, but which exposes the array
// elements as `Ptr<const T>`.
template <typename T>
class PtrArrayRef {
 public:
  using value_type = typename llvm::ArrayRef<Ptr<const T>>::value_type;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using reference = value_type&;
  using const_reference = const value_type&;
  using iterator = const_pointer;
  using const_iterator = const_pointer;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  // Mimic implicit constructors of llvm::ArrayRef, but based on what Carbon
  // uses.
  PtrArrayRef(const std::vector<Ptr<T>>& v) : ref(v) {}

  auto operator[](size_t i) const -> Ptr<const T> { return ref[i]; }

  auto begin() const -> iterator { return ref.begin(); }
  auto end() const -> iterator { return ref.end(); }

  auto rbegin() const -> reverse_iterator { return reverse_iterator(end()); }
  auto rend() const -> reverse_iterator { return reverse_iterator(begin()); }

  auto empty() const -> bool { return ref.empty(); }
  auto size() const -> size_t { return ref.size(); }

 private:
  llvm::ArrayRef<Ptr<T>> ref;
};

}  // namespace Carbon

#endif  // #define EXECUTABLE_SEMANTICS_COMMON_PTR_ARRAY_REF_H_
