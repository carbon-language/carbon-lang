// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/ptr.h"

#include <type_traits>

#include "gtest/gtest.h"

namespace Carbon {
namespace {

struct Stub {
  int i = 3;
};

template <typename T>
inline constexpr bool IsConstRef = std::is_const_v<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool IsConstPtr = std::is_const_v<std::remove_pointer_t<T>>;

TEST(PtrTest, Constness) {
  // A non-const value to point at.
  Stub i;

  // A non-const ptr returns a non-const T.
  Ptr<Stub> p(&i);
  static_assert(!IsConstRef<decltype(*p)>);
  static_assert(!IsConstRef<decltype((p->i))>);
  static_assert(!IsConstPtr<decltype(p.Get())>);

  // A ptr to const returns a const T.
  Ptr<const Stub> p_const(&i);
  static_assert(IsConstRef<decltype(*p_const)>);
  static_assert(IsConstRef<decltype((p_const->i))>);
  static_assert(IsConstPtr<decltype(p_const.Get())>);

  // A const ptr also returns a const T.
  const Ptr<Stub> const_p(&i);
  static_assert(IsConstRef<decltype(*const_p)>);
  static_assert(IsConstRef<decltype((const_p->i))>);
  static_assert(IsConstPtr<decltype(const_p.Get())>);

  // A non-const vector of ptrs gets non-const ptr access.
  std::vector<Ptr<Stub>> v = {Ptr<Stub>(&i)};
  static_assert(!IsConstRef<decltype(*v[0])>);
  static_assert(!IsConstRef<decltype((v[0]->i))>);
  static_assert(!IsConstPtr<decltype(v[0].Get())>);

  // Constness of the vector propagates to ptr access.
  const std::vector<Ptr<Stub>> const_v = v;
  static_assert(IsConstRef<decltype(*const_v[0])>);
  static_assert(IsConstRef<decltype((const_v[0]->i))>);
  static_assert(IsConstPtr<decltype(const_v[0].Get())>);
}

}  // namespace
}  // namespace Carbon
