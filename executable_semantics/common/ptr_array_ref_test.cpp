// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/ptr_array_ref.h"

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

  // A non-const vector of ptrs gets non-const ptr access.
  std::vector<Ptr<Stub>> v = {p};
  static_assert(!IsConstRef<decltype(*v[0])>);
  static_assert(!IsConstRef<decltype((v[0]->i))>);
  static_assert(!IsConstPtr<decltype(v[0].Get())>);

  PtrArrayRef<Stub> ref(v);
  static_assert(IsConstRef<decltype(*ref[0])>);
  static_assert(IsConstRef<decltype((ref[0]->i))>);
  static_assert(IsConstPtr<decltype(ref[0].Get())>);

  // Constness of the vector propagates to ptr access.
  std::vector<Ptr<const Stub>> v_const = {p_const};
  static_assert(IsConstRef<decltype(*v_const[0])>);
  static_assert(IsConstRef<decltype((v_const[0]->i))>);
  static_assert(IsConstPtr<decltype(v_const[0].Get())>);

  PtrArrayRef<const Stub> ref_const(v_const);
  static_assert(IsConstRef<decltype(*ref_const[0])>);
  static_assert(IsConstRef<decltype((ref_const[0]->i))>);
  static_assert(IsConstPtr<decltype(ref_const[0].Get())>);
}

}  // namespace
}  // namespace Carbon
