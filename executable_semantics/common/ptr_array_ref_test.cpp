// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_semantics/common/ptr_array_ref.h"

#include <type_traits>

#include "executable_semantics/common/arena.h"
#include "gtest/gtest.h"
#include "llvm/include/llvm/ADT/STLExtras.h"

namespace Carbon {
namespace {

struct Stub {
  int i = 3;
};

template <typename T>
inline constexpr bool IsConstRef = std::is_const_v<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool IsConstPtr = std::is_const_v<std::remove_pointer_t<T>>;

TEST(PtrArrayRefTest, MutableT) {
  // A non-const value to point at.
  Stub i;

  // The mutable Ptr is mutable.
  Ptr<Stub> p(&i);
  static_assert(!IsConstRef<decltype(*p)>);
  static_assert(!IsConstRef<decltype((p->i))>);
  static_assert(!IsConstPtr<decltype(p.Get())>);

  // Still mutable in a vector.
  std::vector<Ptr<Stub>> v = {p};
  static_assert(!IsConstRef<decltype(*v[0])>);
  static_assert(!IsConstRef<decltype((v[0]->i))>);
  static_assert(!IsConstPtr<decltype(v[0].Get())>);

  // PtrArrayRef returns const references.
  PtrArrayRef<Stub> ref(v);
  static_assert(IsConstRef<decltype(*ref[0])>);
  static_assert(IsConstRef<decltype((ref[0]->i))>);
  static_assert(IsConstPtr<decltype(ref[0].Get())>);
}

TEST(PtrArrayRefTest, ConstT) {
  // A non-const value to point at.
  Stub i;

  // The const Ptr is const.
  Ptr<const Stub> p(&i);
  static_assert(IsConstRef<decltype(*p)>);
  static_assert(IsConstRef<decltype((p->i))>);
  static_assert(IsConstPtr<decltype(p.Get())>);

  // Still const in a vector.
  std::vector<Ptr<const Stub>> v = {p};
  static_assert(IsConstRef<decltype(*v[0])>);
  static_assert(IsConstRef<decltype((v[0]->i))>);
  static_assert(IsConstPtr<decltype(v[0].Get())>);

  // PtrArrayRef is happy to accept const.
  PtrArrayRef<const Stub> ref(v);
  static_assert(IsConstRef<decltype(*ref[0])>);
  static_assert(IsConstRef<decltype((ref[0]->i))>);
  static_assert(IsConstPtr<decltype(ref[0].Get())>);
}

TEST(PtrArrayRefTest, Iterator) {
  Arena arena;
  std::vector<Ptr<int>> v = {arena.New<int>(0), arena.New<int>(1)};
  PtrArrayRef<int> ref(v);

  int i = 0;
  for (Ptr<const int> ptr : ref) {
    EXPECT_EQ(*ptr, i);
    ++i;
  }
}

}  // namespace
}  // namespace Carbon
