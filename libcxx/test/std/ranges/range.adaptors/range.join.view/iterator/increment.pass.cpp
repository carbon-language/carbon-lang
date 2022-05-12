//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// constexpr iterator& operator++();
// constexpr void operator++(int);
// constexpr iterator operator++(int);

#include <cassert>
#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  // This way if we read past end we'll catch the error.
  int buffer1[2][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  int dummy = 42;
  (void) dummy;
  int buffer2[2][4] = {{9, 10, 11, 12}, {13, 14, 15, 16}};

  // operator++(int);
  {
    std::ranges::join_view jv(buffer1);
    auto iter = jv.begin();
    for (int i = 1; i < 9; ++i) {
      assert(*iter++ == i);
    }
  }
  {
    ValueView<int> children[4] = {ValueView(buffer1[0]), ValueView(buffer1[1]), ValueView(buffer2[0]), ValueView(buffer2[1])};
    std::ranges::join_view jv(ValueView<ValueView<int>>{children});
    auto iter = jv.begin();
    for (int i = 1; i < 17; ++i) {
      assert(*iter == i);
      iter++;
    }

    ASSERT_SAME_TYPE(decltype(iter++), void);
  }
  {
    std::ranges::join_view jv(buffer1);
    auto iter = std::next(jv.begin(), 7);
    assert(*iter++ == 8);
    assert(iter == jv.end());
  }
  {
    int small[2][1] = {{1}, {2}};
    std::ranges::join_view jv(small);
    auto iter = jv.begin();
    for (int i = 1; i < 3; ++i) {
      assert(*iter++ == i);
    }
  }
  // Has some empty children.
  {
    CopyableChild children[4] = {CopyableChild(buffer1[0], 4), CopyableChild(buffer1[1], 0), CopyableChild(buffer2[0], 1), CopyableChild(buffer2[1], 0)};
    auto jv = std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    assert(*iter == 1); iter++;
    assert(*iter == 2); iter++;
    assert(*iter == 3); iter++;
    assert(*iter == 4); iter++;
    assert(*iter == 9); iter++;
    assert(iter == jv.end());
  }
  // Parent is empty.
  {
    CopyableChild children[4] = {CopyableChild(buffer1[0]), CopyableChild(buffer1[1]), CopyableChild(buffer2[0]), CopyableChild(buffer2[1])};
    std::ranges::join_view jv(ParentView(children, 0));
    assert(jv.begin() == jv.end());
  }
  // Parent size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer1[0])};
    std::ranges::join_view jv(ParentView(children, 1));
    auto iter = jv.begin();
    assert(*iter == 1); iter++;
    assert(*iter == 2); iter++;
    assert(*iter == 3); iter++;
    assert(*iter == 4); iter++;
    assert(iter == jv.end());
  }
  // Parent and child size is one.
  {
    CopyableChild children[1] = {CopyableChild(buffer1[0], 1)};
    std::ranges::join_view jv(ParentView(children, 1));
    auto iter = jv.begin();
    assert(*iter == 1); iter++;
    assert(iter == jv.end());
  }
  // Parent size is one child is empty
  {
    CopyableChild children[1] = {CopyableChild(buffer1[0], 0)};
    std::ranges::join_view jv(ParentView(children, 1));
    assert(jv.begin() == jv.end());
  }
  // Has all empty children.
  {
    CopyableChild children[4] = {CopyableChild(buffer1[0], 0), CopyableChild(buffer1[1], 0), CopyableChild(buffer2[0], 0), CopyableChild(buffer2[1], 0)};
    auto jv = std::ranges::join_view(ParentView(children));
    assert(jv.begin() == jv.end());
  }
  // First child is empty, others are not.
  {
    CopyableChild children[4] = {CopyableChild(buffer1[0], 4), CopyableChild(buffer1[1], 0), CopyableChild(buffer2[0], 0), CopyableChild(buffer2[1], 0)};
    auto jv = std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    assert(*iter == 1); iter++;
    assert(*iter == 2); iter++;
    assert(*iter == 3); iter++;
    assert(*iter == 4); iter++;
    assert(iter == jv.end());
  }
  // Last child is empty, others are not.
  {
    CopyableChild children[4] = {CopyableChild(buffer1[0], 4), CopyableChild(buffer1[1], 4), CopyableChild(buffer2[0], 4), CopyableChild(buffer2[1], 0)};
    auto jv = std::ranges::join_view(ParentView(children));
    auto iter = jv.begin();
    for (int i = 1; i < 13; ++i) {
      assert(*iter == i);
      iter++;
    }
  }
  // operator++();
  {
    std::ranges::join_view jv(buffer1);
    auto iter = jv.begin();
    for (int i = 2; i < 9; ++i) {
      assert(*++iter == i);
    }
  }
  {
    ValueView<int> children[4] = {ValueView(buffer1[0]), ValueView(buffer1[1]), ValueView(buffer2[0]), ValueView(buffer2[1])};
    std::ranges::join_view jv(ValueView<ValueView<int>>{children});
    auto iter = jv.begin();
    for (int i = 2; i < 17; ++i) {
      assert(*++iter == i);
    }

    ASSERT_SAME_TYPE(decltype(++iter), decltype(iter)&);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
