// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/base/array_stack.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace Carbon::Testing {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

TEST(ArrayStack, Basics) {
  ArrayStack<int> stack;

  // PeekAllValues is valid when there are no arrays.
  EXPECT_THAT(stack.PeekAllValues(), IsEmpty());

  // An array starts empty.
  stack.PushArray();
  EXPECT_THAT(stack.PeekArray(), IsEmpty());
  EXPECT_THAT(stack.PeekAllValues(), IsEmpty());

  // Pushing a couple values works.
  stack.PushValue(1);
  stack.PushValue(2);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // Pushing a new array starts empty, old values are still there.
  stack.PushArray();
  EXPECT_THAT(stack.PeekArray(), IsEmpty());
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // The added value goes to the 2nd array.
  stack.PushValue(3);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(3));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 3));

  // Popping goes back to the 1st array.
  stack.PopArray();
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // Push a couple arrays, then a value on the now-3rd array.
  stack.PushArray();
  stack.PushArray();
  stack.PushValue(4);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(4));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 4));

  // Popping the 3rd array goes to the 2nd array, which is empty.
  stack.PopArray();
  EXPECT_THAT(stack.PeekArray(), IsEmpty());
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // Again back to the 1st array.
  stack.PopArray();
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // Go down to no arrays.
  stack.PopArray();
  EXPECT_THAT(stack.PeekAllValues(), IsEmpty());

  // Add a new 1st array.
  stack.PushArray();
  stack.PushValue(5);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(5));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(5));
}

}  // namespace
}  // namespace Carbon::Testing
