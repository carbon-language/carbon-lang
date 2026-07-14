// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/array_stack.h"

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
  stack.AppendToTop(1);
  stack.AppendToTop(2);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // Pushing a new array starts empty, old values are still there.
  stack.PushArray();
  EXPECT_THAT(stack.PeekArray(), IsEmpty());
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // The added value goes to the 2nd array.
  stack.AppendToTop(3);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(3));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 3));

  // Popping goes back to the 1st array.
  stack.PopArray();
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));

  // Push a couple arrays, then a value on the now-3rd array.
  stack.PushArray();
  stack.PushArray();
  stack.AppendToTop(4);
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
  stack.AppendToTop(5);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(5));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(5));
}

TEST(ArrayStack, AppendArray) {
  ArrayStack<int> stack;

  stack.PushArray();
  stack.AppendToTop(llvm::ArrayRef<int>());
  EXPECT_THAT(stack.PeekArray(), IsEmpty());
  stack.AppendToTop({1, 2});
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2));
}

TEST(ArrayStack, PeekArrayAt) {
  ArrayStack<int> stack;

  // Verify behavior with a single array.
  stack.PushArray();
  stack.AppendToTop(1);
  stack.AppendToTop(2);

  EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(1, 2));

  // Verify behavior with a couple more arrays.
  stack.PushArray();
  stack.PushArray();
  stack.AppendToTop(3);

  EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(1, 2));
  EXPECT_THAT(stack.PeekArrayAt(1), IsEmpty());
  EXPECT_THAT(stack.PeekArrayAt(2), ElementsAre(3));
}

TEST(ArrayStack, MergeTopArrayIntoGrandparent) {
  ArrayStack<int> stack;

  // Basic case with 3 arrays.
  stack.PushArray();
  stack.AppendToTop(1);
  stack.AppendToTop(2);

  stack.PushArray();
  stack.AppendToTop(3);
  stack.AppendToTop(4);
  stack.AppendToTop(5);

  stack.PushArray();
  stack.AppendToTop(6);
  stack.AppendToTop(7);

  stack.MergeTopArrayIntoGrandparent();

  EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(1, 2, 6, 7));
  EXPECT_THAT(stack.PeekArrayAt(1), ElementsAre(3, 4, 5));
  EXPECT_THAT(stack.PeekArrayAt(2), IsEmpty());
  EXPECT_THAT(stack.PeekArray(), IsEmpty());
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 6, 7, 3, 4, 5));

  // Appending to the now-empty top array and popping works as expected.
  stack.AppendToTop(8);
  EXPECT_THAT(stack.PeekArray(), ElementsAre(8));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 6, 7, 3, 4, 5, 8));

  stack.PopArray();
  EXPECT_THAT(stack.PeekArray(), ElementsAre(3, 4, 5));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 6, 7, 3, 4, 5));

  stack.PopArray();
  EXPECT_THAT(stack.PeekArray(), ElementsAre(1, 2, 6, 7));
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 6, 7));
}

TEST(ArrayStack, MergeTopArrayIntoGrandparentDeeperStack) {
  ArrayStack<int> stack;

  // Verify behavior when there are more than 3 arrays on the stack.
  stack.PushArray();
  stack.AppendToTop(10);

  stack.PushArray();
  stack.AppendToTop(20);

  stack.PushArray();
  stack.AppendToTop(30);

  stack.PushArray();
  stack.AppendToTop(40);
  stack.AppendToTop(50);

  stack.MergeTopArrayIntoGrandparent();

  EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(10));
  EXPECT_THAT(stack.PeekArrayAt(1), ElementsAre(20, 40, 50));
  EXPECT_THAT(stack.PeekArrayAt(2), ElementsAre(30));
  EXPECT_THAT(stack.PeekArrayAt(3), IsEmpty());
  EXPECT_THAT(stack.PeekAllValues(), ElementsAre(10, 20, 40, 50, 30));
}

TEST(ArrayStack, MergeTopArrayIntoGrandparentEmptyArrays) {
  // Test when the parent array is initially empty.
  {
    ArrayStack<int> stack;
    stack.PushArray();
    stack.AppendToTop(1);
    stack.PushArray();
    stack.PushArray();
    stack.AppendToTop(2);
    stack.AppendToTop(3);

    stack.MergeTopArrayIntoGrandparent();
    EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(1, 2, 3));
    EXPECT_THAT(stack.PeekArrayAt(1), IsEmpty());
    EXPECT_THAT(stack.PeekArrayAt(2), IsEmpty());
    EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2, 3));
  }

  // Test when the top array is initially empty.
  {
    ArrayStack<int> stack;
    stack.PushArray();
    stack.AppendToTop(1);
    stack.PushArray();
    stack.AppendToTop(2);
    stack.PushArray();

    stack.MergeTopArrayIntoGrandparent();
    EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(1));
    EXPECT_THAT(stack.PeekArrayAt(1), ElementsAre(2));
    EXPECT_THAT(stack.PeekArrayAt(2), IsEmpty());
    EXPECT_THAT(stack.PeekAllValues(), ElementsAre(1, 2));
  }

  // Test when the grandparent array is initially empty.
  {
    ArrayStack<int> stack;
    stack.PushArray();
    stack.PushArray();
    stack.AppendToTop(1);
    stack.AppendToTop(2);
    stack.PushArray();
    stack.AppendToTop(3);
    stack.AppendToTop(4);

    stack.MergeTopArrayIntoGrandparent();
    EXPECT_THAT(stack.PeekArrayAt(0), ElementsAre(3, 4));
    EXPECT_THAT(stack.PeekArrayAt(1), ElementsAre(1, 2));
    EXPECT_THAT(stack.PeekArrayAt(2), IsEmpty());
    EXPECT_THAT(stack.PeekAllValues(), ElementsAre(3, 4, 1, 2));
  }
}

}  // namespace
}  // namespace Carbon::Testing
