//===- RangeAdapterTest.cpp - Unit tests for range adapters  --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

#include <iterator>
#include <list>
#include <vector>

using namespace llvm;

namespace {

// A wrapper around vector which exposes rbegin(), rend().
class ReverseOnlyVector {
  std::vector<int> Vec;
public:
  ReverseOnlyVector(std::initializer_list<int> list) : Vec(list) { }

  typedef std::vector<int>::reverse_iterator reverse_iterator;
  reverse_iterator rbegin() { return Vec.rbegin(); }
  reverse_iterator rend() { return Vec.rend(); }
};

// A wrapper around vector which exposes begin(), end(), rbegin() and rend().
// begin() and end() don't have implementations as this ensures that we will
// get a linker error if reverse() chooses begin()/end() over rbegin(), rend().
class BidirectionalVector {
  std::vector<int> Vec;

public:
  BidirectionalVector(std::initializer_list<int> list) : Vec(list) { }

  typedef std::vector<int>::iterator iterator;
  iterator begin();
  iterator end();

  typedef std::vector<int>::reverse_iterator reverse_iterator;
  reverse_iterator rbegin() { return Vec.rbegin(); }
  reverse_iterator rend() { return Vec.rend(); }
};

// Test fixture
template <typename T>
class RangeAdapterTest : public ::testing::Test { };

typedef ::testing::Types<std::vector<int>,
                         std::list<int>,
                         int[4],
                         ReverseOnlyVector,
                         BidirectionalVector,
                         const std::vector<int>,
                         const std::list<int>,
                         const int[4]> RangeAdapterTestTypes;
TYPED_TEST_CASE(RangeAdapterTest, RangeAdapterTestTypes);

TYPED_TEST(RangeAdapterTest, TrivialOperation) {
  TypeParam v = { 0, 1, 2, 3 };

  int counter = 3;
  for (int i : reverse(v))
    EXPECT_EQ(i, counter--);

  counter = 0;
  for (int i : reverse(reverse(v)))
    EXPECT_EQ(i, counter++);
}

} // anonymous namespace
