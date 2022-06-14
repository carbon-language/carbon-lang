//===----------- ImmutableSetTest.cpp - ImmutableSet unit tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ImmutableSet.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class ImmutableSetTest : public testing::Test {
protected:
  // for callback tests
  static char buffer[10];

  struct MyIter {
    int counter;
    char *ptr;

    MyIter() : counter(0), ptr(buffer) {
      for (unsigned i=0; i<sizeof(buffer);++i) buffer[i]='\0';
    }
    void operator()(char c) {
      *ptr++ = c;
      ++counter;
    }
  };
};
char ImmutableSetTest::buffer[10];


TEST_F(ImmutableSetTest, EmptyIntSetTest) {
  ImmutableSet<int>::Factory f;

  EXPECT_TRUE(f.getEmptySet() == f.getEmptySet());
  EXPECT_FALSE(f.getEmptySet() != f.getEmptySet());
  EXPECT_TRUE(f.getEmptySet().isEmpty());

  ImmutableSet<int> S = f.getEmptySet();
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}


TEST_F(ImmutableSetTest, OneElemIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.getEmptySet();

  ImmutableSet<int> S2 = f.add(S, 3);
  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());
  EXPECT_FALSE(S == S2);
  EXPECT_TRUE(S != S2);
  EXPECT_FALSE(S.contains(3));
  EXPECT_TRUE(S2.contains(3));
  EXPECT_FALSE(S2.begin() == S2.end());
  EXPECT_TRUE(S2.begin() != S2.end());

  ImmutableSet<int> S3 = f.add(S, 2);
  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S3.isEmpty());
  EXPECT_FALSE(S == S3);
  EXPECT_TRUE(S != S3);
  EXPECT_FALSE(S.contains(2));
  EXPECT_TRUE(S3.contains(2));

  EXPECT_FALSE(S2 == S3);
  EXPECT_TRUE(S2 != S3);
  EXPECT_FALSE(S2.contains(2));
  EXPECT_FALSE(S3.contains(3));
}

TEST_F(ImmutableSetTest, MultiElemIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.getEmptySet();

  ImmutableSet<int> S2 = f.add(f.add(f.add(S, 3), 4), 5);
  ImmutableSet<int> S3 = f.add(f.add(f.add(S2, 9), 20), 43);
  ImmutableSet<int> S4 = f.add(S2, 9);

  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());
  EXPECT_FALSE(S3.isEmpty());
  EXPECT_FALSE(S4.isEmpty());

  EXPECT_FALSE(S.contains(3));
  EXPECT_FALSE(S.contains(9));

  EXPECT_TRUE(S2.contains(3));
  EXPECT_TRUE(S2.contains(4));
  EXPECT_TRUE(S2.contains(5));
  EXPECT_FALSE(S2.contains(9));
  EXPECT_FALSE(S2.contains(0));

  EXPECT_TRUE(S3.contains(43));
  EXPECT_TRUE(S3.contains(20));
  EXPECT_TRUE(S3.contains(9));
  EXPECT_TRUE(S3.contains(3));
  EXPECT_TRUE(S3.contains(4));
  EXPECT_TRUE(S3.contains(5));
  EXPECT_FALSE(S3.contains(0));

  EXPECT_TRUE(S4.contains(9));
  EXPECT_TRUE(S4.contains(3));
  EXPECT_TRUE(S4.contains(4));
  EXPECT_TRUE(S4.contains(5));
  EXPECT_FALSE(S4.contains(20));
  EXPECT_FALSE(S4.contains(43));
}

TEST_F(ImmutableSetTest, RemoveIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.getEmptySet();

  ImmutableSet<int> S2 = f.add(f.add(S, 4), 5);
  ImmutableSet<int> S3 = f.add(S2, 3);
  ImmutableSet<int> S4 = f.remove(S3, 3);

  EXPECT_TRUE(S3.contains(3));
  EXPECT_FALSE(S2.contains(3));
  EXPECT_FALSE(S4.contains(3));

  EXPECT_TRUE(S2 == S4);
  EXPECT_TRUE(S3 != S2);
  EXPECT_TRUE(S3 != S4);

  EXPECT_TRUE(S3.contains(4));
  EXPECT_TRUE(S3.contains(5));

  EXPECT_TRUE(S4.contains(4));
  EXPECT_TRUE(S4.contains(5));
}

TEST_F(ImmutableSetTest, IterLongSetTest) {
  ImmutableSet<long>::Factory f;
  ImmutableSet<long> S = f.getEmptySet();

  ImmutableSet<long> S2 = f.add(f.add(f.add(S, 0), 1), 2);
  ImmutableSet<long> S3 = f.add(f.add(f.add(S2, 3), 4), 5);

  int i = 0;
  for (ImmutableSet<long>::iterator I = S.begin(), E = S.end(); I != E; ++I) {
    i++;
  }
  ASSERT_EQ(0, i);

  i = 0;
  for (ImmutableSet<long>::iterator I = S2.begin(), E = S2.end(); I != E; ++I) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(3, i);

  i = 0;
  for (ImmutableSet<long>::iterator I = S3.begin(), E = S3.end(); I != E; I++) {
    ASSERT_EQ(i, *I);
    i++;
  }
  ASSERT_EQ(6, i);
}

}
