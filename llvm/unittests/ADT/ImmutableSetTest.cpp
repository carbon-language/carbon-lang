// llvm/unittest/ADT/ImmutableSetTest.cpp - ImmutableSet unit tests -*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/ImmutableSet.h"

using namespace llvm;

namespace {
class ImmutableSetTest : public testing::Test {
};


TEST_F(ImmutableSetTest, EmptyIntSetTest) {
  ImmutableSet<int>::Factory f;

  EXPECT_TRUE(f.GetEmptySet() == f.GetEmptySet());
  EXPECT_FALSE(f.GetEmptySet() != f.GetEmptySet());
  EXPECT_TRUE(f.GetEmptySet().isEmpty());

  ImmutableSet<int> S = f.GetEmptySet();
  EXPECT_EQ(0u, S.getHeight());
  EXPECT_TRUE(S.begin() == S.end());
  EXPECT_FALSE(S.begin() != S.end());
}


TEST_F(ImmutableSetTest, OneElemIntSetTest) {
  ImmutableSet<int>::Factory f;
  ImmutableSet<int> S = f.GetEmptySet();

  ImmutableSet<int> S2 = f.Add(S, 3);
  EXPECT_TRUE(S.isEmpty());
  EXPECT_FALSE(S2.isEmpty());
  EXPECT_FALSE(S == S2);
  EXPECT_TRUE(S != S2);
  EXPECT_FALSE(S.contains(3));
  EXPECT_TRUE(S2.contains(3));
  EXPECT_FALSE(S2.begin() == S2.end());
  EXPECT_TRUE(S2.begin() != S2.end());

  ImmutableSet<int> S3 = f.Add(S, 2);
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
  ImmutableSet<int> S = f.GetEmptySet();

  ImmutableSet<int> S2 = f.Add(f.Add(f.Add(S, 3), 4), 5);
  ImmutableSet<int> S3 = f.Add(f.Add(f.Add(S2, 9), 20), 43);
  ImmutableSet<int> S4 = f.Add(S2, 9);

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
  ImmutableSet<int> S = f.GetEmptySet();

  ImmutableSet<int> S2 = f.Add(f.Add(S, 4), 5);
  ImmutableSet<int> S3 = f.Add(S2, 3);
  ImmutableSet<int> S4 = f.Remove(S3, 3);

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


static char *ptr; // tmp var
struct MyIter {
  int counter;
  MyIter() : counter(0) {}
  void operator()(char c) {
    *ptr++ = c;
    ++counter;
  }
};

TEST_F(ImmutableSetTest, CallbackCharSetTest) {
  ImmutableSet<char>::Factory f;
  ImmutableSet<char> S = f.GetEmptySet();

  ImmutableSet<char> S2 = f.Add(f.Add(f.Add(S, 'a'), 'e'), 'i');
  ImmutableSet<char> S3 = f.Add(f.Add(S2, 'o'), 'u');

  char buffer[6] = {0};
  ptr = buffer;
  S3.foreach<MyIter>();

  ASSERT_EQ(buffer[0], 'a');
  ASSERT_EQ(buffer[1], 'e');
  ASSERT_EQ(buffer[2], 'i');
  ASSERT_EQ(buffer[3], 'o');
  ASSERT_EQ(buffer[4], 'u');
  ASSERT_EQ(buffer[5], 0);
}

TEST_F(ImmutableSetTest, Callback2CharSetTest) {
  ImmutableSet<char>::Factory f;
  ImmutableSet<char> S = f.GetEmptySet();

  ImmutableSet<char> S2 = f.Add(f.Add(f.Add(S, 'b'), 'c'), 'd');
  ImmutableSet<char> S3 = f.Add(f.Add(f.Add(S2, 'f'), 'g'), 'h');

  char buffer[7] = {0};
  ptr = buffer;
  MyIter obj;
  S3.foreach<MyIter>(obj);

  ASSERT_EQ(buffer[0], 'b');
  ASSERT_EQ(buffer[1], 'c');
  ASSERT_EQ(buffer[2], 'd');
  ASSERT_EQ(buffer[3], 'f');
  ASSERT_EQ(buffer[4], 'g');
  ASSERT_EQ(buffer[5], 'h');
  ASSERT_EQ(buffer[6], 0);

  ASSERT_EQ(obj.counter, 6);

  S2.foreach<MyIter>(obj);
  ASSERT_EQ(obj.counter, 6+3);

  S.foreach<MyIter>(obj);
  ASSERT_EQ(obj.counter, 6+3+0);
}

TEST_F(ImmutableSetTest, IterLongSetTest) {
  ImmutableSet<long>::Factory f;
  ImmutableSet<long> S = f.GetEmptySet();

  ImmutableSet<long> S2 = f.Add(f.Add(f.Add(S, 0), 1), 2);
  ImmutableSet<long> S3 = f.Add(f.Add(f.Add(S2, 3), 4), 5);

  int i = 0;
  for (ImmutableSet<long>::iterator I = S.begin(), E = S.end(); I != E; ++I) {
    ASSERT_EQ(*I, i++);
  }
  ASSERT_EQ(i, 0);

  i = 0;
  for (ImmutableSet<long>::iterator I = S2.begin(), E = S2.end(); I != E; ++I) {
    ASSERT_EQ(*I, i++);
  }
  ASSERT_EQ(i, 3);

  i = 0;
  for (ImmutableSet<long>::iterator I = S3.begin(), E = S3.end(); I != E; I++) {
    ASSERT_EQ(*I, i++);
  }
  ASSERT_EQ(i, 6);
}

}
