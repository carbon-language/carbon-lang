//===- llvm/unittest/ADT/SmallPtrSetTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallPtrSet unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace llvm;

// SmallPtrSet swapping test.
TEST(SmallPtrSetTest, GrowthTest) {
  int i;
  int buf[8];
  for(i=0; i<8; ++i) buf[i]=0;


  SmallPtrSet<int *, 4> s;
  typedef SmallPtrSet<int *, 4>::iterator iter;
  
  s.insert(&buf[0]);
  s.insert(&buf[1]);
  s.insert(&buf[2]);
  s.insert(&buf[3]);
  EXPECT_EQ(4U, s.size());

  i = 0;
  for(iter I=s.begin(), E=s.end(); I!=E; ++I, ++i)
      (**I)++;
  EXPECT_EQ(4, i);
  for(i=0; i<8; ++i)
      EXPECT_EQ(i<4?1:0,buf[i]);

  s.insert(&buf[4]);
  s.insert(&buf[5]);
  s.insert(&buf[6]);
  s.insert(&buf[7]);

  i = 0;
  for(iter I=s.begin(), E=s.end(); I!=E; ++I, ++i)
      (**I)++;
  EXPECT_EQ(8, i);
  s.erase(&buf[4]);
  s.erase(&buf[5]);
  s.erase(&buf[6]);
  s.erase(&buf[7]);
  EXPECT_EQ(4U, s.size());

  i = 0;
  for(iter I=s.begin(), E=s.end(); I!=E; ++I, ++i)
      (**I)++;
  EXPECT_EQ(4, i);
  for(i=0; i<8; ++i)
      EXPECT_EQ(i<4?3:1,buf[i]);

  s.clear();
  for(i=0; i<8; ++i) buf[i]=0;
  for(i=0; i<128; ++i) s.insert(&buf[i%8]); // test repeated entires
  EXPECT_EQ(8U, s.size());
  for(iter I=s.begin(), E=s.end(); I!=E; ++I, ++i)
      (**I)++;
  for(i=0; i<8; ++i)
      EXPECT_EQ(1,buf[i]);
}


TEST(SmallPtrSetTest, SwapTest) {
  int buf[10];

  SmallPtrSet<int *, 2> a;
  SmallPtrSet<int *, 2> b;

  a.insert(&buf[0]);
  a.insert(&buf[1]);
  b.insert(&buf[2]);

  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(2U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));

  b.insert(&buf[3]);
  std::swap(a, b);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(1U, b.size());
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
  EXPECT_TRUE(b.count(&buf[2]));

  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(3U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));
  EXPECT_TRUE(b.count(&buf[3]));

  a.insert(&buf[4]);
  a.insert(&buf[5]);
  a.insert(&buf[6]);

  std::swap(b, a);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(4U, b.size());
  EXPECT_TRUE(b.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[4]));
  EXPECT_TRUE(b.count(&buf[5]));
  EXPECT_TRUE(b.count(&buf[6]));
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
}
