//===- llvm/unittest/ADT/SmallMapTest.cpp ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// SmallMap unit tests.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallMap.h"

using namespace llvm;

// SmallMap test.
TEST(SmallMapTest, GeneralTest) {

  int buf[10];

  SmallMap<int *, int, 3> a;
  SmallMap<int *, int, 3> b;
  SmallMap<int *, int, 3>::iterator found;
  std::pair<SmallMap<int *, int, 3>::iterator, bool> insRes;
  SmallMap<int *, int, 3>::const_iterator foundc;

  a.insert(std::make_pair(&buf[0], 0));
  insRes = a.insert(std::make_pair(&buf[1], 1));
  EXPECT_TRUE(insRes.second);

  // Check insertion, looking up, and data editing in small mode.
  insRes = a.insert(std::make_pair(&buf[1], 6));
  EXPECT_FALSE(insRes.second);
  EXPECT_EQ(insRes.first->second, 1);
  insRes.first->second = 5;
  found = a.find(&buf[1]);
  EXPECT_NE(found, a.end());
  EXPECT_EQ(found->second, 5);
  a[&buf[1]] = 10;
  EXPECT_EQ(found->second, 10);
  // Check "not found" case.
  found = a.find(&buf[8]);
  EXPECT_EQ(found, a.end());

  b.insert(std::make_pair(&buf[2], 2));

  std::swap(a, b);
  a.swap(b);
  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(2U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));

  insRes = b.insert(std::make_pair(&buf[3], 3));
  EXPECT_TRUE(insRes.second);

  // Check insertion, looking up, and data editing in big mode.
  insRes = b.insert(std::make_pair(&buf[3], 6));
  EXPECT_FALSE(insRes.second);
  EXPECT_EQ(insRes.first->second, 3);
  insRes.first->second = 7;
  found = b.find(&buf[3]);
  EXPECT_EQ(found->second, 7);
  b[&buf[3]] = 14;
  EXPECT_EQ(found->second, 14);
  // Check constant looking up.
  foundc = b.find(&buf[3]);
  EXPECT_EQ(foundc->first, &buf[3]);
  EXPECT_EQ(foundc->second, 14);
  // Check not found case.
  found = b.find(&buf[8]);
  EXPECT_EQ(found, b.end());

  std::swap(a, b);
  a.swap(b);
  std::swap(a, b);

  EXPECT_EQ(3U, a.size());
  EXPECT_EQ(1U, b.size());
  EXPECT_TRUE(a.count(&buf[0]));
  EXPECT_TRUE(a.count(&buf[1]));
  EXPECT_TRUE(a.count(&buf[3]));
  EXPECT_TRUE(b.count(&buf[2]));
  EXPECT_EQ(b.find(&buf[2])->second, 2);

  std::swap(a, b);
  a.swap(b);
  std::swap(a, b);

  EXPECT_EQ(1U, a.size());
  EXPECT_EQ(3U, b.size());
  EXPECT_TRUE(a.count(&buf[2]));
  EXPECT_TRUE(b.count(&buf[0]));
  EXPECT_TRUE(b.count(&buf[1]));
  EXPECT_TRUE(b.count(&buf[3]));

  a.insert(std::make_pair(&buf[4], 4));
  a.insert(std::make_pair(&buf[5], 5));
  a.insert(std::make_pair(&buf[6], 6));

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

  // Check findAndConstruct
  SmallMap<int *, int, 3>::value_type Buf7;
  Buf7 = a.FindAndConstruct(&buf[7]);
  EXPECT_EQ(Buf7.second, 0);
  
  // Check increments
  
  SmallMap<int *, int, 2> c;
  c.insert(std::make_pair(&buf[0], 0));
  c.insert(std::make_pair(&buf[1], 1));
  
  // For small mode we know that flat array map is used and we know the
  // order of items.
  unsigned ii = 0;
  for (SmallMap<int *, int, 2>::iterator i = c.begin(), e = c.end();
       i != e; ++i, ++ii) {
    EXPECT_TRUE((i->first == &buf[0] && i->second == 0 && ii == 0) ||
                (i->first == &buf[1] && i->second == 1 && ii == 1));
  }
  
  // For big mode DenseMap is used and final order of items is undefined.
  c.insert(std::make_pair(&buf[2], 2));
  for (SmallMap<int *, int, 2>::iterator i = c.begin(), e = c.end();
       i != e; ++i) {
    EXPECT_TRUE((i->first == &buf[0] && i->second == 0) ||
                (i->first == &buf[1] && i->second == 1) ||
                (i->first == &buf[2] && i->second == 2));
  }
}
