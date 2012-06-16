//===- llvm/unittest/ADT/DenseMapMap.cpp - DenseMap unit tests --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/DenseMap.h"
#include <map>

using namespace llvm;

namespace {

// Test fixture
template <typename T>
class DenseMapTest : public testing::Test {
protected:
  T Map;

  typename T::key_type getKey(int i = 0);
  typename T::mapped_type getValue(int i = 0);
};

template <>
uint32_t DenseMapTest<DenseMap<uint32_t, uint32_t> >::getKey(int i) {
  return i;
}

template <>
uint32_t DenseMapTest<DenseMap<uint32_t, uint32_t> >::getValue(int i) {
  return 42 + i;
}

template <>
uint32_t *DenseMapTest<DenseMap<uint32_t *, uint32_t *> >::getKey(int i) {
  static uint32_t dummy_arr1[8192];
  assert(i < 8192 && "Only support 8192 dummy keys.");
  return &dummy_arr1[i];
}

template <>
uint32_t *DenseMapTest<DenseMap<uint32_t *, uint32_t *> >::getValue(int i) {
  static uint32_t dummy_arr2[8192];
  assert(i < 8192 && "Only support 8192 dummy values.");
  return &dummy_arr2[i];
}

// Register these types for testing.
typedef ::testing::Types<DenseMap<uint32_t, uint32_t>,
                         DenseMap<uint32_t *, uint32_t *> > DenseMapTestTypes;
TYPED_TEST_CASE(DenseMapTest, DenseMapTestTypes);

// Empty map tests
TYPED_TEST(DenseMapTest, EmptyIntMapTest) {
  // Size tests
  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());

  // Iterator tests
  EXPECT_TRUE(this->Map.begin() == this->Map.end());

  // Lookup tests
  EXPECT_FALSE(this->Map.count(this->getKey()));
  EXPECT_TRUE(this->Map.find(this->getKey()) == this->Map.end());
#ifndef _MSC_VER
  EXPECT_EQ(typename TypeParam::mapped_type(),
            this->Map.lookup(this->getKey()));
#else
  // MSVC, at least old versions, cannot parse the typename to disambiguate
  // TypeParam::mapped_type as a type. However, because MSVC doesn't implement
  // two-phase name lookup, it also doesn't require the typename. Deal with
  // this mutual incompatibility through specialized code.
  EXPECT_EQ(TypeParam::mapped_type(),
            this->Map.lookup(this->getKey()));
#endif
}

// Constant map tests
TYPED_TEST(DenseMapTest, ConstEmptyMapTest) {
  const TypeParam &ConstMap = this->Map;
  EXPECT_EQ(0u, ConstMap.size());
  EXPECT_TRUE(ConstMap.empty());
  EXPECT_TRUE(ConstMap.begin() == ConstMap.end());
}

// A map with a single entry
TYPED_TEST(DenseMapTest, SingleEntryMapTest) {
  this->Map[this->getKey()] = this->getValue();

  // Size tests
  EXPECT_EQ(1u, this->Map.size());
  EXPECT_FALSE(this->Map.begin() == this->Map.end());
  EXPECT_FALSE(this->Map.empty());

  // Iterator tests
  typename TypeParam::iterator it = this->Map.begin();
  EXPECT_EQ(this->getKey(), it->first);
  EXPECT_EQ(this->getValue(), it->second);
  ++it;
  EXPECT_TRUE(it == this->Map.end());

  // Lookup tests
  EXPECT_TRUE(this->Map.count(this->getKey()));
  EXPECT_TRUE(this->Map.find(this->getKey()) == this->Map.begin());
  EXPECT_EQ(this->getValue(), this->Map.lookup(this->getKey()));
  EXPECT_EQ(this->getValue(), this->Map[this->getKey()]);
}

// Test clear() method
TYPED_TEST(DenseMapTest, ClearTest) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.clear();

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_TRUE(this->Map.begin() == this->Map.end());
}

// Test erase(iterator) method
TYPED_TEST(DenseMapTest, EraseTest) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.erase(this->Map.begin());

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_TRUE(this->Map.begin() == this->Map.end());
}

// Test erase(value) method
TYPED_TEST(DenseMapTest, EraseTest2) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.erase(this->getKey());

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
  EXPECT_TRUE(this->Map.begin() == this->Map.end());
}

// Test insert() method
TYPED_TEST(DenseMapTest, InsertTest) {
  this->Map.insert(std::make_pair(this->getKey(), this->getValue()));
  EXPECT_EQ(1u, this->Map.size());
  EXPECT_EQ(this->getValue(), this->Map[this->getKey()]);
}

// Test copy constructor method
TYPED_TEST(DenseMapTest, CopyConstructorTest) {
  this->Map[this->getKey()] = this->getValue();
  TypeParam copyMap(this->Map);

  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(this->getValue(), copyMap[this->getKey()]);
}

// Test assignment operator method
TYPED_TEST(DenseMapTest, AssignmentTest) {
  this->Map[this->getKey()] = this->getValue();
  TypeParam copyMap = this->Map;

  EXPECT_EQ(1u, copyMap.size());
  EXPECT_EQ(this->getValue(), copyMap[this->getKey()]);
}

// A more complex iteration test
TYPED_TEST(DenseMapTest, IterationTest) {
  bool visited[100];
  std::map<typename TypeParam::key_type, unsigned> visitedIndex;

  // Insert 100 numbers into the map
  for (int i = 0; i < 100; ++i) {
    visited[i] = false;
    visitedIndex[this->getKey(i)] = i;

    this->Map[this->getKey(i)] = this->getValue(i);
  }

  // Iterate over all numbers and mark each one found.
  for (typename TypeParam::iterator it = this->Map.begin();
       it != this->Map.end(); ++it)
    visited[visitedIndex[it->first]] = true;

  // Ensure every number was visited.
  for (int i = 0; i < 100; ++i)
    ASSERT_TRUE(visited[i]) << "Entry #" << i << " was never visited";
}

// const_iterator test
TYPED_TEST(DenseMapTest, ConstIteratorTest) {
  // Check conversion from iterator to const_iterator.
  typename TypeParam::iterator it = this->Map.begin();
  typename TypeParam::const_iterator cit(it);
  EXPECT_TRUE(it == cit);

  // Check copying of const_iterators.
  typename TypeParam::const_iterator cit2(cit);
  EXPECT_TRUE(cit == cit2);
}

// Key traits that allows lookup with either an unsigned or char* key;
// In the latter case, "a" == 0, "b" == 1 and so on.
struct TestDenseMapInfo {
  static inline unsigned getEmptyKey() { return ~0; }
  static inline unsigned getTombstoneKey() { return ~0U - 1; }
  static unsigned getHashValue(const unsigned& Val) { return Val * 37U; }
  static unsigned getHashValue(const char* Val) {
    return (unsigned)(Val[0] - 'a') * 37U;
  }
  static bool isEqual(const unsigned& LHS, const unsigned& RHS) {
    return LHS == RHS;
  }
  static bool isEqual(const char* LHS, const unsigned& RHS) {
    return (unsigned)(LHS[0] - 'a') == RHS;
  }
};

// find_as() tests
TEST(DenseMapCustomTest, FindAsTest) {
  DenseMap<unsigned, unsigned, TestDenseMapInfo> map;
  map[0] = 1;
  map[1] = 2;
  map[2] = 3;

  // Size tests
  EXPECT_EQ(3u, map.size());

  // Normal lookup tests
  EXPECT_EQ(1, map.count(1));
  EXPECT_EQ(1u, map.find(0)->second);
  EXPECT_EQ(2u, map.find(1)->second);
  EXPECT_EQ(3u, map.find(2)->second);
  EXPECT_TRUE(map.find(3) == map.end());

  // find_as() tests
  EXPECT_EQ(1u, map.find_as("a")->second);
  EXPECT_EQ(2u, map.find_as("b")->second);
  EXPECT_EQ(3u, map.find_as("c")->second);
  EXPECT_TRUE(map.find_as("d") == map.end());
}

}
