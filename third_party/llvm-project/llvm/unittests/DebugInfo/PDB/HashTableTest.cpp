//===- llvm/unittest/DebugInfo/PDB/HashTableTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Native/HashTable.h"

#include "llvm/DebugInfo/PDB/Native/Hash.h"
#include "llvm/DebugInfo/PDB/Native/NamedStreamMap.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

#include <vector>

using namespace llvm;
using namespace llvm::pdb;
using namespace llvm::support;

namespace {

struct IdentityHashTraits {
  uint32_t hashLookupKey(uint32_t N) const { return N; }
  uint32_t storageKeyToLookupKey(uint32_t N) const { return N; }
  uint32_t lookupKeyToStorageKey(uint32_t N) { return N; }
};

template <class T = uint32_t>
class HashTableInternals : public HashTable<T> {
public:
  using HashTable<T>::Buckets;
  using HashTable<T>::Present;
  using HashTable<T>::Deleted;
};
}

TEST(HashTableTest, TestSimple) {
  HashTableInternals<> Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  IdentityHashTraits Traits;
  Table.set_as(3u, 7, Traits);
  EXPECT_EQ(1u, Table.size());
  ASSERT_NE(Table.end(), Table.find_as(3u, Traits));
  EXPECT_EQ(7u, Table.get(3u, Traits));
}

TEST(HashTableTest, TestCollision) {
  HashTableInternals<> Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  // We use knowledge of the hash table's implementation details to make sure
  // to add another value that is the equivalent to the first value modulo the
  // hash table's capacity.
  uint32_t N1 = Table.capacity() + 1;
  uint32_t N2 = 2 * N1;

  IdentityHashTraits Traits;
  Table.set_as(N1, 7, Traits);
  Table.set_as(N2, 12, Traits);
  EXPECT_EQ(2u, Table.size());
  ASSERT_NE(Table.end(), Table.find_as(N1, Traits));
  ASSERT_NE(Table.end(), Table.find_as(N2, Traits));

  EXPECT_EQ(7u, Table.get(N1, Traits));
  EXPECT_EQ(12u, Table.get(N2, Traits));
}

TEST(HashTableTest, TestRemove) {
  HashTableInternals<> Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  IdentityHashTraits Traits;
  Table.set_as(1u, 2, Traits);
  Table.set_as(3u, 4, Traits);
  EXPECT_EQ(2u, Table.size());
  ASSERT_NE(Table.end(), Table.find_as(1u, Traits));
  ASSERT_NE(Table.end(), Table.find_as(3u, Traits));

  EXPECT_EQ(2u, Table.get(1u, Traits));
  EXPECT_EQ(4u, Table.get(3u, Traits));
}

TEST(HashTableTest, TestCollisionAfterMultipleProbes) {
  HashTableInternals<> Table;
  EXPECT_EQ(0u, Table.size());
  EXPECT_GT(Table.capacity(), 0u);

  // Probing looks for the first available slot.  A slot may already be filled
  // as a result of an item with a *different* hash value already being there.
  // Test that when this happens, the probe still finds the value.
  uint32_t N1 = Table.capacity() + 1;
  uint32_t N2 = N1 + 1;
  uint32_t N3 = 2 * N1;

  IdentityHashTraits Traits;
  Table.set_as(N1, 7, Traits);
  Table.set_as(N2, 11, Traits);
  Table.set_as(N3, 13, Traits);
  EXPECT_EQ(3u, Table.size());
  ASSERT_NE(Table.end(), Table.find_as(N1, Traits));
  ASSERT_NE(Table.end(), Table.find_as(N2, Traits));
  ASSERT_NE(Table.end(), Table.find_as(N3, Traits));

  EXPECT_EQ(7u, Table.get(N1, Traits));
  EXPECT_EQ(11u, Table.get(N2, Traits));
  EXPECT_EQ(13u, Table.get(N3, Traits));
}

TEST(HashTableTest, Grow) {
  // So that we are independent of the load factor, `capacity` items, which is
  // guaranteed to trigger a grow.  Then verify that the size is the same, the
  // capacity is larger, and all the original items are still in the table.

  HashTableInternals<> Table;
  IdentityHashTraits Traits;
  uint32_t OldCapacity = Table.capacity();
  for (uint32_t I = 0; I < OldCapacity; ++I) {
    Table.set_as(OldCapacity + I * 2 + 1, I * 2 + 3, Traits);
  }
  EXPECT_EQ(OldCapacity, Table.size());
  EXPECT_GT(Table.capacity(), OldCapacity);
  for (uint32_t I = 0; I < OldCapacity; ++I) {
    ASSERT_NE(Table.end(), Table.find_as(OldCapacity + I * 2 + 1, Traits));
    EXPECT_EQ(I * 2 + 3, Table.get(OldCapacity + I * 2 + 1, Traits));
  }
}

TEST(HashTableTest, Serialization) {
  HashTableInternals<> Table;
  IdentityHashTraits Traits;
  uint32_t Cap = Table.capacity();
  for (uint32_t I = 0; I < Cap; ++I) {
    Table.set_as(Cap + I * 2 + 1, I * 2 + 3, Traits);
  }

  std::vector<uint8_t> Buffer(Table.calculateSerializedLength());
  MutableBinaryByteStream Stream(Buffer, little);
  BinaryStreamWriter Writer(Stream);
  EXPECT_THAT_ERROR(Table.commit(Writer), Succeeded());
  // We should have written precisely the number of bytes we calculated earlier.
  EXPECT_EQ(Buffer.size(), Writer.getOffset());

  HashTableInternals<> Table2;
  BinaryStreamReader Reader(Stream);
  EXPECT_THAT_ERROR(Table2.load(Reader), Succeeded());
  // We should have read precisely the number of bytes we calculated earlier.
  EXPECT_EQ(Buffer.size(), Reader.getOffset());

  EXPECT_EQ(Table.size(), Table2.size());
  EXPECT_EQ(Table.capacity(), Table2.capacity());
  EXPECT_EQ(Table.Buckets, Table2.Buckets);
  EXPECT_EQ(Table.Present, Table2.Present);
  EXPECT_EQ(Table.Deleted, Table2.Deleted);
}

TEST(HashTableTest, NamedStreamMap) {
  std::vector<StringRef> Streams = {"One",  "Two", "Three", "Four",
                                    "Five", "Six", "Seven"};
  StringMap<uint32_t> ExpectedIndices;
  for (uint32_t I = 0; I < Streams.size(); ++I)
    ExpectedIndices[Streams[I]] = I + 1;

  // To verify the hash table actually works, we want to verify that insertion
  // order doesn't matter.  So try inserting in every possible order of 7 items.
  do {
    NamedStreamMap NSM;
    for (StringRef S : Streams)
      NSM.set(S, ExpectedIndices[S]);

    EXPECT_EQ(Streams.size(), NSM.size());

    uint32_t N;
    EXPECT_TRUE(NSM.get("One", N));
    EXPECT_EQ(1U, N);

    EXPECT_TRUE(NSM.get("Two", N));
    EXPECT_EQ(2U, N);

    EXPECT_TRUE(NSM.get("Three", N));
    EXPECT_EQ(3U, N);

    EXPECT_TRUE(NSM.get("Four", N));
    EXPECT_EQ(4U, N);

    EXPECT_TRUE(NSM.get("Five", N));
    EXPECT_EQ(5U, N);

    EXPECT_TRUE(NSM.get("Six", N));
    EXPECT_EQ(6U, N);

    EXPECT_TRUE(NSM.get("Seven", N));
    EXPECT_EQ(7U, N);
  } while (std::next_permutation(Streams.begin(), Streams.end()));
}

struct FooBar {
  uint32_t X;
  uint32_t Y;

  bool operator==(const FooBar &RHS) const {
    return X == RHS.X && Y == RHS.Y;
  }
};

struct FooBarHashTraits {
  std::vector<char> Buffer;

  FooBarHashTraits() { Buffer.push_back(0); }

  uint32_t hashLookupKey(StringRef S) const {
    return llvm::pdb::hashStringV1(S);
  }

  StringRef storageKeyToLookupKey(uint32_t N) const {
    if (N >= Buffer.size())
      return StringRef();

    return StringRef(Buffer.data() + N);
  }

  uint32_t lookupKeyToStorageKey(StringRef S) {
    uint32_t N = Buffer.size();
    Buffer.insert(Buffer.end(), S.begin(), S.end());
    Buffer.push_back('\0');
    return N;
  }
};

TEST(HashTableTest, NonTrivialValueType) {
  HashTableInternals<FooBar> Table;
  FooBarHashTraits Traits;
  uint32_t Cap = Table.capacity();
  for (uint32_t I = 0; I < Cap; ++I) {
    FooBar F;
    F.X = I;
    F.Y = I + 1;
    Table.set_as(utostr(I), F, Traits);
  }

  std::vector<uint8_t> Buffer(Table.calculateSerializedLength());
  MutableBinaryByteStream Stream(Buffer, little);
  BinaryStreamWriter Writer(Stream);
  EXPECT_THAT_ERROR(Table.commit(Writer), Succeeded());
  // We should have written precisely the number of bytes we calculated earlier.
  EXPECT_EQ(Buffer.size(), Writer.getOffset());

  HashTableInternals<FooBar> Table2;
  BinaryStreamReader Reader(Stream);
  EXPECT_THAT_ERROR(Table2.load(Reader), Succeeded());
  // We should have read precisely the number of bytes we calculated earlier.
  EXPECT_EQ(Buffer.size(), Reader.getOffset());

  EXPECT_EQ(Table.size(), Table2.size());
  EXPECT_EQ(Table.capacity(), Table2.capacity());
  EXPECT_EQ(Table.Buckets, Table2.Buckets);
  EXPECT_EQ(Table.Present, Table2.Present);
  EXPECT_EQ(Table.Deleted, Table2.Deleted);
}
