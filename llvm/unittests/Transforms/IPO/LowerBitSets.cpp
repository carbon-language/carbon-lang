//===- LowerBitSets.cpp - Unit tests for bitset lowering ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/LowerBitSets.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(LowerBitSets, BitSetBuilder) {
  struct {
    std::vector<uint64_t> Offsets;
    std::vector<uint8_t> Bits;
    uint64_t ByteOffset;
    uint64_t BitSize;
    unsigned AlignLog2;
    bool IsSingleOffset;
    bool IsAllOnes;
  } BSBTests[] = {
      {{}, {0}, 0, 1, 0, false, false},
      {{0}, {1}, 0, 1, 0, true, true},
      {{4}, {1}, 4, 1, 0, true, true},
      {{37}, {1}, 37, 1, 0, true, true},
      {{0, 1}, {3}, 0, 2, 0, false, true},
      {{0, 4}, {3}, 0, 2, 2, false, true},
      {{0, uint64_t(1) << 33}, {3}, 0, 2, 33, false, true},
      {{3, 7}, {3}, 3, 2, 2, false, true},
      {{0, 1, 7}, {131}, 0, 8, 0, false, false},
      {{0, 2, 14}, {131}, 0, 8, 1, false, false},
      {{0, 1, 8}, {3, 1}, 0, 9, 0, false, false},
      {{0, 2, 16}, {3, 1}, 0, 9, 1, false, false},
      {{0, 1, 2, 3, 4, 5, 6, 7}, {255}, 0, 8, 0, false, true},
      {{0, 1, 2, 3, 4, 5, 6, 7, 8}, {255, 1}, 0, 9, 0, false, true},
  };

  for (auto &&T : BSBTests) {
    BitSetBuilder BSB;
    for (auto Offset : T.Offsets)
      BSB.addOffset(Offset);

    BitSetInfo BSI = BSB.build();

    EXPECT_EQ(T.Bits, BSI.Bits);
    EXPECT_EQ(T.ByteOffset, BSI.ByteOffset);
    EXPECT_EQ(T.BitSize, BSI.BitSize);
    EXPECT_EQ(T.AlignLog2, BSI.AlignLog2);
    EXPECT_EQ(T.IsSingleOffset, BSI.isSingleOffset());
    EXPECT_EQ(T.IsAllOnes, BSI.isAllOnes());

    for (auto Offset : T.Offsets)
      EXPECT_TRUE(BSI.containsGlobalOffset(Offset));

    auto I = T.Offsets.begin();
    for (uint64_t NonOffset = 0; NonOffset != 256; ++NonOffset) {
      if (I != T.Offsets.end() && *I == NonOffset) {
        ++I;
        continue;
      }

      EXPECT_FALSE(BSI.containsGlobalOffset(NonOffset));
    }
  }
}

TEST(LowerBitSets, GlobalLayoutBuilder) {
  struct {
    uint64_t NumObjects;
    std::vector<std::set<uint64_t>> Fragments;
    std::vector<uint64_t> WantLayout;
  } GLBTests[] = {
    {0, {}, {}},
    {4, {{0, 1}, {2, 3}}, {0, 1, 2, 3}},
    {3, {{0, 1}, {1, 2}}, {0, 1, 2}},
    {4, {{0, 1}, {1, 2}, {2, 3}}, {0, 1, 2, 3}},
    {4, {{0, 1}, {2, 3}, {1, 2}}, {0, 1, 2, 3}},
    {6, {{2, 5}, {0, 1, 2, 3, 4, 5}}, {0, 1, 2, 5, 3, 4}},
  };

  for (auto &&T : GLBTests) {
    GlobalLayoutBuilder GLB(T.NumObjects);
    for (auto &&F : T.Fragments)
      GLB.addFragment(F);

    std::vector<uint64_t> ComputedLayout;
    for (auto &&F : GLB.Fragments)
      ComputedLayout.insert(ComputedLayout.end(), F.begin(), F.end());

    EXPECT_EQ(T.WantLayout, ComputedLayout);
  }
}
