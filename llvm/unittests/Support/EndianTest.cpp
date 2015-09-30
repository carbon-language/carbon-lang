//===- unittests/Support/EndianTest.cpp - Endian.h tests ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Endian.h"
#include "llvm/Support/DataTypes.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <ctime>
using namespace llvm;
using namespace support;

#undef max

namespace {

TEST(Endian, Read) {
  // These are 5 bytes so we can be sure at least one of the reads is unaligned.
  unsigned char bigval[] = {0x00, 0x01, 0x02, 0x03, 0x04};
  unsigned char littleval[] = {0x00, 0x04, 0x03, 0x02, 0x01};
  int32_t BigAsHost = 0x00010203;
  EXPECT_EQ(BigAsHost, (endian::read<int32_t, big, unaligned>(bigval)));
  int32_t LittleAsHost = 0x02030400;
  EXPECT_EQ(LittleAsHost,(endian::read<int32_t, little, unaligned>(littleval)));

  EXPECT_EQ((endian::read<int32_t, big, unaligned>(bigval + 1)),
            (endian::read<int32_t, little, unaligned>(littleval + 1)));
}

TEST(Endian, ReadBitAligned) {
  // Simple test to make sure we properly pull out the 0x0 word.
  unsigned char littleval[] = {0x3f, 0x00, 0x00, 0x00, 0xc0, 0xff, 0xff, 0xff};
  unsigned char bigval[] = {0x00, 0x00, 0x00, 0x3f, 0xff, 0xff, 0xff, 0xc0};
  EXPECT_EQ(
      (endian::readAtBitAlignment<int, little, unaligned>(&littleval[0], 6)),
      0x0);
  EXPECT_EQ((endian::readAtBitAlignment<int, big, unaligned>(&bigval[0], 6)),
            0x0);
  // Test to make sure that signed right shift of 0xf0000000 is masked
  // properly.
  unsigned char littleval2[] = {0x00, 0x00, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x00};
  unsigned char bigval2[] = {0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  EXPECT_EQ(
      (endian::readAtBitAlignment<int, little, unaligned>(&littleval2[0], 4)),
      0x0f000000);
  EXPECT_EQ((endian::readAtBitAlignment<int, big, unaligned>(&bigval2[0], 4)),
            0x0f000000);
}

TEST(Endian, WriteBitAligned) {
  // This test ensures that signed right shift of 0xffffaa is masked
  // properly.
  unsigned char bigval[8] = {0x00};
  endian::writeAtBitAlignment<int32_t, big, unaligned>(bigval, (int)0xffffaaaa,
                                                       4);
  EXPECT_EQ(bigval[0], 0xff);
  EXPECT_EQ(bigval[1], 0xfa);
  EXPECT_EQ(bigval[2], 0xaa);
  EXPECT_EQ(bigval[3], 0xa0);
  EXPECT_EQ(bigval[4], 0x00);
  EXPECT_EQ(bigval[5], 0x00);
  EXPECT_EQ(bigval[6], 0x00);
  EXPECT_EQ(bigval[7], 0x0f);

  unsigned char littleval[8] = {0x00};
  endian::writeAtBitAlignment<int32_t, little, unaligned>(littleval,
                                                          (int)0xffffaaaa, 4);
  EXPECT_EQ(littleval[0], 0xa0);
  EXPECT_EQ(littleval[1], 0xaa);
  EXPECT_EQ(littleval[2], 0xfa);
  EXPECT_EQ(littleval[3], 0xff);
  EXPECT_EQ(littleval[4], 0x0f);
  EXPECT_EQ(littleval[5], 0x00);
  EXPECT_EQ(littleval[6], 0x00);
  EXPECT_EQ(littleval[7], 0x00);
}

TEST(Endian, Write) {
  unsigned char data[5];
  endian::write<int32_t, big, unaligned>(data, -1362446643);
  EXPECT_EQ(data[0], 0xAE);
  EXPECT_EQ(data[1], 0xCA);
  EXPECT_EQ(data[2], 0xB6);
  EXPECT_EQ(data[3], 0xCD);
  endian::write<int32_t, big, unaligned>(data + 1, -1362446643);
  EXPECT_EQ(data[1], 0xAE);
  EXPECT_EQ(data[2], 0xCA);
  EXPECT_EQ(data[3], 0xB6);
  EXPECT_EQ(data[4], 0xCD);

  endian::write<int32_t, little, unaligned>(data, -1362446643);
  EXPECT_EQ(data[0], 0xCD);
  EXPECT_EQ(data[1], 0xB6);
  EXPECT_EQ(data[2], 0xCA);
  EXPECT_EQ(data[3], 0xAE);
  endian::write<int32_t, little, unaligned>(data + 1, -1362446643);
  EXPECT_EQ(data[1], 0xCD);
  EXPECT_EQ(data[2], 0xB6);
  EXPECT_EQ(data[3], 0xCA);
  EXPECT_EQ(data[4], 0xAE);
}

TEST(Endian, PackedEndianSpecificIntegral) {
  // These are 5 bytes so we can be sure at least one of the reads is unaligned.
  unsigned char big[] = {0x00, 0x01, 0x02, 0x03, 0x04};
  unsigned char little[] = {0x00, 0x04, 0x03, 0x02, 0x01};
  big32_t    *big_val    =
    reinterpret_cast<big32_t *>(big + 1);
  little32_t *little_val =
    reinterpret_cast<little32_t *>(little + 1);

  EXPECT_EQ(*big_val, *little_val);
}

} // end anon namespace
