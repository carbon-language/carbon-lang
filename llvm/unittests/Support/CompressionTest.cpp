//===- llvm/unittest/Support/CompressionTest.cpp - Compression tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the Compression functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compression.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

#if LLVM_ENABLE_ZLIB == 1 && HAVE_LIBZ

void TestZlibCompression(StringRef Input, zlib::CompressionLevel Level) {
  OwningPtr<MemoryBuffer> Compressed;
  OwningPtr<MemoryBuffer> Uncompressed;
  EXPECT_EQ(zlib::StatusOK, zlib::compress(Input, Compressed, Level));
  // Check that uncompressed buffer is the same as original.
  EXPECT_EQ(zlib::StatusOK, zlib::uncompress(Compressed->getBuffer(),
                                             Uncompressed, Input.size()));
  EXPECT_EQ(Input.size(), Uncompressed->getBufferSize());
  EXPECT_EQ(0,
            memcmp(Input.data(), Uncompressed->getBufferStart(), Input.size()));
  if (Input.size() > 0) {
    // Uncompression fails if expected length is too short.
    EXPECT_EQ(zlib::StatusBufferTooShort,
              zlib::uncompress(Compressed->getBuffer(), Uncompressed,
                               Input.size() - 1));
  }
}

TEST(CompressionTest, Zlib) {
  TestZlibCompression("", zlib::DefaultCompression);

  TestZlibCompression("hello, world!", zlib::NoCompression);
  TestZlibCompression("hello, world!", zlib::BestSizeCompression);
  TestZlibCompression("hello, world!", zlib::BestSpeedCompression);
  TestZlibCompression("hello, world!", zlib::DefaultCompression);

  const size_t kSize = 1024;
  char BinaryData[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    BinaryData[i] = i & 255;
  }
  StringRef BinaryDataStr(BinaryData, kSize);

  TestZlibCompression(BinaryDataStr, zlib::NoCompression);
  TestZlibCompression(BinaryDataStr, zlib::BestSizeCompression);
  TestZlibCompression(BinaryDataStr, zlib::BestSpeedCompression);
  TestZlibCompression(BinaryDataStr, zlib::DefaultCompression);
}

#endif

}
