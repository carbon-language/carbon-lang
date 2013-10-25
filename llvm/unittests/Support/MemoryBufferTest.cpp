//===- llvm/unittest/Support/MemoryBufferTest.cpp - MemoryBuffer tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the MemoryBuffer support class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/OwningPtr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class MemoryBufferTest : public testing::Test {
protected:
  MemoryBufferTest()
  : data("this is some data")
  { }

  virtual void SetUp() { }

  /// Common testing for different modes of getOpenFileSlice.
  /// Creates a temporary file with known contents, and uses
  /// MemoryBuffer::getOpenFileSlice to map it.
  /// If \p Reopen is true, the file is closed after creating and reopened
  /// anew before using MemoryBuffer.
  void testGetOpenFileSlice(bool Reopen);

  typedef OwningPtr<MemoryBuffer> OwningBuffer;

  std::string data;
};

TEST_F(MemoryBufferTest, get) {
  // Default name and null-terminator flag
  OwningBuffer MB1(MemoryBuffer::getMemBuffer(data));
  EXPECT_TRUE(0 != MB1.get());

  // RequiresNullTerminator = false
  OwningBuffer MB2(MemoryBuffer::getMemBuffer(data, "one", false));
  EXPECT_TRUE(0 != MB2.get());

  // RequiresNullTerminator = true
  OwningBuffer MB3(MemoryBuffer::getMemBuffer(data, "two", true));
  EXPECT_TRUE(0 != MB3.get());

  // verify all 3 buffers point to the same address
  EXPECT_EQ(MB1->getBufferStart(), MB2->getBufferStart());
  EXPECT_EQ(MB2->getBufferStart(), MB3->getBufferStart());

  // verify the original data is unmodified after deleting the buffers
  MB1.reset();
  MB2.reset();
  MB3.reset();
  EXPECT_EQ("this is some data", data);
}

TEST_F(MemoryBufferTest, NullTerminator4K) {
  // Test that a file with size that is a multiple of the page size can be null
  // terminated correctly by MemoryBuffer.
  int TestFD;
  SmallString<64> TestPath;
  sys::fs::createTemporaryFile("MemoryBufferTest_NullTerminator4K", "temp",
                               TestFD, TestPath);
  raw_fd_ostream OF(TestFD, true, /*unbuffered=*/true);
  for (unsigned i = 0; i < 4096 / 16; ++i) {
    OF << "0123456789abcdef";
  }
  OF.close();

  OwningPtr<MemoryBuffer> MB;
  error_code EC = MemoryBuffer::getFile(TestPath.c_str(), MB);
  ASSERT_FALSE(EC);

  const char *BufData = MB->getBufferStart();
  EXPECT_EQ('f', BufData[4095]);
  EXPECT_EQ('\0', BufData[4096]);
}

TEST_F(MemoryBufferTest, copy) {
  // copy with no name
  OwningBuffer MBC1(MemoryBuffer::getMemBufferCopy(data));
  EXPECT_TRUE(0 != MBC1.get());

  // copy with a name
  OwningBuffer MBC2(MemoryBuffer::getMemBufferCopy(data, "copy"));
  EXPECT_TRUE(0 != MBC2.get());

  // verify the two copies do not point to the same place
  EXPECT_NE(MBC1->getBufferStart(), MBC2->getBufferStart());
}

TEST_F(MemoryBufferTest, make_new) {
  // 0-sized buffer
  OwningBuffer Zero(MemoryBuffer::getNewUninitMemBuffer(0));
  EXPECT_TRUE(0 != Zero.get());

  // uninitialized buffer with no name
  OwningBuffer One(MemoryBuffer::getNewUninitMemBuffer(321));
  EXPECT_TRUE(0 != One.get());

  // uninitialized buffer with name
  OwningBuffer Two(MemoryBuffer::getNewUninitMemBuffer(123, "bla"));
  EXPECT_TRUE(0 != Two.get());

  // 0-initialized buffer with no name
  OwningBuffer Three(MemoryBuffer::getNewMemBuffer(321, data));
  EXPECT_TRUE(0 != Three.get());
  for (size_t i = 0; i < 321; ++i)
    EXPECT_EQ(0, Three->getBufferStart()[0]);

  // 0-initialized buffer with name
  OwningBuffer Four(MemoryBuffer::getNewMemBuffer(123, "zeros"));
  EXPECT_TRUE(0 != Four.get());
  for (size_t i = 0; i < 123; ++i)
    EXPECT_EQ(0, Four->getBufferStart()[0]);
}

void MemoryBufferTest::testGetOpenFileSlice(bool Reopen) {
  // Test that MemoryBuffer::getOpenFile works properly when no null
  // terminator is requested and the size is large enough to trigger
  // the usage of memory mapping.
  int TestFD;
  SmallString<64> TestPath;
  // Create a temporary file and write data into it.
  sys::fs::createTemporaryFile("prefix", "temp", TestFD, TestPath);
  // OF is responsible for closing the file; If the file is not
  // reopened, it will be unbuffered so that the results are
  // immediately visible through the fd.
  raw_fd_ostream OF(TestFD, true, !Reopen);
  for (int i = 0; i < 60000; ++i) {
    OF << "0123456789";
  }

  if (Reopen) {
    OF.close();
    EXPECT_FALSE(sys::fs::openFileForRead(TestPath.c_str(), TestFD));
  }

  OwningBuffer Buf;
  error_code EC = MemoryBuffer::getOpenFileSlice(TestFD, TestPath.c_str(), Buf,
                                                 40000, // Size
                                                 80000  // Offset
                                                 );
  EXPECT_FALSE(EC);

  StringRef BufData = Buf->getBuffer();
  EXPECT_EQ(BufData.size(), 40000U);
  EXPECT_EQ(BufData[0], '0');
  EXPECT_EQ(BufData[9], '9');
}

TEST_F(MemoryBufferTest, getOpenFileNoReopen) {
  testGetOpenFileSlice(false);
}

TEST_F(MemoryBufferTest, getOpenFileReopened) {
  testGetOpenFileSlice(true);
}

}
