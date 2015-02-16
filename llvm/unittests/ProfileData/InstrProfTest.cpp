//===- unittest/ProfileData/InstrProfTest.cpp -------------------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(InstrProfTest, write_and_read_empty_profile) {
  InstrProfWriter Writer;
  std::string Profile = Writer.writeString();
  auto ReaderOrErr =
      IndexedInstrProfReader::create(MemoryBuffer::getMemBuffer(Profile));
  ASSERT_EQ(std::error_code(), ReaderOrErr.getError());
  auto Reader = std::move(ReaderOrErr.get());
  ASSERT_TRUE(Reader->begin() == Reader->end());
}

TEST(InstrProfTest, write_and_read_one_function) {
  InstrProfWriter Writer;
  Writer.addFunctionCounts("foo", 0x1234, {1, 2, 3, 4});
  std::string Profile = Writer.writeString();
  auto ReaderOrErr =
      IndexedInstrProfReader::create(MemoryBuffer::getMemBuffer(Profile));
  ASSERT_EQ(std::error_code(), ReaderOrErr.getError());
  auto Reader = std::move(ReaderOrErr.get());

  auto I = Reader->begin(), E = Reader->end();
  ASSERT_TRUE(I != E);
  ASSERT_EQ(StringRef("foo"), I->Name);
  ASSERT_EQ(0x1234U, I->Hash);
  ASSERT_EQ(4U, I->Counts.size());
  ASSERT_EQ(1U, I->Counts[0]);
  ASSERT_EQ(2U, I->Counts[1]);
  ASSERT_EQ(3U, I->Counts[2]);
  ASSERT_EQ(4U, I->Counts[3]);
  ASSERT_TRUE(++I == E);
}

TEST(InstrProfTest, get_function_counts) {
  InstrProfWriter Writer;
  Writer.addFunctionCounts("foo", 0x1234, {1, 2});
  std::string Profile = Writer.writeString();
  auto ReaderOrErr =
      IndexedInstrProfReader::create(MemoryBuffer::getMemBuffer(Profile));
  ASSERT_EQ(std::error_code(), ReaderOrErr.getError());
  auto Reader = std::move(ReaderOrErr.get());

  std::vector<uint64_t> Counts;
  std::error_code EC;

  EC = Reader->getFunctionCounts("foo", 0x1234, Counts);
  ASSERT_EQ(instrprof_error::success, EC);
  ASSERT_EQ(2U, Counts.size());
  ASSERT_EQ(1U, Counts[0]);
  ASSERT_EQ(2U, Counts[1]);

  EC = Reader->getFunctionCounts("foo", 0x5678, Counts);
  ASSERT_EQ(instrprof_error::hash_mismatch, EC);

  EC = Reader->getFunctionCounts("bar", 0x1234, Counts);
  ASSERT_EQ(instrprof_error::unknown_function, EC);
}

TEST(InstrProfTest, get_max_function_count) {
  InstrProfWriter Writer;
  Writer.addFunctionCounts("foo", 0x1234, {1ULL << 31, 2});
  Writer.addFunctionCounts("bar", 0, {1ULL << 63});
  Writer.addFunctionCounts("baz", 0x5678, {0, 0, 0, 0});
  std::string Profile = Writer.writeString();
  auto ReaderOrErr =
      IndexedInstrProfReader::create(MemoryBuffer::getMemBuffer(Profile));
  ASSERT_EQ(std::error_code(), ReaderOrErr.getError());
  auto Reader = std::move(ReaderOrErr.get());

  ASSERT_EQ(1ULL << 63, Reader->getMaximumFunctionCount());
}

} // end anonymous namespace
