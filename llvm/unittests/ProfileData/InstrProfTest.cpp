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

struct InstrProfTest : ::testing::Test {
  InstrProfWriter Writer;
  std::unique_ptr<IndexedInstrProfReader> Reader;

  void addCounts(StringRef Name, uint64_t Hash, int NumCounts, ...) {
    SmallVector<uint64_t, 8> Counts;
    va_list Args;
    va_start(Args, NumCounts);
    for (int I = 0; I < NumCounts; ++I)
      Counts.push_back(va_arg(Args, uint64_t));
    va_end(Args);
    Writer.addFunctionCounts(Name, Hash, Counts);
  }

  std::string writeProfile() { return Writer.writeString(); }
  void readProfile(std::string Profile) {
    auto ReaderOrErr =
        IndexedInstrProfReader::create(MemoryBuffer::getMemBuffer(Profile));
    ASSERT_EQ(std::error_code(), ReaderOrErr.getError());
    Reader = std::move(ReaderOrErr.get());
  }
};

TEST_F(InstrProfTest, write_and_read_empty_profile) {
  std::string Profile = writeProfile();
  readProfile(Profile);
  ASSERT_TRUE(Reader->begin() == Reader->end());
}

TEST_F(InstrProfTest, write_and_read_one_function) {
  addCounts("foo", 0x1234, 4, 1ULL, 2ULL, 3ULL, 4ULL);
  std::string Profile = writeProfile();
  readProfile(Profile);

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

TEST_F(InstrProfTest, get_function_counts) {
  addCounts("foo", 0x1234, 2, 1ULL, 2ULL);
  std::string Profile = writeProfile();
  readProfile(Profile);

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

TEST_F(InstrProfTest, get_max_function_count) {
  addCounts("foo", 0x1234, 2, 1ULL << 31, 2ULL);
  addCounts("bar", 0, 1, 1ULL << 63);
  addCounts("baz", 0x5678, 4, 0ULL, 0ULL, 0ULL, 0ULL);
  std::string Profile = writeProfile();
  readProfile(Profile);

  ASSERT_EQ(1ULL << 63, Reader->getMaximumFunctionCount());
}

} // end anonymous namespace
