//===- unittest/ProfileData/CoverageMappingTest.cpp -------------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/ProfileData/CoverageMappingWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <sstream>

using namespace llvm;
using namespace coverage;

namespace llvm {
namespace coverage {
void PrintTo(const Counter &C, ::std::ostream *os) {
  if (C.isZero())
    *os << "Zero";
  else if (C.isExpression())
    *os << "Expression " << C.getExpressionID();
  else
    *os << "Counter " << C.getCounterID();
}
}
}

namespace {

struct CoverageMappingTest : ::testing::Test {
  StringMap<unsigned> Files;
  unsigned NextFile;
  std::vector<CounterMappingRegion> InputCMRs;

  std::vector<StringRef> OutputFiles;
  std::vector<CounterExpression> OutputExpressions;
  std::vector<CounterMappingRegion> OutputCMRs;

  void SetUp() override {
    NextFile = 0;
  }

  unsigned getFile(StringRef Name) {
    auto R = Files.find(Name);
    if (R != Files.end())
      return R->second;
    Files[Name] = NextFile;
    return NextFile++;
  }

  void addCMR(Counter C, StringRef File, unsigned LS, unsigned CS, unsigned LE,
              unsigned CE) {
    InputCMRs.push_back(
        CounterMappingRegion::makeRegion(C, getFile(File), LS, CS, LE, CE));
  }

  void addExpansionCMR(StringRef File, StringRef ExpandedFile, unsigned LS,
                       unsigned CS, unsigned LE, unsigned CE) {
    InputCMRs.push_back(CounterMappingRegion::makeExpansion(
        getFile(File), getFile(ExpandedFile), LS, CS, LE, CE));
  }

  std::string writeCoverageRegions() {
    SmallVector<unsigned, 8> FileIDs;
    for (const auto &E : Files)
      FileIDs.push_back(E.getValue());
    std::string Coverage;
    llvm::raw_string_ostream OS(Coverage);
    CoverageMappingWriter(FileIDs, None, InputCMRs).write(OS);
    return OS.str();
  }

  void readCoverageRegions(std::string Coverage) {
    SmallVector<StringRef, 8> Filenames;
    for (const auto &E : Files)
      Filenames.push_back(E.getKey());
    RawCoverageMappingReader Reader(Coverage, Filenames, OutputFiles,
                                    OutputExpressions, OutputCMRs);
    std::error_code EC = Reader.read();
    ASSERT_EQ(instrprof_error::success, EC);
  }
};

TEST_F(CoverageMappingTest, basic_write_read) {
  addCMR(Counter::getCounter(0), "foo", 1, 1, 1, 1);
  addCMR(Counter::getCounter(1), "foo", 2, 1, 2, 2);
  addCMR(Counter::getZero(),     "foo", 3, 1, 3, 4);
  addCMR(Counter::getCounter(2), "foo", 4, 1, 4, 8);
  addCMR(Counter::getCounter(3), "bar", 1, 2, 3, 4);
  std::string Coverage = writeCoverageRegions();
  readCoverageRegions(Coverage);

  size_t N = makeArrayRef(InputCMRs).size();
  ASSERT_EQ(N, OutputCMRs.size());
  for (size_t I = 0; I < N; ++I) {
    ASSERT_EQ(InputCMRs[I].Count,      OutputCMRs[I].Count);
    ASSERT_EQ(InputCMRs[I].FileID,     OutputCMRs[I].FileID);
    ASSERT_EQ(InputCMRs[I].startLoc(), OutputCMRs[I].startLoc());
    ASSERT_EQ(InputCMRs[I].endLoc(),   OutputCMRs[I].endLoc());
    ASSERT_EQ(InputCMRs[I].Kind,       OutputCMRs[I].Kind);
  }
}

TEST_F(CoverageMappingTest, expansion_gets_first_counter) {
  addCMR(Counter::getCounter(1), "foo", 10, 1, 10, 2);
  // This starts earlier in "foo", so the expansion should get its counter.
  addCMR(Counter::getCounter(2), "foo", 1, 1, 20, 1);
  addExpansionCMR("bar", "foo", 3, 3, 3, 3);
  std::string Coverage = writeCoverageRegions();
  readCoverageRegions(Coverage);

  ASSERT_EQ(CounterMappingRegion::ExpansionRegion, OutputCMRs[2].Kind);
  ASSERT_EQ(Counter::getCounter(2), OutputCMRs[2].Count);
  ASSERT_EQ(3U, OutputCMRs[2].LineStart);
}


} // end anonymous namespace
