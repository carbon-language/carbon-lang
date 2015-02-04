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

static std::string writeCoverage(MutableArrayRef<CounterMappingRegion> Regions,
                                 int NumFiles) {
  SmallVector<unsigned, 8> FileIDs;
  for (int I = 0; I < NumFiles; ++I)
    FileIDs.push_back(I);

  std::string Coverage;
  llvm::raw_string_ostream OS(Coverage);
  CoverageMappingWriter(FileIDs, None, Regions).write(OS);
  OS.flush();

  return Coverage;
}

static std::vector<CounterMappingRegion>
readCoverageRegions(std::string Coverage, int NumFiles) {
  SmallVector<std::string, 8> Filenames;
  SmallVector<StringRef, 8> FilenameRefs;
  for (int I = 0; I < NumFiles; ++I) {
    Filenames.push_back("file" + std::to_string(I));
    FilenameRefs.push_back(Filenames.back());
  }

  std::vector<StringRef> FuncFiles;
  std::vector<CounterExpression> Expressions;
  std::vector<CounterMappingRegion> Regions;
  RawCoverageMappingReader Reader(Coverage, FilenameRefs, FuncFiles,
                                  Expressions, Regions);
  if (Reader.read())
    // ASSERT doesn't work here, we'll just return an empty vector.
    Regions.clear();
  return Regions;
}

TEST(CoverageMappingTest, basic_write_read) {
  int NumFiles = 2;
  CounterMappingRegion InputRegions[] = {
      CounterMappingRegion::makeRegion(Counter::getCounter(0), 0, 1, 1, 1, 1),
      CounterMappingRegion::makeRegion(Counter::getCounter(1), 0, 2, 1, 2, 2),
      CounterMappingRegion::makeRegion(Counter::getZero(), 0, 3, 1, 3, 4),
      CounterMappingRegion::makeRegion(Counter::getCounter(2), 0, 4, 1, 4, 8),
      CounterMappingRegion::makeRegion(Counter::getCounter(3), 1, 1, 2, 3, 4),
  };
  std::string Coverage = writeCoverage(InputRegions, NumFiles);
  std::vector<CounterMappingRegion> OutputRegions =
      readCoverageRegions(Coverage, NumFiles);
  ASSERT_FALSE(OutputRegions.empty());

  size_t N = makeArrayRef(InputRegions).size();
  ASSERT_EQ(N, OutputRegions.size());
  for (size_t I = 0; I < N; ++I) {
    ASSERT_EQ(InputRegions[I].Count, OutputRegions[I].Count);
    ASSERT_EQ(InputRegions[I].FileID, OutputRegions[I].FileID);
    ASSERT_EQ(InputRegions[I].startLoc(), OutputRegions[I].startLoc());
    ASSERT_EQ(InputRegions[I].endLoc(), OutputRegions[I].endLoc());
    ASSERT_EQ(InputRegions[I].Kind, OutputRegions[I].Kind);
  }
}

TEST(CoverageMappingTest, expansion_gets_first_counter) {
  int NumFiles = 2;
  CounterMappingRegion InputRegions[] = {
      CounterMappingRegion::makeRegion(Counter::getCounter(1), 0, 10, 1, 10, 2),
      // This starts earlier in file 0, so the expansion should get its counter.
      CounterMappingRegion::makeRegion(Counter::getCounter(2), 0, 1, 1, 20, 1),
      CounterMappingRegion::makeExpansion(1, 0, 3, 3, 3, 3),
  };
  std::string Coverage = writeCoverage(InputRegions, NumFiles);
  std::vector<CounterMappingRegion> OutputRegions =
      readCoverageRegions(Coverage, NumFiles);
  ASSERT_FALSE(OutputRegions.empty());

  ASSERT_EQ(CounterMappingRegion::ExpansionRegion, OutputRegions[2].Kind);
  ASSERT_EQ(Counter::getCounter(2), OutputRegions[2].Count);
  ASSERT_EQ(3U, OutputRegions[2].LineStart);
}


} // end anonymous namespace
