#include "gtest/gtest.h"
#include "Core/PerfSupport.h"

using namespace llvm;
using namespace clang;

class TransformA : public Transform {
public:
  TransformA(const TransformOptions &Options)
      : Transform("TransformA", Options) {}

  virtual int apply(FileOverrides &,
                    const tooling::CompilationDatabase &,
                    const std::vector<std::string> &) {
    return 0;
  }

  void addTiming(StringRef Label, TimeRecord Duration) {
    Transform::addTiming(Label, Duration);
  }
};

class TransformB : public Transform {
public:
  TransformB(const TransformOptions &Options)
      : Transform("TransformB", Options) {}

  virtual int apply(FileOverrides &,
                    const tooling::CompilationDatabase &,
                    const std::vector<std::string> &) {
    return 0;
  }

  void addTiming(StringRef Label, TimeRecord Duration) {
    Transform::addTiming(Label, Duration);
  }
};

struct ExpectedResults {
  const char *SourceName;
  unsigned DataCount;
  struct Datum {
    const char *Label;
    float Duration;
  } Data[2];
};

TEST(PerfSupport, collectSourcePerfData) {
  TransformOptions Options;
  TransformA A(Options);
  TransformB B(Options);
  
  // The actual durations don't matter. Below only their relative ordering is
  // tested to ensure times, labels, and sources all stay together properly.
  A.addTiming("FileA.cpp", TimeRecord::getCurrentTime(/*Start=*/true));
  A.addTiming("FileC.cpp", TimeRecord::getCurrentTime(/*Start=*/true));
  B.addTiming("FileC.cpp", TimeRecord::getCurrentTime(/*Start=*/true));
  B.addTiming("FileB.cpp", TimeRecord::getCurrentTime(/*Start=*/true));

  SourcePerfData PerfData;
  collectSourcePerfData(A, PerfData);

  SourcePerfData::const_iterator FileAI = PerfData.find("FileA.cpp");
  EXPECT_NE(FileAI, PerfData.end());
  SourcePerfData::const_iterator FileCI = PerfData.find("FileC.cpp");
  EXPECT_NE(FileCI, PerfData.end());
  EXPECT_EQ(2u, PerfData.size());

  EXPECT_EQ(1u, FileAI->second.size());
  EXPECT_EQ("TransformA", FileAI->second[0].Label);
  EXPECT_EQ(1u, FileCI->second.size());
  EXPECT_EQ("TransformA", FileCI->second[0].Label);
  EXPECT_LE(FileAI->second[0].Duration, FileCI->second[0].Duration);

  collectSourcePerfData(B, PerfData);

  SourcePerfData::const_iterator FileBI = PerfData.find("FileB.cpp");
  EXPECT_NE(FileBI, PerfData.end());
  EXPECT_EQ(3u, PerfData.size());

  EXPECT_EQ(1u, FileAI->second.size());
  EXPECT_EQ("TransformA", FileAI->second[0].Label);
  EXPECT_EQ(2u, FileCI->second.size());
  EXPECT_EQ("TransformA", FileCI->second[0].Label);
  EXPECT_EQ("TransformB", FileCI->second[1].Label);
  EXPECT_LE(FileCI->second[0].Duration, FileCI->second[1].Duration);
  EXPECT_EQ(1u, FileBI->second.size());
  EXPECT_EQ("TransformB", FileBI->second[0].Label);
  EXPECT_LE(FileCI->second[1].Duration, FileBI->second[0].Duration);
}
