//===-- StatisticReporter.cpp - Easy way to expose stats information -------==//
//
// This file implements the 'Statistic' class, which is designed to be an easy
// way to expose various success metrics from passes.  These statistics are
// printed at the end of a run, when the -stats command line option is enabled
// on the command line.
//
// This is useful for reporting information like the number of instructions
// simplified, optimized or removed by various transformations, like this:
//
// static Statistic<> NumInstEliminated("GCSE - Number of instructions killed");
//
// Later, in the code: ++NumInstEliminated;
//
//===----------------------------------------------------------------------===//

#include "Support/StatisticReporter.h"
#include "Support/CommandLine.h"
#include <iostream>

bool DebugFlag;  // DebugFlag - Exported boolean set by the -debug option

static cl::Flag Enabled("stats", "Enable statistics output from program");
static cl::Flag Debug(DebugFlag, "debug", "Enable debug output", cl::Hidden);

// Print information when destroyed, iff command line option is specified
void StatisticBase::destroy() const {
  if (Enabled && hasSomeData()) {
    std::cerr.width(7);
    printValue(std::cerr);
    std::cerr.width(0);
    std::cerr << "\t" << Name << "\n";
  }
}
