//===-- llvm/Analysis/Writer.h - Printer for Analysis routines ---*- C++ -*--=//
//
// This library provides routines to print out various analysis results to 
// an output stream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_WRITER_H
#define LLVM_ANALYSIS_WRITER_H

#include "llvm/Assembly/Writer.h"

namespace cfg {

  // This library provides support for printing out Intervals.
  class Interval;
  class IntervalPartition;

  void WriteToOutput(const Interval *I, ostream &o);
  inline ostream &operator <<(ostream &o, const Interval *I) {
    WriteToOutput(I, o); return o;
  }

  void WriteToOutput(const IntervalPartition &IP, ostream &o);
  inline ostream &operator <<(ostream &o, const IntervalPartition &IP) {
    WriteToOutput(IP, o); return o;
  }

  // Stuff for printing out Dominator data structures...
  class DominatorSet;
  class ImmediateDominators;
  class DominatorTree;
  class DominanceFrontier;

  void WriteToOutput(const DominatorSet &, ostream &o);
  inline ostream &operator <<(ostream &o, const DominatorSet &DS) {
    WriteToOutput(DS, o); return o;
  }

  void WriteToOutput(const ImmediateDominators &, ostream &o);
  inline ostream &operator <<(ostream &o, const ImmediateDominators &ID) {
    WriteToOutput(ID, o); return o;
  }

  void WriteToOutput(const DominatorTree &, ostream &o);
  inline ostream &operator <<(ostream &o, const DominatorTree &DT) {
    WriteToOutput(DT, o); return o;
  }

  void WriteToOutput(const DominanceFrontier &, ostream &o);
  inline ostream &operator <<(ostream &o, const DominanceFrontier &DF) {
    WriteToOutput(DF, o); return o;
  }
}  // End namespace CFG

#endif
