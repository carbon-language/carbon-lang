//===-- llvm/Analysis/Writer.h - Printer for Analysis routines ---*- C++ -*--=//
//
// This library provides routines to print out various analysis results to 
// an output stream.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_WRITER_H
#define LLVM_ANALYSIS_WRITER_H

#include <iosfwd>

// This library provides support for printing out Intervals.
class Interval;
class IntervalPartition;

void WriteToOutput(const Interval *I, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const Interval *I) {
  WriteToOutput(I, o); return o;
}

void WriteToOutput(const IntervalPartition &IP, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o,
                                 const IntervalPartition &IP) {
  WriteToOutput(IP, o); return o;
}

// Stuff for printing out Dominator data structures...
class DominatorSetBase;
class ImmediateDominatorsBase;
class DominatorTreeBase;
class DominanceFrontierBase;

void WriteToOutput(const DominatorSetBase &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const DominatorSetBase &DS) {
  WriteToOutput(DS, o); return o;
}

void WriteToOutput(const ImmediateDominatorsBase &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o,
                                 const ImmediateDominatorsBase &ID) {
  WriteToOutput(ID, o); return o;
}

void WriteToOutput(const DominatorTreeBase &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const DominatorTreeBase &DT) {
  WriteToOutput(DT, o); return o;
}

void WriteToOutput(const DominanceFrontierBase &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o,
                                 const DominanceFrontierBase &DF) {
  WriteToOutput(DF, o); return o;
}

// Stuff for printing out Loop information
class Loop;
class LoopInfo;

void WriteToOutput(const LoopInfo &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const LoopInfo &LI) {
  WriteToOutput(LI, o); return o;
}

void WriteToOutput(const Loop *, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const Loop *L) {
  WriteToOutput(L, o); return o;
}

class InductionVariable;
void WriteToOutput(const InductionVariable &, std::ostream &o);
inline std::ostream &operator <<(std::ostream &o, const InductionVariable &IV) {
  WriteToOutput(IV, o); return o;
}

#endif
