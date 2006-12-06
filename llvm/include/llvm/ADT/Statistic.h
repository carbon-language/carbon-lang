//===-- llvm/ADT/Statistic.h - Easy way to expose stats ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the 'Statistic' class, which is designed to be an easy way
// to expose various metrics from passes.  These statistics are printed at the
// end of a run (from llvm_shutdown), when the -stats command line option is
// passed on the command line.
//
// This is useful for reporting information like the number of instructions
// simplified, optimized or removed by various transformations, like this:
//
// static Statistic NumInstsKilled("gcse", "Number of instructions killed");
//
// Later, in the code: ++NumInstsKilled;
//
// NOTE: Statistics *must* be declared as global variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STATISTIC_H
#define LLVM_ADT_STATISTIC_H

namespace llvm {

class Statistic {
  const char *Name;
  const char *Desc;
  unsigned Value;
  static unsigned NumStats;
public:
  // Normal constructor, default initialize data item...
  Statistic(const char *name, const char *desc)
    : Name(name), Desc(desc), Value(0) {
    ++NumStats;  // Keep track of how many stats are created...
  }

  // Print information when destroyed, iff command line option is specified
  ~Statistic();

  // Allow use of this class as the value itself...
  operator unsigned() const { return Value; }
  const Statistic &operator=(unsigned Val) { Value = Val; return *this; }
  const Statistic &operator++() { ++Value; return *this; }
  unsigned operator++(int) { return Value++; }
  const Statistic &operator--() { --Value; return *this; }
  unsigned operator--(int) { return Value--; }
  const Statistic &operator+=(const unsigned &V) { Value += V; return *this; }
  const Statistic &operator-=(const unsigned &V) { Value -= V; return *this; }
  const Statistic &operator*=(const unsigned &V) { Value *= V; return *this; }
  const Statistic &operator/=(const unsigned &V) { Value /= V; return *this; }
};

} // End llvm namespace

#endif
