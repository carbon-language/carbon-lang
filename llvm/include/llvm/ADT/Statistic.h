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

#include <ostream>
#include "llvm/Support/Compiler.h"

namespace llvm {

// StatisticBase - Nontemplated base class for Statistic class...
class StatisticBase {
  const char *Name;
  const char *Desc;
  static unsigned NumStats;
protected:
  StatisticBase(const char *name, const char *desc) : Name(name), Desc(desc) {
    ++NumStats;  // Keep track of how many stats are created...
  }
  // Out of line virtual dtor, to give the vtable etc a home.
  virtual ~StatisticBase();

  // destroy - Called by subclass dtor so that we can still invoke virtual
  // functions on the subclass.
  void destroy() const;

  // printValue - Overridden by template class to print out the value type...
  virtual void printValue(std::ostream &o) const = 0;

  // hasSomeData - Return true if some data has been aquired.  Avoid printing
  // lots of zero counts.
  //
  virtual bool hasSomeData() const = 0;
};

// Statistic Class - templated on the data type we are monitoring...
class Statistic : private StatisticBase {
  unsigned Value;

  virtual void printValue(std::ostream &o) const { o << Value; }
  virtual bool hasSomeData() const { return Value != 0; }
public:
  // Normal constructor, default initialize data item...
  Statistic(const char *name, const char *desc)
    : StatisticBase(name, desc), Value(0) {}

  // Constructor to provide an initial value...
  Statistic(const unsigned &Val, const char *name, const char *desc)
    : StatisticBase(name, desc), Value(Val) {}

  // Print information when destroyed, iff command line option is specified
  ~Statistic() { destroy(); }

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
