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

class StatisticBase {
public:
  const char *Name;
  const char *Desc;
  unsigned Value : 31;
  bool Initialized : 1;

  unsigned getValue() const { return Value; }
  const char *getName() const { return Name; }
  const char *getDesc() const { return Desc; }
  
  // Allow use of this class as the value itself.
  operator unsigned() const { return Value; }
  const StatisticBase &operator=(unsigned Val) { Value = Val; return init(); }
  const StatisticBase &operator++() { ++Value; return init(); }
  unsigned operator++(int) { init(); return Value++; }
  const StatisticBase &operator--() { --Value; return init(); }
  unsigned operator--(int) { init(); return Value--; }
  const StatisticBase &operator+=(const unsigned &V) {Value += V;return init();}
  const StatisticBase &operator-=(const unsigned &V) {Value -= V;return init();}
  const StatisticBase &operator*=(const unsigned &V) {Value *= V;return init();}
  const StatisticBase &operator/=(const unsigned &V) {Value /= V;return init();}
  
private:
  StatisticBase &init() {
    if (!Initialized) RegisterStatistic();
    return *this;
  }
  void RegisterStatistic();
};
  
struct Statistic : public StatisticBase {
  Statistic(const char *name, const char *desc) {
    Name = name; Desc = desc; Value = 0; Initialized = 0;
  }
};

  
// STATISTIC - A macro to make definition of statistics really simple.  This
// automatically passes the DEBUG_TYPE of the file into the statistic.
#define STATISTIC(VARNAME, DESC) \
  static StatisticBase VARNAME = { DEBUG_TYPE, DESC, 0, 0 }

} // End llvm namespace

#endif
