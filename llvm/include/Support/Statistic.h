//===-- Support/Statistic.h - Easy way to expose stats ----------*- C++ -*-===//
//
// This file defines the 'Statistic' class, which is designed to be an easy way
// to expose various success metrics from passes.  These statistics are printed
// at the end of a run, when the -stats command line option is enabled on the
// command line.
//
// This is useful for reporting information like the number of instructions
// simplified, optimized or removed by various transformations, like this:
//
// static Statistic<> NumInstEliminated("GCSE - Number of instructions killed");
//
// Later, in the code: ++NumInstEliminated;
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_STATISTIC_H
#define SUPPORT_STATISTIC_H

#include <iosfwd>

// DebugFlag - This boolean is set to true if the '-debug' command line option
// is specified.  This should probably not be referenced directly, instead, use
// the DEBUG macro below.
//
extern bool DebugFlag;

// DEBUG macro - This macro should be used by passes to emit debug information.
// In the '-debug' option is specified on the commandline, and if this is a
// debug build, then the code specified as the option to the macro will be
// executed.  Otherwise it will not be.  Example:
//
// DEBUG(cerr << "Bitset contains: " << Bitset << "\n");
//
#ifdef NDEBUG
#define DEBUG(X)
#else
#define DEBUG(X) \
  do { if (DebugFlag) { X; } } while (0)
#endif


// StatisticBase - Nontemplated base class for Statistic<> class...
class StatisticBase {
  const char *Name;
  const char *Desc;
  static unsigned NumStats;
protected:
  StatisticBase(const char *name, const char *desc) : Name(name), Desc(desc) {
    ++NumStats;  // Keep track of how many stats are created...
  }
  virtual ~StatisticBase() {}

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
template <typename DataType=unsigned>
class Statistic : private StatisticBase {
  DataType Value;

  virtual void printValue(std::ostream &o) const { o << Value; }
  virtual bool hasSomeData() const { return Value != DataType(); }
public:
  // Normal constructor, default initialize data item...
  Statistic(const char *name, const char *desc)
    : StatisticBase(name, desc), Value(DataType()) {}

  // Constructor to provide an initial value...
  Statistic(const DataType &Val, const char *name, const char *desc)
    : StatisticBase(name, desc), Value(Val) {}

  // Print information when destroyed, iff command line option is specified
  ~Statistic() { destroy(); }

  // Allow use of this class as the value itself...
  operator DataType() const { return Value; }
  const Statistic &operator=(DataType Val) { Value = Val; return *this; }
  const Statistic &operator++() { ++Value; return *this; }
  DataType operator++(int) { return Value++; }
  const Statistic &operator+=(const DataType &V) { Value += V; return *this; }
  const Statistic &operator-=(const DataType &V) { Value -= V; return *this; }
};

#endif
