//===-- tsan_report.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_REPORT_H
#define TSAN_REPORT_H

#include "tsan_defs.h"
#include "tsan_vector.h"

namespace __tsan {

enum ReportType {
  ReportTypeRace,
  ReportTypeUseAfterFree,
  ReportTypeThreadLeak,
  ReportTypeMutexDestroyLocked,
  ReportTypeSignalUnsafe,
  ReportTypeErrnoInSignal
};

struct ReportStack {
  ReportStack *next;
  char *module;
  uptr offset;
  uptr pc;
  char *func;
  char *file;
  int line;
  int col;
};

struct ReportMopMutex {
  u64 id;
  bool write;
};

struct ReportMop {
  int tid;
  uptr addr;
  int size;
  bool write;
  Vector<ReportMopMutex> mset;
  ReportStack *stack;

  ReportMop();
};

enum ReportLocationType {
  ReportLocationGlobal,
  ReportLocationHeap,
  ReportLocationStack
};

struct ReportLocation {
  ReportLocationType type;
  uptr addr;
  uptr size;
  char *module;
  uptr offset;
  int tid;
  char *name;
  char *file;
  int line;
  ReportStack *stack;
};

struct ReportThread {
  int id;
  uptr pid;
  bool running;
  char *name;
  ReportStack *stack;
};

struct ReportMutex {
  u64 id;
  bool destroyed;
  ReportStack *stack;
};

class ReportDesc {
 public:
  ReportType typ;
  Vector<ReportStack*> stacks;
  Vector<ReportMop*> mops;
  Vector<ReportLocation*> locs;
  Vector<ReportMutex*> mutexes;
  Vector<ReportThread*> threads;
  ReportStack *sleep;

  ReportDesc();
  ~ReportDesc();

 private:
  ReportDesc(const ReportDesc&);
  void operator = (const ReportDesc&);
};

// Format and output the report to the console/log. No additional logic.
void PrintReport(const ReportDesc *rep);
void PrintStack(const ReportStack *stack);

}  // namespace __tsan

#endif  // TSAN_REPORT_H
