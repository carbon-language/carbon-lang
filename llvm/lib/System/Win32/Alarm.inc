//===-- Alarm.inc - Implement Win32 Alarm Support ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 Alarm support.
//
//===----------------------------------------------------------------------===//

#include <cassert>
using namespace llvm;

/// NestedSOI - Sanity check.  Alarms cannot be nested or run in parallel.
/// This ensures that they never do.
static bool NestedSOI = false;

void sys::SetupAlarm(unsigned seconds) {
  assert(!NestedSOI && "sys::SetupAlarm calls cannot be nested!");
  NestedSOI = true;
  // FIXME: Implement for Win32
}

void sys::TerminateAlarm() {
  assert(NestedSOI && "sys::TerminateAlarm called without sys::SetupAlarm!");
  // FIXME: Implement for Win32
  NestedSOI = false;
}

int sys::AlarmStatus() {
  // FIXME: Implement for Win32
  return 0;
}

// Don't pull in all of the Windows headers.
extern "C"  void __stdcall Sleep(unsigned long);

void sys::Sleep(unsigned n) {
  ::Sleep(n*1000);
}
