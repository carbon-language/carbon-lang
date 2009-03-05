//===- PrettyStackTrace.cpp - Pretty Crash Handling -----------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for dealing with the possibility of
// Unix signals occuring while your program is running.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/System/Signals.h"
using namespace llvm;

// FIXME: This should be thread local when llvm supports threads.
static const PrettyStackTraceEntry *PrettyStackTraceHead = 0;

/// CrashHandler - This callback is run if a fatal signal is delivered to the
/// process, it prints the pretty stack trace.
static void CrashHandler(void *Cookie) {
  // If there are pretty stack frames registered, walk and emit them.
  raw_ostream &OS = errs();
  OS << "Stack dump:\n";
  
  unsigned i = 0;
  for (const PrettyStackTraceEntry *Entry = PrettyStackTraceHead; Entry;
       Entry = Entry->getNextEntry(), ++i) {
    OS << i << ".\t";
    Entry->print(OS);
  }
  OS.flush();
}

static bool RegisterCrashPrinter() {
  sys::AddSignalHandler(CrashHandler, 0);
  return false;
}

PrettyStackTraceEntry::PrettyStackTraceEntry() {
  // The first time this is called, we register the crash printer.
  static bool HandlerRegistered = RegisterCrashPrinter();
  HandlerRegistered = HandlerRegistered;
    
  // Link ourselves.
  NextEntry = PrettyStackTraceHead;
  PrettyStackTraceHead = this;
}

PrettyStackTraceEntry::~PrettyStackTraceEntry() {
  assert(PrettyStackTraceHead == this &&
         "Pretty stack trace entry destruction is out of order");
  PrettyStackTraceHead = getNextEntry();
}

void PrettyStackTraceString::print(raw_ostream &OS) const {
  OS << Str << "\n";
}

void PrettyStackTraceProgram::print(raw_ostream &OS) const {
  OS << "Program arguments: ";
  // Print the argument list.
  for (unsigned i = 0, e = ArgC; i != e; ++i)
    OS << ArgV[i] << ' ';
  OS << '\n';
}

