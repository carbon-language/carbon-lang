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
#include "llvm/System/ThreadLocal.h"
#include "llvm/ADT/SmallString.h"
using namespace llvm;

namespace llvm {
  bool DisablePrettyStackTrace = false;
}

// FIXME: This should be thread local when llvm supports threads.
static sys::ThreadLocal<const PrettyStackTraceEntry> PrettyStackTraceHead;

static unsigned PrintStack(const PrettyStackTraceEntry *Entry, raw_ostream &OS){
  unsigned NextID = 0;
  if (Entry->getNextEntry())
    NextID = PrintStack(Entry->getNextEntry(), OS);
  OS << NextID << ".\t";
  Entry->print(OS);
  
  return NextID+1;
}

/// PrintCurStackTrace - Print the current stack trace to the specified stream.
static void PrintCurStackTrace(raw_ostream &OS) {
  // Don't print an empty trace.
  if (PrettyStackTraceHead.get() == 0) return;
  
  // If there are pretty stack frames registered, walk and emit them.
  OS << "Stack dump:\n";
  
  PrintStack(PrettyStackTraceHead.get(), OS);
  OS.flush();
}

// Integrate with crash reporter.
#ifdef __APPLE__
static const char *__crashreporter_info__ = 0;
asm(".desc ___crashreporter_info__, 0x10");
#endif


/// CrashHandler - This callback is run if a fatal signal is delivered to the
/// process, it prints the pretty stack trace.
static void CrashHandler(void *Cookie) {
#ifndef __APPLE__
  // On non-apple systems, just emit the crash stack trace to stderr.
  PrintCurStackTrace(errs());
#else
  // Otherwise, emit to a smallvector of chars, send *that* to stderr, but also
  // put it into __crashreporter_info__.
  SmallString<2048> TmpStr;
  {
    raw_svector_ostream Stream(TmpStr);
    PrintCurStackTrace(Stream);
  }
  
  if (!TmpStr.empty()) {
    __crashreporter_info__ = strdup(std::string(TmpStr.str()).c_str());
    errs() << TmpStr.str();
  }
  
#endif
}

static bool RegisterCrashPrinter() {
  if (!DisablePrettyStackTrace)
    sys::AddSignalHandler(CrashHandler, 0);
  return false;
}

PrettyStackTraceEntry::PrettyStackTraceEntry() {
  // The first time this is called, we register the crash printer.
  static bool HandlerRegistered = RegisterCrashPrinter();
  HandlerRegistered = HandlerRegistered;
    
  // Link ourselves.
  NextEntry = PrettyStackTraceHead.get();
  PrettyStackTraceHead.set(this);
}

PrettyStackTraceEntry::~PrettyStackTraceEntry() {
  assert(PrettyStackTraceHead.get() == this &&
         "Pretty stack trace entry destruction is out of order");
  PrettyStackTraceHead.set(getNextEntry());
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

