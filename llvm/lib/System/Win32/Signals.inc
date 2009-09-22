//===- Win32/Signals.cpp - Win32 Signals Implementation ---------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file provides the Win32 specific implementation of the Signals class.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <stdio.h>
#include <vector>
#include <algorithm>

#ifdef __MINGW32__
 #include <imagehlp.h>
#else
 #include <dbghelp.h>
#endif
#include <psapi.h>

#ifdef __MINGW32__
 #if ((HAVE_LIBIMAGEHLP != 1) || (HAVE_LIBPSAPI != 1))
  #error "libimagehlp.a & libpsapi.a should be present"
 #endif
#else
 #pragma comment(lib, "psapi.lib")
 #pragma comment(lib, "dbghelp.lib")
#endif

// Forward declare.
static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep);
static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType);

// InterruptFunction - The function to call if ctrl-c is pressed.
static void (*InterruptFunction)() = 0;

static std::vector<llvm::sys::Path> *FilesToRemove = NULL;
static std::vector<std::pair<void(*)(void*), void*> > *CallBacksToRun = 0;
static bool RegisteredUnhandledExceptionFilter = false;
static bool CleanupExecuted = false;
static bool ExitOnUnhandledExceptions = false;
static PTOP_LEVEL_EXCEPTION_FILTER OldFilter = NULL;

// Windows creates a new thread to execute the console handler when an event
// (such as CTRL/C) occurs.  This causes concurrency issues with the above
// globals which this critical section addresses.
static CRITICAL_SECTION CriticalSection;

namespace llvm {

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only Win32 specific code 
//===          and must not be UNIX code
//===----------------------------------------------------------------------===//

/// CRTReportHook - Function called on a CRT debugging event.
static int CRTReportHook(int ReportType, char *Message, int *Return) {
  // Don't cause a DebugBreak() on return.
  if (Return)
    *Return = 0;

  switch (ReportType) {
  default:
  case _CRT_ASSERT:
    fprintf(stderr, "CRT assert: %s\n", Message);
    // FIXME: Is there a way to just crash? Perhaps throw to the unhandled
    // exception code? Perhaps SetErrorMode() handles this.
    _exit(3);
    break;
  case _CRT_ERROR:
    fprintf(stderr, "CRT error: %s\n", Message);
    // FIXME: Is there a way to just crash? Perhaps throw to the unhandled
    // exception code? Perhaps SetErrorMode() handles this.
    _exit(3);
    break;
  case _CRT_WARN:
    fprintf(stderr, "CRT warn: %s\n", Message);
    break;
  }

  // Don't call _CrtDbgReport.
  return TRUE;
}

static void RegisterHandler() {
  if (RegisteredUnhandledExceptionFilter) {
    EnterCriticalSection(&CriticalSection);
    return;
  }

  // Now's the time to create the critical section.  This is the first time
  // through here, and there's only one thread.
  InitializeCriticalSection(&CriticalSection);

  // Enter it immediately.  Now if someone hits CTRL/C, the console handler
  // can't proceed until the globals are updated.
  EnterCriticalSection(&CriticalSection);

  RegisteredUnhandledExceptionFilter = true;
  OldFilter = SetUnhandledExceptionFilter(LLVMUnhandledExceptionFilter);
  SetConsoleCtrlHandler(LLVMConsoleCtrlHandler, TRUE);

  // Environment variable to disable any kind of crash dialog.
  if (getenv("LLVM_DISABLE_CRT_DEBUG")) {
    _CrtSetReportHook(CRTReportHook);
	ExitOnUnhandledExceptions = true;
  }

  // IMPORTANT NOTE: Caller must call LeaveCriticalSection(&CriticalSection) or
  // else multi-threading problems will ensue.
}

// RemoveFileOnSignal - The public API
bool sys::RemoveFileOnSignal(const sys::Path &Filename, std::string* ErrMsg) {
  RegisterHandler();

  if (CleanupExecuted) {
    if (ErrMsg)
      *ErrMsg = "Process terminating -- cannot register for removal";
    return true;
  }

  if (FilesToRemove == NULL)
    FilesToRemove = new std::vector<sys::Path>;

  FilesToRemove->push_back(Filename);

  LeaveCriticalSection(&CriticalSection);
  return false;
}

/// PrintStackTraceOnErrorSignal - When an error signal (such as SIBABRT or
/// SIGSEGV) is delivered to the process, print a stack trace and then exit.
void sys::PrintStackTraceOnErrorSignal() {
  RegisterHandler();
  LeaveCriticalSection(&CriticalSection);
}


void sys::SetInterruptFunction(void (*IF)()) {
  RegisterHandler();
  InterruptFunction = IF;
  LeaveCriticalSection(&CriticalSection);
}


/// AddSignalHandler - Add a function to be called when a signal is delivered
/// to the process.  The handler can have a cookie passed to it to identify
/// what instance of the handler it is.
void sys::AddSignalHandler(void (*FnPtr)(void *), void *Cookie) {
  if (CallBacksToRun == 0)
    CallBacksToRun = new std::vector<std::pair<void(*)(void*), void*> >();
  CallBacksToRun->push_back(std::make_pair(FnPtr, Cookie));
  RegisterHandler();
}
}

static void Cleanup() {
  EnterCriticalSection(&CriticalSection);

  // Prevent other thread from registering new files and directories for
  // removal, should we be executing because of the console handler callback.
  CleanupExecuted = true;

  // FIXME: open files cannot be deleted.

  if (FilesToRemove != NULL)
    while (!FilesToRemove->empty()) {
      FilesToRemove->back().eraseFromDisk();
      FilesToRemove->pop_back();
    }

  if (CallBacksToRun)
    for (unsigned i = 0, e = CallBacksToRun->size(); i != e; ++i)
      (*CallBacksToRun)[i].first((*CallBacksToRun)[i].second);

  LeaveCriticalSection(&CriticalSection);
}

static LONG WINAPI LLVMUnhandledExceptionFilter(LPEXCEPTION_POINTERS ep) {
  try {
    Cleanup();
    
#ifdef _WIN64
  // TODO: provide a x64 friendly version of the following
#else
    
    // Initialize the STACKFRAME structure.
    STACKFRAME StackFrame;
    memset(&StackFrame, 0, sizeof(StackFrame));

    StackFrame.AddrPC.Offset = ep->ContextRecord->Eip;
    StackFrame.AddrPC.Mode = AddrModeFlat;
    StackFrame.AddrStack.Offset = ep->ContextRecord->Esp;
    StackFrame.AddrStack.Mode = AddrModeFlat;
    StackFrame.AddrFrame.Offset = ep->ContextRecord->Ebp;
    StackFrame.AddrFrame.Mode = AddrModeFlat;

    HANDLE hProcess = GetCurrentProcess();
    HANDLE hThread = GetCurrentThread();

    // Initialize the symbol handler.
    SymSetOptions(SYMOPT_DEFERRED_LOADS|SYMOPT_LOAD_LINES);
    SymInitialize(hProcess, NULL, TRUE);

    while (true) {
      if (!StackWalk(IMAGE_FILE_MACHINE_I386, hProcess, hThread, &StackFrame,
                     ep->ContextRecord, NULL, SymFunctionTableAccess,
                     SymGetModuleBase, NULL)) {
        break;
      }

      if (StackFrame.AddrFrame.Offset == 0)
        break;

      // Print the PC in hexadecimal.
      DWORD PC = StackFrame.AddrPC.Offset;
      fprintf(stderr, "%08lX", PC);

      // Print the parameters.  Assume there are four.
      fprintf(stderr, " (0x%08lX 0x%08lX 0x%08lX 0x%08lX)", StackFrame.Params[0],
              StackFrame.Params[1], StackFrame.Params[2], StackFrame.Params[3]);

      // Verify the PC belongs to a module in this process.
      if (!SymGetModuleBase(hProcess, PC)) {
        fputs(" <unknown module>\n", stderr);
        continue;
      }

      // Print the symbol name.
      char buffer[512];
      IMAGEHLP_SYMBOL *symbol = reinterpret_cast<IMAGEHLP_SYMBOL *>(buffer);
      memset(symbol, 0, sizeof(IMAGEHLP_SYMBOL));
      symbol->SizeOfStruct = sizeof(IMAGEHLP_SYMBOL);
      symbol->MaxNameLength = 512 - sizeof(IMAGEHLP_SYMBOL);

      DWORD dwDisp;
      if (!SymGetSymFromAddr(hProcess, PC, &dwDisp, symbol)) {
        fputc('\n', stderr);
        continue;
      }

      buffer[511] = 0;
      if (dwDisp > 0)
        fprintf(stderr, ", %s()+%04lu bytes(s)", symbol->Name, dwDisp);
      else
        fprintf(stderr, ", %s", symbol->Name);

      // Print the source file and line number information.
      IMAGEHLP_LINE line;
      memset(&line, 0, sizeof(line));
      line.SizeOfStruct = sizeof(line);
      if (SymGetLineFromAddr(hProcess, PC, &dwDisp, &line)) {
        fprintf(stderr, ", %s, line %lu", line.FileName, line.LineNumber);
        if (dwDisp > 0)
          fprintf(stderr, "+%04lu byte(s)", dwDisp);
      }

      fputc('\n', stderr);
    }

#endif

  } catch (...) {
      assert(0 && "Crashed in LLVMUnhandledExceptionFilter");
  }

  if (ExitOnUnhandledExceptions)
  	_exit(-3);

  // Allow dialog box to pop up allowing choice to start debugger.
  if (OldFilter)
    return (*OldFilter)(ep);
  else
    return EXCEPTION_CONTINUE_SEARCH;
}

static BOOL WINAPI LLVMConsoleCtrlHandler(DWORD dwCtrlType) {
  // We are running in our very own thread, courtesy of Windows.
  EnterCriticalSection(&CriticalSection);
  Cleanup();

  // If an interrupt function has been set, go and run one it; otherwise,
  // the process dies.
  void (*IF)() = InterruptFunction;
  InterruptFunction = 0;      // Don't run it on another CTRL-C.

  if (IF) {
    // Note: if the interrupt function throws an exception, there is nothing
    // to catch it in this thread so it will kill the process.
    IF();                     // Run it now.
    LeaveCriticalSection(&CriticalSection);
    return TRUE;              // Don't kill the process.
  }

  // Allow normal processing to take place; i.e., the process dies.
  LeaveCriticalSection(&CriticalSection);
  return FALSE;
}

