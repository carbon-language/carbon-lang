//===- Interpreter.cpp - Top-Level LLVM Interpreter Implementation --------===//
//
// This file implements the top-level functionality for the LLVM interpreter.
// This interpreter is designed to be a very simple, portable, inefficient
// interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Module.h"

/// create - Create a new interpreter object.  This can never fail.
///
ExecutionEngine *Interpreter::create(Module *M, bool DebugMode, bool TraceMode){
  bool isLittleEndian;
  switch (M->getEndianness()) {
  case Module::LittleEndian: isLittleEndian = true; break;
  case Module::BigEndian:    isLittleEndian = false; break;
  case Module::AnyPointerSize:
    int Test = 0;
    *(char*)&Test = 1;    // Return true if the host is little endian
    isLittleEndian = (Test == 1);
    break;
  }

  bool isLongPointer;
  switch (M->getPointerSize()) {
  case Module::Pointer32: isLongPointer = false; break;
  case Module::Pointer64: isLongPointer = true; break;
  case Module::AnyPointerSize:
    isLongPointer = (sizeof(void*) == 8);  // Follow host
    break;
  }

  return new Interpreter(M, isLittleEndian, isLongPointer, DebugMode,TraceMode);
}

//===----------------------------------------------------------------------===//
// Interpreter ctor - Initialize stuff
//
Interpreter::Interpreter(Module *M, bool isLittleEndian, bool isLongPointer,
                         bool DebugMode, bool TraceMode)
  : ExecutionEngine(M), ExitCode(0), Debug(DebugMode), Trace(TraceMode),
    CurFrame(-1), TD("lli", isLittleEndian, isLongPointer ? 8 : 4,
                     isLongPointer ? 8 : 4, isLongPointer ? 8 : 4) {

  setTargetData(TD);
  // Initialize the "backend"
  initializeExecutionEngine();
  initializeExternalFunctions();
  CW.setModule(M);  // Update Writer
  emitGlobals();
}

/// run - Start execution with the specified function and arguments.
///
int Interpreter::run(const std::string &MainFunction,
		     const std::vector<std::string> &Args,
                     const char ** envp) {
  // Start interpreter into the main function...
  //
  if (!callMainFunction(MainFunction, Args) && !Debug) {
    // If not in debug mode and if the call succeeded, run the code now...
    run();
  }

  do {
    // If debug mode, allow the user to interact... also, if the user pressed 
    // ctrl-c or execution hit an error, enter the event loop...
    if (Debug || isStopped())
      handleUserInput();

    // If the program has exited, run atexit handlers...
    if (ECStack.empty() && !AtExitHandlers.empty()) {
      callFunction(AtExitHandlers.back(), std::vector<GenericValue>());
      AtExitHandlers.pop_back();
      run();
    }
  } while (!ECStack.empty());

  PerformExitStuff();
  return ExitCode;
}

