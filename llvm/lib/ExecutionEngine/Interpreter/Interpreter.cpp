//===- Interpreter.cpp - Top-Level LLVM Interpreter Implementation --------===//
//
// This file implements the top-level functionality for the LLVM interpreter.
// This interpreter is designed to be a very simple, portable, inefficient
// interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Target/TargetMachineImpls.h"

/// createInterpreter - Create a new interpreter object.  This can never fail.
///
ExecutionEngine *ExecutionEngine::createInterpreter(Module *M,
						    unsigned Config,
						    bool DebugMode,
						    bool TraceMode) {
  return new Interpreter(M, Config, DebugMode, TraceMode);
}

//===----------------------------------------------------------------------===//
// Interpreter ctor - Initialize stuff
//
Interpreter::Interpreter(Module *M, unsigned Config,
			 bool DebugMode, bool TraceMode)
  : ExecutionEngine(M), ExitCode(0), Debug(DebugMode), Trace(TraceMode),
    CurFrame(-1), TD("lli", (Config & TM::EndianMask) == TM::LittleEndian,
		     1, 4,
		     (Config & TM::PtrSizeMask) == TM::PtrSize64 ? 8 : 4,
		     (Config & TM::PtrSizeMask) == TM::PtrSize64 ? 8 : 4,
		     (Config & TM::PtrSizeMask) == TM::PtrSize64 ? 8 : 4) {

  setTargetData(TD);
  // Initialize the "backend"
  initializeExecutionEngine();
  initializeExternalMethods();
  CW.setModule(M);  // Update Writer
}

/// run - Start execution with the specified function and arguments.
///
int Interpreter::run(const std::string &MainFunction,
		     const std::vector<std::string> &Args) {
  // Start interpreter into the main function...
  //
  if (!callMainMethod(MainFunction, Args) && !Debug) {
    // If not in debug mode and if the call succeeded, run the code now...
    run();
  }

  // If debug mode, allow the user to interact... also, if the user pressed 
  // ctrl-c or execution hit an error, enter the event loop...
  if (Debug || isStopped())
    handleUserInput();
  return ExitCode;
}

