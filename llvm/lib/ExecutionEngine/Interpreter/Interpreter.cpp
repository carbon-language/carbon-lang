//===- Interpreter.cpp - Top-Level LLVM Interpreter Implementation --------===//
//
// This file implements the top-level functionality for the LLVM interpreter.
// This interpreter is designed to be a very simple, portable, inefficient
// interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"

/// create - Create a new interpreter object.  This can never fail.
///
ExecutionEngine *Interpreter::create(Module *M, bool TraceMode){
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

  return new Interpreter(M, isLittleEndian, isLongPointer, TraceMode);
}

//===----------------------------------------------------------------------===//
// Interpreter ctor - Initialize stuff
//
Interpreter::Interpreter(Module *M, bool isLittleEndian, bool isLongPointer,
                         bool TraceMode)
  : ExecutionEngine(M), ExitCode(0), Trace(TraceMode),
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
  if (!callMainFunction(MainFunction, Args)) {
    // If the call succeeded, run the code now...
    run();
  }

  do {
    // If the program has exited, run atexit handlers...
    if (ECStack.empty() && !AtExitHandlers.empty()) {
      callFunction(AtExitHandlers.back(), std::vector<GenericValue>());
      AtExitHandlers.pop_back();
      run();
    }
  } while (!ECStack.empty());

  return ExitCode;
}


// callMainFunction - Construct call to typical C main() function and
// call it using callFunction().
//
bool Interpreter::callMainFunction(const std::string &Name,
                                   const std::vector<std::string> &InputArgv) {
  Function *M = getModule().getNamedFunction(Name);
  if (M == 0) {
    std::cerr << "Could not find function '" << Name << "' in module!\n";
    return 1;
  }
  const FunctionType *MT = M->getFunctionType();

  std::vector<GenericValue> Args;
  if (MT->getParamTypes().size() >= 2) {
    PointerType *SPP = PointerType::get(PointerType::get(Type::SByteTy));
    if (MT->getParamTypes()[1] != SPP) {
      CW << "Second argument of '" << Name << "' should have type: '"
	 << SPP << "'!\n";
      return true;
    }
    Args.push_back(PTOGV(CreateArgv(InputArgv)));
  }

  if (MT->getParamTypes().size() >= 1) {
    if (!MT->getParamTypes()[0]->isInteger()) {
      std::cout << "First argument of '" << Name
		<< "' should be an integer!\n";
      return true;
    } else {
      GenericValue GV; GV.UIntVal = InputArgv.size();
      Args.insert(Args.begin(), GV);
    }
  }

  callFunction(M, Args);  // Start executing it...

  // Reset the current frame location to the top of stack
  CurFrame = ECStack.size()-1;

  return false;
}
