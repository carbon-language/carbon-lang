//===- Interpreter.cpp - Top-Level LLVM Interpreter Implementation --------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the top-level functionality for the LLVM interpreter.
// This interpreter is designed to be a very simple, portable, inefficient
// interpreter.
//
//===----------------------------------------------------------------------===//

#include "Interpreter.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
using namespace llvm;

/// create - Create a new interpreter object.  This can never fail.
///
ExecutionEngine *Interpreter::create(Module *M, IntrinsicLowering *IL) {
  bool isLittleEndian = false;
  switch (M->getEndianness()) {
  case Module::LittleEndian: isLittleEndian = true; break;
  case Module::BigEndian:    isLittleEndian = false; break;
  case Module::AnyPointerSize:
    int Test = 0;
    *(char*)&Test = 1;    // Return true if the host is little endian
    isLittleEndian = (Test == 1);
    break;
  }

  bool isLongPointer = false;
  switch (M->getPointerSize()) {
  case Module::Pointer32: isLongPointer = false; break;
  case Module::Pointer64: isLongPointer = true; break;
  case Module::AnyPointerSize:
    isLongPointer = (sizeof(void*) == 8);  // Follow host
    break;
  }

  return new Interpreter(M, isLittleEndian, isLongPointer, IL);
}

//===----------------------------------------------------------------------===//
// Interpreter ctor - Initialize stuff
//
Interpreter::Interpreter(Module *M, bool isLittleEndian, bool isLongPointer,
                         IntrinsicLowering *il)
  : ExecutionEngine(M), ExitCode(0), 
    TD("lli", isLittleEndian, isLongPointer ? 8 : 4, isLongPointer ? 8 : 4,
       isLongPointer ? 8 : 4), IL(il) {

  setTargetData(TD);
  // Initialize the "backend"
  initializeExecutionEngine();
  initializeExternalFunctions();
  emitGlobals();

  if (IL == 0) IL = new DefaultIntrinsicLowering();
}

Interpreter::~Interpreter() {
  delete IL;
}

void Interpreter::runAtExitHandlers () {
  while (!AtExitHandlers.empty()) {
    callFunction(AtExitHandlers.back(), std::vector<GenericValue>());
    AtExitHandlers.pop_back();
    run();
  }
}

/// run - Start execution with the specified function and arguments.
///
GenericValue Interpreter::runFunction(Function *F,
			      const std::vector<GenericValue> &ArgValues) {
  assert (F && "Function *F was null at entry to run()");

  // Try extra hard not to pass extra args to a function that isn't
  // expecting them.  C programmers frequently bend the rules and
  // declare main() with fewer parameters than it actually gets
  // passed, and the interpreter barfs if you pass a function more
  // parameters than it is declared to take. This does not attempt to
  // take into account gratuitous differences in declared types,
  // though.
  std::vector<GenericValue> ActualArgs;
  const unsigned ArgCount = F->getFunctionType()->getNumParams();
  for (unsigned i = 0; i < ArgCount; ++i)
    ActualArgs.push_back(ArgValues[i]);
  
  // Set up the function call.
  callFunction(F, ActualArgs);

  // Start executing the function.
  run();
  
  GenericValue rv;
  rv.IntVal = ExitCode;
  return rv;
}

