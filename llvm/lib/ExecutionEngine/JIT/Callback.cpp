//===-- Callback.cpp - Trap handler for function resolution ---------------===//
//
// This file defines the SIGSEGV handler which is invoked when a reference to a
// non-codegen'd function is found.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "Support/Statistic.h"
#include <iostream>

static VM *TheVM = 0;

// CompilationCallback - Invoked the first time that a call site is found,
// which causes lazy compilation of the target function.
// 
void VM::CompilationCallback() {
#if defined(i386) || defined(__i386__) || defined(__x86__)
  unsigned *StackPtr = (unsigned*)__builtin_frame_address(0);
  unsigned RetAddr = (unsigned)__builtin_return_address(0);

  assert(StackPtr[1] == RetAddr &&
         "Could not find return address on the stack!");

  // The call instruction should have pushed the return value onto the stack...
  RetAddr -= 4;  // Backtrack to the reference itself...

  DEBUG(std::cerr << "In callback! Addr=0x" << std::hex << RetAddr
                  << " ESP=0x" << (unsigned)StackPtr << std::dec
                  << ": Resolving call to function: "
                  << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n");

  // Sanity check to make sure this really is a call instruction...
  assert(((unsigned char*)RetAddr)[-1] == 0xE8 && "Not a call instr!");
  
  unsigned NewVal = (unsigned)TheVM->resolveFunctionReference((void*)RetAddr);

  // Rewrite the call target... so that we don't fault every time we execute
  // the call.
  *(unsigned*)RetAddr = NewVal-RetAddr-4;    

  // Change the return address to reexecute the call instruction...
  StackPtr[1] -= 5;
#else
  abort();
#endif
}

void VM::registerCallback() {
  TheVM = this;
}


