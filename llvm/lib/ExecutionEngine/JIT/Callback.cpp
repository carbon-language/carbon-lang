//===-- Callback.cpp - Trap handler for function resolution ---------------===//
//
// This file defines the handler which is invoked when a reference to a
// non-codegen'd function is found.  This file defines target specific code
// which is used by the JIT.
//
//===----------------------------------------------------------------------===//

#include "VM.h"
#include "Support/Statistic.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
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
  bool isStub = ((unsigned char*)RetAddr)[0] == 0xCD;  // Interrupt marker?

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

  if (isStub) {
    // If this is a stub, rewrite the call into an unconditional branch
    // instruction so that two return addresses are not pushed onto the stack
    // when the requested function finally gets called.  This also makes the
    // 0xCD byte (interrupt) dead, so the marker doesn't effect anything.
    ((unsigned char*)RetAddr)[-1] = 0xE9;
  }

  // Change the return address to reexecute the call instruction...
  StackPtr[1] -= 5;
#else
  abort();
#endif
}

/// emitStubForFunction - This virtual method is used by the JIT when it needs
/// to emit the address of a function for a function whose code has not yet
/// been generated.  In order to do this, it generates a stub which jumps to
/// the lazy function compiler, which will eventually get fixed to call the
/// function directly.
///
void *VM::emitStubForFunction(const Function &F) {
#if defined(i386) || defined(__i386__) || defined(__x86__)
  MCE->startFunctionStub(F, 6);
  MCE->emitByte(0xE8);   // Call with 32 bit pc-rel destination...
  MCE->emitGlobalAddress((GlobalValue*)&F, true);
  MCE->emitByte(0xCD);   // Interrupt - Just a marker identifying the stub!
  return MCE->finishFunctionStub(F);
#else
  abort();
#endif
}

void VM::registerCallback() {
  TheVM = this;
}


