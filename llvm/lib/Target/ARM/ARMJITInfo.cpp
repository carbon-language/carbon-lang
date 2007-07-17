//===-- ARMJITInfo.cpp - Implement the JIT interfaces for the ARM target --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Raul Herbster and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the ARM target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "ARMJITInfo.h"
#include "ARMRelocations.h"
#include "ARMSubtarget.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include <cstdlib>
using namespace llvm;

void ARMJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  unsigned char *OldByte = (unsigned char *)Old;
  *OldByte++ = 0xEA;                // Emit B opcode.
  unsigned *OldWord = (unsigned *)OldByte;
  unsigned NewAddr = (intptr_t)New;
  unsigned OldAddr = (intptr_t)OldWord;
  *OldWord = NewAddr - OldAddr - 4; // Emit PC-relative addr of New code.
}

/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

// CompilationCallback stub - We can't use a C function with inline assembly in
// it, because we the prolog/epilog inserted by GCC won't work for us.  Instead,
// write our own wrapper, which does things our way, so we have complete control
// over register saving and restoring.
extern "C" {
#if defined(__arm__)
  void ARMCompilationCallback(void);
  asm(
    ".text\n"
    ".align 2\n"
    ".globl ARMCompilationCallback\n"
    "ARMCompilationCallback:\n"
    // save main registers
    "mov    ip, sp\n"
    "stmfd  sp!, {fp, ip, lr, pc}\n"
    "sub    fp, ip, #4\n"
    // arguments to Compilation Callback
    // r0 - our lr (address of the call instruction in stub plus 4)
    // r1 - stub's lr (address of instruction that called the stub plus 4)
    "mov    r0, fp\n"  // stub's frame
    "mov    r1, lr\n"  // stub's lr
    "bl     ARMCompilationCallbackC\n"
    // restore main registers
    "ldmfd  sp, {fp, sp, pc}\n");
#else // Not an ARM host
  void ARMCompilationCallback() {
    assert(0 && "Cannot call ARMCompilationCallback() on a non-ARM arch!\n");
    abort();
  }
#endif
}

/// ARMCompilationCallbackC - This is the target-specific function invoked by the
/// function stub when we did not know the real target of a call.  This function
/// must locate the start of the stub or call site and pass it into the JIT
/// compiler function.
extern "C" void ARMCompilationCallbackC(intptr_t *StackPtr, intptr_t RetAddr) {
  intptr_t *RetAddrLoc = &StackPtr[-1];

  assert(*RetAddrLoc == RetAddr &&
         "Could not find return address on the stack!");
#if 0
  DOUT << "In callback! Addr=" << (void*)RetAddr
       << " FP=" << (void*)StackPtr
       << ": Resolving call to function: "
       << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n";
#endif

  // Sanity check to make sure this really is a branch and link instruction.
  assert(((unsigned char*)RetAddr-1)[3] == 0xEB && "Not a branch and link instr!");

  intptr_t NewVal = (intptr_t)JITCompilerFunction((void*)RetAddr);

  // Rewrite the call target... so that we don't end up here every time we
  // execute the call.
  *(intptr_t *)RetAddr = (intptr_t)(NewVal-RetAddr-4);

  // Change the return address to reexecute the branch and link instruction...
  *RetAddrLoc -= 1;
}

TargetJITInfo::LazyResolverFn
ARMJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return ARMCompilationCallback;
}

void *ARMJITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
  unsigned addr = (intptr_t)Fn-MCE.getCurrentPCValue()-4;
  // If this is just a call to an external function, emit a branch instead of a
  // call.  The code is the same except for one bit of the last instruction.
  if (Fn != (void*)(intptr_t)ARMCompilationCallback) {
    MCE.startFunctionStub(4, 2);
    MCE.emitByte(0xEA);  // branch to the corresponding function addr 
    MCE.emitByte((unsigned char)(addr >>  0));
    MCE.emitByte((unsigned char)(addr >>  8));
    MCE.emitByte((unsigned char)(addr >>  16));
    return MCE.finishFunctionStub(0);
  } else {
    MCE.startFunctionStub(5, 2);
    MCE.emitByte(0xEB);  // branch and link to the corresponding function addr
  }
  MCE.emitByte((unsigned char)(addr >>  0));
  MCE.emitByte((unsigned char)(addr >>  8));
  MCE.emitByte((unsigned char)(addr >>  16));

  return MCE.finishFunctionStub(0);
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void ARMJITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {

}
