//===-- ARMJITInfo.cpp - Implement the JIT interfaces for the ARM target --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include <cstdlib>
using namespace llvm;

void ARMJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  abort();
}

/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

// Get the ASMPREFIX for the current host.  This is often '_'.
#ifndef __USER_LABEL_PREFIX__
#define __USER_LABEL_PREFIX__
#endif
#define GETASMPREFIX2(X) #X
#define GETASMPREFIX(X) GETASMPREFIX2(X)
#define ASMPREFIX GETASMPREFIX(__USER_LABEL_PREFIX__)

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
    ".globl " ASMPREFIX "ARMCompilationCallback\n"
    ASMPREFIX "ARMCompilationCallback:\n"
    // save main registers
#if defined(__APPLE__)
    "stmfd  sp!, {r4, r5, r6, r7, lr}\n"
    "mov    r0, r7\n"  // stub's frame
    "stmfd  sp!, {r8, r10, r11}\n"
#else
    "mov    ip, sp\n"
    "stmfd  sp!, {fp, ip, lr, pc}\n"
    "sub    fp, ip, #4\n"
#endif // __APPLE__
    // arguments to Compilation Callback
    // r0 - our lr (address of the call instruction in stub plus 4)
    // r1 - stub's lr (address of instruction that called the stub plus 4)
#if defined(__APPLE__)
    "mov    r0, r7\n"  // stub's frame
#else
    "mov    r0, fp\n"  // stub's frame
#endif // __APPLE__
    "mov    r1, lr\n"  // stub's lr
    "bl     " ASMPREFIX "ARMCompilationCallbackC\n"
    // restore main registers
#if defined(__APPLE__)
    "ldmfd  sp!, {r8, r10, r11}\n"
    "ldmfd  sp!, {r4, r5, r6, r7, pc}\n"
#else
    "ldmfd  sp, {fp, sp, pc}\n"
#endif // __APPLE__
      );
#else  // Not an ARM host
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
  intptr_t Addr = RetAddr - 4;

  intptr_t NewVal = (intptr_t)JITCompilerFunction((void*)Addr);

  // Rewrite the call target... so that we don't end up here every time we
  // execute the call.
  *(intptr_t *)Addr = NewVal;

  // Change the return address to reexecute the branch and link instruction...
  *RetAddrLoc -= 12;
}

TargetJITInfo::LazyResolverFn
ARMJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return ARMCompilationCallback;
}

void *ARMJITInfo::emitFunctionStub(const Function* F, void *Fn,
                                   MachineCodeEmitter &MCE) {
  unsigned addr = (intptr_t)Fn;
  // If this is just a call to an external function, emit a branch instead of a
  // call.  The code is the same except for one bit of the last instruction.
  if (Fn != (void*)(intptr_t)ARMCompilationCallback) {
    // branch to the corresponding function addr
    // the stub is 8-byte size and 4-aligned
    MCE.startFunctionStub(F, 8, 4);
    MCE.emitWordLE(0xE51FF004); // LDR PC, [PC,#-4]
    MCE.emitWordLE(addr);       // addr of function
  } else {
    // branch and link to the corresponding function addr
    // the stub is 20-byte size and 4-aligned
    MCE.startFunctionStub(F, 20, 4);
    MCE.emitWordLE(0xE92D4800); // STMFD SP!, [R11, LR]
    MCE.emitWordLE(0xE28FE004); // ADD LR, PC, #4
    MCE.emitWordLE(0xE51FF004); // LDR PC, [PC,#-4]
    MCE.emitWordLE(addr);       // addr of function
    MCE.emitWordLE(0xE8BD8800); // LDMFD SP!, [R11, PC]
  }

  return MCE.finishFunctionStub(F);
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void ARMJITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*)Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t)MR->getResultPointer();
    switch ((ARM::RelocationType)MR->getRelocationType()) {
    case ARM::reloc_arm_relative: {
      // It is necessary to calculate the correct PC relative value. We
      // subtract the base addr from the target addr to form a byte offset.
      ResultPtr = ResultPtr-(intptr_t)RelocPos-8;
      // If the result is positive, set bit U(23) to 1.
      if (ResultPtr >= 0)
        *((unsigned*)RelocPos) |= 1 << 23;
      else {
      // otherwise, obtain the absolute value and set
      // bit U(23) to 0.
        ResultPtr *= -1;
        *((unsigned*)RelocPos) &= 0xFF7FFFFF;
      }
      // set the immed value calculated
      *((unsigned*)RelocPos) |= (unsigned)ResultPtr;
      // set register Rn to PC
      *((unsigned*)RelocPos) |= 0xF << 16;
      break;
    }
    case ARM::reloc_arm_branch: {
      // It is necessary to calculate the correct value of signed_immed_24
      // field. We subtract the base addr from the target addr to form a
      // byte offset, which must be inside the range -33554432 and +33554428.
      // Then, we set the signed_immed_24 field of the instruction to bits
      // [25:2] of the byte offset. More details ARM-ARM p. A4-11.
      ResultPtr = ResultPtr-(intptr_t)RelocPos-8;
      ResultPtr = (ResultPtr & 0x03FFFFFC) >> 2;
      assert(ResultPtr >= -33554432 && ResultPtr <= 33554428);
      *((unsigned*)RelocPos) |= ResultPtr;
      break;
    }
    }
  }
}
