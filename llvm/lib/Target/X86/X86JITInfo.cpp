//===-- X86JITInfo.cpp - Implement the JIT interfaces for the X86 target --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the X86 target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include <cstdlib>
#include "X86JITInfo.h"
#include "X86Relocations.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
using namespace llvm;

void X86JITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  unsigned char *OldByte = (unsigned char *)Old;
  *OldByte++ = 0xE9;                // Emit JMP opcode.
  unsigned *OldWord = (unsigned *)OldByte;
  unsigned NewAddr = (intptr_t)New;
  unsigned OldAddr = (intptr_t)OldWord;
  *OldWord = NewAddr - OldAddr - 4; // Emit PC-relative addr of New code.
}


/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

// Provide a wrapper for X86CompilationCallback2 that saves non-traditional
// callee saved registers, for the fastcc calling convention.
extern "C" {
#if defined(__i386__) || defined(i386) || defined(_M_IX86)
#ifndef _MSC_VER
  void X86CompilationCallback(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl X86CompilationCallback\n"
  "X86CompilationCallback:\n"
    "pushl   %ebp\n"
    "movl    %esp, %ebp\n"    // Standard prologue
    "pushl   %eax\n"
    "pushl   %edx\n"          // save EAX/EDX
    "call X86CompilationCallback2\n"
    "popl    %edx\n"
    "popl    %eax\n"
    "popl    %ebp\n"
    "ret\n");
#else
  extern "C" void *_AddressOfReturnAddress(void);
  #pragma intrinsic(_AddressOfReturnAddress)

  void X86CompilationCallback2(void);

  _declspec(naked) void X86CompilationCallback(void) {
    __asm {
      push  eax
      push  edx
      call  X86CompilationCallback2
      pop   edx
      pop   eax
      ret
    }
  }
#endif

#else
  // Not an i386 host
  void X86CompilationCallback() {
    assert(0 && "This is not a X86, you can't execute this!");
    abort();
  }
#endif
}

/// X86CompilationCallback - This is the target-specific function invoked by the
/// function stub when we did not know the real target of a call.  This function
/// must locate the start of the stub or call site and pass it into the JIT
/// compiler function.
extern "C" void X86CompilationCallback2() {
#ifdef _MSC_VER
  assert(sizeof(size_t) == 4); // FIXME: handle Win64
  unsigned *RetAddrLoc = (unsigned *)_AddressOfReturnAddress();
  RetAddrLoc += 3;  // skip over ret addr, edx, eax
  unsigned RetAddr = *RetAddrLoc;
#else
  unsigned *StackPtr = (unsigned*)__builtin_frame_address(1);
  unsigned RetAddr = (unsigned)(intptr_t)__builtin_return_address(1);
  unsigned *RetAddrLoc = &StackPtr[1];

  // NOTE: __builtin_frame_address doesn't work if frame pointer elimination has
  // been performed.  Having a variable sized alloca disables frame pointer
  // elimination currently, even if it's dead.  This is a gross hack.
  alloca(10+(RetAddr >> 31));

#endif
  assert(*RetAddrLoc == RetAddr &&
         "Could not find return address on the stack!");

  // It's a stub if there is an interrupt marker after the call.
  bool isStub = ((unsigned char*)(intptr_t)RetAddr)[0] == 0xCD;

  // The call instruction should have pushed the return value onto the stack...
  RetAddr -= 4;  // Backtrack to the reference itself...

#if 0
  DEBUG(std::cerr << "In callback! Addr=" << (void*)RetAddr
                  << " ESP=" << (void*)StackPtr
                  << ": Resolving call to function: "
                  << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n");
#endif

  // Sanity check to make sure this really is a call instruction.
  assert(((unsigned char*)(intptr_t)RetAddr)[-1] == 0xE8 &&"Not a call instr!");

  unsigned NewVal = (intptr_t)JITCompilerFunction((void*)(intptr_t)RetAddr);

  // Rewrite the call target... so that we don't end up here every time we
  // execute the call.
  *(unsigned*)(intptr_t)RetAddr = NewVal-RetAddr-4;

  if (isStub) {
    // If this is a stub, rewrite the call into an unconditional branch
    // instruction so that two return addresses are not pushed onto the stack
    // when the requested function finally gets called.  This also makes the
    // 0xCD byte (interrupt) dead, so the marker doesn't effect anything.
    ((unsigned char*)(intptr_t)RetAddr)[-1] = 0xE9;
  }

  // Change the return address to reexecute the call instruction...
  *RetAddrLoc -= 5;
}

TargetJITInfo::LazyResolverFn
X86JITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return X86CompilationCallback;
}

void *X86JITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
  if (Fn != X86CompilationCallback) {
    MCE.startFunctionStub(5);
    MCE.emitByte(0xE9);
    MCE.emitWord((intptr_t)Fn-MCE.getCurrentPCValue()-4);
    return MCE.finishFunctionStub(0);
  }

  MCE.startFunctionStub(6);
  MCE.emitByte(0xE8);   // Call with 32 bit pc-rel destination...

  MCE.emitWord((intptr_t)Fn-MCE.getCurrentPCValue()-4);

  MCE.emitByte(0xCD);   // Interrupt - Just a marker identifying the stub!
  return MCE.finishFunctionStub(0);
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void X86JITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*)Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t)MR->getResultPointer();
    switch ((X86::RelocationType)MR->getRelocationType()) {
    case X86::reloc_pcrel_word:
      // PC relative relocation, add the relocated value to the value already in
      // memory, after we adjust it for where the PC is.
      ResultPtr = ResultPtr-(intptr_t)RelocPos-4;
      *((intptr_t*)RelocPos) += ResultPtr;
      break;
    case X86::reloc_absolute_word:
      // Absolute relocation, just add the relocated value to the value already
      // in memory.
      *((intptr_t*)RelocPos) += ResultPtr;
      break;
    }
  }
}
