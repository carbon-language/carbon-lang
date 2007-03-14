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
#include "X86JITInfo.h"
#include "X86Relocations.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include <cstdlib>
using namespace llvm;

#ifdef _MSC_VER
  extern "C" void *_AddressOfReturnAddress(void);
  #pragma intrinsic(_AddressOfReturnAddress)
#endif

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

// Get the ASMPREFIX for the current host.  This is often '_'.
#ifndef __USER_LABEL_PREFIX__
#define __USER_LABEL_PREFIX__
#endif
#define GETASMPREFIX2(X) #X
#define GETASMPREFIX(X) GETASMPREFIX2(X)
#define ASMPREFIX GETASMPREFIX(__USER_LABEL_PREFIX__)

// Provide a wrapper for X86CompilationCallback2 that saves non-traditional
// callee saved registers, for the fastcc calling convention.
extern "C" {
#if defined(__x86_64__)
  // No need to save EAX/EDX for X86-64.
  void X86CompilationCallback(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl " ASMPREFIX "X86CompilationCallback\n"
  ASMPREFIX "X86CompilationCallback:\n"
    // Save RBP
    "pushq   %rbp\n"
    // Save RSP
    "movq    %rsp, %rbp\n"
    // Save all int arg registers
    "pushq   %rdi\n"
    "pushq   %rsi\n"
    "pushq   %rdx\n"
    "pushq   %rcx\n"
    "pushq   %r8\n"
    "pushq   %r9\n"
    // Align stack on 16-byte boundary. ESP might not be properly aligned
    // (8 byte) if this is called from an indirect stub.
    "andq    $-16, %rsp\n"
    // Save all XMM arg registers
    "subq    $128, %rsp\n"
    "movaps  %xmm0, (%rsp)\n"
    "movaps  %xmm1, 16(%rsp)\n"
    "movaps  %xmm2, 32(%rsp)\n"
    "movaps  %xmm3, 48(%rsp)\n"
    "movaps  %xmm4, 64(%rsp)\n"
    "movaps  %xmm5, 80(%rsp)\n"
    "movaps  %xmm6, 96(%rsp)\n"
    "movaps  %xmm7, 112(%rsp)\n"
    // JIT callee
    "movq    %rbp, %rdi\n"    // Pass prev frame and return address
    "movq    8(%rbp), %rsi\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
    // Restore all XMM arg registers
    "movaps  112(%rsp), %xmm7\n"
    "movaps  96(%rsp), %xmm6\n"
    "movaps  80(%rsp), %xmm5\n"
    "movaps  64(%rsp), %xmm4\n"
    "movaps  48(%rsp), %xmm3\n"
    "movaps  32(%rsp), %xmm2\n"
    "movaps  16(%rsp), %xmm1\n"
    "movaps  (%rsp), %xmm0\n"
    // Restore RSP
    "movq    %rbp, %rsp\n"
    // Restore all int arg registers
    "subq    $48, %rsp\n"
    "popq    %r9\n"
    "popq    %r8\n"
    "popq    %rcx\n"
    "popq    %rdx\n"
    "popq    %rsi\n"
    "popq    %rdi\n"
    // Restore RBP
    "popq    %rbp\n"
    "ret\n");
#elif defined(__i386__) || defined(i386) || defined(_M_IX86)
#ifndef _MSC_VER
  void X86CompilationCallback(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl " ASMPREFIX  "X86CompilationCallback\n"
  ASMPREFIX "X86CompilationCallback:\n"
    "pushl   %ebp\n"
    "movl    %esp, %ebp\n"    // Standard prologue
    "pushl   %eax\n"
    "pushl   %edx\n"          // Save EAX/EDX/ECX
    "pushl   %ecx\n"
#if defined(__APPLE__)
    "andl    $-16, %esp\n"    // Align ESP on 16-byte boundary
#endif
    "subl    $16, %esp\n"
    "movl    4(%ebp), %eax\n" // Pass prev frame and return address
    "movl    %eax, 4(%esp)\n"
    "movl    %ebp, (%esp)\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
    "movl    %ebp, %esp\n"    // Restore ESP
    "subl    $12, %esp\n"
    "popl    %ecx\n"
    "popl    %edx\n"
    "popl    %eax\n"
    "popl    %ebp\n"
    "ret\n");

  // Same as X86CompilationCallback but also saves XMM argument registers.
  void X86CompilationCallback_SSE(void);
  asm(
    ".text\n"
    ".align 8\n"
    ".globl " ASMPREFIX  "X86CompilationCallback_SSE\n"
  ASMPREFIX "X86CompilationCallback_SSE:\n"
    "pushl   %ebp\n"
    "movl    %esp, %ebp\n"    // Standard prologue
    "pushl   %eax\n"
    "pushl   %edx\n"          // Save EAX/EDX/ECX
    "pushl   %ecx\n"
    "andl    $-16, %esp\n"    // Align ESP on 16-byte boundary
    // Save all XMM arg registers
    "subl    $64, %esp\n"
    "movaps  %xmm0, (%esp)\n"
    "movaps  %xmm1, 16(%esp)\n"
    "movaps  %xmm2, 32(%esp)\n"
    "movaps  %xmm3, 48(%esp)\n"
    "subl    $16, %esp\n"
    "movl    4(%ebp), %eax\n" // Pass prev frame and return address
    "movl    %eax, 4(%esp)\n"
    "movl    %ebp, (%esp)\n"
    "call    " ASMPREFIX "X86CompilationCallback2\n"
    "addl    $16, %esp\n"
    "movaps  48(%esp), %xmm3\n"
    "movaps  32(%esp), %xmm2\n"
    "movaps  16(%esp), %xmm1\n"
    "movaps  (%esp), %xmm0\n"
    "movl    %ebp, %esp\n"    // Restore ESP
    "subl    $12, %esp\n"
    "popl    %ecx\n"
    "popl    %edx\n"
    "popl    %eax\n"
    "popl    %ebp\n"
    "ret\n");
#else
  void X86CompilationCallback2(void);

  _declspec(naked) void X86CompilationCallback(void) {
    __asm {
      push  eax
      push  edx
      push  ecx
      call  X86CompilationCallback2
      pop   ecx
      pop   edx
      pop   eax
      ret
    }
  }
#endif // _MSC_VER

#else // Not an i386 host
  void X86CompilationCallback() {
    assert(0 && "Cannot call X86CompilationCallback() on a non-x86 arch!\n");
    abort();
  }
#endif
}

/// X86CompilationCallback - This is the target-specific function invoked by the
/// function stub when we did not know the real target of a call.  This function
/// must locate the start of the stub or call site and pass it into the JIT
/// compiler function.
#ifdef _MSC_VER
extern "C" void X86CompilationCallback2() {
  assert(sizeof(size_t) == 4); // FIXME: handle Win64
  intptr_t *RetAddrLoc = (intptr_t *)_AddressOfReturnAddress();
  RetAddrLoc += 4;  // skip over ret addr, edx, eax, ecx
  intptr_t RetAddr = *RetAddrLoc;
#else
extern "C" void X86CompilationCallback2(intptr_t *StackPtr, intptr_t RetAddr) {
  intptr_t *RetAddrLoc = &StackPtr[1];
#endif
  assert(*RetAddrLoc == RetAddr &&
         "Could not find return address on the stack!");

  // It's a stub if there is an interrupt marker after the call.
  bool isStub = ((unsigned char*)RetAddr)[0] == 0xCD;

  // The call instruction should have pushed the return value onto the stack...
#ifdef __x86_64__
  RetAddr--;     // Backtrack to the reference itself...
#else
  RetAddr -= 4;  // Backtrack to the reference itself...
#endif

#if 0
  DOUT << "In callback! Addr=" << (void*)RetAddr
       << " ESP=" << (void*)StackPtr
       << ": Resolving call to function: "
       << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n";
#endif

  // Sanity check to make sure this really is a call instruction.
#ifdef __x86_64__
  assert(((unsigned char*)RetAddr)[-2] == 0x41 &&"Not a call instr!");
  assert(((unsigned char*)RetAddr)[-1] == 0xFF &&"Not a call instr!");
#else
  assert(((unsigned char*)RetAddr)[-1] == 0xE8 &&"Not a call instr!");
#endif

  intptr_t NewVal = (intptr_t)JITCompilerFunction((void*)RetAddr);

  // Rewrite the call target... so that we don't end up here every time we
  // execute the call.
#ifdef __x86_64__
  *(intptr_t *)(RetAddr - 0xa) = NewVal;
#else
  *(intptr_t *)RetAddr = (intptr_t)(NewVal-RetAddr-4);
#endif

  if (isStub) {
    // If this is a stub, rewrite the call into an unconditional branch
    // instruction so that two return addresses are not pushed onto the stack
    // when the requested function finally gets called.  This also makes the
    // 0xCD byte (interrupt) dead, so the marker doesn't effect anything.
#ifdef __x86_64__
    ((unsigned char*)RetAddr)[0] = (2 | (4 << 3) | (3 << 6));
#else
    ((unsigned char*)RetAddr)[-1] = 0xE9;
#endif
  }

  // Change the return address to reexecute the call instruction...
#ifdef __x86_64__
  *RetAddrLoc -= 0xd;
#else
  *RetAddrLoc -= 5;
#endif
}

TargetJITInfo::LazyResolverFn
X86JITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;

#if (defined(__i386__) || defined(i386) || defined(_M_IX86)) && \
  !defined(_MSC_VER) && !defined(__x86_64__)
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  union {
    unsigned u[3];
    char     c[12];
  } text;

  if (!X86::GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1)) {
    // FIXME: support for AMD family of processors.
    if (memcmp(text.c, "GenuineIntel", 12) == 0) {
      X86::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);
      if ((EDX >> 25) & 0x1)
        return X86CompilationCallback_SSE;
    }
  }
#endif

  return X86CompilationCallback;
}

void *X86JITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
  // Note, we cast to intptr_t here to silence a -pedantic warning that 
  // complains about casting a function pointer to a normal pointer.
#if (defined(__i386__) || defined(i386) || defined(_M_IX86)) && \
  !defined(_MSC_VER) && !defined(__x86_64__)
  bool NotCC = (Fn != (void*)(intptr_t)X86CompilationCallback &&
                Fn != (void*)(intptr_t)X86CompilationCallback_SSE);
#else
  bool NotCC = Fn != (void*)(intptr_t)X86CompilationCallback;
#endif
  if (NotCC) {
#ifdef __x86_64__
    MCE.startFunctionStub(13, 4);
    MCE.emitByte(0x49);          // REX prefix
    MCE.emitByte(0xB8+2);        // movabsq r10
    MCE.emitWordLE(((unsigned *)&Fn)[0]);
    MCE.emitWordLE(((unsigned *)&Fn)[1]);
    MCE.emitByte(0x41);          // REX prefix
    MCE.emitByte(0xFF);          // jmpq *r10
    MCE.emitByte(2 | (4 << 3) | (3 << 6));
#else
    MCE.startFunctionStub(5, 4);
    MCE.emitByte(0xE9);
    MCE.emitWordLE((intptr_t)Fn-MCE.getCurrentPCValue()-4);
#endif
    return MCE.finishFunctionStub(0);
  }

#ifdef __x86_64__
  MCE.startFunctionStub(14, 4);
  MCE.emitByte(0x49);          // REX prefix
  MCE.emitByte(0xB8+2);        // movabsq r10
  MCE.emitWordLE(((unsigned *)&Fn)[0]);
  MCE.emitWordLE(((unsigned *)&Fn)[1]);
  MCE.emitByte(0x41);          // REX prefix
  MCE.emitByte(0xFF);          // callq *r10
  MCE.emitByte(2 | (2 << 3) | (3 << 6));
#else
  MCE.startFunctionStub(6, 4);
  MCE.emitByte(0xE8);   // Call with 32 bit pc-rel destination...

  MCE.emitWordLE((intptr_t)Fn-MCE.getCurrentPCValue()-4);
#endif

  MCE.emitByte(0xCD);   // Interrupt - Just a marker identifying the stub!
  return MCE.finishFunctionStub(0);
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void X86JITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*)Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t)MR->getResultPointer();
    switch ((X86::RelocationType)MR->getRelocationType()) {
    case X86::reloc_pcrel_word: {
      // PC relative relocation, add the relocated value to the value already in
      // memory, after we adjust it for where the PC is.
      ResultPtr = ResultPtr-(intptr_t)RelocPos-4-MR->getConstantVal();
      *((unsigned*)RelocPos) += (unsigned)ResultPtr;
      break;
    }
    case X86::reloc_absolute_word:
      // Absolute relocation, just add the relocated value to the value already
      // in memory.
      *((unsigned*)RelocPos) += (unsigned)ResultPtr;
      break;
    case X86::reloc_absolute_dword:
      *((intptr_t*)RelocPos) += ResultPtr;
      break;
    }
  }
}
