//===- MipsJITInfo.cpp - Implement the JIT interfaces for the Mips target -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the Mips target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "MipsJITInfo.h"
#include "MipsInstrInfo.h"
#include "MipsRelocations.h"
#include "MipsSubtarget.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Memory.h"
#include <cstdlib>
using namespace llvm;


void MipsJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  report_fatal_error("MipsJITInfo::replaceMachineCodeForFunction");
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

// save registers, call MipsCompilationCallbackC, restore registers
extern "C" {
#if defined (__mips__)
void MipsCompilationCallback();

  asm(
    ".text\n"
    ".align 2\n"
    ".globl " ASMPREFIX "MipsCompilationCallback\n"
    ASMPREFIX "MipsCompilationCallback:\n"
    ".ent " ASMPREFIX "MipsCompilationCallback\n"
    ".set  noreorder\n"
    ".cpload $t9\n"
    ".frame  $29, 32, $31\n"

    "addiu $sp, $sp, -40\n"
    "sw $a0, 4($sp)\n"
    "sw $a1, 8($sp)\n"
    "sw $a2, 12($sp)\n"
    "sw $a3, 20($sp)\n"
    "sw $ra, 24($sp)\n"
    "sw $v0, 28($sp)\n"
    "sw $v1, 32($sp)\n"
    "sw $t8, 36($sp)\n"
    ".cprestore 16\n"

    "addiu $a0, $t8, -16\n"
    "jal   " ASMPREFIX "MipsCompilationCallbackC\n"
    "nop\n"

    "lw $a0, 4($sp)\n"
    "lw $a1, 8($sp)\n"
    "lw $a2, 12($sp)\n"
    "lw $a3, 20($sp)\n"
    "lw $ra, 24($sp)\n"
    "lw $v0, 28($sp)\n"
    "lw $v1, 32($sp)\n"
    "lw $t8, 36($sp)\n"
    "addiu $sp, $sp, 40\n"

    "addiu $t8, $t8, -16\n"
    "jr $t8\n"
    "nop\n"

    ".set  reorder\n"
    ".end " ASMPREFIX "MipsCompilationCallback\n"
      );
#else  // host != Mips
  void MipsCompilationCallback() {
    llvm_unreachable(
      "Cannot call MipsCompilationCallback() on a non-Mips arch!");
  }
#endif
}

/// MipsCompilationCallbackC - This is the target-specific function invoked
/// by the function stub when we did not know the real target of a call.
/// This function must locate the start of the stub or call site and pass
/// it into the JIT compiler function.
extern "C" void MipsCompilationCallbackC(intptr_t StubAddr) {

  // Get the address of the compiled code for this function.
  intptr_t NewVal = (intptr_t) JITCompilerFunction((void*) StubAddr);

  *(intptr_t *) (StubAddr) = 2 << 26 | ((NewVal & 0x0fffffff) >> 2); // J NewVal
  *(intptr_t *) (StubAddr + 4) = 0; // NOP
  *(intptr_t *) (StubAddr + 8) = 0; // NOP
  *(intptr_t *) (StubAddr + 12) = 0; // NOP

  sys::Memory::InvalidateInstructionCache((void*) StubAddr, 16);
}

TargetJITInfo::LazyResolverFn MipsJITInfo::getLazyResolverFunction(
    JITCompilerFn F) {
  JITCompilerFunction = F;
  return MipsCompilationCallback;
}

TargetJITInfo::StubLayout MipsJITInfo::getStubLayout() {
  StubLayout Result = { 24, 4 }; // {Size. Alignment} (of FunctionStub)
  return Result;
}

void *MipsJITInfo::emitFunctionStub(const Function* F, void *Fn,
    JITCodeEmitter &JCE) {
  JCE.emitAlignment(4);
  void *Addr = (void*) (JCE.getCurrentPCValue());

  unsigned arg0 = ((intptr_t) MipsCompilationCallback >> 16);
  if ((((intptr_t) MipsCompilationCallback & 0xffff) >> 15) == 1) {
    arg0 += 1;  // same hack as in relocate()
  }

  // LUI t9, %hi(MipsCompilationCallback)
  JCE.emitWordLE(0xf << 26 | 25 << 16 | arg0);
  // ADDiu t9, t9, %lo(MipsCompilationCallback)
  JCE.emitWordLE(9 << 26 | 25 << 21 | 25 << 16
          | ((intptr_t) MipsCompilationCallback & 0xffff));
  // JALR t8, t9
  JCE.emitWordLE(25 << 21 | 24 << 11 | 9);
  JCE.emitWordLE(0);  // NOP

  sys::Memory::InvalidateInstructionCache((void*) Addr, 16);

  return Addr;
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void MipsJITInfo::relocate(void *Function, MachineRelocation *MR,
    unsigned NumRelocs, unsigned char* GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {

    void *RelocPos = (char*) Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t) MR->getResultPointer();

    switch ((Mips::RelocationType) MR->getRelocationType()) {
    case Mips::reloc_mips_pcrel:
      ResultPtr = (((ResultPtr - (intptr_t) RelocPos) - 4) >> 2) & 0xffff;
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
      break;

    case Mips::reloc_mips_j_jal: {
      ResultPtr = (ResultPtr & 0x0fffffff) >> 2;
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
    }
      break;

    case Mips::reloc_mips_hi: {
      ResultPtr = ResultPtr >> 16;

      // see See MIPS Run Linux, chapter 9.4
      if ((((intptr_t) (MR->getResultPointer()) & 0xffff) >> 15) == 1) {
        ResultPtr += 1;
      }

      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
    }
      break;

    case Mips::reloc_mips_lo:
      ResultPtr = ResultPtr & 0xffff;
      *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
      break;

    default:
      assert(0 && "MipsJITInfo.unknown relocation;");
    }
  }
}
