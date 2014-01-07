//===-- SparcJITInfo.cpp - Implement the Sparc JIT Interface --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the Sparc target.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "jit"
#include "SparcJITInfo.h"
#include "SparcRelocations.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/Support/Memory.h"

using namespace llvm;

/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

extern "C" void SparcCompilationCallback();

extern "C" {
#if defined (__sparc__)
  asm(
      ".text\n"
      "\t.align 4\n"
      "\t.global SparcCompilationCallback\n"
      "\t.type SparcCompilationCallback, #function\n"
      "SparcCompilationCallback:\n"
      // Save current register window.
      "\tsave %sp, -192, %sp\n"
      // stubaddr+4 is in %g1.
      "\tcall SparcCompilationCallbackC\n"
      "\t  sub %g1, 4, %o0\n"
      // restore original register window and
      // copy %o0 to %g1
      "\t  restore %o0, 0, %g1\n"
      // call the new stub
      "\tjmp %g1\n"
      "\t  nop\n"
      "\t.size   SparcCompilationCallback, .-SparcCompilationCallback"
      );

#else
  void SparcCompilationCallback() {
    llvm_unreachable(
      "Cannot call SparcCompilationCallback() on a non-sparc arch!");
  }
#endif
}

#define HI(Val) (((unsigned)(Val)) >> 10)
#define LO(Val) (((unsigned)(Val)) & 0x3FF)

#define SETHI_INST(imm, rd)    (0x01000000 | ((rd) << 25) | ((imm) & 0x3FFFFF))
#define JMP_INST(rs1, imm, rd) (0x80000000 | ((rd) << 25) | (0x38 << 19) \
                                | ((rs1) << 14) | (1 << 13) | ((imm) & 0x1FFF))
#define NOP_INST               SETHI_INST(0, 0)

extern "C" void *SparcCompilationCallbackC(intptr_t StubAddr) {
  // Get the address of the compiled code for this function.
  intptr_t NewVal = (intptr_t) JITCompilerFunction((void*) StubAddr);

  // Rewrite the function stub so that we don't end up here every time we
  // execute the call. We're replacing the first three instructions of the
  // stub with code that jumps to the compiled function:
  //   sethi %hi(NewVal), %g1
  //   jmp %g1+%lo(NewVal)
  //   nop

  *(intptr_t *)(StubAddr)      = SETHI_INST(HI(NewVal), 1);
  *(intptr_t *)(StubAddr + 4)  = JMP_INST(1, LO(NewVal), 0);
  *(intptr_t *)(StubAddr + 8)  = NOP_INST;

  sys::Memory::InvalidateInstructionCache((void*) StubAddr, 12);
  return (void*)StubAddr;
}

void SparcJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  assert(0 && "FIXME: Implement SparcJITInfo::replaceMachineCodeForFunction");
}


TargetJITInfo::StubLayout SparcJITInfo::getStubLayout() {
  // The stub contains 3 4-byte instructions, aligned at 4 bytes. See
  // emitFunctionStub for details.

  StubLayout Result = { 3*4, 4 };
  return Result;
}

void *SparcJITInfo::emitFunctionStub(const Function *F, void *Fn,
                                     JITCodeEmitter &JCE)
{
  JCE.emitAlignment(4);
  void *Addr = (void*) (JCE.getCurrentPCValue());
  if (!sys::Memory::setRangeWritable(Addr, 12))
    llvm_unreachable("ERROR: Unable to mark stub writable.");

  intptr_t EmittedAddr;
  if (Fn != (void*)(intptr_t)SparcCompilationCallback)
    EmittedAddr = (intptr_t)Fn;
  else
    EmittedAddr = (intptr_t)SparcCompilationCallback;

  // sethi %hi(EmittedAddr), %g1
  // jmp   %g1+%lo(EmittedAddr), %g1
  // nop

  JCE.emitWordBE(SETHI_INST(HI(EmittedAddr), 1));
  JCE.emitWordBE(JMP_INST(1, LO(EmittedAddr), 1));
  JCE.emitWordBE(NOP_INST);

  sys::Memory::InvalidateInstructionCache(Addr, 12);
  if (!sys::Memory::setRangeExecutable(Addr, 12))
    llvm_unreachable("ERROR: Unable to mark stub executable.");

  return Addr;
}

TargetJITInfo::LazyResolverFn
SparcJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return SparcCompilationCallback;
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void SparcJITInfo::relocate(void *Function, MachineRelocation *MR,
                            unsigned NumRelocs, unsigned char *GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*) Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t) MR->getResultPointer();

    switch ((SP::RelocationType) MR->getRelocationType()) {
    case SP::reloc_sparc_hi:
      ResultPtr = (ResultPtr >> 10) & 0x3fffff;
      break;

    case SP::reloc_sparc_lo:
      ResultPtr = (ResultPtr & 0x3ff);
      break;

    case SP::reloc_sparc_pc30:
      ResultPtr = ((ResultPtr - (intptr_t)RelocPos) >> 2) & 0x3fffffff;
      break;

    case SP::reloc_sparc_pc22:
      ResultPtr = ((ResultPtr - (intptr_t)RelocPos) >> 2) & 0x3fffff;
      break;

    case SP::reloc_sparc_pc19:
      ResultPtr = ((ResultPtr - (intptr_t)RelocPos) >> 2) & 0x7ffff;
      break;
    }
    *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
  }
}
