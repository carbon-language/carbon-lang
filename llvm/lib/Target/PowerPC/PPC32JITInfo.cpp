//===-- PPC32JITInfo.cpp - Implement the JIT interfaces for the PowerPC ---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the 32-bit PowerPC target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "PPC32JITInfo.h"
#include "PPC32Relocations.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
using namespace llvm;

static TargetJITInfo::JITCompilerFn JITCompilerFunction;

#define BUILD_ADDIS(RD,RS,IMM16) \
  ((15 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 65535))
#define BUILD_ORI(RD,RS,UIMM16) \
  ((24 << 26) | ((RS) << 21) | ((RD) << 16) | ((UIMM16) & 65535))
#define BUILD_MTSPR(RS,SPR)      \
  ((31 << 26) | ((RS) << 21) | ((SPR) << 16) | (467 << 1))
#define BUILD_BCCTRx(BO,BI,LINK) \
  ((19 << 26) | ((BO) << 21) | ((BI) << 16) | (528 << 1) | ((LINK) & 1))

// Pseudo-ops
#define BUILD_LIS(RD,IMM16)    BUILD_ADDIS(RD,0,IMM16)
#define BUILD_MTCTR(RS)        BUILD_MTSPR(RS,9)
#define BUILD_BCTR(LINK)       BUILD_BCCTRx(20,0,LINK)

static void CompilationCallback() {
  //JITCompilerFunction
  assert(0 && "CompilationCallback not implemented yet!");
}


TargetJITInfo::LazyResolverFn 
PPC32JITInfo::getLazyResolverFunction(JITCompilerFn Fn) {
  return CompilationCallback;
}

static void EmitBranchToAt(void *At, void *To) {
  intptr_t Addr = (intptr_t)To;

  // FIXME: should special case the short branch case.
  unsigned *AtI = (unsigned*)At;

  AtI[0] = BUILD_LIS(12, Addr >> 16);   // lis r12, hi16(address)
  AtI[1] = BUILD_ORI(12, 12, Addr);     // ori r12, r12, low16(address)
  AtI[2] = BUILD_MTCTR(12);             // mtctr r12
  AtI[3] = BUILD_BCTR(0);               // bctr
}

void *PPC32JITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
  // If this is just a call to an external function, emit a branch instead of a
  // call.  The code is the same except for one bit of the last instruction.
  if (Fn != CompilationCallback) {
    MCE.startFunctionStub(4*4);
    void *Addr = (void*)(intptr_t)MCE.getCurrentPCValue();
    MCE.emitWord(0);
    MCE.emitWord(0);
    MCE.emitWord(0);
    MCE.emitWord(0);
    EmitBranchToAt(Addr, Fn);
    return MCE.finishFunctionStub(0);
  }

  MCE.startFunctionStub(4*4);
  return MCE.finishFunctionStub(0);
}


void PPC32JITInfo::relocate(void *Function, MachineRelocation *MR,
                            unsigned NumRelocs) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    unsigned *RelocPos = (unsigned*)Function + MR->getMachineCodeOffset()/4;
    intptr_t ResultPtr = (intptr_t)MR->getResultPointer();
    switch ((PPC::RelocationType)MR->getRelocationType()) {
    default: assert(0 && "Unknown relocation type!");
    case PPC::reloc_pcrel_bx:
      // PC-relative relocation for b and bl instructions.
      ResultPtr = (ResultPtr-(intptr_t)RelocPos) >> 2;
      assert(ResultPtr >= -(1 << 23) && ResultPtr < (1 << 23) &&
             "Relocation out of range!");
      *RelocPos |= (ResultPtr & ((1 << 24)-1))  << 2;
      break;
    case PPC::reloc_absolute_loadhi:   // Relocate high bits into addis
    case PPC::reloc_absolute_la:       // Relocate low bits into addi
      ResultPtr += MR->getConstantVal();

      if (MR->getRelocationType() == PPC::reloc_absolute_loadhi) {
        // If the low part will have a carry (really a borrow) from the low
        // 16-bits into the high 16, add a bit to borrow from.
        if (((int)ResultPtr << 16) < 0)
          ResultPtr += 1 << 16;
        ResultPtr >>= 16;
      }

      // Do the addition then mask, so the addition does not overflow the 16-bit
      // immediate section of the instruction.
      unsigned LowBits  = (*RelocPos + ResultPtr) & 65535;
      unsigned HighBits = *RelocPos & ~65535;
      *RelocPos = LowBits | HighBits;  // Slam into low 16-bits
      break;
    }
  }
}

void PPC32JITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  EmitBranchToAt(Old, New);
}
