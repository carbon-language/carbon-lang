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


static void EmitBranchToAt(void *At, void *To, bool isCall) {
  intptr_t Addr = (intptr_t)To;

  // FIXME: should special case the short branch case.
  unsigned *AtI = (unsigned*)At;

  AtI[0] = BUILD_LIS(12, Addr >> 16);   // lis r12, hi16(address)
  AtI[1] = BUILD_ORI(12, 12, Addr);     // ori r12, r12, low16(address)
  AtI[2] = BUILD_MTCTR(12);             // mtctr r12
  AtI[3] = BUILD_BCTR(isCall);          // bctr/bctrl
}

static void CompilationCallback() {
  // Save R3-R31, since we want to restore arguments and nonvolatile regs used
  // by the compiler.  We also save and restore the FP regs, although this is
  // probably just paranoia (gcc is unlikely to emit code that uses them for
  // for this function.
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  unsigned IntRegs[29];
  double FPRegs[13];
  __asm__ __volatile__ (
  "stmw r3, 0(%0)\n"
  "stfd f1, 0(%1)\n"  "stfd f2, 8(%1)\n"  "stfd f3, 16(%1)\n" 
  "stfd f4, 24(%1)\n" "stfd f5, 32(%1)\n" "stfd f6, 40(%1)\n" 
  "stfd f7, 48(%1)\n" "stfd f8, 56(%1)\n" "stfd f9, 64(%1)\n" 
  "stfd f10, 72(%1)\n" "stfd f11, 80(%1)\n" "stfd f12, 88(%1)\n"
  "stfd f13, 96(%1)\n" :: "b" (IntRegs), "b" (FPRegs) );
  /// FIXME: Need to safe and restore the rest of the FP regs!
#endif

  unsigned *CameFromStub = (unsigned*)__builtin_return_address(0);
  unsigned *CameFromOrig = (unsigned*)__builtin_return_address(1);
  unsigned *CCStackPtr   = (unsigned*)__builtin_frame_address(0);
//unsigned *StubStackPtr = (unsigned*)__builtin_frame_address(1);
  unsigned *OrigStackPtr = (unsigned*)__builtin_frame_address(2);

  // Adjust pointer to the branch, not the return address.
  --CameFromStub;

  void *Target = JITCompilerFunction(CameFromStub);

  // Check to see if CameFromOrig[-1] is a 'bl' instruction, and if we can
  // rewrite it to branch directly to the destination.  If so, rewrite it so it
  // does not need to go through the stub anymore.
  unsigned CameFromOrigInst = CameFromOrig[-1];
  if ((CameFromOrigInst >> 26) == 18) {     // Direct call.
    intptr_t Offset = ((intptr_t)Target-(intptr_t)CameFromOrig+4) >> 2;
    if (Offset >= -(1 << 23) && Offset < (1 << 23)) {   // In range?
      // Clear the original target out.
      CameFromOrigInst &= (63 << 26) | 3;
      // Fill in the new target.
      CameFromOrigInst |= (Offset & ((1 << 24)-1)) << 2;
      // Replace the call.
      CameFromOrig[-1] = CameFromOrigInst;
    }
  }
  
  // Locate the start of the stub.  If this is a short call, adjust backwards
  // the short amount, otherwise the full amount.
  bool isShortStub = (*CameFromStub >> 26) == 18;
  CameFromStub -= isShortStub ? 2 : 6;

  // Rewrite the stub with an unconditional branch to the target, for any users
  // who took the address of the stub.
  EmitBranchToAt(CameFromStub, Target, false);

  // Change the SP so that we pop two stack frames off when we return.
  *CCStackPtr = (intptr_t)OrigStackPtr;

  // Put the address of the stub and the LR value that originally came into the
  // stub in a place that is easy to get on the stack after we restore all regs.
  CCStackPtr[2] = (intptr_t)Target;
  CCStackPtr[1] = (intptr_t)CameFromOrig;

  // Note, this is not a standard epilog!
#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
  register unsigned *IRR asm ("r2") = IntRegs;
  register double   *FRR asm ("r3") = FPRegs;
  __asm__ __volatile__ (
  "lfd f1, 0(%0)\n"  "lfd f2, 8(%0)\n"  "lfd f3, 16(%0)\n" 
  "lfd f4, 24(%0)\n" "lfd f5, 32(%0)\n" "lfd f6, 40(%0)\n" 
  "lfd f7, 48(%0)\n" "lfd f8, 56(%0)\n" "lfd f9, 64(%0)\n" 
  "lfd f10, 72(%0)\n" "lfd f11, 80(%0)\n" "lfd f12, 88(%0)\n"
  "lfd f13, 96(%0)\n"
  "lmw r3, 0(%1)\n"  // Load all integer regs
  "lwz r0,4(r1)\n"   // Get CameFromOrig (LR into stub)
  "mtlr r0\n"        // Put it in the LR register
  "lwz r0,8(r1)\n"   // Get target function pointer
  "mtctr r0\n"       // Put it into the CTR register
  "lwz r1,0(r1)\n"   // Pop two frames off
  "bctr\n" ::        // Return to stub!
  "b" (FRR), "b" (IRR)); 
#endif
}



TargetJITInfo::LazyResolverFn 
PPC32JITInfo::getLazyResolverFunction(JITCompilerFn Fn) {
  JITCompilerFunction = Fn;
  return CompilationCallback;
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
    EmitBranchToAt(Addr, Fn, false);
    return MCE.finishFunctionStub(0);
  }

  MCE.startFunctionStub(4*7);
  MCE.emitWord(0x9421ffe0);     // stwu    r1,-32(r1)
  MCE.emitWord(0x7d6802a6);     // mflr r11
  MCE.emitWord(0x91610028);     // stw r11, 40(r1)
  void *Addr = (void*)(intptr_t)MCE.getCurrentPCValue();
  MCE.emitWord(0);
  MCE.emitWord(0);
  MCE.emitWord(0);
  MCE.emitWord(0);
  EmitBranchToAt(Addr, Fn, true/*is call*/);
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
  EmitBranchToAt(Old, New, false);
}
