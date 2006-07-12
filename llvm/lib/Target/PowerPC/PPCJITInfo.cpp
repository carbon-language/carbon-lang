//===-- PPCJITInfo.cpp - Implement the JIT interfaces for the PowerPC -----===//
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
#include "PPCJITInfo.h"
#include "PPCRelocations.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include <set>
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

extern "C" void PPC32CompilationCallback();

#if defined(__POWERPC__) || defined (__ppc__) || defined(_POWER)
// CompilationCallback stub - We can't use a C function with inline assembly in
// it, because we the prolog/epilog inserted by GCC won't work for us.  Instead,
// write our own wrapper, which does things our way, so we have complete control
// over register saving and restoring.
asm(
    ".text\n"
    ".align 2\n"
    ".globl _PPC32CompilationCallback\n"
"_PPC32CompilationCallback:\n"
    // Make space for 8 ints r[3-10] and 13 doubles f[1-13] and the 
    // FIXME: need to save v[0-19] for altivec?
    // Set up a proper stack frame
    "stwu r1, -208(r1)\n"
    "mflr r0\n"
    "stw r0,  216(r1)\n"
    // Save all int arg registers
    "stw r10, 204(r1)\n"    "stw r9,  200(r1)\n"
    "stw r8,  196(r1)\n"    "stw r7,  192(r1)\n"
    "stw r6,  188(r1)\n"    "stw r5,  184(r1)\n"
    "stw r4,  180(r1)\n"    "stw r3,  176(r1)\n"
    // Save all call-clobbered FP regs.
    "stfd f13, 168(r1)\n"   "stfd f12, 160(r1)\n"
    "stfd f11, 152(r1)\n"   "stfd f10, 144(r1)\n"
    "stfd f9,  136(r1)\n"   "stfd f8,  128(r1)\n"
    "stfd f7,  120(r1)\n"   "stfd f6,  112(r1)\n"
    "stfd f5,  104(r1)\n"   "stfd f4,   96(r1)\n"
    "stfd f3,   88(r1)\n"   "stfd f2,   80(r1)\n"
    "stfd f1,   72(r1)\n"
    // Arguments to Compilation Callback:
    // r3 - our lr (address of the call instruction in stub plus 4)
    // r4 - stub's lr (address of instruction that called the stub plus 4)
    "mr   r3, r0\n"
    "lwz  r2, 208(r1)\n" // stub's frame
    "lwz  r4, 8(r2)\n" // stub's lr
    "bl _PPC32CompilationCallbackC\n"
    "mtctr r3\n"
    // Restore all int arg registers
    "lwz r10, 204(r1)\n"    "lwz r9,  200(r1)\n"
    "lwz r8,  196(r1)\n"    "lwz r7,  192(r1)\n"
    "lwz r6,  188(r1)\n"    "lwz r5,  184(r1)\n"
    "lwz r4,  180(r1)\n"    "lwz r3,  176(r1)\n"
    // Restore all FP arg registers
    "lfd f13, 168(r1)\n"    "lfd f12, 160(r1)\n"
    "lfd f11, 152(r1)\n"    "lfd f10, 144(r1)\n"
    "lfd f9,  136(r1)\n"    "lfd f8,  128(r1)\n"
    "lfd f7,  120(r1)\n"    "lfd f6,  112(r1)\n"
    "lfd f5,  104(r1)\n"    "lfd f4,   96(r1)\n"
    "lfd f3,   88(r1)\n"    "lfd f2,   80(r1)\n"
    "lfd f1,   72(r1)\n"
    // Pop 3 frames off the stack and branch to target
    "lwz  r1, 208(r1)\n"
    "lwz  r2, 8(r1)\n"
    "mtlr r2\n"
    "bctr\n"
    );
#else
void PPC32CompilationCallback() {
  assert(0 && "This is not a power pc, you can't execute this!");
  abort();
}
#endif

extern "C" unsigned *PPC32CompilationCallbackC(unsigned *StubCallAddrPlus4,
                                               unsigned *OrigCallAddrPlus4) {
  // Adjust the pointer to the address of the call instruction in the stub
  // emitted by emitFunctionStub, rather than the instruction after it.
  unsigned *StubCallAddr = StubCallAddrPlus4 - 1;
  unsigned *OrigCallAddr = OrigCallAddrPlus4 - 1;

  void *Target = JITCompilerFunction(StubCallAddr);

  // Check to see if *OrigCallAddr is a 'bl' instruction, and if we can rewrite
  // it to branch directly to the destination.  If so, rewrite it so it does not
  // need to go through the stub anymore.
  unsigned OrigCallInst = *OrigCallAddr;
  if ((OrigCallInst >> 26) == 18) {     // Direct call.
    intptr_t Offset = ((intptr_t)Target - (intptr_t)OrigCallAddr) >> 2;
    
    if (Offset >= -(1 << 23) && Offset < (1 << 23)) {   // In range?
      // Clear the original target out.
      OrigCallInst &= (63 << 26) | 3;
      // Fill in the new target.
      OrigCallInst |= (Offset & ((1 << 24)-1)) << 2;
      // Replace the call.
      *OrigCallAddr = OrigCallInst;
    }
  }

  // Assert that we are coming from a stub that was created with our
  // emitFunctionStub.
  assert((*StubCallAddr >> 26) == 19 && "Call in stub is not indirect!");
  StubCallAddr -= 6;

  // Rewrite the stub with an unconditional branch to the target, for any users
  // who took the address of the stub.
  EmitBranchToAt(StubCallAddr, Target, false);

  // Put the address of the target function to call and the address to return to
  // after calling the target function in a place that is easy to get on the
  // stack after we restore all regs.
  return (unsigned *)Target;
}



TargetJITInfo::LazyResolverFn
PPCJITInfo::getLazyResolverFunction(JITCompilerFn Fn) {
  JITCompilerFunction = Fn;
  return PPC32CompilationCallback;
}

void *PPCJITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
  // If this is just a call to an external function, emit a branch instead of a
  // call.  The code is the same except for one bit of the last instruction.
  if (Fn != (void*)(intptr_t)PPC32CompilationCallback) {
    MCE.startFunctionStub(4*4);
    void *Addr = (void*)(intptr_t)MCE.getCurrentPCValue();
    MCE.emitWordBE(0);
    MCE.emitWordBE(0);
    MCE.emitWordBE(0);
    MCE.emitWordBE(0);
    EmitBranchToAt(Addr, Fn, false);
    return MCE.finishFunctionStub(0);
  }

  MCE.startFunctionStub(4*7);
  MCE.emitWordBE(0x9421ffe0);     // stwu    r1,-32(r1)
  MCE.emitWordBE(0x7d6802a6);     // mflr r11
  MCE.emitWordBE(0x91610028);     // stw r11, 40(r1)
  void *Addr = (void*)(intptr_t)MCE.getCurrentPCValue();
  MCE.emitWordBE(0);
  MCE.emitWordBE(0);
  MCE.emitWordBE(0);
  MCE.emitWordBE(0);
  EmitBranchToAt(Addr, Fn, true/*is call*/);
  return MCE.finishFunctionStub(0);
}


void PPCJITInfo::relocate(void *Function, MachineRelocation *MR,
                          unsigned NumRelocs, unsigned char* GOTBase) {
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
    case PPC::reloc_absolute_ptr_high: // Pointer relocations.
    case PPC::reloc_absolute_ptr_low:
    case PPC::reloc_absolute_high:     // high bits of ref -> low 16 of instr
    case PPC::reloc_absolute_low: {    // low bits of ref  -> low 16 of instr
      ResultPtr += MR->getConstantVal();

      // If this is a high-part access, get the high-part.
      if (MR->getRelocationType() == PPC::reloc_absolute_high ||
          MR->getRelocationType() == PPC::reloc_absolute_ptr_high) {
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
    case PPC::reloc_absolute_low_ix: {  // low bits of ref  -> low 14 of instr
      ResultPtr += MR->getConstantVal();
      // Do the addition then mask, so the addition does not overflow the 16-bit
      // immediate section of the instruction.
      unsigned LowBits  = (*RelocPos + ResultPtr) & 0xFFFC;
      unsigned HighBits = *RelocPos & 0xFFFF0003;
      *RelocPos = LowBits | HighBits;  // Slam into low 14-bits.
      break;
    }
    }
  }
}

void PPCJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  EmitBranchToAt(Old, New, false);
}
