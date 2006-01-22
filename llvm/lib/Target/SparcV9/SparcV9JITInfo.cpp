//===-- SparcJITInfo.cpp - Implement the JIT interfaces for SparcV9 -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the SparcV9 target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "SparcV9JITInfo.h"
#include "SparcV9Relocations.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

/// BUILD_SETHI/BUILD_ORI/BUILD_BA/BUILD_CALL - These macros build sparc machine
/// instructions using lots of magic defined by the Sparc ISA.
#define BUILD_SETHI(RD, C)   (((RD) << 25) | (4 << 22) | (C & ((1 << 22)-1)))
#define BUILD_ORI(RS, C, RD) ((2 << 30) | (RD << 25) | (2 << 19) | (RS << 14) |\
                              (1 << 13) | (C & ((1 << 12)-1)))
#define BUILD_BA(DISP)       ((8 << 25) | (2 << 22) | (DISP & ((1 << 22)-1)))
#define BUILD_CALL(OFFSET)   ((1 << 30) | (OFFSET & (1 << 30)-1))

static void InsertJumpAtAddr(int64_t JumpTarget, unsigned *Addr) {
  // If the target function is close enough to fit into the 19bit disp of
  // BA, we should use this version, as it's much cheaper to generate.
  int64_t BranchTarget = (JumpTarget-(intptr_t)Addr) >> 2;
  if (BranchTarget < (1 << 19) && BranchTarget > -(1 << 19)) {
    // ba <target>
    Addr[0] = BUILD_BA(BranchTarget);

    // nop
    Addr[1] = 0x01000000;
  } else {
    enum { G0 = 0, G1 = 1, G5 = 5 };
    // Get address to branch into %g1, using %g5 as a temporary
    //
    // sethi %uhi(Target), %g5   ;; get upper 22 bits of Target into %g5
    Addr[0] = BUILD_SETHI(G5, JumpTarget >> 42);
    // or %g5, %ulo(Target), %g5 ;; get 10 lower bits of upper word into %1
    Addr[1] = BUILD_ORI(G5, JumpTarget >> 32, G5);
    // sllx %g5, 32, %g5         ;; shift those 10 bits to the upper word
    Addr[2] = 0x8B297020;
    // sethi %hi(Target), %g1    ;; extract bits 10-31 into the dest reg
    Addr[3] = BUILD_SETHI(G1, JumpTarget >> 10);
    // or %g5, %g1, %g1          ;; get upper word (in %g5) into %g1
    Addr[4] = 0x82114001;
    // or %g1, %lo(Target), %g1  ;; get lowest 10 bits of Target into %g1
    Addr[5] = BUILD_ORI(G1, JumpTarget, G1);

    // jmpl %g1, %g0, %g0          ;; indirect branch on %g1
    Addr[6] = 0x81C00001;
    // nop                         ;; delay slot
    Addr[7] = 0x01000000;
  }
}

void SparcV9JITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  InsertJumpAtAddr((intptr_t)New, (unsigned*)Old);
}


static void SaveRegisters(uint64_t DoubleFP[], uint64_t CC[],
                          uint64_t Globals[]) {
#if defined(__sparcv9)

  __asm__ __volatile__ (// Save condition-code registers
                        "stx %%fsr, %0;\n\t"
                        "rd %%fprs, %1;\n\t"
                        "rd %%ccr,  %2;\n\t"
                        : "=m"(CC[0]), "=r"(CC[1]), "=r"(CC[2]));

  __asm__ __volatile__ (// Save globals g1 and g5
                        "stx %%g1, %0;\n\t"
                        "stx %%g5, %0;\n\t"
                        : "=m"(Globals[0]), "=m"(Globals[1]));

  // GCC says: `asm' only allows up to thirty parameters!
  __asm__ __volatile__ (// Save Single/Double FP registers, part 1
                        "std  %%f0,  %0;\n\t"  "std  %%f2,  %1;\n\t"
                        "std  %%f4,  %2;\n\t"  "std  %%f6,  %3;\n\t"
                        "std  %%f8,  %4;\n\t"  "std  %%f10, %5;\n\t"
                        "std  %%f12, %6;\n\t"  "std  %%f14, %7;\n\t"
                        "std  %%f16, %8;\n\t"  "std  %%f18, %9;\n\t"
                        "std  %%f20, %10;\n\t" "std  %%f22, %11;\n\t"
                        "std  %%f24, %12;\n\t" "std  %%f26, %13;\n\t"
                        "std  %%f28, %14;\n\t" "std  %%f30, %15;\n\t"
                        : "=m"(DoubleFP[ 0]), "=m"(DoubleFP[ 1]),
                          "=m"(DoubleFP[ 2]), "=m"(DoubleFP[ 3]),
                          "=m"(DoubleFP[ 4]), "=m"(DoubleFP[ 5]),
                          "=m"(DoubleFP[ 6]), "=m"(DoubleFP[ 7]),
                          "=m"(DoubleFP[ 8]), "=m"(DoubleFP[ 9]),
                          "=m"(DoubleFP[10]), "=m"(DoubleFP[11]),
                          "=m"(DoubleFP[12]), "=m"(DoubleFP[13]),
                          "=m"(DoubleFP[14]), "=m"(DoubleFP[15]));

  __asm__ __volatile__ (// Save Double FP registers, part 2
                        "std %%f32, %0;\n\t"  "std %%f34, %1;\n\t"
                        "std %%f36, %2;\n\t"  "std %%f38, %3;\n\t"
                        "std %%f40, %4;\n\t"  "std %%f42, %5;\n\t"
                        "std %%f44, %6;\n\t"  "std %%f46, %7;\n\t"
                        "std %%f48, %8;\n\t"  "std %%f50, %9;\n\t"
                        "std %%f52, %10;\n\t" "std %%f54, %11;\n\t"
                        "std %%f56, %12;\n\t" "std %%f58, %13;\n\t"
                        "std %%f60, %14;\n\t" "std %%f62, %15;\n\t"
                        : "=m"(DoubleFP[16]), "=m"(DoubleFP[17]),
                          "=m"(DoubleFP[18]), "=m"(DoubleFP[19]),
                          "=m"(DoubleFP[20]), "=m"(DoubleFP[21]),
                          "=m"(DoubleFP[22]), "=m"(DoubleFP[23]),
                          "=m"(DoubleFP[24]), "=m"(DoubleFP[25]),
                          "=m"(DoubleFP[26]), "=m"(DoubleFP[27]),
                          "=m"(DoubleFP[28]), "=m"(DoubleFP[29]),
                          "=m"(DoubleFP[30]), "=m"(DoubleFP[31]));
#else
  std::cerr << "ERROR: RUNNING CODE THAT ONLY WORKS ON A SPARCV9 HOST!\n";
  abort();
#endif
}

static void RestoreRegisters(uint64_t DoubleFP[], uint64_t CC[],
                             uint64_t Globals[]) {
#if defined(__sparcv9)

  __asm__ __volatile__ (// Restore condition-code registers
                        "ldx %0,    %%fsr;\n\t"
                        "wr  %1, 0, %%fprs;\n\t"
                        "wr  %2, 0, %%ccr;\n\t"
                        :: "m"(CC[0]), "r"(CC[1]), "r"(CC[2]));

  __asm__ __volatile__ (// Restore globals g1 and g5
                        "ldx %0, %%g1;\n\t"
                        "ldx %0, %%g5;\n\t"
                        :: "m"(Globals[0]), "m"(Globals[1]));

  // GCC says: `asm' only allows up to thirty parameters!
  __asm__ __volatile__ (// Restore Single/Double FP registers, part 1
                        "ldd %0,  %%f0;\n\t"   "ldd %1, %%f2;\n\t"
                        "ldd %2,  %%f4;\n\t"   "ldd %3, %%f6;\n\t"
                        "ldd %4,  %%f8;\n\t"   "ldd %5, %%f10;\n\t"
                        "ldd %6,  %%f12;\n\t"  "ldd %7, %%f14;\n\t"
                        "ldd %8,  %%f16;\n\t"  "ldd %9, %%f18;\n\t"
                        "ldd %10, %%f20;\n\t" "ldd %11, %%f22;\n\t"
                        "ldd %12, %%f24;\n\t" "ldd %13, %%f26;\n\t"
                        "ldd %14, %%f28;\n\t" "ldd %15, %%f30;\n\t"
                        :: "m"(DoubleFP[0]), "m"(DoubleFP[1]),
                           "m"(DoubleFP[2]), "m"(DoubleFP[3]),
                           "m"(DoubleFP[4]), "m"(DoubleFP[5]),
                           "m"(DoubleFP[6]), "m"(DoubleFP[7]),
                           "m"(DoubleFP[8]), "m"(DoubleFP[9]),
                           "m"(DoubleFP[10]), "m"(DoubleFP[11]),
                           "m"(DoubleFP[12]), "m"(DoubleFP[13]),
                           "m"(DoubleFP[14]), "m"(DoubleFP[15]));

  __asm__ __volatile__ (// Restore Double FP registers, part 2
                        "ldd %0, %%f32;\n\t"  "ldd %1, %%f34;\n\t"
                        "ldd %2, %%f36;\n\t"  "ldd %3, %%f38;\n\t"
                        "ldd %4, %%f40;\n\t"  "ldd %5, %%f42;\n\t"
                        "ldd %6, %%f44;\n\t"  "ldd %7, %%f46;\n\t"
                        "ldd %8, %%f48;\n\t"  "ldd %9, %%f50;\n\t"
                        "ldd %10, %%f52;\n\t" "ldd %11, %%f54;\n\t"
                        "ldd %12, %%f56;\n\t" "ldd %13, %%f58;\n\t"
                        "ldd %14, %%f60;\n\t" "ldd %15, %%f62;\n\t"
                        :: "m"(DoubleFP[16]), "m"(DoubleFP[17]),
                           "m"(DoubleFP[18]), "m"(DoubleFP[19]),
                           "m"(DoubleFP[20]), "m"(DoubleFP[21]),
                           "m"(DoubleFP[22]), "m"(DoubleFP[23]),
                           "m"(DoubleFP[24]), "m"(DoubleFP[25]),
                           "m"(DoubleFP[26]), "m"(DoubleFP[27]),
                           "m"(DoubleFP[28]), "m"(DoubleFP[29]),
                           "m"(DoubleFP[30]), "m"(DoubleFP[31]));
#else
  std::cerr << "ERROR: RUNNING CODE THAT ONLY WORKS ON A SPARCV9 HOST!\n";
  abort();
#endif
}


static void CompilationCallback() {
  // Local space to save the registers
  uint64_t DoubleFP[32];
  uint64_t CC[3];
  uint64_t Globals[2];

  SaveRegisters(DoubleFP, CC, Globals);

  unsigned *CameFrom = (unsigned*)__builtin_return_address(0);
  unsigned *CameFrom1 = (unsigned*)__builtin_return_address(1);

  int64_t Target = (intptr_t)JITCompilerFunction(CameFrom);

  DEBUG(std::cerr << "In callback! Addr=" << (void*)CameFrom << "\n");

  // If we can rewrite the ORIGINAL caller, we eliminate the whole need for a
  // trampoline function stub!!
  unsigned OrigCallInst = *CameFrom1;
  int64_t OrigTarget = (Target-(intptr_t)CameFrom1) >> 2;
  if ((OrigCallInst >> 30) == 1 &&
      (OrigTarget <= (1 << 30) && OrigTarget >= -(1 << 30))) {
    // The original call instruction was CALL <immed>, which means we can
    // overwrite it directly, since the offset will fit into 30 bits
    *CameFrom1 = BUILD_CALL(OrigTarget);
    //++OverwrittenCalls;
  } else {
    //++UnmodifiedCalls;
  }

  // Rewrite the call target so that we don't fault every time we execute it.
  //
  unsigned OrigStubCallInst = *CameFrom;

  // Subtract enough to overwrite up to the 'save' instruction
  // This depends on whether we made a short call (1 instruction) or the
  // farCall (7 instructions)
  int Offset = ((OrigStubCallInst >> 30) == 1) ? 1 : 7;
  unsigned *CodeBegin = CameFrom - Offset;

  // FIXME: __builtin_frame_address doesn't work if frame pointer elimination
  // has been performed.  Having a variable sized alloca disables frame pointer
  // elimination currently, even if it's dead.  This is a gross hack.
  alloca(42+Offset);

  // Make sure that what we're about to overwrite is indeed "save".
  if (*CodeBegin != 0x9DE3BF40) {
    std::cerr << "About to overwrite smthg not a save instr!";
    abort();
  }

  // Overwrite it
  InsertJumpAtAddr(Target, CodeBegin);

  // Flush the I-Cache: FLUSH clears out a doubleword at a given address
  // Self-modifying code MUST clear out the I-Cache to be portable
#if defined(__sparcv9)
  for (int i = -Offset*4, e = 32-((int64_t)Offset*4); i < e; i += 8)
    __asm__ __volatile__ ("flush %%i7 + %0" : : "r" (i));
#endif

  // Change the return address to re-execute the restore, then the jump.
  DEBUG(std::cerr << "Callback returning to: 0x"
                  << std::hex << (CameFrom-Offset*4-12) << "\n");
#if defined(__sparcv9)
  __asm__ __volatile__ ("sub %%i7, %0, %%i7" : : "r" (Offset*4+12));
#endif

  RestoreRegisters(DoubleFP, CC, Globals);
}


/// emitStubForFunction - This method is used by the JIT when it needs to emit
/// the address of a function for a function whose code has not yet been
/// generated.  In order to do this, it generates a stub which jumps to the lazy
/// function compiler, which will eventually get fixed to call the function
/// directly.
///
void *SparcV9JITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
  if (Fn != CompilationCallback) {
    // If this is just a call to an external function,
    MCE.startFunctionStub(4*8);
    unsigned *Stub = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
    for (unsigned i = 0; i != 8; ++i)
      MCE.emitWord(0);
    InsertJumpAtAddr((intptr_t)Fn, Stub);
    return MCE.finishFunctionStub(0); // 1 instr past the restore
  }

  MCE.startFunctionStub(44);
  MCE.emitWord(0x81e82000); // restore %g0, 0, %g0
  MCE.emitWord(0x9DE3BF40); // save %sp, -192, %sp

  int64_t CurrPC = MCE.getCurrentPCValue();
  int64_t Addr = (intptr_t)Fn;
  int64_t CallTarget = (Addr-CurrPC) >> 2;
  if (CallTarget < (1 << 29) && CallTarget > -(1 << 29)) {
    // call CallTarget
    MCE.emitWord((0x01 << 30) | CallTarget);
  } else {
    enum {G5 = 5, G1 = 1 };
    // Otherwise, we need to emit a sequence of instructions to call a distant
    // function.  We use %g5 as a temporary, and compute the value into %g1

    // sethi %uhi(Target), %g5   ;; get upper 22 bits of Target into %g5
    MCE.emitWord(BUILD_SETHI(G5, Addr >> 42));
    // or %g5, %ulo(Target), %g5 ;; get 10 lower bits of upper word into %1
    MCE.emitWord(BUILD_ORI(G5, Addr >> 32, G5));
    // sllx %g5, 32, %g5         ;; shift those 10 bits to the upper word
    MCE.emitWord(0x8B297020);
    // sethi %hi(Target), %g1    ;; extract bits 10-31 into the dest reg
    MCE.emitWord(BUILD_SETHI(G1, Addr >> 10));
    // or %g5, %g1, %g1          ;; get upper word (in %g5) into %g1
    MCE.emitWord(0x82114001);
    // or %g1, %lo(Target), %g1  ;; get lowest 10 bits of Target into %g1
    MCE.emitWord(BUILD_ORI(G1, Addr, G1));

    // call %g1                  ;; indirect call on %g1
    MCE.emitWord(0x9FC04000);
  }

  // nop                         ;; call delay slot
  MCE.emitWord(0x1000000);

  // FIXME: Should have a restore and return!

  MCE.emitWord(0xDEADBEEF);    // marker so that we know it's really a stub
  return (char*)MCE.finishFunctionStub(0)+4; // 1 instr past the restore
}



TargetJITInfo::LazyResolverFn
SparcV9JITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return CompilationCallback;
}

void SparcV9JITInfo::relocate(void *Function, MachineRelocation *MR,
                              unsigned NumRelocs, unsigned char* GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    unsigned *RelocPos = (unsigned*)Function + MR->getMachineCodeOffset()/4;
    intptr_t ResultPtr = (intptr_t)MR->getResultPointer();
    switch ((V9::RelocationType)MR->getRelocationType()) {
    default: assert(0 && "Unknown relocation type!");
    case V9::reloc_pcrel_call:
      ResultPtr = (ResultPtr-(intptr_t)RelocPos) >> 2;   // PC relative.
      assert((ResultPtr < (1 << 29) && ResultPtr > -(1 << 29)) &&
             "reloc_pcrel_call is out of range!");
      // The high two bits of the call are always set to 01.
      *RelocPos = (1 << 30) | (ResultPtr & ((1 << 30)-1)) ;
      break;
    case V9::reloc_sethi_hh:
    case V9::reloc_sethi_lm:
      ResultPtr >>= (MR->getRelocationType() == V9::reloc_sethi_hh ? 32 : 0);
      ResultPtr >>= 10;
      ResultPtr &= (1 << 22)-1;
      *RelocPos |= (unsigned)ResultPtr;
      break;
    case V9::reloc_or_hm:
    case V9::reloc_or_lo:
      ResultPtr >>= (MR->getRelocationType() == V9::reloc_or_hm ? 32 : 0);
      ResultPtr &= (1 << 12)-1;
      *RelocPos |= (unsigned)ResultPtr;
      break;
    }
  }
}
