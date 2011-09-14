//===- Memory.cpp - Memory Handling Support ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines some helpful functions for allocating memory and dealing
// with memory mapped files
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Memory.h"
#include "llvm/Support/Valgrind.h"
#include "llvm/Config/config.h"

namespace llvm {
using namespace sys;
}

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Memory.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Memory.inc"
#endif

extern "C" void sys_icache_invalidate(const void *Addr, size_t len);

/// ClearMipsCache - Invalidates instruction cache for Mips. This assembly code
/// is copied from the MIPS32 Instruction Set Reference. Since the code ends
/// with the return instruction "jr.hb ra" (Jump Register with Hazard Barrier),
/// it must be implemented as a function (which is called from the
/// InvalidateInstructionCache function). It cannot be directly inlined into
/// InvalidateInstructionCache function, because in that case the epilog of
/// InvalidateInstructionCache will not be executed.
#if defined(__mips__)
extern "C" void ClearMipsCache(const void* Addr, size_t Size);
  asm volatile(
    ".text\n"
    ".align 2\n"
    ".globl ClearMipsCache\n"
    "ClearMipsCache:\n"
    ".set       noreorder\n"
    "beq $a1, $zero, 20f\n"          /* If size==0, branch around */
    "nop\n"
    "addu       $a1, $a0, $a1\n"     /* Calculate end address + 1 */
    "rdhwr      $v0, $1\n"           /* Get step size for SYNCI */
                                     /* $1 is $HW_SYNCI_Step */
    "beq        $v0, $zero, 20f\n"   /* If no caches require synchronization, */
                                     /* branch around */
    "nop\n"
    "10: synci  0($a0)\n"            /* Synchronize all caches around address */
    "sltu       $v1, $a0, $a1\n"     /* Compare current with end address */
    "bne        $v1, $zero, 10b\n"   /* Branch if more to do */
    "addu       $a0, $a0, $v0\n"     /* Add step size in delay slot */
    "sync\n"                         /* Clear memory hazards */
    "20: jr.hb      $ra\n"           /* Return, clearing instruction hazards */
    "nop\n"
  );
#endif

/// InvalidateInstructionCache - Before the JIT can run a block of code
/// that has been emitted it must invalidate the instruction cache on some
/// platforms.
void llvm::sys::Memory::InvalidateInstructionCache(const void *Addr,
                                                   size_t Len) {

// icache invalidation for PPC and ARM.
#if defined(__APPLE__)

#  if (defined(__POWERPC__) || defined (__ppc__) || \
     defined(_POWER) || defined(_ARCH_PPC)) || defined(__arm__)
  sys_icache_invalidate(Addr, Len);
#  endif

#else

#  if (defined(__POWERPC__) || defined (__ppc__) || \
       defined(_POWER) || defined(_ARCH_PPC)) && defined(__GNUC__)
  const size_t LineSize = 32;

  const intptr_t Mask = ~(LineSize - 1);
  const intptr_t StartLine = ((intptr_t) Addr) & Mask;
  const intptr_t EndLine = ((intptr_t) Addr + Len + LineSize - 1) & Mask;

  for (intptr_t Line = StartLine; Line < EndLine; Line += LineSize)
    asm volatile("dcbf 0, %0" : : "r"(Line));
  asm volatile("sync");

  for (intptr_t Line = StartLine; Line < EndLine; Line += LineSize)
    asm volatile("icbi 0, %0" : : "r"(Line));
  asm volatile("isync");
#  elif defined(__arm__) && defined(__GNUC__)
  // FIXME: Can we safely always call this for __GNUC__ everywhere?
  char *Start = (char*) Addr;
  char *End = Start + Len;
  __clear_cache(Start, End);
#  elif defined(__mips__)
  ClearMipsCache(Addr, Len);
#  endif

#endif  // end apple

  ValgrindDiscardTranslations(Addr, Len);
}
