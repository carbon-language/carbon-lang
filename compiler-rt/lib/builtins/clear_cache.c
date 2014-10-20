/* ===-- clear_cache.c - Implement __clear_cache ---------------------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */

#include "int_lib.h"

#if __APPLE__
  #include <libkern/OSCacheControl.h>
#endif
#if defined(__NetBSD__) && defined(__arm__)
  #include <machine/sysarch.h>
#endif

#if defined(__ANDROID__) && defined(__mips__)
  #include <sys/cachectl.h>
#endif

#if defined(__ANDROID__) && defined(__arm__)
  #include <asm/unistd.h>
#endif

/*
 * The compiler generates calls to __clear_cache() when creating 
 * trampoline functions on the stack for use with nested functions.
 * It is expected to invalidate the instruction cache for the 
 * specified range.
 */

void __clear_cache(void *start, void *end) {
#if __i386__ || __x86_64__
/*
 * Intel processors have a unified instruction and data cache
 * so there is nothing to do
 */
#elif defined(__arm__) && !defined(__APPLE__)
    #if defined(__NetBSD__)
        struct arm_sync_icache_args arg;

        arg.addr = (uintptr_t)start;
        arg.len = (uintptr_t)end - (uintptr_t)start;

        sysarch(ARM_SYNC_ICACHE, &arg);
    #elif defined(__ANDROID__)
         const register int start_reg __asm("r0") = (int) (intptr_t) start;
         const register int end_reg __asm("r1") = (int) (intptr_t) end;
         const register int flags __asm("r2") = 0;
         const register int syscall_nr __asm("r7") = __ARM_NR_cacheflush;
        __asm __volatile("svc 0x0" : "=r"(start_reg)
            : "r"(syscall_nr), "r"(start_reg), "r"(end_reg), "r"(flags) : "r0");
         if (start_reg != 0) {
             compilerrt_abort();
         }
    #else
        compilerrt_abort();
    #endif
#elif defined(__ANDROID__) && defined(__mips__)
  const uintptr_t start_int = (uintptr_t) start;
  const uintptr_t end_int = (uintptr_t) end;
  _flush_cache(start, (end_int - start_int), BCACHE);
#elif defined(__aarch64__) && !defined(__APPLE__)
  uint64_t xstart = (uint64_t)(uintptr_t) start;
  uint64_t xend = (uint64_t)(uintptr_t) end;

  // Get Cache Type Info
  uint64_t ctr_el0;
  __asm __volatile("mrs %0, ctr_el0" : "=r"(ctr_el0));

  /*
   * dc & ic instructions must use 64bit registers so we don't use
   * uintptr_t in case this runs in an IPL32 environment.
   */
  const size_t dcache_line_size = 4 << ((ctr_el0 >> 16) & 15);
  for (uint64_t addr = xstart; addr < xend; addr += dcache_line_size)
    __asm __volatile("dc cvau, %0" :: "r"(addr));
  __asm __volatile("dsb ish");

  const size_t icache_line_size = 4 << ((ctr_el0 >> 0) & 15);
  for (uint64_t addr = xstart; addr < xend; addr += icache_line_size)
    __asm __volatile("ic ivau, %0" :: "r"(addr));
  __asm __volatile("isb sy");
#else
    #if __APPLE__
        /* On Darwin, sys_icache_invalidate() provides this functionality */
        sys_icache_invalidate(start, end-start);
    #else
        compilerrt_abort();
    #endif
#endif
}

