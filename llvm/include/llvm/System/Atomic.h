//===- llvm/System/Atomic.h - Atomic Operations -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys atomic operations.
//
// Portions of this file use code from libatomic_ops, for which the following 
// license applies:
//
// Copyright (c) 2003 by Hewlett-Packard Company.  All rights reserved.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_ATOMIC_H
#define LLVM_SYSTEM_ATOMIC_H

#include <stdint.h>

#if defined(_HPUX_SOURCE) && defined(__ia64)
#include <machine/sys/inline.h>
#elif defined(_MSC_VER)
#include <windows.h>
#endif // defined(_HPUX_SOURCE) && defined(__ia64)


namespace llvm {
  namespace sys {
    
    inline void CompilerFence() {
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
      __asm__ __volatile__("" : : : "memory");
#elif defined(_MSC_VER)
      __asm { };
#elif defined(__INTEL_COMPILER)
      __memory_barrier(); /* Too strong? IA64-only? */
#else
    /* We conjecture that the following usually gives us the right 	*/
    /* semantics or an error.						*/
      asm("");
#endif // defined(__GNUC__) && !defined(__INTEL_COMPILER)
}

#if !defined(ENABLE_THREADS) || ENABLE_THREADS == 0
    inline void MemoryFence() {
      CompilerFence();
    }

    typedef uint32_t cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* dest, cas_flag exc, cas_flag c) {
      cas_flag result = *dest;
      if (result == c)
        *dest = exc;
      return result;
    }

#elif defined(__GNUC__)

    inline void MemoryFence() {
#  if defined(__i386__) || defined(__x86_64__)
#    if defined(__SSE2__)
      __asm__ __volatile__("mfence" : : : "memory");
#    else
      unsigned char dummy = 0;
      volatile unsigned char* addr = &dummy;
      unsigned char oldval;
      __asm __ __volatile__("xchgb %0, %1" : "=r"(oldval),
        "=m"(*addr), "0"(0xff), "m"(*addr) : "memory");
#    endif // defined(__SSE2__)
#  elif defined(__ia64__)
      __asm__ __volatile__("mf" : : : "memory");
# elif defined(__alpha__)
      __asm__ __volatile__("mb" : : : "memory");
# elif defined(__sparc__)
      __asm__ __volatile__("membar	#StoreStore | #LoadStore | #LoadLoad | #StoreLoad");
# elif defined(__powerpc__) || defined(__ppc__)
      __asm__ __volatile__("sync" : : : "memory");
# elif defined(__arm__)
      __asm__ __volatile__ ("mcr	p15, 0, r0, c7, c10, 5	@ dmb");
# endif
    } // defined(__i386__) || defined(__x86_64__)
    
    typedef unsigned long cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* ptr,
                                   cas_flag new_value,
                                   cas_flag old_value) {
      cas_flag prev;
#  if defined(__i386__) || defined(__x86_64__)
      __asm__ __volatile__("lock; cmpxchgl %1,%2"
                           : "=a" (prev)
                           : "q" (new_value), "m" (*ptr), "0" (old_value)
                           : "memory");
#  elif defined(__ia64__)
      MemoryFence();
#  if defined(_ILP32)
      __asm__("zxt4 %1=%1": "=r"(prev) : "0"(prev));
      __asm__ __volatile__("addp4 %1=0,%1;;\n"
        "mov ar.ccv=%[old] ;; cmpxchg 4"
        ".acq %0=[%1],%[new_val],ar.ccv"
        : "=r"(prev) "1"(addr),
        : "=r"(addr), [new_value]"r"(new_value), [old_value]"r"(old_value)
        : "memory");
#  else
      __asm__ __volatile__(
        "mov ar.ccv=%[old] ;; cmpxchg 8"
        ".acq %0=[%1],%[new_val],ar.ccv"
        : "=r"(prev)
        : "r"(ptr), [new_value]"r"(new_value), 
        [old_value]"r"(old_value)
        : "memory");
#  endif // defined(_ILP32)
#  elif defined(__alpha__)
      cas_flag was_equal;
      __asm__ __volatile__(
        "1:     ldq_l %0,%1\n"
        "       cmpeq %0,%4,%2\n"
        "	     mov %3,%0\n"
        "       beq %2,2f\n"
        "       stq_c %0,%1\n"
        "       beq %0,1b\n"
        "2:\n"
        :"=&r" (prev), "=m" (*ptr), "=&r" (was_equal)
        : "r" (new_value), "Ir" (old_value)
        :"memory");
#elif defined(__sparc__)
#error No CAS implementation for SPARC yet.
#elif defined(__powerpc__) || defined(__ppc__)
      int result = 0;
      __asm__ __volatile__(
        "1:lwarx %0,0,%2\n"   /* load and reserve              */
        "cmpw %0, %4\n"      /* if load is not equal to 	*/
        "bne 2f\n"            /*   old, fail			*/
        "stwcx. %3,0,%2\n"    /* else store conditional         */
        "bne- 1b\n"           /* retry if lost reservation      */
        "li %1,1\n"	     /* result = 1;			*/
        "2:\n"
        : "=&r"(prev), "=&r"(result)
        : "r"(ptr), "r"(new_value), "r"(old_value), "1"(result)
        : "memory", "cc");
#elif defined(__arm__)
      int result;
      __asm__ __volatile__ (
        "\n"
        "0:\t"
        "ldr     %1,[%2] \n\t"
        "mov     %0,#0 \n\t"
        "cmp     %1,%4 \n\t"
        "bne     1f \n\t"
        "swp     %0,%3,[%2] \n\t"
        "cmp     %1,%0 \n\t"
        "swpne   %1,%0,[%2] \n\t"
        "bne     0b \n\t"
        "mov     %0,#1 \n"
        "1:\n\t"
        ""
        : "=&r"(result), "=&r"(prev) 
        : "r" ptr), "r" (new_value), "r" (old_value) 
        : "cc", "memory");
#endif // defined(__i386__)
      return prev;
    }

#elif defined(_MSC_VER) && _M_IX86 > 400
    inline void MemoryFence() {
      LONG dummy = 0;
      InterlockedExchanged((LONG volatile *)&dummy, (LONG)0);
    }
    
    typedef DWORD cas_flag;
    inline cas_flag CompareAndSwap(cas_flag* ptr,
                                   cas_flag new_value,
                                   cas_flag old_value) {
      /* FIXME - This is nearly useless on win64.			*/
      /* Use InterlockedCompareExchange64 for win64?	*/
      return InterlockedCompareExchange((DWORD volatile *)addr,
                                        (DWORD)new_value, (DWORD) old_value)
    }
#else
#error No atomics implementation found for your platform.
#endif // !defined(ENABLE_THREADS) || ENABLE_THREADS == 0

  }
}

#endif
