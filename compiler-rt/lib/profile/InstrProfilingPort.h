/*===- InstrProfilingPort.h- Support library for PGO instrumentation ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#ifndef PROFILE_INSTRPROFILING_PORT_H_
#define PROFILE_INSTRPROFILING_PORT_H_

#ifdef _MSC_VER
# define LLVM_ALIGNAS(x) __declspec(align(x))
#elif __GNUC__
#define LLVM_ALIGNAS(x) __attribute__((aligned(x)))
#endif

#define LLVM_LIBRARY_VISIBILITY __attribute__((visibility("hidden")))
#define LLVM_SECTION(Sect) __attribute__((section(Sect)))

#define PROF_ERR(Format, ...)                                                  \
  if (GetEnvHook && GetEnvHook("LLVM_PROFILE_VERBOSE_ERRORS"))                 \
    fprintf(stderr, Format, __VA_ARGS__);

#if COMPILER_RT_HAS_ATOMICS == 1
#define BOOL_CMPXCHG(Ptr, OldV, NewV)                                          \
  __sync_bool_compare_and_swap(Ptr, OldV, NewV)
#else
#define BOOL_CMPXCHG(Ptr, OldV, NewV) BoolCmpXchg((void **)Ptr, OldV, NewV)
#endif

#if defined(__FreeBSD__) && defined(__i386__)

/* System headers define 'size_t' incorrectly on x64 FreeBSD (prior to
 * FreeBSD 10, r232261) when compiled in 32-bit mode.
 */
#define PRIu64 "llu"
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef uint32_t uintptr_t;
#elif defined(__FreeBSD__) && defined(__x86_64__)
#define PRIu64 "lu"
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long int uintptr_t;

#else /* defined(__FreeBSD__) && defined(__i386__) */

#include <inttypes.h>
#include <stdint.h>

#endif /* defined(__FreeBSD__) && defined(__i386__) */

#endif /* PROFILE_INSTRPROFILING_PORT_H_ */
