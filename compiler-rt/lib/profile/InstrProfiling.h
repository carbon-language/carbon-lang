/*===- InstrProfiling.h- Support library for PGO instrumentation ----------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include <stdio.h>
#include <stdlib.h>

#define I386_FREEBSD (defined(__FreeBSD__) && defined(__i386__))

#if !I386_FREEBSD
#include <inttypes.h>
#endif

#if !defined(_MSC_VER) && !I386_FREEBSD
#include <stdint.h>
#endif

#if defined(_MSC_VER)
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#elif I386_FREEBSD
/* System headers define 'size_t' incorrectly on x64 FreeBSD (prior to
 * FreeBSD 10, r232261) when compiled in 32-bit mode.
 */
#define PRIu64 "llu"
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#endif

typedef struct __llvm_pgo_data {
  const uint32_t NameSize;
  const uint32_t NumCounters;
  const uint64_t FuncHash;
  const char *const Name;
  const uint64_t *const Counters;
} __llvm_pgo_data;

/* TODO: void __llvm_pgo_get_size_for_buffer(void);  */

/*!
 * \brief Write instrumentation data to the given buffer.
 *
 * This function is currently broken:  it shouldn't rely on libc, but it does.
 * It should be changed to take a char* buffer, and write binary data directly
 * to it.
 */
void __llvm_pgo_write_buffer(FILE *OutputFile);
