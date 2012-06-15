//===-- sanitizer_common.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// It declares common functions and classes that are used in both runtimes.
// Implementation of some functions are provided in sanitizer_common, while
// others must be defined by run-time library itself.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_COMMON_H
#define SANITIZER_COMMON_H

#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// Constants.
const uptr kWordSize = __WORDSIZE / 8;
const uptr kWordSizeInBits = 8 * kWordSize;
const uptr kPageSizeBits = 12;
const uptr kPageSize = 1UL << kPageSizeBits;
#ifndef _WIN32
const uptr kMmapGranularity = kPageSize;
#else
const uptr kMmapGranularity = 1UL << 16;
#endif

// Threads
int GetPid();
uptr GetThreadSelf();
void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom);

// Memory management
void *MmapOrDie(uptr size, const char *mem_type);
void UnmapOrDie(void *addr, uptr size);
void *MmapFixedNoReserve(uptr fixed_addr, uptr size);
void *Mprotect(uptr fixed_addr, uptr size);
// Used to check if we can map shadow memory to a fixed location.
bool MemoryRangeIsAvailable(uptr range_start, uptr range_end);

// Internal allocator
void *InternalAlloc(uptr size);
void InternalFree(void *addr);

// IO
void RawWrite(const char *buffer);
void Printf(const char *format, ...);
int SNPrintf(char *buffer, uptr length, const char *format, ...);
void Report(const char *format, ...);

// Opens the file 'file_name" and reads up to 'max_len' bytes.
// The resulting buffer is mmaped and stored in '*buff'.
// The size of the mmaped region is stored in '*buff_size',
// Returns the number of read bytes or 0 if file can not be opened.
uptr ReadFileToBuffer(const char *file_name, char **buff,
                      uptr *buff_size, uptr max_len);
const char *GetEnv(const char *name);

// Other
void DisableCoreDumper();
void DumpProcessMap();
void SleepForSeconds(int seconds);
void NORETURN Exit(int exitcode);
void NORETURN Abort();
int Atexit(void (*function)(void));
void SortArray(uptr *array, uptr size);

// Atomics
int AtomicInc(int *a);
u16 AtomicExchange(u16 *a, u16 new_val);
u8 AtomicExchange(u8 *a, u8 new_val);

// Math
inline bool IsPowerOfTwo(uptr x) {
  return (x & (x - 1)) == 0;
}
inline uptr RoundUpTo(uptr size, uptr boundary) {
  CHECK(IsPowerOfTwo(boundary));
  return (size + boundary - 1) & ~(boundary - 1);
}
// Don't use std::min and std::max, to minimize dependency on libstdc++.
template<class T> T Min(T a, T b) { return a < b ? a : b; }
template<class T> T Max(T a, T b) { return a > b ? a : b; }

#if __WORDSIZE == 64
# define FIRST_32_SECOND_64(a, b) (b)
#else
# define FIRST_32_SECOND_64(a, b) (a)
#endif

}  // namespace __sanitizer

#endif  // SANITIZER_COMMON_H
