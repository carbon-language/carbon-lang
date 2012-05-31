//===-- asan_malloc_win.cc --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Windows-specific malloc interception.
//===----------------------------------------------------------------------===//
#ifdef _WIN32

#include "asan_allocator.h"
#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_stack.h"

#include "interception/interception.h"

// ---------------------- Replacement functions ---------------- {{{1
using namespace __asan;  // NOLINT

// FIXME: Simply defining functions with the same signature in *.obj
// files overrides the standard functions in *.lib
// This works well for simple helloworld-like tests but might need to be
// revisited in the future.

extern "C" {
void free(void *ptr) {
  GET_STACK_TRACE_HERE_FOR_FREE(ptr);
  return asan_free(ptr, &stack);
}

void _free_dbg(void* ptr, int) {
  free(ptr);
}

void cfree(void *ptr) {
  CHECK(!"cfree() should not be used on Windows?");
}

void *malloc(size_t size) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_malloc(size, &stack);
}

void* _malloc_dbg(size_t size, int , const char*, int) {
  return malloc(size);
}

void *calloc(size_t nmemb, size_t size) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_calloc(nmemb, size, &stack);
}

void* _calloc_dbg(size_t n, size_t size, int, const char*, int) {
  return calloc(n, size);
}

void *_calloc_impl(size_t nmemb, size_t size, int *errno_tmp) {
  return calloc(nmemb, size);
}

void *realloc(void *ptr, size_t size) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_realloc(ptr, size, &stack);
}

void *_realloc_dbg(void *ptr, size_t size, int) {
  CHECK(!"_realloc_dbg should not exist!");
  return 0;
}

void* _recalloc(void* p, size_t n, size_t elem_size) {
  if (!p)
    return calloc(n, elem_size);
  const size_t size = n * elem_size;
  if (elem_size != 0 && size / elem_size != n)
    return 0;
  return realloc(p, size);
}

size_t _msize(void *ptr) {
  GET_STACK_TRACE_HERE_FOR_MALLOC;
  return asan_malloc_usable_size(ptr, &stack);
}

int _CrtDbgReport(int, const char*, int,
                  const char*, const char*, ...) {
  ShowStatsAndAbort();
}

int _CrtDbgReportW(int reportType, const wchar_t*, int,
                   const wchar_t*, const wchar_t*, ...) {
  ShowStatsAndAbort();
}

int _CrtSetReportMode(int, int) {
  return 0;
}
}  // extern "C"

using __interception::GetRealFunctionAddress;

// We don't want to include "windows.h" in this file to avoid extra attributes
// set on malloc/free etc (e.g. dllimport), so declare a few things manually:
extern "C" int __stdcall VirtualProtect(void* addr, size_t size,
                                        DWORD prot, DWORD *old_prot);
const int PAGE_EXECUTE_READWRITE = 0x40;

namespace __asan {
void ReplaceSystemMalloc() {
#if defined(_DLL)
# ifdef _WIN64
#  error ReplaceSystemMalloc was not tested on x64
# endif
  char *crt_malloc;
  if (GetRealFunctionAddress("malloc", (void**)&crt_malloc)) {
    // Replace malloc in the CRT dll with a jump to our malloc.
    DWORD old_prot, unused;
    CHECK(VirtualProtect(crt_malloc, 16, PAGE_EXECUTE_READWRITE, &old_prot));
    REAL(memset)(crt_malloc, 0xCC /* int 3 */, 16);  // just in case.

    ptrdiff_t jmp_offset = (char*)malloc - (char*)crt_malloc - 5;
    crt_malloc[0] = 0xE9;  // jmp, should be followed by an offset.
    REAL(memcpy)(crt_malloc + 1, &jmp_offset, sizeof(jmp_offset));

    CHECK(VirtualProtect(crt_malloc, 16, old_prot, &unused));

    // FYI: FlushInstructionCache is needed on Itanium etc but not on x86/x64.
  }

  // FIXME: investigate whether anything else is needed.
#endif
}
}  // namespace __asan

#endif  // _WIN32
