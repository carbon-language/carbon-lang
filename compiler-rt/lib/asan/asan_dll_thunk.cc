//===-- asan_dll_thunk.cc -------------------------------------------------===//
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
// This file defines a family of thunks that should be statically linked into
// the DLLs that have ASan instrumentation in order to delegate the calls to the
// shared runtime that lives in the main binary.
// See https://code.google.com/p/address-sanitizer/issues/detail?id=209 for the
// details.
//===----------------------------------------------------------------------===//

// Only compile this code when buidling asan_dll_thunk.lib
// Using #ifdef rather than relying on Makefiles etc.
// simplifies the build procedure.
#ifdef ASAN_DLL_THUNK
#include <windows.h>

// ------------------------- Helper macros ------------------ {{{1

static FARPROC getRealProcAddressOrDie(const char *name) {
  FARPROC ret = GetProcAddress(GetModuleHandle(NULL), name);
  if (!ret)
    abort();
  return ret;
}

#define WRAP_VV(name)                                                          \
  extern "C" void name() {                                                     \
    typedef void (*fntype)();                                                  \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    fn();                                                                      \
  }

#define WRAP_VW(name)                                                          \
  extern "C" void name(void *arg) {                                            \
    typedef void (*fntype)(void *arg);                                         \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    fn(arg);                                                                   \
  }

#define WRAP_VWW(name)                                                         \
  extern "C" void name(void *arg1, void *arg2) {                               \
    typedef void (*fntype)(void *, void *);                                    \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    fn(arg1, arg2);                                                            \
  }
// }}}

WRAP_VV(__asan_init_v3)

WRAP_VW(__asan_report_store1)
WRAP_VW(__asan_report_store2)
WRAP_VW(__asan_report_store4)
WRAP_VW(__asan_report_store8)
WRAP_VW(__asan_report_store16)
WRAP_VWW(__asan_report_store_n)

WRAP_VW(__asan_report_load1)
WRAP_VW(__asan_report_load2)
WRAP_VW(__asan_report_load4)
WRAP_VW(__asan_report_load8)
WRAP_VW(__asan_report_load16)
WRAP_VWW(__asan_report_load_n)

WRAP_VWW(__asan_register_globals)
WRAP_VWW(__asan_unregister_globals)

// TODO(timurrrr): Add more interface functions on the as-needed basis.

// TODO(timurrrr): Add malloc & friends (see asan_malloc_win.cc).

#endif // ASAN_DLL_THUNK
