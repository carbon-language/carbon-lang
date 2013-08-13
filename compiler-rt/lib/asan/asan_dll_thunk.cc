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

// ----------------- Helper functions and macros --------------------- {{{1
extern "C" {
void *__stdcall GetModuleHandleA(const char *module_name);
void *__stdcall GetProcAddress(void *module, const char *proc_name);
void abort();
}

static void *getRealProcAddressOrDie(const char *name) {
  void *ret = GetProcAddress(GetModuleHandleA(0), name);
  if (!ret)
    abort();
  return ret;
}

#define WRAP_V_V(name)                                                         \
  extern "C" void name() {                                                     \
    typedef void (*fntype)();                                                  \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    fn();                                                                      \
  }

#define WRAP_V_W(name)                                                         \
  extern "C" void name(void *arg) {                                            \
    typedef void (*fntype)(void *arg);                                         \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    fn(arg);                                                                   \
  }

#define WRAP_V_WW(name)                                                        \
  extern "C" void name(void *arg1, void *arg2) {                               \
    typedef void (*fntype)(void *, void *);                                    \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    fn(arg1, arg2);                                                            \
  }

#define WRAP_W_W(name)                                                         \
  extern "C" void *name(void *arg) {                                           \
    typedef void *(*fntype)(void *arg);                                        \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    return fn(arg);                                                            \
  }

#define WRAP_W_WW(name)                                                        \
  extern "C" void *name(void *arg1, void *arg2) {                              \
    typedef void *(*fntype)(void *, void *);                                   \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    return fn(arg1, arg2);                                                     \
  }

#define WRAP_W_WWW(name)                                                       \
  extern "C" void *name(void *arg1, void *arg2, void *arg3) {                  \
    typedef void *(*fntype)(void *, void *, void *);                           \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    return fn(arg1, arg2, arg3);                                               \
  }

#define WRAP_W_WWWW(name)                                                      \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4) {      \
    typedef void *(*fntype)(void *, void *, void *, void *);                   \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    return fn(arg1, arg2, arg3, arg4);                                         \
  }

#define WRAP_W_WWWWW(name)                                                     \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4,        \
                        void *arg5) {                                          \
    typedef void *(*fntype)(void *, void *, void *, void *, void *);           \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    return fn(arg1, arg2, arg3, arg4, arg5);                                   \
  }

#define WRAP_W_WWWWWW(name)                                                    \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4,        \
                        void *arg5, void *arg6) {                              \
    typedef void *(*fntype)(void *, void *, void *, void *, void *, void *);   \
    fntype fn = (fntype)getRealProcAddressOrDie(#name);                        \
    return fn(arg1, arg2, arg3, arg4, arg5, arg6);                             \
  }
// }}}

// ----------------- ASan own interface functions --------------------
WRAP_V_V(__asan_init_v3)

WRAP_V_W(__asan_report_store1)
WRAP_V_W(__asan_report_store2)
WRAP_V_W(__asan_report_store4)
WRAP_V_W(__asan_report_store8)
WRAP_V_W(__asan_report_store16)
WRAP_V_WW(__asan_report_store_n)

WRAP_V_W(__asan_report_load1)
WRAP_V_W(__asan_report_load2)
WRAP_V_W(__asan_report_load4)
WRAP_V_W(__asan_report_load8)
WRAP_V_W(__asan_report_load16)
WRAP_V_WW(__asan_report_load_n)

WRAP_V_WW(__asan_register_globals)
WRAP_V_WW(__asan_unregister_globals)

// TODO(timurrrr): Add more interface functions on the as-needed basis.

// ----------------- Memory allocation functions ---------------------
WRAP_V_W(free)
WRAP_V_WW(_free_dbg)

WRAP_W_W(malloc)
WRAP_W_WWWW(_malloc_dbg)

WRAP_W_WW(calloc)
WRAP_W_WWWWW(_calloc_dbg)
WRAP_W_WWW(_calloc_impl)

WRAP_W_WW(realloc)
WRAP_W_WWW(_realloc_dbg)
WRAP_W_WWW(_recalloc)

WRAP_W_W(_msize)

// TODO(timurrrr): Do we need to add _Crt* stuff here? (see asan_malloc_win.cc).

#endif // ASAN_DLL_THUNK
