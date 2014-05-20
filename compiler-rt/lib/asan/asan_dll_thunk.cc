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
#include "sanitizer_common/sanitizer_interception.h"

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
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn();                                                                      \
  }

#define WRAP_V_W(name)                                                         \
  extern "C" void name(void *arg) {                                            \
    typedef void (*fntype)(void *arg);                                         \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn(arg);                                                                   \
  }

#define WRAP_V_WW(name)                                                        \
  extern "C" void name(void *arg1, void *arg2) {                               \
    typedef void (*fntype)(void *, void *);                                    \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn(arg1, arg2);                                                            \
  }

#define WRAP_V_WWW(name)                                                       \
  extern "C" void name(void *arg1, void *arg2, void *arg3) {                   \
    typedef void *(*fntype)(void *, void *, void *);                           \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    fn(arg1, arg2, arg3);                                                      \
  }

#define WRAP_W_V(name)                                                         \
  extern "C" void *name() {                                                    \
    typedef void *(*fntype)();                                                 \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn();                                                               \
  }

#define WRAP_W_W(name)                                                         \
  extern "C" void *name(void *arg) {                                           \
    typedef void *(*fntype)(void *arg);                                        \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg);                                                            \
  }

#define WRAP_W_WW(name)                                                        \
  extern "C" void *name(void *arg1, void *arg2) {                              \
    typedef void *(*fntype)(void *, void *);                                   \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2);                                                     \
  }

#define WRAP_W_WWW(name)                                                       \
  extern "C" void *name(void *arg1, void *arg2, void *arg3) {                  \
    typedef void *(*fntype)(void *, void *, void *);                           \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3);                                               \
  }

#define WRAP_W_WWWW(name)                                                      \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4) {      \
    typedef void *(*fntype)(void *, void *, void *, void *);                   \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3, arg4);                                         \
  }

#define WRAP_W_WWWWW(name)                                                     \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4,        \
                        void *arg5) {                                          \
    typedef void *(*fntype)(void *, void *, void *, void *, void *);           \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3, arg4, arg5);                                   \
  }

#define WRAP_W_WWWWWW(name)                                                    \
  extern "C" void *name(void *arg1, void *arg2, void *arg3, void *arg4,        \
                        void *arg5, void *arg6) {                              \
    typedef void *(*fntype)(void *, void *, void *, void *, void *, void *);   \
    static fntype fn = (fntype)getRealProcAddressOrDie(#name);                 \
    return fn(arg1, arg2, arg3, arg4, arg5, arg6);                             \
  }
// }}}

// --------- Interface interception helper functions and macros ----------- {{{1
// We need to intercept the ASan interface exported by the DLL thunk and forward
// all the functions to the runtime in the main module.
// However, we don't want to keep two lists of interface functions.
// To avoid that, the list of interface functions should be defined using the
// INTERFACE_FUNCTION macro. Then, all the interface can be intercepted at once
// by calling INTERCEPT_ASAN_INTERFACE().

// Use macro+template magic to automatically generate the list of interface
// functions.  Each interface function at line LINE defines a template class
// with a static InterfaceInteceptor<LINE>::Execute() method intercepting the
// function.  The default implementation of InterfaceInteceptor<LINE> is to call
// the Execute() method corresponding to the previous line.
template<int LINE>
struct InterfaceInteceptor {
  static void Execute() { InterfaceInteceptor<LINE-1>::Execute(); }
};

// There shouldn't be any interface function with negative line number.
template<>
struct InterfaceInteceptor<0> {
  static void Execute() {}
};

#define INTERFACE_FUNCTION(name)                                               \
  extern "C" void name() { __debugbreak(); }                                   \
  template<> struct InterfaceInteceptor<__LINE__> {                            \
    static void Execute() {                                                    \
      void *wrapper = getRealProcAddressOrDie(#name);                          \
      if (!__interception::OverrideFunction((uptr)name, (uptr)wrapper, 0))     \
        abort();                                                               \
      InterfaceInteceptor<__LINE__-1>::Execute();                              \
    }                                                                          \
  };

// INTERCEPT_ASAN_INTERFACE must be used after the last INTERFACE_FUNCTION.
#define INTERCEPT_ASAN_INTERFACE InterfaceInteceptor<__LINE__>::Execute

static void InterceptASanInterface();
// }}}

// ----------------- ASan own interface functions --------------------
// Don't use the INTERFACE_FUNCTION machinery for this function as we actually
// want to call it in the __asan_init interceptor.
WRAP_W_V(__asan_should_detect_stack_use_after_return)

extern "C" {
  int __asan_option_detect_stack_use_after_return;

  // Manually wrap __asan_init as we need to initialize
  // __asan_option_detect_stack_use_after_return afterwards.
  void __asan_init_v3() {
    typedef void (*fntype)();
    static fntype fn = 0;
    if (fn) return;

    fn = (fntype)getRealProcAddressOrDie("__asan_init_v3");
    fn();
    __asan_option_detect_stack_use_after_return =
        (__asan_should_detect_stack_use_after_return() != 0);

    InterceptASanInterface();
  }
}

INTERFACE_FUNCTION(__asan_handle_no_return)

INTERFACE_FUNCTION(__asan_report_store1)
INTERFACE_FUNCTION(__asan_report_store2)
INTERFACE_FUNCTION(__asan_report_store4)
INTERFACE_FUNCTION(__asan_report_store8)
INTERFACE_FUNCTION(__asan_report_store16)
INTERFACE_FUNCTION(__asan_report_store_n)

INTERFACE_FUNCTION(__asan_report_load1)
INTERFACE_FUNCTION(__asan_report_load2)
INTERFACE_FUNCTION(__asan_report_load4)
INTERFACE_FUNCTION(__asan_report_load8)
INTERFACE_FUNCTION(__asan_report_load16)
INTERFACE_FUNCTION(__asan_report_load_n)

INTERFACE_FUNCTION(__asan_memcpy);
INTERFACE_FUNCTION(__asan_memset);
INTERFACE_FUNCTION(__asan_memmove);

INTERFACE_FUNCTION(__asan_register_globals)
INTERFACE_FUNCTION(__asan_unregister_globals)

INTERFACE_FUNCTION(__asan_before_dynamic_init)
INTERFACE_FUNCTION(__asan_after_dynamic_init)

INTERFACE_FUNCTION(__asan_poison_stack_memory)
INTERFACE_FUNCTION(__asan_unpoison_stack_memory)

INTERFACE_FUNCTION(__asan_poison_memory_region)
INTERFACE_FUNCTION(__asan_unpoison_memory_region)

INTERFACE_FUNCTION(__asan_get_current_fake_stack)
INTERFACE_FUNCTION(__asan_addr_is_in_fake_stack)

INTERFACE_FUNCTION(__asan_stack_malloc_0)
INTERFACE_FUNCTION(__asan_stack_malloc_1)
INTERFACE_FUNCTION(__asan_stack_malloc_2)
INTERFACE_FUNCTION(__asan_stack_malloc_3)
INTERFACE_FUNCTION(__asan_stack_malloc_4)
INTERFACE_FUNCTION(__asan_stack_malloc_5)
INTERFACE_FUNCTION(__asan_stack_malloc_6)
INTERFACE_FUNCTION(__asan_stack_malloc_7)
INTERFACE_FUNCTION(__asan_stack_malloc_8)
INTERFACE_FUNCTION(__asan_stack_malloc_9)
INTERFACE_FUNCTION(__asan_stack_malloc_10)

INTERFACE_FUNCTION(__asan_stack_free_0)
INTERFACE_FUNCTION(__asan_stack_free_1)
INTERFACE_FUNCTION(__asan_stack_free_2)
INTERFACE_FUNCTION(__asan_stack_free_4)
INTERFACE_FUNCTION(__asan_stack_free_5)
INTERFACE_FUNCTION(__asan_stack_free_6)
INTERFACE_FUNCTION(__asan_stack_free_7)
INTERFACE_FUNCTION(__asan_stack_free_8)
INTERFACE_FUNCTION(__asan_stack_free_9)
INTERFACE_FUNCTION(__asan_stack_free_10)

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
WRAP_W_W(_expand)
WRAP_W_W(_expand_dbg)

// TODO(timurrrr): Might want to add support for _aligned_* allocation
// functions to detect a bit more bugs.  Those functions seem to wrap malloc().

// TODO(timurrrr): Do we need to add _Crt* stuff here? (see asan_malloc_win.cc).

void InterceptASanInterface() {
  INTERCEPT_ASAN_INTERFACE();
}

#endif // ASAN_DLL_THUNK
