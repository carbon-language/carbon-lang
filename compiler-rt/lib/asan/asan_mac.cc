//===-- asan_mac.cc -------------------------------------------------------===//
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
// Mac-specific details.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"
#if SANITIZER_MAC

#include "asan_interceptors.h"
#include "asan_internal.h"
#include "asan_mapping.h"
#include "asan_stack.h"
#include "asan_thread.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_mac.h"

#if !SANITIZER_IOS
#include <crt_externs.h>  // for _NSGetArgv and _NSGetEnviron
#else
extern "C" {
  extern char ***_NSGetArgv(void);
}
#endif

#include <dlfcn.h>  // for dladdr()
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/ucontext.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>  // for free()
#include <unistd.h>
#include <libkern/OSAtomic.h>

namespace __asan {

void InitializePlatformInterceptors() {}

bool PlatformHasDifferentMemcpyAndMemmove() {
  // On OS X 10.7 memcpy() and memmove() are both resolved
  // into memmove$VARIANT$sse42.
  // See also http://code.google.com/p/address-sanitizer/issues/detail?id=34.
  // TODO(glider): need to check dynamically that memcpy() and memmove() are
  // actually the same function.
  return GetMacosVersion() == MACOS_VERSION_SNOW_LEOPARD;
}

extern "C"
void __asan_init();

static const char kDyldInsertLibraries[] = "DYLD_INSERT_LIBRARIES";
LowLevelAllocator allocator_for_env;

// Change the value of the env var |name|, leaking the original value.
// If |name_value| is NULL, the variable is deleted from the environment,
// otherwise the corresponding "NAME=value" string is replaced with
// |name_value|.
void LeakyResetEnv(const char *name, const char *name_value) {
  char **env = GetEnviron();
  uptr name_len = internal_strlen(name);
  while (*env != 0) {
    uptr len = internal_strlen(*env);
    if (len > name_len) {
      const char *p = *env;
      if (!internal_memcmp(p, name, name_len) && p[name_len] == '=') {
        // Match.
        if (name_value) {
          // Replace the old value with the new one.
          *env = const_cast<char*>(name_value);
        } else {
          // Shift the subsequent pointers back.
          char **del = env;
          do {
            del[0] = del[1];
          } while (*del++);
        }
      }
    }
    env++;
  }
}

static bool reexec_disabled = false;

void DisableReexec() {
  reexec_disabled = true;
}

bool DyldNeedsEnvVariable() {
// If running on OS X 10.11+ or iOS 9.0+, dyld will interpose even if
// DYLD_INSERT_LIBRARIES is not set.

#if SANITIZER_IOSSIM
  // GetMacosVersion will not work for the simulator, whose kernel version
  // is tied to the host. Use a weak linking hack for the simulator.
  // This API was introduced in the same version of the OS as the dyld
  // optimization.

  // Check for presence of a symbol that is available on OS X 10.11+, iOS 9.0+.
  return (dlsym(RTLD_NEXT, "mach_memory_info") == nullptr);
#else
  return (GetMacosVersion() <= MACOS_VERSION_YOSEMITE);
#endif
}

void MaybeReexec() {
  if (reexec_disabled) return;

  // Make sure the dynamic ASan runtime library is preloaded so that the
  // wrappers work. If it is not, set DYLD_INSERT_LIBRARIES and re-exec
  // ourselves.
  Dl_info info;
  CHECK(dladdr((void*)((uptr)__asan_init), &info));
  char *dyld_insert_libraries =
      const_cast<char*>(GetEnv(kDyldInsertLibraries));
  uptr old_env_len = dyld_insert_libraries ?
      internal_strlen(dyld_insert_libraries) : 0;
  uptr fname_len = internal_strlen(info.dli_fname);
  const char *dylib_name = StripModuleName(info.dli_fname);
  uptr dylib_name_len = internal_strlen(dylib_name);

  bool lib_is_in_env =
      dyld_insert_libraries && REAL(strstr)(dyld_insert_libraries, dylib_name);
  if (DyldNeedsEnvVariable() && !lib_is_in_env) {
    // DYLD_INSERT_LIBRARIES is not set or does not contain the runtime
    // library.
    char program_name[1024];
    uint32_t buf_size = sizeof(program_name);
    _NSGetExecutablePath(program_name, &buf_size);
    char *new_env = const_cast<char*>(info.dli_fname);
    if (dyld_insert_libraries) {
      // Append the runtime dylib name to the existing value of
      // DYLD_INSERT_LIBRARIES.
      new_env = (char*)allocator_for_env.Allocate(old_env_len + fname_len + 2);
      internal_strncpy(new_env, dyld_insert_libraries, old_env_len);
      new_env[old_env_len] = ':';
      // Copy fname_len and add a trailing zero.
      internal_strncpy(new_env + old_env_len + 1, info.dli_fname,
                       fname_len + 1);
      // Ok to use setenv() since the wrappers don't depend on the value of
      // asan_inited.
      setenv(kDyldInsertLibraries, new_env, /*overwrite*/1);
    } else {
      // Set DYLD_INSERT_LIBRARIES equal to the runtime dylib name.
      setenv(kDyldInsertLibraries, info.dli_fname, /*overwrite*/0);
    }
    VReport(1, "exec()-ing the program with\n");
    VReport(1, "%s=%s\n", kDyldInsertLibraries, new_env);
    VReport(1, "to enable ASan wrappers.\n");
    execv(program_name, *_NSGetArgv());

    // We get here only if execv() failed.
    Report("ERROR: The process is launched without DYLD_INSERT_LIBRARIES, "
           "which is required for ASan to work. ASan tried to set the "
           "environment variable and re-execute itself, but execv() failed, "
           "possibly because of sandbox restrictions. Make sure to launch the "
           "executable with:\n%s=%s\n", kDyldInsertLibraries, new_env);
    CHECK("execv failed" && 0);
  }

  if (!lib_is_in_env)
    return;

  // DYLD_INSERT_LIBRARIES is set and contains the runtime library. Let's remove
  // the dylib from the environment variable, because interceptors are installed
  // and we don't want our children to inherit the variable.

  uptr env_name_len = internal_strlen(kDyldInsertLibraries);
  // Allocate memory to hold the previous env var name, its value, the '='
  // sign and the '\0' char.
  char *new_env = (char*)allocator_for_env.Allocate(
      old_env_len + 2 + env_name_len);
  CHECK(new_env);
  internal_memset(new_env, '\0', old_env_len + 2 + env_name_len);
  internal_strncpy(new_env, kDyldInsertLibraries, env_name_len);
  new_env[env_name_len] = '=';
  char *new_env_pos = new_env + env_name_len + 1;

  // Iterate over colon-separated pieces of |dyld_insert_libraries|.
  char *piece_start = dyld_insert_libraries;
  char *piece_end = NULL;
  char *old_env_end = dyld_insert_libraries + old_env_len;
  do {
    if (piece_start[0] == ':') piece_start++;
    piece_end = REAL(strchr)(piece_start, ':');
    if (!piece_end) piece_end = dyld_insert_libraries + old_env_len;
    if ((uptr)(piece_start - dyld_insert_libraries) > old_env_len) break;
    uptr piece_len = piece_end - piece_start;

    char *filename_start =
        (char *)internal_memrchr(piece_start, '/', piece_len);
    uptr filename_len = piece_len;
    if (filename_start) {
      filename_start += 1;
      filename_len = piece_len - (filename_start - piece_start);
    } else {
      filename_start = piece_start;
    }

    // If the current piece isn't the runtime library name,
    // append it to new_env.
    if ((dylib_name_len != filename_len) ||
        (internal_memcmp(filename_start, dylib_name, dylib_name_len) != 0)) {
      if (new_env_pos != new_env + env_name_len + 1) {
        new_env_pos[0] = ':';
        new_env_pos++;
      }
      internal_strncpy(new_env_pos, piece_start, piece_len);
      new_env_pos += piece_len;
    }
    // Move on to the next piece.
    piece_start = piece_end;
  } while (piece_start < old_env_end);

  // Can't use setenv() here, because it requires the allocator to be
  // initialized.
  // FIXME: instead of filtering DYLD_INSERT_LIBRARIES here, do it in
  // a separate function called after InitializeAllocator().
  if (new_env_pos == new_env + env_name_len + 1) new_env = NULL;
  LeakyResetEnv(kDyldInsertLibraries, new_env);
}

// No-op. Mac does not support static linkage anyway.
void *AsanDoesNotSupportStaticLinkage() {
  return 0;
}

// No-op. Mac does not support static linkage anyway.
void AsanCheckDynamicRTPrereqs() {}

// No-op. Mac does not support static linkage anyway.
void AsanCheckIncompatibleRT() {}

void AsanPlatformThreadInit() {
}

void ReadContextStack(void *context, uptr *stack, uptr *ssize) {
  UNIMPLEMENTED();
}

// Support for the following functions from libdispatch on Mac OS:
//   dispatch_async_f()
//   dispatch_async()
//   dispatch_sync_f()
//   dispatch_sync()
//   dispatch_after_f()
//   dispatch_after()
//   dispatch_group_async_f()
//   dispatch_group_async()
// TODO(glider): libdispatch API contains other functions that we don't support
// yet.
//
// dispatch_sync() and dispatch_sync_f() are synchronous, although chances are
// they can cause jobs to run on a thread different from the current one.
// TODO(glider): if so, we need a test for this (otherwise we should remove
// them).
//
// The following functions use dispatch_barrier_async_f() (which isn't a library
// function but is exported) and are thus supported:
//   dispatch_source_set_cancel_handler_f()
//   dispatch_source_set_cancel_handler()
//   dispatch_source_set_event_handler_f()
//   dispatch_source_set_event_handler()
//
// The reference manual for Grand Central Dispatch is available at
//   http://developer.apple.com/library/mac/#documentation/Performance/Reference/GCD_libdispatch_Ref/Reference/reference.html
// The implementation details are at
//   http://libdispatch.macosforge.org/trac/browser/trunk/src/queue.c

typedef void* dispatch_group_t;
typedef void* dispatch_queue_t;
typedef void* dispatch_source_t;
typedef u64 dispatch_time_t;
typedef void (*dispatch_function_t)(void *block);
typedef void* (*worker_t)(void *block);

// A wrapper for the ObjC blocks used to support libdispatch.
typedef struct {
  void *block;
  dispatch_function_t func;
  u32 parent_tid;
} asan_block_context_t;

ALWAYS_INLINE
void asan_register_worker_thread(int parent_tid, StackTrace *stack) {
  AsanThread *t = GetCurrentThread();
  if (!t) {
    t = AsanThread::Create(/* start_routine */ nullptr, /* arg */ nullptr,
                           parent_tid, stack, /* detached */ true);
    t->Init();
    asanThreadRegistry().StartThread(t->tid(), 0, 0);
    SetCurrentThread(t);
  }
}

// For use by only those functions that allocated the context via
// alloc_asan_context().
extern "C"
void asan_dispatch_call_block_and_release(void *block) {
  GET_STACK_TRACE_THREAD;
  asan_block_context_t *context = (asan_block_context_t*)block;
  VReport(2,
          "asan_dispatch_call_block_and_release(): "
          "context: %p, pthread_self: %p\n",
          block, pthread_self());
  asan_register_worker_thread(context->parent_tid, &stack);
  // Call the original dispatcher for the block.
  context->func(context->block);
  asan_free(context, &stack, FROM_MALLOC);
}

}  // namespace __asan

using namespace __asan;  // NOLINT

// Wrap |ctxt| and |func| into an asan_block_context_t.
// The caller retains control of the allocated context.
extern "C"
asan_block_context_t *alloc_asan_context(void *ctxt, dispatch_function_t func,
                                         BufferedStackTrace *stack) {
  asan_block_context_t *asan_ctxt =
      (asan_block_context_t*) asan_malloc(sizeof(asan_block_context_t), stack);
  asan_ctxt->block = ctxt;
  asan_ctxt->func = func;
  asan_ctxt->parent_tid = GetCurrentTidOrInvalid();
  return asan_ctxt;
}

// Define interceptor for dispatch_*_f function with the three most common
// parameters: dispatch_queue_t, context, dispatch_function_t.
#define INTERCEPT_DISPATCH_X_F_3(dispatch_x_f)                                \
  INTERCEPTOR(void, dispatch_x_f, dispatch_queue_t dq, void *ctxt,            \
                                  dispatch_function_t func) {                 \
    GET_STACK_TRACE_THREAD;                                                   \
    asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack); \
    if (Verbosity() >= 2) {                                     \
      Report(#dispatch_x_f "(): context: %p, pthread_self: %p\n",             \
             asan_ctxt, pthread_self());                                      \
      PRINT_CURRENT_STACK();                                                  \
    }                                                                         \
    return REAL(dispatch_x_f)(dq, (void*)asan_ctxt,                           \
                              asan_dispatch_call_block_and_release);          \
  }

INTERCEPT_DISPATCH_X_F_3(dispatch_async_f)
INTERCEPT_DISPATCH_X_F_3(dispatch_sync_f)
INTERCEPT_DISPATCH_X_F_3(dispatch_barrier_async_f)

INTERCEPTOR(void, dispatch_after_f, dispatch_time_t when,
                                    dispatch_queue_t dq, void *ctxt,
                                    dispatch_function_t func) {
  GET_STACK_TRACE_THREAD;
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (Verbosity() >= 2) {
    Report("dispatch_after_f: %p\n", asan_ctxt);
    PRINT_CURRENT_STACK();
  }
  return REAL(dispatch_after_f)(when, dq, (void*)asan_ctxt,
                                asan_dispatch_call_block_and_release);
}

INTERCEPTOR(void, dispatch_group_async_f, dispatch_group_t group,
                                          dispatch_queue_t dq, void *ctxt,
                                          dispatch_function_t func) {
  GET_STACK_TRACE_THREAD;
  asan_block_context_t *asan_ctxt = alloc_asan_context(ctxt, func, &stack);
  if (Verbosity() >= 2) {
    Report("dispatch_group_async_f(): context: %p, pthread_self: %p\n",
           asan_ctxt, pthread_self());
    PRINT_CURRENT_STACK();
  }
  REAL(dispatch_group_async_f)(group, dq, (void*)asan_ctxt,
                               asan_dispatch_call_block_and_release);
}

#if !defined(MISSING_BLOCKS_SUPPORT)
extern "C" {
void dispatch_async(dispatch_queue_t dq, void(^work)(void));
void dispatch_group_async(dispatch_group_t dg, dispatch_queue_t dq,
                          void(^work)(void));
void dispatch_after(dispatch_time_t when, dispatch_queue_t queue,
                    void(^work)(void));
void dispatch_source_set_cancel_handler(dispatch_source_t ds,
                                        void(^work)(void));
void dispatch_source_set_event_handler(dispatch_source_t ds, void(^work)(void));
}

#define GET_ASAN_BLOCK(work) \
  void (^asan_block)(void);  \
  int parent_tid = GetCurrentTidOrInvalid(); \
  asan_block = ^(void) { \
    GET_STACK_TRACE_THREAD; \
    asan_register_worker_thread(parent_tid, &stack); \
    work(); \
  }

INTERCEPTOR(void, dispatch_async,
            dispatch_queue_t dq, void(^work)(void)) {
  ENABLE_FRAME_POINTER;
  GET_ASAN_BLOCK(work);
  REAL(dispatch_async)(dq, asan_block);
}

INTERCEPTOR(void, dispatch_group_async,
            dispatch_group_t dg, dispatch_queue_t dq, void(^work)(void)) {
  ENABLE_FRAME_POINTER;
  GET_ASAN_BLOCK(work);
  REAL(dispatch_group_async)(dg, dq, asan_block);
}

INTERCEPTOR(void, dispatch_after,
            dispatch_time_t when, dispatch_queue_t queue, void(^work)(void)) {
  ENABLE_FRAME_POINTER;
  GET_ASAN_BLOCK(work);
  REAL(dispatch_after)(when, queue, asan_block);
}

INTERCEPTOR(void, dispatch_source_set_cancel_handler,
            dispatch_source_t ds, void(^work)(void)) {
  if (!work) {
    REAL(dispatch_source_set_cancel_handler)(ds, work);
    return;
  }
  ENABLE_FRAME_POINTER;
  GET_ASAN_BLOCK(work);
  REAL(dispatch_source_set_cancel_handler)(ds, asan_block);
}

INTERCEPTOR(void, dispatch_source_set_event_handler,
            dispatch_source_t ds, void(^work)(void)) {
  ENABLE_FRAME_POINTER;
  GET_ASAN_BLOCK(work);
  REAL(dispatch_source_set_event_handler)(ds, asan_block);
}
#endif

#endif  // SANITIZER_MAC
