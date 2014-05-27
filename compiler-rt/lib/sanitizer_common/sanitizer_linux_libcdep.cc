//===-- sanitizer_linux_libcdep.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements linux-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_FREEBSD || SANITIZER_LINUX

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_atomic.h"
#include "sanitizer_symbolizer.h"

#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#if SANITIZER_FREEBSD
#define _GNU_SOURCE  // to declare _Unwind_Backtrace() from <unwind.h>
#endif
#include <unwind.h>

#if SANITIZER_FREEBSD
#include <pthread_np.h>
#define pthread_getattr_np pthread_attr_get_np
#endif

#if SANITIZER_LINUX
#include <sys/prctl.h>
#endif

#if !SANITIZER_ANDROID
#include <elf.h>
#include <link.h>
#include <unistd.h>
#endif

namespace __sanitizer {

// This function is defined elsewhere if we intercepted pthread_attr_getstack.
extern "C" {
SANITIZER_WEAK_ATTRIBUTE int
real_pthread_attr_getstack(void *attr, void **addr, size_t *size);
}  // extern "C"

static int my_pthread_attr_getstack(void *attr, void **addr, size_t *size) {
  if (real_pthread_attr_getstack)
    return real_pthread_attr_getstack((pthread_attr_t *)attr, addr, size);
  return pthread_attr_getstack((pthread_attr_t *)attr, addr, size);
}

SANITIZER_WEAK_ATTRIBUTE int
real_sigaction(int signum, const void *act, void *oldact);

int internal_sigaction(int signum, const void *act, void *oldact) {
  if (real_sigaction)
    return real_sigaction(signum, act, oldact);
  return sigaction(signum, (struct sigaction *)act, (struct sigaction *)oldact);
}

void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom) {
  CHECK(stack_top);
  CHECK(stack_bottom);
  if (at_initialization) {
    // This is the main thread. Libpthread may not be initialized yet.
    struct rlimit rl;
    CHECK_EQ(getrlimit(RLIMIT_STACK, &rl), 0);

    // Find the mapping that contains a stack variable.
    MemoryMappingLayout proc_maps(/*cache_enabled*/true);
    uptr start, end, offset;
    uptr prev_end = 0;
    while (proc_maps.Next(&start, &end, &offset, 0, 0, /* protection */0)) {
      if ((uptr)&rl < end)
        break;
      prev_end = end;
    }
    CHECK((uptr)&rl >= start && (uptr)&rl < end);

    // Get stacksize from rlimit, but clip it so that it does not overlap
    // with other mappings.
    uptr stacksize = rl.rlim_cur;
    if (stacksize > end - prev_end)
      stacksize = end - prev_end;
    // When running with unlimited stack size, we still want to set some limit.
    // The unlimited stack size is caused by 'ulimit -s unlimited'.
    // Also, for some reason, GNU make spawns subprocesses with unlimited stack.
    if (stacksize > kMaxThreadStackSize)
      stacksize = kMaxThreadStackSize;
    *stack_top = end;
    *stack_bottom = end - stacksize;
    return;
  }
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  uptr stacksize = 0;
  void *stackaddr = 0;
  my_pthread_attr_getstack(&attr, &stackaddr, (size_t*)&stacksize);
  pthread_attr_destroy(&attr);

  CHECK_LE(stacksize, kMaxThreadStackSize);  // Sanity check.
  *stack_top = (uptr)stackaddr + stacksize;
  *stack_bottom = (uptr)stackaddr;
}

bool SetEnv(const char *name, const char *value) {
  void *f = dlsym(RTLD_NEXT, "setenv");
  if (f == 0)
    return false;
  typedef int(*setenv_ft)(const char *name, const char *value, int overwrite);
  setenv_ft setenv_f;
  CHECK_EQ(sizeof(setenv_f), sizeof(f));
  internal_memcpy(&setenv_f, &f, sizeof(f));
  return IndirectExternCall(setenv_f)(name, value, 1) == 0;
}

bool SanitizerSetThreadName(const char *name) {
#ifdef PR_SET_NAME
  return 0 == prctl(PR_SET_NAME, (unsigned long)name, 0, 0, 0);  // NOLINT
#else
  return false;
#endif
}

bool SanitizerGetThreadName(char *name, int max_len) {
#ifdef PR_GET_NAME
  char buff[17];
  if (prctl(PR_GET_NAME, (unsigned long)buff, 0, 0, 0))  // NOLINT
    return false;
  internal_strncpy(name, buff, max_len);
  name[max_len] = 0;
  return true;
#else
  return false;
#endif
}

//------------------------- SlowUnwindStack -----------------------------------

typedef struct {
  uptr absolute_pc;
  uptr stack_top;
  uptr stack_size;
} backtrace_frame_t;

extern "C" {
typedef void *(*acquire_my_map_info_list_func)();
typedef void (*release_my_map_info_list_func)(void *map);
typedef sptr (*unwind_backtrace_signal_arch_func)(
    void *siginfo, void *sigcontext, void *map_info_list,
    backtrace_frame_t *backtrace, uptr ignore_depth, uptr max_depth);
acquire_my_map_info_list_func acquire_my_map_info_list;
release_my_map_info_list_func release_my_map_info_list;
unwind_backtrace_signal_arch_func unwind_backtrace_signal_arch;
} // extern "C"

#if SANITIZER_ANDROID
void SanitizerInitializeUnwinder() {
  void *p = dlopen("libcorkscrew.so", RTLD_LAZY);
  if (!p) {
    VReport(1,
            "Failed to open libcorkscrew.so. You may see broken stack traces "
            "in SEGV reports.");
    return;
  }
  acquire_my_map_info_list =
      (acquire_my_map_info_list_func)(uptr)dlsym(p, "acquire_my_map_info_list");
  release_my_map_info_list =
      (release_my_map_info_list_func)(uptr)dlsym(p, "release_my_map_info_list");
  unwind_backtrace_signal_arch = (unwind_backtrace_signal_arch_func)(uptr)dlsym(
      p, "unwind_backtrace_signal_arch");
  if (!acquire_my_map_info_list || !release_my_map_info_list ||
      !unwind_backtrace_signal_arch) {
    VReport(1,
            "Failed to find one of the required symbols in libcorkscrew.so. "
            "You may see broken stack traces in SEGV reports.");
    acquire_my_map_info_list = NULL;
    unwind_backtrace_signal_arch = NULL;
    release_my_map_info_list = NULL;
  }
}
#endif

#ifdef __arm__
#define UNWIND_STOP _URC_END_OF_STACK
#define UNWIND_CONTINUE _URC_NO_REASON
#else
#define UNWIND_STOP _URC_NORMAL_STOP
#define UNWIND_CONTINUE _URC_NO_REASON
#endif

uptr Unwind_GetIP(struct _Unwind_Context *ctx) {
#ifdef __arm__
  uptr val;
  _Unwind_VRS_Result res = _Unwind_VRS_Get(ctx, _UVRSC_CORE,
      15 /* r15 = PC */, _UVRSD_UINT32, &val);
  CHECK(res == _UVRSR_OK && "_Unwind_VRS_Get failed");
  // Clear the Thumb bit.
  return val & ~(uptr)1;
#else
  return _Unwind_GetIP(ctx);
#endif
}

struct UnwindTraceArg {
  StackTrace *stack;
  uptr max_depth;
};

_Unwind_Reason_Code Unwind_Trace(struct _Unwind_Context *ctx, void *param) {
  UnwindTraceArg *arg = (UnwindTraceArg*)param;
  CHECK_LT(arg->stack->size, arg->max_depth);
  uptr pc = Unwind_GetIP(ctx);
  arg->stack->trace[arg->stack->size++] = pc;
  if (arg->stack->size == arg->max_depth) return UNWIND_STOP;
  return UNWIND_CONTINUE;
}

void StackTrace::SlowUnwindStack(uptr pc, uptr max_depth) {
  CHECK_GE(max_depth, 2);
  size = 0;
  UnwindTraceArg arg = {this, Min(max_depth + 1, kStackTraceMax)};
  _Unwind_Backtrace(Unwind_Trace, &arg);
  // We need to pop a few frames so that pc is on top.
  uptr to_pop = LocatePcInTrace(pc);
  // trace[0] belongs to the current function so we always pop it.
  if (to_pop == 0)
    to_pop = 1;
  PopStackFrames(to_pop);
  trace[0] = pc;
}

void StackTrace::SlowUnwindStackWithContext(uptr pc, void *context,
                                            uptr max_depth) {
  CHECK_GE(max_depth, 2);
  if (!unwind_backtrace_signal_arch) {
    SlowUnwindStack(pc, max_depth);
    return;
  }

  void *map = acquire_my_map_info_list();
  CHECK(map);
  InternalScopedBuffer<backtrace_frame_t> frames(kStackTraceMax);
  // siginfo argument appears to be unused.
  sptr res = unwind_backtrace_signal_arch(/* siginfo */ NULL, context, map,
                                          frames.data(),
                                          /* ignore_depth */ 0, max_depth);
  release_my_map_info_list(map);
  if (res < 0) return;
  CHECK_LE((uptr)res, kStackTraceMax);

  size = 0;
  // +2 compensate for libcorkscrew unwinder returning addresses of call
  // instructions instead of raw return addresses.
  for (sptr i = 0; i < res; ++i)
    trace[size++] = frames[i].absolute_pc + 2;
}

#if !SANITIZER_FREEBSD
static uptr g_tls_size;
#endif

#ifdef __i386__
# define DL_INTERNAL_FUNCTION __attribute__((regparm(3), stdcall))
#else
# define DL_INTERNAL_FUNCTION
#endif

void InitTlsSize() {
#if !SANITIZER_FREEBSD && !SANITIZER_ANDROID
  typedef void (*get_tls_func)(size_t*, size_t*) DL_INTERNAL_FUNCTION;
  get_tls_func get_tls;
  void *get_tls_static_info_ptr = dlsym(RTLD_NEXT, "_dl_get_tls_static_info");
  CHECK_EQ(sizeof(get_tls), sizeof(get_tls_static_info_ptr));
  internal_memcpy(&get_tls, &get_tls_static_info_ptr,
                  sizeof(get_tls_static_info_ptr));
  CHECK_NE(get_tls, 0);
  size_t tls_size = 0;
  size_t tls_align = 0;
  IndirectExternCall(get_tls)(&tls_size, &tls_align);
  g_tls_size = tls_size;
#endif  // !SANITIZER_FREEBSD && !SANITIZER_ANDROID
}

#if (defined(__x86_64__) || defined(__i386__)) && SANITIZER_LINUX
// sizeof(struct thread) from glibc.
static atomic_uintptr_t kThreadDescriptorSize;

uptr ThreadDescriptorSize() {
  char buf[64];
  uptr val = atomic_load(&kThreadDescriptorSize, memory_order_relaxed);
  if (val)
    return val;
#ifdef _CS_GNU_LIBC_VERSION
  uptr len = confstr(_CS_GNU_LIBC_VERSION, buf, sizeof(buf));
  if (len < sizeof(buf) && internal_strncmp(buf, "glibc 2.", 8) == 0) {
    char *end;
    int minor = internal_simple_strtoll(buf + 8, &end, 10);
    if (end != buf + 8 && (*end == '\0' || *end == '.')) {
      /* sizeof(struct thread) values from various glibc versions.  */
      if (SANITIZER_X32)
        val = 1728;  // Assume only one particular version for x32.
      else if (minor <= 3)
        val = FIRST_32_SECOND_64(1104, 1696);
      else if (minor == 4)
        val = FIRST_32_SECOND_64(1120, 1728);
      else if (minor == 5)
        val = FIRST_32_SECOND_64(1136, 1728);
      else if (minor <= 9)
        val = FIRST_32_SECOND_64(1136, 1712);
      else if (minor == 10)
        val = FIRST_32_SECOND_64(1168, 1776);
      else if (minor <= 12)
        val = FIRST_32_SECOND_64(1168, 2288);
      else
        val = FIRST_32_SECOND_64(1216, 2304);
    }
    if (val)
      atomic_store(&kThreadDescriptorSize, val, memory_order_relaxed);
    return val;
  }
#endif
  return 0;
}

// The offset at which pointer to self is located in the thread descriptor.
const uptr kThreadSelfOffset = FIRST_32_SECOND_64(8, 16);

uptr ThreadSelfOffset() {
  return kThreadSelfOffset;
}

uptr ThreadSelf() {
  uptr descr_addr;
# if defined(__i386__)
  asm("mov %%gs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
# elif defined(__x86_64__)
  asm("mov %%fs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
# else
#  error "unsupported CPU arch"
# endif
  return descr_addr;
}
#endif  // (defined(__x86_64__) || defined(__i386__)) && SANITIZER_LINUX

#if SANITIZER_FREEBSD
static void **ThreadSelfSegbase() {
  void **segbase = 0;
# if defined(__i386__)
  // sysarch(I386_GET_GSBASE, segbase);
  __asm __volatile("mov %%gs:0, %0" : "=r" (segbase));
# elif defined(__x86_64__)
  // sysarch(AMD64_GET_FSBASE, segbase);
  __asm __volatile("movq %%fs:0, %0" : "=r" (segbase));
# else
#  error "unsupported CPU arch for FreeBSD platform"
# endif
  return segbase;
}

uptr ThreadSelf() {
  return (uptr)ThreadSelfSegbase()[2];
}
#endif  // SANITIZER_FREEBSD

static void GetTls(uptr *addr, uptr *size) {
#if SANITIZER_LINUX
# if defined(__x86_64__) || defined(__i386__)
  *addr = ThreadSelf();
  *size = GetTlsSize();
  *addr -= *size;
  *addr += ThreadDescriptorSize();
# else
  *addr = 0;
  *size = 0;
# endif
#elif SANITIZER_FREEBSD
  void** segbase = ThreadSelfSegbase();
  *addr = 0;
  *size = 0;
  if (segbase != 0) {
    // tcbalign = 16
    // tls_size = round(tls_static_space, tcbalign);
    // dtv = segbase[1];
    // dtv[2] = segbase - tls_static_space;
    void **dtv = (void**) segbase[1];
    *addr = (uptr) dtv[2];
    *size = (*addr == 0) ? 0 : ((uptr) segbase[0] - (uptr) dtv[2]);
  }
#else
# error "Unknown OS"
#endif
}

uptr GetTlsSize() {
#if SANITIZER_FREEBSD
  uptr addr, size;
  GetTls(&addr, &size);
  return size;
#else
  return g_tls_size;
#endif
}

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
  GetTls(tls_addr, tls_size);

  uptr stack_top, stack_bottom;
  GetThreadStackTopAndBottom(main, &stack_top, &stack_bottom);
  *stk_addr = stack_bottom;
  *stk_size = stack_top - stack_bottom;

  if (!main) {
    // If stack and tls intersect, make them non-intersecting.
    if (*tls_addr > *stk_addr && *tls_addr < *stk_addr + *stk_size) {
      CHECK_GT(*tls_addr + *tls_size, *stk_addr);
      CHECK_LE(*tls_addr + *tls_size, *stk_addr + *stk_size);
      *stk_size -= *tls_size;
      *tls_addr = *stk_addr + *stk_size;
    }
  }
}

void AdjustStackSize(void *attr_) {
  pthread_attr_t *attr = (pthread_attr_t *)attr_;
  uptr stackaddr = 0;
  size_t stacksize = 0;
  my_pthread_attr_getstack(attr, (void**)&stackaddr, &stacksize);
  // GLibC will return (0 - stacksize) as the stack address in the case when
  // stacksize is set, but stackaddr is not.
  bool stack_set = (stackaddr != 0) && (stackaddr + stacksize != 0);
  // We place a lot of tool data into TLS, account for that.
  const uptr minstacksize = GetTlsSize() + 128*1024;
  if (stacksize < minstacksize) {
    if (!stack_set) {
      if (stacksize != 0) {
        VPrintf(1, "Sanitizer: increasing stacksize %zu->%zu\n", stacksize,
                minstacksize);
        pthread_attr_setstacksize(attr, minstacksize);
      }
    } else {
      Printf("Sanitizer: pre-allocated stack size is insufficient: "
             "%zu < %zu\n", stacksize, minstacksize);
      Printf("Sanitizer: pthread_create is likely to fail.\n");
    }
  }
}

#if SANITIZER_ANDROID
uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  MemoryMappingLayout memory_mapping(false);
  return memory_mapping.DumpListOfModules(modules, max_modules, filter);
}
#else  // SANITIZER_ANDROID
# if !SANITIZER_FREEBSD
typedef ElfW(Phdr) Elf_Phdr;
# endif

struct DlIteratePhdrData {
  LoadedModule *modules;
  uptr current_n;
  bool first;
  uptr max_n;
  string_predicate_t filter;
};

static int dl_iterate_phdr_cb(dl_phdr_info *info, size_t size, void *arg) {
  DlIteratePhdrData *data = (DlIteratePhdrData*)arg;
  if (data->current_n == data->max_n)
    return 0;
  InternalScopedBuffer<char> module_name(kMaxPathLength);
  module_name.data()[0] = '\0';
  if (data->first) {
    data->first = false;
    // First module is the binary itself.
    ReadBinaryName(module_name.data(), module_name.size());
  } else if (info->dlpi_name) {
    internal_strncpy(module_name.data(), info->dlpi_name, module_name.size());
  }
  if (module_name.data()[0] == '\0')
    return 0;
  if (data->filter && !data->filter(module_name.data()))
    return 0;
  void *mem = &data->modules[data->current_n];
  LoadedModule *cur_module = new(mem) LoadedModule(module_name.data(),
                                                   info->dlpi_addr);
  data->current_n++;
  for (int i = 0; i < info->dlpi_phnum; i++) {
    const Elf_Phdr *phdr = &info->dlpi_phdr[i];
    if (phdr->p_type == PT_LOAD) {
      uptr cur_beg = info->dlpi_addr + phdr->p_vaddr;
      uptr cur_end = cur_beg + phdr->p_memsz;
      cur_module->addAddressRange(cur_beg, cur_end);
    }
  }
  return 0;
}

uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  CHECK(modules);
  DlIteratePhdrData data = {modules, 0, true, max_modules, filter};
  dl_iterate_phdr(dl_iterate_phdr_cb, &data);
  return data.current_n;
}
#endif  // SANITIZER_ANDROID

uptr indirect_call_wrapper;

void SetIndirectCallWrapper(uptr wrapper) {
  CHECK(!indirect_call_wrapper);
  CHECK(wrapper);
  indirect_call_wrapper = wrapper;
}

void PrepareForSandboxing(__sanitizer_sandbox_arguments *args) {
  // Some kinds of sandboxes may forbid filesystem access, so we won't be able
  // to read the file mappings from /proc/self/maps. Luckily, neither the
  // process will be able to load additional libraries, so it's fine to use the
  // cached mappings.
  MemoryMappingLayout::CacheMemoryMappings();
  // Same for /proc/self/exe in the symbolizer.
#if !SANITIZER_GO
  if (Symbolizer *sym = Symbolizer::GetOrNull())
    sym->PrepareForSandboxing();
  CovPrepareForSandboxing(args);
#endif
}

}  // namespace __sanitizer

#endif  // SANITIZER_FREEBSD || SANITIZER_LINUX
