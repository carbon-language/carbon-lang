//===-- tsan_interceptors.cc ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// FIXME: move as many interceptors as possible into
// sanitizer_common/sanitizer_common_interceptors.inc
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_platform_limits_posix.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "interception/interception.h"
#include "tsan_interface.h"
#include "tsan_platform.h"
#include "tsan_rtl.h"
#include "tsan_mman.h"
#include "tsan_fd.h"

using namespace __tsan;  // NOLINT

const int kSigCount = 64;

struct my_siginfo_t {
  // The size is determined by looking at sizeof of real siginfo_t on linux.
  u64 opaque[128 / sizeof(u64)];
};

struct sigset_t {
  // The size is determined by looking at sizeof of real sigset_t on linux.
  u64 val[128 / sizeof(u64)];
};

struct ucontext_t {
  // The size is determined by looking at sizeof of real ucontext_t on linux.
  u64 opaque[936 / sizeof(u64) + 1];
};

extern "C" int pthread_attr_init(void *attr);
extern "C" int pthread_attr_destroy(void *attr);
extern "C" int pthread_attr_getdetachstate(void *attr, int *v);
extern "C" int pthread_attr_setstacksize(void *attr, uptr stacksize);
extern "C" int pthread_attr_getstacksize(void *attr, uptr *stacksize);
extern "C" int pthread_key_create(unsigned *key, void (*destructor)(void* v));
extern "C" int pthread_setspecific(unsigned key, const void *v);
extern "C" int pthread_mutexattr_gettype(void *a, int *type);
extern "C" int pthread_yield();
extern "C" int pthread_sigmask(int how, const sigset_t *set, sigset_t *oldset);
extern "C" int sigfillset(sigset_t *set);
extern "C" void *pthread_self();
extern "C" void _exit(int status);
extern "C" int *__errno_location();
extern "C" int fileno_unlocked(void *stream);
extern "C" void *__libc_malloc(uptr size);
extern "C" void *__libc_calloc(uptr size, uptr n);
extern "C" void *__libc_realloc(void *ptr, uptr size);
extern "C" void __libc_free(void *ptr);
extern "C" int mallopt(int param, int value);
const int PTHREAD_MUTEX_RECURSIVE = 1;
const int PTHREAD_MUTEX_RECURSIVE_NP = 1;
const int kPthreadAttrSize = 56;
const int EINVAL = 22;
const int EBUSY = 16;
const int EPOLL_CTL_ADD = 1;
const int SIGILL = 4;
const int SIGABRT = 6;
const int SIGFPE = 8;
const int SIGSEGV = 11;
const int SIGPIPE = 13;
const int SIGBUS = 7;
void *const MAP_FAILED = (void*)-1;
const int PTHREAD_BARRIER_SERIAL_THREAD = -1;
const int MAP_FIXED = 0x10;
typedef long long_t;  // NOLINT

// From /usr/include/unistd.h
# define F_ULOCK 0      /* Unlock a previously locked region.  */
# define F_LOCK  1      /* Lock a region for exclusive use.  */
# define F_TLOCK 2      /* Test and lock a region for exclusive use.  */
# define F_TEST  3      /* Test a region for other processes locks.  */

typedef void (*sighandler_t)(int sig);

#define errno (*__errno_location())

struct sigaction_t {
  union {
    sighandler_t sa_handler;
    void (*sa_sigaction)(int sig, my_siginfo_t *siginfo, void *uctx);
  };
  sigset_t sa_mask;
  int sa_flags;
  void (*sa_restorer)();
};

const sighandler_t SIG_DFL = (sighandler_t)0;
const sighandler_t SIG_IGN = (sighandler_t)1;
const sighandler_t SIG_ERR = (sighandler_t)-1;
const int SA_SIGINFO = 4;
const int SIG_SETMASK = 2;

namespace std {
struct nothrow_t {};
}  // namespace std

static sigaction_t sigactions[kSigCount];

namespace __tsan {
struct SignalDesc {
  bool armed;
  bool sigaction;
  my_siginfo_t siginfo;
  ucontext_t ctx;
};

struct SignalContext {
  int in_blocking_func;
  int int_signal_send;
  int pending_signal_count;
  SignalDesc pending_signals[kSigCount];
};
}  // namespace __tsan

static SignalContext *SigCtx(ThreadState *thr) {
  SignalContext *ctx = (SignalContext*)thr->signal_ctx;
  if (ctx == 0 && thr->is_alive) {
    ScopedInRtl in_rtl;
    ctx = (SignalContext*)MmapOrDie(sizeof(*ctx), "SignalContext");
    MemoryResetRange(thr, (uptr)&SigCtx, (uptr)ctx, sizeof(*ctx));
    thr->signal_ctx = ctx;
  }
  return ctx;
}

static unsigned g_thread_finalize_key;

class ScopedInterceptor {
 public:
  ScopedInterceptor(ThreadState *thr, const char *fname, uptr pc);
  ~ScopedInterceptor();
 private:
  ThreadState *const thr_;
  const int in_rtl_;
};

ScopedInterceptor::ScopedInterceptor(ThreadState *thr, const char *fname,
                                     uptr pc)
    : thr_(thr)
    , in_rtl_(thr->in_rtl) {
  if (thr_->in_rtl == 0) {
    Initialize(thr);
    FuncEntry(thr, pc);
    thr_->in_rtl++;
    DPrintf("#%d: intercept %s()\n", thr_->tid, fname);
  } else {
    thr_->in_rtl++;
  }
}

ScopedInterceptor::~ScopedInterceptor() {
  thr_->in_rtl--;
  if (thr_->in_rtl == 0) {
    FuncExit(thr_);
    ProcessPendingSignals(thr_);
  }
  CHECK_EQ(in_rtl_, thr_->in_rtl);
}

#define SCOPED_INTERCEPTOR_RAW(func, ...) \
    ThreadState *thr = cur_thread(); \
    StatInc(thr, StatInterceptor); \
    StatInc(thr, StatInt_##func); \
    const uptr caller_pc = GET_CALLER_PC(); \
    ScopedInterceptor si(thr, #func, caller_pc); \
    const uptr pc = __sanitizer::StackTrace::GetPreviousInstructionPc( \
        __sanitizer::StackTrace::GetCurrentPc()); \
    (void)pc; \
/**/

#define SCOPED_TSAN_INTERCEPTOR(func, ...) \
    SCOPED_INTERCEPTOR_RAW(func, __VA_ARGS__); \
    if (REAL(func) == 0) { \
      Printf("FATAL: ThreadSanitizer: failed to intercept %s\n", #func); \
      Die(); \
    } \
    if (thr->in_rtl > 1) \
      return REAL(func)(__VA_ARGS__); \
/**/

#define TSAN_INTERCEPTOR(ret, func, ...) INTERCEPTOR(ret, func, __VA_ARGS__)
#define TSAN_INTERCEPT(func) INTERCEPT_FUNCTION(func)

#define BLOCK_REAL(name) (BlockingCall(thr), REAL(name))

struct BlockingCall {
  explicit BlockingCall(ThreadState *thr)
      : ctx(SigCtx(thr)) {
    ctx->in_blocking_func++;
  }

  ~BlockingCall() {
    ctx->in_blocking_func--;
  }

  SignalContext *ctx;
};

TSAN_INTERCEPTOR(unsigned, sleep, unsigned sec) {
  SCOPED_TSAN_INTERCEPTOR(sleep, sec);
  unsigned res = BLOCK_REAL(sleep)(sec);
  AfterSleep(thr, pc);
  return res;
}

TSAN_INTERCEPTOR(int, usleep, long_t usec) {
  SCOPED_TSAN_INTERCEPTOR(usleep, usec);
  int res = BLOCK_REAL(usleep)(usec);
  AfterSleep(thr, pc);
  return res;
}

TSAN_INTERCEPTOR(int, nanosleep, void *req, void *rem) {
  SCOPED_TSAN_INTERCEPTOR(nanosleep, req, rem);
  int res = BLOCK_REAL(nanosleep)(req, rem);
  AfterSleep(thr, pc);
  return res;
}

class AtExitContext {
 public:
  AtExitContext()
    : mtx_(MutexTypeAtExit, StatMtxAtExit)
    , pos_() {
  }

  typedef void(*atexit_t)();

  int atexit(ThreadState *thr, uptr pc, bool is_on_exit,
             atexit_t f, void *arg) {
    Lock l(&mtx_);
    if (pos_ == kMaxAtExit)
      return 1;
    Release(thr, pc, (uptr)this);
    stack_[pos_] = f;
    args_[pos_] = arg;
    is_on_exits_[pos_] = is_on_exit;
    pos_++;
    return 0;
  }

  void exit(ThreadState *thr, uptr pc) {
    CHECK_EQ(thr->in_rtl, 0);
    for (;;) {
      atexit_t f = 0;
      void *arg = 0;
      bool is_on_exit = false;
      {
        Lock l(&mtx_);
        if (pos_) {
          pos_--;
          f = stack_[pos_];
          arg = args_[pos_];
          is_on_exit = is_on_exits_[pos_];
          ScopedInRtl in_rtl;
          Acquire(thr, pc, (uptr)this);
        }
      }
      if (f == 0)
        break;
      DPrintf("#%d: executing atexit func %p\n", thr->tid, f);
      CHECK_EQ(thr->in_rtl, 0);
      if (is_on_exit)
        ((void(*)(int status, void *arg))f)(0, arg);
      else
        ((void(*)(void *arg, void *dso))f)(arg, 0);
    }
  }

 private:
  static const int kMaxAtExit = 128;
  Mutex mtx_;
  atexit_t stack_[kMaxAtExit];
  void *args_[kMaxAtExit];
  bool is_on_exits_[kMaxAtExit];
  int pos_;
};

static AtExitContext *atexit_ctx;

TSAN_INTERCEPTOR(int, atexit, void (*f)()) {
  if (cur_thread()->in_symbolizer)
    return 0;
  SCOPED_TSAN_INTERCEPTOR(atexit, f);
  return atexit_ctx->atexit(thr, pc, false, (void(*)())f, 0);
}

TSAN_INTERCEPTOR(int, on_exit, void(*f)(int, void*), void *arg) {
  if (cur_thread()->in_symbolizer)
    return 0;
  SCOPED_TSAN_INTERCEPTOR(on_exit, f, arg);
  return atexit_ctx->atexit(thr, pc, true, (void(*)())f, arg);
}

TSAN_INTERCEPTOR(int, __cxa_atexit, void (*f)(void *a), void *arg, void *dso) {
  if (cur_thread()->in_symbolizer)
    return 0;
  SCOPED_TSAN_INTERCEPTOR(__cxa_atexit, f, arg, dso);
  if (dso)
    return REAL(__cxa_atexit)(f, arg, dso);
  return atexit_ctx->atexit(thr, pc, false, (void(*)())f, arg);
}

// Cleanup old bufs.
static void JmpBufGarbageCollect(ThreadState *thr, uptr sp) {
  for (uptr i = 0; i < thr->jmp_bufs.Size(); i++) {
    JmpBuf *buf = &thr->jmp_bufs[i];
    if (buf->sp <= sp) {
      uptr sz = thr->jmp_bufs.Size();
      thr->jmp_bufs[i] = thr->jmp_bufs[sz - 1];
      thr->jmp_bufs.PopBack();
      i--;
    }
  }
}

static void SetJmp(ThreadState *thr, uptr sp, uptr mangled_sp) {
  if (thr->shadow_stack_pos == 0)  // called from libc guts during bootstrap
    return;
  // Cleanup old bufs.
  JmpBufGarbageCollect(thr, sp);
  // Remember the buf.
  JmpBuf *buf = thr->jmp_bufs.PushBack();
  buf->sp = sp;
  buf->mangled_sp = mangled_sp;
  buf->shadow_stack_pos = thr->shadow_stack_pos;
}

static void LongJmp(ThreadState *thr, uptr *env) {
  uptr mangled_sp = env[6];
  // Find the saved buf by mangled_sp.
  for (uptr i = 0; i < thr->jmp_bufs.Size(); i++) {
    JmpBuf *buf = &thr->jmp_bufs[i];
    if (buf->mangled_sp == mangled_sp) {
      CHECK_GE(thr->shadow_stack_pos, buf->shadow_stack_pos);
      // Unwind the stack.
      while (thr->shadow_stack_pos > buf->shadow_stack_pos)
        FuncExit(thr);
      JmpBufGarbageCollect(thr, buf->sp - 1);  // do not collect buf->sp
      return;
    }
  }
  Printf("ThreadSanitizer: can't find longjmp buf\n");
  CHECK(0);
}

extern "C" void __tsan_setjmp(uptr sp, uptr mangled_sp) {
  ScopedInRtl in_rtl;
  SetJmp(cur_thread(), sp, mangled_sp);
}

// Not called.  Merely to satisfy TSAN_INTERCEPT().
extern "C" int __interceptor_setjmp(void *env) {
  CHECK(0);
  return 0;
}

extern "C" int __interceptor__setjmp(void *env) {
  CHECK(0);
  return 0;
}

extern "C" int __interceptor_sigsetjmp(void *env) {
  CHECK(0);
  return 0;
}

extern "C" int __interceptor___sigsetjmp(void *env) {
  CHECK(0);
  return 0;
}

extern "C" int setjmp(void *env);
extern "C" int _setjmp(void *env);
extern "C" int sigsetjmp(void *env);
extern "C" int __sigsetjmp(void *env);
DEFINE_REAL(int, setjmp, void *env)
DEFINE_REAL(int, _setjmp, void *env)
DEFINE_REAL(int, sigsetjmp, void *env)
DEFINE_REAL(int, __sigsetjmp, void *env)

TSAN_INTERCEPTOR(void, longjmp, uptr *env, int val) {
  {
    SCOPED_TSAN_INTERCEPTOR(longjmp, env, val);
  }
  LongJmp(cur_thread(), env);
  REAL(longjmp)(env, val);
}

TSAN_INTERCEPTOR(void, siglongjmp, uptr *env, int val) {
  {
    SCOPED_TSAN_INTERCEPTOR(siglongjmp, env, val);
  }
  LongJmp(cur_thread(), env);
  REAL(siglongjmp)(env, val);
}

TSAN_INTERCEPTOR(void*, malloc, uptr size) {
  if (cur_thread()->in_symbolizer)
    return __libc_malloc(size);
  void *p = 0;
  {
    SCOPED_INTERCEPTOR_RAW(malloc, size);
    p = user_alloc(thr, pc, size);
  }
  invoke_malloc_hook(p, size);
  return p;
}

TSAN_INTERCEPTOR(void*, __libc_memalign, uptr align, uptr sz) {
  SCOPED_TSAN_INTERCEPTOR(__libc_memalign, align, sz);
  return user_alloc(thr, pc, sz, align);
}

TSAN_INTERCEPTOR(void*, calloc, uptr size, uptr n) {
  if (cur_thread()->in_symbolizer)
    return __libc_calloc(size, n);
  if (__sanitizer::CallocShouldReturnNullDueToOverflow(size, n)) return 0;
  void *p = 0;
  {
    SCOPED_INTERCEPTOR_RAW(calloc, size, n);
    p = user_alloc(thr, pc, n * size);
    if (p)
      internal_memset(p, 0, n * size);
  }
  invoke_malloc_hook(p, n * size);
  return p;
}

TSAN_INTERCEPTOR(void*, realloc, void *p, uptr size) {
  if (cur_thread()->in_symbolizer)
    return __libc_realloc(p, size);
  if (p)
    invoke_free_hook(p);
  {
    SCOPED_INTERCEPTOR_RAW(realloc, p, size);
    p = user_realloc(thr, pc, p, size);
  }
  invoke_malloc_hook(p, size);
  return p;
}

TSAN_INTERCEPTOR(void, free, void *p) {
  if (p == 0)
    return;
  if (cur_thread()->in_symbolizer)
    return __libc_free(p);
  invoke_free_hook(p);
  SCOPED_INTERCEPTOR_RAW(free, p);
  user_free(thr, pc, p);
}

TSAN_INTERCEPTOR(void, cfree, void *p) {
  if (p == 0)
    return;
  if (cur_thread()->in_symbolizer)
    return __libc_free(p);
  invoke_free_hook(p);
  SCOPED_INTERCEPTOR_RAW(cfree, p);
  user_free(thr, pc, p);
}

TSAN_INTERCEPTOR(uptr, malloc_usable_size, void *p) {
  SCOPED_INTERCEPTOR_RAW(malloc_usable_size, p);
  return user_alloc_usable_size(thr, pc, p);
}

#define OPERATOR_NEW_BODY(mangled_name) \
  if (cur_thread()->in_symbolizer) \
    return __libc_malloc(size); \
  void *p = 0; \
  {  \
    SCOPED_INTERCEPTOR_RAW(mangled_name, size); \
    p = user_alloc(thr, pc, size); \
  }  \
  invoke_malloc_hook(p, size);  \
  return p;

void *operator new(__sanitizer::uptr size) {
  OPERATOR_NEW_BODY(_Znwm);
}
void *operator new[](__sanitizer::uptr size) {
  OPERATOR_NEW_BODY(_Znam);
}
void *operator new(__sanitizer::uptr size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(_ZnwmRKSt9nothrow_t);
}
void *operator new[](__sanitizer::uptr size, std::nothrow_t const&) {
  OPERATOR_NEW_BODY(_ZnamRKSt9nothrow_t);
}

#define OPERATOR_DELETE_BODY(mangled_name) \
  if (ptr == 0) return;  \
  if (cur_thread()->in_symbolizer) \
    return __libc_free(ptr); \
  invoke_free_hook(ptr);  \
  SCOPED_INTERCEPTOR_RAW(mangled_name, ptr);  \
  user_free(thr, pc, ptr);

void operator delete(void *ptr) {
  OPERATOR_DELETE_BODY(_ZdlPv);
}
void operator delete[](void *ptr) {
  OPERATOR_DELETE_BODY(_ZdlPvRKSt9nothrow_t);
}
void operator delete(void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(_ZdaPv);
}
void operator delete[](void *ptr, std::nothrow_t const&) {
  OPERATOR_DELETE_BODY(_ZdaPvRKSt9nothrow_t);
}

TSAN_INTERCEPTOR(uptr, strlen, const char *s) {
  SCOPED_TSAN_INTERCEPTOR(strlen, s);
  uptr len = internal_strlen(s);
  MemoryAccessRange(thr, pc, (uptr)s, len + 1, false);
  return len;
}

TSAN_INTERCEPTOR(void*, memset, void *dst, int v, uptr size) {
  SCOPED_TSAN_INTERCEPTOR(memset, dst, v, size);
  MemoryAccessRange(thr, pc, (uptr)dst, size, true);
  return internal_memset(dst, v, size);
}

TSAN_INTERCEPTOR(void*, memcpy, void *dst, const void *src, uptr size) {
  SCOPED_TSAN_INTERCEPTOR(memcpy, dst, src, size);
  MemoryAccessRange(thr, pc, (uptr)dst, size, true);
  MemoryAccessRange(thr, pc, (uptr)src, size, false);
  return internal_memcpy(dst, src, size);
}

TSAN_INTERCEPTOR(int, memcmp, const void *s1, const void *s2, uptr n) {
  SCOPED_TSAN_INTERCEPTOR(memcmp, s1, s2, n);
  int res = 0;
  uptr len = 0;
  for (; len < n; len++) {
    if ((res = ((unsigned char*)s1)[len] - ((unsigned char*)s2)[len]))
      break;
  }
  MemoryAccessRange(thr, pc, (uptr)s1, len < n ? len + 1 : n, false);
  MemoryAccessRange(thr, pc, (uptr)s2, len < n ? len + 1 : n, false);
  return res;
}

TSAN_INTERCEPTOR(int, strcmp, const char *s1, const char *s2) {
  SCOPED_TSAN_INTERCEPTOR(strcmp, s1, s2);
  uptr len = 0;
  for (; s1[len] && s2[len]; len++) {
    if (s1[len] != s2[len])
      break;
  }
  MemoryAccessRange(thr, pc, (uptr)s1, len + 1, false);
  MemoryAccessRange(thr, pc, (uptr)s2, len + 1, false);
  return s1[len] - s2[len];
}

TSAN_INTERCEPTOR(int, strncmp, const char *s1, const char *s2, uptr n) {
  SCOPED_TSAN_INTERCEPTOR(strncmp, s1, s2, n);
  uptr len = 0;
  for (; len < n && s1[len] && s2[len]; len++) {
    if (s1[len] != s2[len])
      break;
  }
  MemoryAccessRange(thr, pc, (uptr)s1, len < n ? len + 1 : n, false);
  MemoryAccessRange(thr, pc, (uptr)s2, len < n ? len + 1 : n, false);
  return len == n ? 0 : s1[len] - s2[len];
}

TSAN_INTERCEPTOR(void*, memchr, void *s, int c, uptr n) {
  SCOPED_TSAN_INTERCEPTOR(memchr, s, c, n);
  void *res = REAL(memchr)(s, c, n);
  uptr len = res ? (char*)res - (char*)s + 1 : n;
  MemoryAccessRange(thr, pc, (uptr)s, len, false);
  return res;
}

TSAN_INTERCEPTOR(void*, memrchr, char *s, int c, uptr n) {
  SCOPED_TSAN_INTERCEPTOR(memrchr, s, c, n);
  MemoryAccessRange(thr, pc, (uptr)s, n, false);
  return REAL(memrchr)(s, c, n);
}

TSAN_INTERCEPTOR(void*, memmove, void *dst, void *src, uptr n) {
  SCOPED_TSAN_INTERCEPTOR(memmove, dst, src, n);
  MemoryAccessRange(thr, pc, (uptr)dst, n, true);
  MemoryAccessRange(thr, pc, (uptr)src, n, false);
  return REAL(memmove)(dst, src, n);
}

TSAN_INTERCEPTOR(char*, strchr, char *s, int c) {
  SCOPED_TSAN_INTERCEPTOR(strchr, s, c);
  char *res = REAL(strchr)(s, c);
  uptr len = res ? (char*)res - (char*)s + 1 : internal_strlen(s) + 1;
  MemoryAccessRange(thr, pc, (uptr)s, len, false);
  return res;
}

TSAN_INTERCEPTOR(char*, strchrnul, char *s, int c) {
  SCOPED_TSAN_INTERCEPTOR(strchrnul, s, c);
  char *res = REAL(strchrnul)(s, c);
  uptr len = (char*)res - (char*)s + 1;
  MemoryAccessRange(thr, pc, (uptr)s, len, false);
  return res;
}

TSAN_INTERCEPTOR(char*, strrchr, char *s, int c) {
  SCOPED_TSAN_INTERCEPTOR(strrchr, s, c);
  MemoryAccessRange(thr, pc, (uptr)s, internal_strlen(s) + 1, false);
  return REAL(strrchr)(s, c);
}

TSAN_INTERCEPTOR(char*, strcpy, char *dst, const char *src) {  // NOLINT
  SCOPED_TSAN_INTERCEPTOR(strcpy, dst, src);  // NOLINT
  uptr srclen = internal_strlen(src);
  MemoryAccessRange(thr, pc, (uptr)dst, srclen + 1, true);
  MemoryAccessRange(thr, pc, (uptr)src, srclen + 1, false);
  return REAL(strcpy)(dst, src);  // NOLINT
}

TSAN_INTERCEPTOR(char*, strncpy, char *dst, char *src, uptr n) {
  SCOPED_TSAN_INTERCEPTOR(strncpy, dst, src, n);
  uptr srclen = internal_strnlen(src, n);
  MemoryAccessRange(thr, pc, (uptr)dst, n, true);
  MemoryAccessRange(thr, pc, (uptr)src, min(srclen + 1, n), false);
  return REAL(strncpy)(dst, src, n);
}

TSAN_INTERCEPTOR(const char*, strstr, const char *s1, const char *s2) {
  SCOPED_TSAN_INTERCEPTOR(strstr, s1, s2);
  const char *res = REAL(strstr)(s1, s2);
  uptr len1 = internal_strlen(s1);
  uptr len2 = internal_strlen(s2);
  MemoryAccessRange(thr, pc, (uptr)s1, len1 + 1, false);
  MemoryAccessRange(thr, pc, (uptr)s2, len2 + 1, false);
  return res;
}

static bool fix_mmap_addr(void **addr, long_t sz, int flags) {
  if (*addr) {
    if (!IsAppMem((uptr)*addr) || !IsAppMem((uptr)*addr + sz - 1)) {
      if (flags & MAP_FIXED) {
        errno = EINVAL;
        return false;
      } else {
        *addr = 0;
      }
    }
  }
  return true;
}

TSAN_INTERCEPTOR(void*, mmap, void *addr, long_t sz, int prot,
                         int flags, int fd, unsigned off) {
  SCOPED_TSAN_INTERCEPTOR(mmap, addr, sz, prot, flags, fd, off);
  if (!fix_mmap_addr(&addr, sz, flags))
    return MAP_FAILED;
  void *res = REAL(mmap)(addr, sz, prot, flags, fd, off);
  if (res != MAP_FAILED) {
    if (fd > 0)
      FdAccess(thr, pc, fd);
    MemoryRangeImitateWrite(thr, pc, (uptr)res, sz);
  }
  return res;
}

TSAN_INTERCEPTOR(void*, mmap64, void *addr, long_t sz, int prot,
                           int flags, int fd, u64 off) {
  SCOPED_TSAN_INTERCEPTOR(mmap64, addr, sz, prot, flags, fd, off);
  if (!fix_mmap_addr(&addr, sz, flags))
    return MAP_FAILED;
  void *res = REAL(mmap64)(addr, sz, prot, flags, fd, off);
  if (res != MAP_FAILED) {
    if (fd > 0)
      FdAccess(thr, pc, fd);
    MemoryRangeImitateWrite(thr, pc, (uptr)res, sz);
  }
  return res;
}

TSAN_INTERCEPTOR(int, munmap, void *addr, long_t sz) {
  SCOPED_TSAN_INTERCEPTOR(munmap, addr, sz);
  DontNeedShadowFor((uptr)addr, sz);
  int res = REAL(munmap)(addr, sz);
  return res;
}

TSAN_INTERCEPTOR(void*, memalign, uptr align, uptr sz) {
  SCOPED_TSAN_INTERCEPTOR(memalign, align, sz);
  return user_alloc(thr, pc, sz, align);
}

TSAN_INTERCEPTOR(void*, valloc, uptr sz) {
  SCOPED_TSAN_INTERCEPTOR(valloc, sz);
  return user_alloc(thr, pc, sz, GetPageSizeCached());
}

TSAN_INTERCEPTOR(void*, pvalloc, uptr sz) {
  SCOPED_TSAN_INTERCEPTOR(pvalloc, sz);
  sz = RoundUp(sz, GetPageSizeCached());
  return user_alloc(thr, pc, sz, GetPageSizeCached());
}

TSAN_INTERCEPTOR(int, posix_memalign, void **memptr, uptr align, uptr sz) {
  SCOPED_TSAN_INTERCEPTOR(posix_memalign, memptr, align, sz);
  *memptr = user_alloc(thr, pc, sz, align);
  return 0;
}

// Used in thread-safe function static initialization.
extern "C" int INTERFACE_ATTRIBUTE __cxa_guard_acquire(atomic_uint32_t *g) {
  SCOPED_INTERCEPTOR_RAW(__cxa_guard_acquire, g);
  for (;;) {
    u32 cmp = atomic_load(g, memory_order_acquire);
    if (cmp == 0) {
      if (atomic_compare_exchange_strong(g, &cmp, 1<<16, memory_order_relaxed))
        return 1;
    } else if (cmp == 1) {
      Acquire(thr, pc, (uptr)g);
      return 0;
    } else {
      internal_sched_yield();
    }
  }
}

extern "C" void INTERFACE_ATTRIBUTE __cxa_guard_release(atomic_uint32_t *g) {
  SCOPED_INTERCEPTOR_RAW(__cxa_guard_release, g);
  Release(thr, pc, (uptr)g);
  atomic_store(g, 1, memory_order_release);
}

extern "C" void INTERFACE_ATTRIBUTE __cxa_guard_abort(atomic_uint32_t *g) {
  SCOPED_INTERCEPTOR_RAW(__cxa_guard_abort, g);
  atomic_store(g, 0, memory_order_relaxed);
}

static void thread_finalize(void *v) {
  uptr iter = (uptr)v;
  if (iter > 1) {
    if (pthread_setspecific(g_thread_finalize_key, (void*)(iter - 1))) {
      Printf("ThreadSanitizer: failed to set thread key\n");
      Die();
    }
    return;
  }
  {
    ScopedInRtl in_rtl;
    ThreadState *thr = cur_thread();
    ThreadFinish(thr);
    SignalContext *sctx = thr->signal_ctx;
    if (sctx) {
      thr->signal_ctx = 0;
      UnmapOrDie(sctx, sizeof(*sctx));
    }
  }
}


struct ThreadParam {
  void* (*callback)(void *arg);
  void *param;
  atomic_uintptr_t tid;
};

extern "C" void *__tsan_thread_start_func(void *arg) {
  ThreadParam *p = (ThreadParam*)arg;
  void* (*callback)(void *arg) = p->callback;
  void *param = p->param;
  int tid = 0;
  {
    ThreadState *thr = cur_thread();
    ScopedInRtl in_rtl;
    if (pthread_setspecific(g_thread_finalize_key, (void*)4)) {
      Printf("ThreadSanitizer: failed to set thread key\n");
      Die();
    }
    while ((tid = atomic_load(&p->tid, memory_order_acquire)) == 0)
      pthread_yield();
    atomic_store(&p->tid, 0, memory_order_release);
    ThreadStart(thr, tid, GetTid());
    CHECK_EQ(thr->in_rtl, 1);
  }
  void *res = callback(param);
  // Prevent the callback from being tail called,
  // it mixes up stack traces.
  volatile int foo = 42;
  foo++;
  return res;
}

TSAN_INTERCEPTOR(int, pthread_create,
    void *th, void *attr, void *(*callback)(void*), void * param) {
  SCOPED_TSAN_INTERCEPTOR(pthread_create, th, attr, callback, param);
  __sanitizer_pthread_attr_t myattr;
  if (attr == 0) {
    pthread_attr_init(&myattr);
    attr = &myattr;
  }
  int detached = 0;
  pthread_attr_getdetachstate(attr, &detached);

#if defined(TSAN_DEBUG_OUTPUT)
  int verbosity = (TSAN_DEBUG_OUTPUT);
#else
  int verbosity = 0;
#endif
  AdjustStackSizeLinux(attr, verbosity);

  ThreadParam p;
  p.callback = callback;
  p.param = param;
  atomic_store(&p.tid, 0, memory_order_relaxed);
  int res = REAL(pthread_create)(th, attr, __tsan_thread_start_func, &p);
  if (res == 0) {
    int tid = ThreadCreate(thr, pc, *(uptr*)th, detached);
    CHECK_NE(tid, 0);
    atomic_store(&p.tid, tid, memory_order_release);
    while (atomic_load(&p.tid, memory_order_acquire) != 0)
      pthread_yield();
  }
  if (attr == &myattr)
    pthread_attr_destroy(&myattr);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_join, void *th, void **ret) {
  SCOPED_TSAN_INTERCEPTOR(pthread_join, th, ret);
  int tid = ThreadTid(thr, pc, (uptr)th);
  int res = BLOCK_REAL(pthread_join)(th, ret);
  if (res == 0) {
    ThreadJoin(thr, pc, tid);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_detach, void *th) {
  SCOPED_TSAN_INTERCEPTOR(pthread_detach, th);
  int tid = ThreadTid(thr, pc, (uptr)th);
  int res = REAL(pthread_detach)(th);
  if (res == 0) {
    ThreadDetach(thr, pc, tid);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_mutex_init, void *m, void *a) {
  SCOPED_TSAN_INTERCEPTOR(pthread_mutex_init, m, a);
  int res = REAL(pthread_mutex_init)(m, a);
  if (res == 0) {
    bool recursive = false;
    if (a) {
      int type = 0;
      if (pthread_mutexattr_gettype(a, &type) == 0)
        recursive = (type == PTHREAD_MUTEX_RECURSIVE
            || type == PTHREAD_MUTEX_RECURSIVE_NP);
    }
    MutexCreate(thr, pc, (uptr)m, false, recursive, false);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_mutex_destroy, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_mutex_destroy, m);
  int res = REAL(pthread_mutex_destroy)(m);
  if (res == 0 || res == EBUSY) {
    MutexDestroy(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_mutex_lock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_mutex_lock, m);
  int res = REAL(pthread_mutex_lock)(m);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_mutex_trylock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_mutex_trylock, m);
  int res = REAL(pthread_mutex_trylock)(m);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_mutex_timedlock, void *m, void *abstime) {
  SCOPED_TSAN_INTERCEPTOR(pthread_mutex_timedlock, m, abstime);
  int res = REAL(pthread_mutex_timedlock)(m, abstime);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_mutex_unlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_mutex_unlock, m);
  MutexUnlock(thr, pc, (uptr)m);
  int res = REAL(pthread_mutex_unlock)(m);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_spin_init, void *m, int pshared) {
  SCOPED_TSAN_INTERCEPTOR(pthread_spin_init, m, pshared);
  int res = REAL(pthread_spin_init)(m, pshared);
  if (res == 0) {
    MutexCreate(thr, pc, (uptr)m, false, false, false);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_spin_destroy, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_spin_destroy, m);
  int res = REAL(pthread_spin_destroy)(m);
  if (res == 0) {
    MutexDestroy(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_spin_lock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_spin_lock, m);
  int res = REAL(pthread_spin_lock)(m);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_spin_trylock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_spin_trylock, m);
  int res = REAL(pthread_spin_trylock)(m);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_spin_unlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_spin_unlock, m);
  MutexUnlock(thr, pc, (uptr)m);
  int res = REAL(pthread_spin_unlock)(m);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_init, void *m, void *a) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_init, m, a);
  int res = REAL(pthread_rwlock_init)(m, a);
  if (res == 0) {
    MutexCreate(thr, pc, (uptr)m, true, false, false);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_destroy, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_destroy, m);
  int res = REAL(pthread_rwlock_destroy)(m);
  if (res == 0) {
    MutexDestroy(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_rdlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_rdlock, m);
  int res = REAL(pthread_rwlock_rdlock)(m);
  if (res == 0) {
    MutexReadLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_tryrdlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_tryrdlock, m);
  int res = REAL(pthread_rwlock_tryrdlock)(m);
  if (res == 0) {
    MutexReadLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_timedrdlock, void *m, void *abstime) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_timedrdlock, m, abstime);
  int res = REAL(pthread_rwlock_timedrdlock)(m, abstime);
  if (res == 0) {
    MutexReadLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_wrlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_wrlock, m);
  int res = REAL(pthread_rwlock_wrlock)(m);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_trywrlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_trywrlock, m);
  int res = REAL(pthread_rwlock_trywrlock)(m);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_timedwrlock, void *m, void *abstime) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_timedwrlock, m, abstime);
  int res = REAL(pthread_rwlock_timedwrlock)(m, abstime);
  if (res == 0) {
    MutexLock(thr, pc, (uptr)m);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_rwlock_unlock, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_rwlock_unlock, m);
  MutexReadOrWriteUnlock(thr, pc, (uptr)m);
  int res = REAL(pthread_rwlock_unlock)(m);
  return res;
}

// libpthread.so contains several versions of pthread_cond_init symbol.
// When we just dlsym() it, we get the wrong (old) version.
/*
TSAN_INTERCEPTOR(int, pthread_cond_init, void *c, void *a) {
  SCOPED_TSAN_INTERCEPTOR(pthread_cond_init, c, a);
  int res = REAL(pthread_cond_init)(c, a);
  return res;
}
*/

TSAN_INTERCEPTOR(int, pthread_cond_destroy, void *c) {
  SCOPED_TSAN_INTERCEPTOR(pthread_cond_destroy, c);
  int res = REAL(pthread_cond_destroy)(c);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_cond_signal, void *c) {
  SCOPED_TSAN_INTERCEPTOR(pthread_cond_signal, c);
  int res = REAL(pthread_cond_signal)(c);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_cond_broadcast, void *c) {
  SCOPED_TSAN_INTERCEPTOR(pthread_cond_broadcast, c);
  int res = REAL(pthread_cond_broadcast)(c);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_cond_wait, void *c, void *m) {
  SCOPED_TSAN_INTERCEPTOR(pthread_cond_wait, c, m);
  MutexUnlock(thr, pc, (uptr)m);
  int res = REAL(pthread_cond_wait)(c, m);
  MutexLock(thr, pc, (uptr)m);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_cond_timedwait, void *c, void *m, void *abstime) {
  SCOPED_TSAN_INTERCEPTOR(pthread_cond_timedwait, c, m, abstime);
  MutexUnlock(thr, pc, (uptr)m);
  int res = REAL(pthread_cond_timedwait)(c, m, abstime);
  MutexLock(thr, pc, (uptr)m);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_barrier_init, void *b, void *a, unsigned count) {
  SCOPED_TSAN_INTERCEPTOR(pthread_barrier_init, b, a, count);
  MemoryWrite(thr, pc, (uptr)b, kSizeLog1);
  int res = REAL(pthread_barrier_init)(b, a, count);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_barrier_destroy, void *b) {
  SCOPED_TSAN_INTERCEPTOR(pthread_barrier_destroy, b);
  MemoryWrite(thr, pc, (uptr)b, kSizeLog1);
  int res = REAL(pthread_barrier_destroy)(b);
  return res;
}

TSAN_INTERCEPTOR(int, pthread_barrier_wait, void *b) {
  SCOPED_TSAN_INTERCEPTOR(pthread_barrier_wait, b);
  Release(thr, pc, (uptr)b);
  MemoryRead(thr, pc, (uptr)b, kSizeLog1);
  int res = REAL(pthread_barrier_wait)(b);
  MemoryRead(thr, pc, (uptr)b, kSizeLog1);
  if (res == 0 || res == PTHREAD_BARRIER_SERIAL_THREAD) {
    Acquire(thr, pc, (uptr)b);
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_once, void *o, void (*f)()) {
  SCOPED_TSAN_INTERCEPTOR(pthread_once, o, f);
  if (o == 0 || f == 0)
    return EINVAL;
  atomic_uint32_t *a = static_cast<atomic_uint32_t*>(o);
  u32 v = atomic_load(a, memory_order_acquire);
  if (v == 0 && atomic_compare_exchange_strong(a, &v, 1,
                                               memory_order_relaxed)) {
    const int old_in_rtl = thr->in_rtl;
    thr->in_rtl = 0;
    (*f)();
    CHECK_EQ(thr->in_rtl, 0);
    thr->in_rtl = old_in_rtl;
    Release(thr, pc, (uptr)o);
    atomic_store(a, 2, memory_order_release);
  } else {
    while (v != 2) {
      pthread_yield();
      v = atomic_load(a, memory_order_acquire);
    }
    Acquire(thr, pc, (uptr)o);
  }
  return 0;
}

TSAN_INTERCEPTOR(int, sem_init, void *s, int pshared, unsigned value) {
  SCOPED_TSAN_INTERCEPTOR(sem_init, s, pshared, value);
  int res = REAL(sem_init)(s, pshared, value);
  return res;
}

TSAN_INTERCEPTOR(int, sem_destroy, void *s) {
  SCOPED_TSAN_INTERCEPTOR(sem_destroy, s);
  int res = REAL(sem_destroy)(s);
  return res;
}

TSAN_INTERCEPTOR(int, sem_wait, void *s) {
  SCOPED_TSAN_INTERCEPTOR(sem_wait, s);
  int res = BLOCK_REAL(sem_wait)(s);
  if (res == 0) {
    Acquire(thr, pc, (uptr)s);
  }
  return res;
}

TSAN_INTERCEPTOR(int, sem_trywait, void *s) {
  SCOPED_TSAN_INTERCEPTOR(sem_trywait, s);
  int res = BLOCK_REAL(sem_trywait)(s);
  if (res == 0) {
    Acquire(thr, pc, (uptr)s);
  }
  return res;
}

TSAN_INTERCEPTOR(int, sem_timedwait, void *s, void *abstime) {
  SCOPED_TSAN_INTERCEPTOR(sem_timedwait, s, abstime);
  int res = BLOCK_REAL(sem_timedwait)(s, abstime);
  if (res == 0) {
    Acquire(thr, pc, (uptr)s);
  }
  return res;
}

TSAN_INTERCEPTOR(int, sem_post, void *s) {
  SCOPED_TSAN_INTERCEPTOR(sem_post, s);
  Release(thr, pc, (uptr)s);
  int res = REAL(sem_post)(s);
  return res;
}

TSAN_INTERCEPTOR(int, sem_getvalue, void *s, int *sval) {
  SCOPED_TSAN_INTERCEPTOR(sem_getvalue, s, sval);
  int res = REAL(sem_getvalue)(s, sval);
  if (res == 0) {
    Acquire(thr, pc, (uptr)s);
  }
  return res;
}

TSAN_INTERCEPTOR(int, __xstat, int version, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__xstat, version, path, buf);
  return REAL(__xstat)(version, path, buf);
}

TSAN_INTERCEPTOR(int, stat, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__xstat, 0, path, buf);
  return REAL(__xstat)(0, path, buf);
}

TSAN_INTERCEPTOR(int, __xstat64, int version, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__xstat64, version, path, buf);
  return REAL(__xstat64)(version, path, buf);
}

TSAN_INTERCEPTOR(int, stat64, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__xstat64, 0, path, buf);
  return REAL(__xstat64)(0, path, buf);
}

TSAN_INTERCEPTOR(int, __lxstat, int version, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__lxstat, version, path, buf);
  return REAL(__lxstat)(version, path, buf);
}

TSAN_INTERCEPTOR(int, lstat, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__lxstat, 0, path, buf);
  return REAL(__lxstat)(0, path, buf);
}

TSAN_INTERCEPTOR(int, __lxstat64, int version, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__lxstat64, version, path, buf);
  return REAL(__lxstat64)(version, path, buf);
}

TSAN_INTERCEPTOR(int, lstat64, const char *path, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__lxstat64, 0, path, buf);
  return REAL(__lxstat64)(0, path, buf);
}

TSAN_INTERCEPTOR(int, __fxstat, int version, int fd, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__fxstat, version, fd, buf);
  if (fd > 0)
    FdAccess(thr, pc, fd);
  return REAL(__fxstat)(version, fd, buf);
}

TSAN_INTERCEPTOR(int, fstat, int fd, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__fxstat, 0, fd, buf);
  if (fd > 0)
    FdAccess(thr, pc, fd);
  return REAL(__fxstat)(0, fd, buf);
}

TSAN_INTERCEPTOR(int, __fxstat64, int version, int fd, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__fxstat64, version, fd, buf);
  if (fd > 0)
    FdAccess(thr, pc, fd);
  return REAL(__fxstat64)(version, fd, buf);
}

TSAN_INTERCEPTOR(int, fstat64, int fd, void *buf) {
  SCOPED_TSAN_INTERCEPTOR(__fxstat64, 0, fd, buf);
  if (fd > 0)
    FdAccess(thr, pc, fd);
  return REAL(__fxstat64)(0, fd, buf);
}

TSAN_INTERCEPTOR(int, open, const char *name, int flags, int mode) {
  SCOPED_TSAN_INTERCEPTOR(open, name, flags, mode);
  int fd = REAL(open)(name, flags, mode);
  if (fd >= 0)
    FdFileCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, open64, const char *name, int flags, int mode) {
  SCOPED_TSAN_INTERCEPTOR(open64, name, flags, mode);
  int fd = REAL(open64)(name, flags, mode);
  if (fd >= 0)
    FdFileCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, creat, const char *name, int mode) {
  SCOPED_TSAN_INTERCEPTOR(creat, name, mode);
  int fd = REAL(creat)(name, mode);
  if (fd >= 0)
    FdFileCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, creat64, const char *name, int mode) {
  SCOPED_TSAN_INTERCEPTOR(creat64, name, mode);
  int fd = REAL(creat64)(name, mode);
  if (fd >= 0)
    FdFileCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, dup, int oldfd) {
  SCOPED_TSAN_INTERCEPTOR(dup, oldfd);
  int newfd = REAL(dup)(oldfd);
  if (oldfd >= 0 && newfd >= 0 && newfd != oldfd)
    FdDup(thr, pc, oldfd, newfd);
  return newfd;
}

TSAN_INTERCEPTOR(int, dup2, int oldfd, int newfd) {
  SCOPED_TSAN_INTERCEPTOR(dup2, oldfd, newfd);
  int newfd2 = REAL(dup2)(oldfd, newfd);
  if (oldfd >= 0 && newfd2 >= 0 && newfd2 != oldfd)
    FdDup(thr, pc, oldfd, newfd2);
  return newfd2;
}

TSAN_INTERCEPTOR(int, dup3, int oldfd, int newfd, int flags) {
  SCOPED_TSAN_INTERCEPTOR(dup3, oldfd, newfd, flags);
  int newfd2 = REAL(dup3)(oldfd, newfd, flags);
  if (oldfd >= 0 && newfd2 >= 0 && newfd2 != oldfd)
    FdDup(thr, pc, oldfd, newfd2);
  return newfd2;
}

TSAN_INTERCEPTOR(int, eventfd, unsigned initval, int flags) {
  SCOPED_TSAN_INTERCEPTOR(eventfd, initval, flags);
  int fd = REAL(eventfd)(initval, flags);
  if (fd >= 0)
    FdEventCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, signalfd, int fd, void *mask, int flags) {
  SCOPED_TSAN_INTERCEPTOR(signalfd, fd, mask, flags);
  if (fd >= 0)
    FdClose(thr, pc, fd);
  fd = REAL(signalfd)(fd, mask, flags);
  if (fd >= 0)
    FdSignalCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, inotify_init, int fake) {
  SCOPED_TSAN_INTERCEPTOR(inotify_init, fake);
  int fd = REAL(inotify_init)(fake);
  if (fd >= 0)
    FdInotifyCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, inotify_init1, int flags) {
  SCOPED_TSAN_INTERCEPTOR(inotify_init1, flags);
  int fd = REAL(inotify_init1)(flags);
  if (fd >= 0)
    FdInotifyCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, socket, int domain, int type, int protocol) {
  SCOPED_TSAN_INTERCEPTOR(socket, domain, type, protocol);
  int fd = REAL(socket)(domain, type, protocol);
  if (fd >= 0)
    FdSocketCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, socketpair, int domain, int type, int protocol, int *fd) {
  SCOPED_TSAN_INTERCEPTOR(socketpair, domain, type, protocol, fd);
  int res = REAL(socketpair)(domain, type, protocol, fd);
  if (res == 0 && fd[0] >= 0 && fd[1] >= 0)
    FdPipeCreate(thr, pc, fd[0], fd[1]);
  return res;
}

TSAN_INTERCEPTOR(int, connect, int fd, void *addr, unsigned addrlen) {
  SCOPED_TSAN_INTERCEPTOR(connect, fd, addr, addrlen);
  FdSocketConnecting(thr, pc, fd);
  int res = REAL(connect)(fd, addr, addrlen);
  if (res == 0 && fd >= 0)
    FdSocketConnect(thr, pc, fd);
  return res;
}

TSAN_INTERCEPTOR(int, bind, int fd, void *addr, unsigned addrlen) {
  SCOPED_TSAN_INTERCEPTOR(bind, fd, addr, addrlen);
  int res = REAL(bind)(fd, addr, addrlen);
  if (fd > 0 && res == 0)
    FdAccess(thr, pc, fd);
  return res;
}

TSAN_INTERCEPTOR(int, listen, int fd, int backlog) {
  SCOPED_TSAN_INTERCEPTOR(listen, fd, backlog);
  int res = REAL(listen)(fd, backlog);
  if (fd > 0 && res == 0)
    FdAccess(thr, pc, fd);
  return res;
}

TSAN_INTERCEPTOR(int, accept, int fd, void *addr, unsigned *addrlen) {
  SCOPED_TSAN_INTERCEPTOR(accept, fd, addr, addrlen);
  int fd2 = REAL(accept)(fd, addr, addrlen);
  if (fd >= 0 && fd2 >= 0)
    FdSocketAccept(thr, pc, fd, fd2);
  return fd2;
}

TSAN_INTERCEPTOR(int, accept4, int fd, void *addr, unsigned *addrlen, int f) {
  SCOPED_TSAN_INTERCEPTOR(accept4, fd, addr, addrlen, f);
  int fd2 = REAL(accept4)(fd, addr, addrlen, f);
  if (fd >= 0 && fd2 >= 0)
    FdSocketAccept(thr, pc, fd, fd2);
  return fd2;
}

TSAN_INTERCEPTOR(int, epoll_create, int size) {
  SCOPED_TSAN_INTERCEPTOR(epoll_create, size);
  int fd = REAL(epoll_create)(size);
  if (fd >= 0)
    FdPollCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, epoll_create1, int flags) {
  SCOPED_TSAN_INTERCEPTOR(epoll_create1, flags);
  int fd = REAL(epoll_create1)(flags);
  if (fd >= 0)
    FdPollCreate(thr, pc, fd);
  return fd;
}

TSAN_INTERCEPTOR(int, close, int fd) {
  SCOPED_TSAN_INTERCEPTOR(close, fd);
  if (fd >= 0)
    FdClose(thr, pc, fd);
  return REAL(close)(fd);
}

TSAN_INTERCEPTOR(int, __close, int fd) {
  SCOPED_TSAN_INTERCEPTOR(__close, fd);
  if (fd >= 0)
    FdClose(thr, pc, fd);
  return REAL(__close)(fd);
}

// glibc guts
TSAN_INTERCEPTOR(void, __res_iclose, void *state, bool free_addr) {
  SCOPED_TSAN_INTERCEPTOR(__res_iclose, state, free_addr);
  int fds[64];
  int cnt = ExtractResolvFDs(state, fds, ARRAY_SIZE(fds));
  for (int i = 0; i < cnt; i++) {
    if (fds[i] > 0)
      FdClose(thr, pc, fds[i]);
  }
  REAL(__res_iclose)(state, free_addr);
}

TSAN_INTERCEPTOR(int, pipe, int *pipefd) {
  SCOPED_TSAN_INTERCEPTOR(pipe, pipefd);
  int res = REAL(pipe)(pipefd);
  if (res == 0 && pipefd[0] >= 0 && pipefd[1] >= 0)
    FdPipeCreate(thr, pc, pipefd[0], pipefd[1]);
  return res;
}

TSAN_INTERCEPTOR(int, pipe2, int *pipefd, int flags) {
  SCOPED_TSAN_INTERCEPTOR(pipe2, pipefd, flags);
  int res = REAL(pipe2)(pipefd, flags);
  if (res == 0 && pipefd[0] >= 0 && pipefd[1] >= 0)
    FdPipeCreate(thr, pc, pipefd[0], pipefd[1]);
  return res;
}

TSAN_INTERCEPTOR(long_t, readv, int fd, void *vec, int cnt) {
  SCOPED_TSAN_INTERCEPTOR(readv, fd, vec, cnt);
  int res = REAL(readv)(fd, vec, cnt);
  if (res >= 0 && fd >= 0) {
    FdAcquire(thr, pc, fd);
  }
  return res;
}

TSAN_INTERCEPTOR(long_t, preadv64, int fd, void *vec, int cnt, u64 off) {
  SCOPED_TSAN_INTERCEPTOR(preadv64, fd, vec, cnt, off);
  int res = REAL(preadv64)(fd, vec, cnt, off);
  if (res >= 0 && fd >= 0) {
    FdAcquire(thr, pc, fd);
  }
  return res;
}

TSAN_INTERCEPTOR(long_t, writev, int fd, void *vec, int cnt) {
  SCOPED_TSAN_INTERCEPTOR(writev, fd, vec, cnt);
  if (fd >= 0)
    FdRelease(thr, pc, fd);
  int res = REAL(writev)(fd, vec, cnt);
  return res;
}

TSAN_INTERCEPTOR(long_t, pwritev64, int fd, void *vec, int cnt, u64 off) {
  SCOPED_TSAN_INTERCEPTOR(pwritev64, fd, vec, cnt, off);
  if (fd >= 0)
    FdRelease(thr, pc, fd);
  int res = REAL(pwritev64)(fd, vec, cnt, off);
  return res;
}

TSAN_INTERCEPTOR(long_t, send, int fd, void *buf, long_t len, int flags) {
  SCOPED_TSAN_INTERCEPTOR(send, fd, buf, len, flags);
  if (fd >= 0)
    FdRelease(thr, pc, fd);
  int res = REAL(send)(fd, buf, len, flags);
  return res;
}

TSAN_INTERCEPTOR(long_t, sendmsg, int fd, void *msg, int flags) {
  SCOPED_TSAN_INTERCEPTOR(sendmsg, fd, msg, flags);
  if (fd >= 0)
    FdRelease(thr, pc, fd);
  int res = REAL(sendmsg)(fd, msg, flags);
  return res;
}

TSAN_INTERCEPTOR(long_t, recv, int fd, void *buf, long_t len, int flags) {
  SCOPED_TSAN_INTERCEPTOR(recv, fd, buf, len, flags);
  int res = REAL(recv)(fd, buf, len, flags);
  if (res >= 0 && fd >= 0) {
    FdAcquire(thr, pc, fd);
  }
  return res;
}

TSAN_INTERCEPTOR(long_t, recvmsg, int fd, void *msg, int flags) {
  SCOPED_TSAN_INTERCEPTOR(recvmsg, fd, msg, flags);
  int res = REAL(recvmsg)(fd, msg, flags);
  if (res >= 0 && fd >= 0) {
    FdAcquire(thr, pc, fd);
  }
  return res;
}

TSAN_INTERCEPTOR(int, unlink, char *path) {
  SCOPED_TSAN_INTERCEPTOR(unlink, path);
  Release(thr, pc, File2addr(path));
  int res = REAL(unlink)(path);
  return res;
}

TSAN_INTERCEPTOR(void*, fopen, char *path, char *mode) {
  SCOPED_TSAN_INTERCEPTOR(fopen, path, mode);
  void *res = REAL(fopen)(path, mode);
  Acquire(thr, pc, File2addr(path));
  if (res) {
    int fd = fileno_unlocked(res);
    if (fd >= 0)
      FdFileCreate(thr, pc, fd);
  }
  return res;
}

TSAN_INTERCEPTOR(void*, freopen, char *path, char *mode, void *stream) {
  SCOPED_TSAN_INTERCEPTOR(freopen, path, mode, stream);
  if (stream) {
    int fd = fileno_unlocked(stream);
    if (fd >= 0)
      FdClose(thr, pc, fd);
  }
  void *res = REAL(freopen)(path, mode, stream);
  Acquire(thr, pc, File2addr(path));
  if (res) {
    int fd = fileno_unlocked(res);
    if (fd >= 0)
      FdFileCreate(thr, pc, fd);
  }
  return res;
}

TSAN_INTERCEPTOR(int, fclose, void *stream) {
  {
    SCOPED_TSAN_INTERCEPTOR(fclose, stream);
    if (stream) {
      int fd = fileno_unlocked(stream);
      if (fd >= 0)
        FdClose(thr, pc, fd);
    }
  }
  return REAL(fclose)(stream);
}

TSAN_INTERCEPTOR(uptr, fread, void *ptr, uptr size, uptr nmemb, void *f) {
  {
    SCOPED_TSAN_INTERCEPTOR(fread, ptr, size, nmemb, f);
    MemoryAccessRange(thr, pc, (uptr)ptr, size * nmemb, true);
  }
  return REAL(fread)(ptr, size, nmemb, f);
}

TSAN_INTERCEPTOR(uptr, fwrite, const void *p, uptr size, uptr nmemb, void *f) {
  {
    SCOPED_TSAN_INTERCEPTOR(fwrite, p, size, nmemb, f);
    MemoryAccessRange(thr, pc, (uptr)p, size * nmemb, false);
  }
  return REAL(fwrite)(p, size, nmemb, f);
}

TSAN_INTERCEPTOR(int, fflush, void *stream) {
  SCOPED_TSAN_INTERCEPTOR(fflush, stream);
  return REAL(fflush)(stream);
}

TSAN_INTERCEPTOR(void, abort, int fake) {
  SCOPED_TSAN_INTERCEPTOR(abort, fake);
  REAL(fflush)(0);
  REAL(abort)(fake);
}

TSAN_INTERCEPTOR(int, puts, const char *s) {
  SCOPED_TSAN_INTERCEPTOR(puts, s);
  MemoryAccessRange(thr, pc, (uptr)s, internal_strlen(s), false);
  return REAL(puts)(s);
}

TSAN_INTERCEPTOR(int, rmdir, char *path) {
  SCOPED_TSAN_INTERCEPTOR(rmdir, path);
  Release(thr, pc, Dir2addr(path));
  int res = REAL(rmdir)(path);
  return res;
}

TSAN_INTERCEPTOR(void*, opendir, char *path) {
  SCOPED_TSAN_INTERCEPTOR(opendir, path);
  void *res = REAL(opendir)(path);
  if (res != 0)
    Acquire(thr, pc, Dir2addr(path));
  return res;
}

TSAN_INTERCEPTOR(int, epoll_ctl, int epfd, int op, int fd, void *ev) {
  SCOPED_TSAN_INTERCEPTOR(epoll_ctl, epfd, op, fd, ev);
  if (op == EPOLL_CTL_ADD && epfd >= 0) {
    FdRelease(thr, pc, epfd);
  }
  int res = REAL(epoll_ctl)(epfd, op, fd, ev);
  if (fd >= 0)
    FdAccess(thr, pc, fd);
  return res;
}

TSAN_INTERCEPTOR(int, epoll_wait, int epfd, void *ev, int cnt, int timeout) {
  SCOPED_TSAN_INTERCEPTOR(epoll_wait, epfd, ev, cnt, timeout);
  int res = BLOCK_REAL(epoll_wait)(epfd, ev, cnt, timeout);
  if (res > 0 && epfd >= 0) {
    FdAcquire(thr, pc, epfd);
  }
  return res;
}

TSAN_INTERCEPTOR(int, poll, void *fds, long_t nfds, int timeout) {
  SCOPED_TSAN_INTERCEPTOR(poll, fds, nfds, timeout);
  int res = BLOCK_REAL(poll)(fds, nfds, timeout);
  return res;
}

void ALWAYS_INLINE rtl_generic_sighandler(bool sigact, int sig,
    my_siginfo_t *info, void *ctx) {
  ThreadState *thr = cur_thread();
  SignalContext *sctx = SigCtx(thr);
  // Don't mess with synchronous signals.
  if (sig == SIGSEGV || sig == SIGBUS || sig == SIGILL ||
      sig == SIGABRT || sig == SIGFPE || sig == SIGPIPE ||
      // If we are sending signal to ourselves, we must process it now.
      (sctx && sig == sctx->int_signal_send) ||
      // If we are in blocking function, we can safely process it now
      // (but check if we are in a recursive interceptor,
      // i.e. pthread_join()->munmap()).
      (sctx && sctx->in_blocking_func == 1 && thr->in_rtl == 1)) {
    int in_rtl = thr->in_rtl;
    thr->in_rtl = 0;
    CHECK_EQ(thr->in_signal_handler, false);
    thr->in_signal_handler = true;
    if (sigact)
      sigactions[sig].sa_sigaction(sig, info, ctx);
    else
      sigactions[sig].sa_handler(sig);
    CHECK_EQ(thr->in_signal_handler, true);
    thr->in_signal_handler = false;
    thr->in_rtl = in_rtl;
    return;
  }

  if (sctx == 0)
    return;
  SignalDesc *signal = &sctx->pending_signals[sig];
  if (signal->armed == false) {
    signal->armed = true;
    signal->sigaction = sigact;
    if (info)
      internal_memcpy(&signal->siginfo, info, sizeof(*info));
    if (ctx)
      internal_memcpy(&signal->ctx, ctx, sizeof(signal->ctx));
    sctx->pending_signal_count++;
  }
}

static void rtl_sighandler(int sig) {
  rtl_generic_sighandler(false, sig, 0, 0);
}

static void rtl_sigaction(int sig, my_siginfo_t *info, void *ctx) {
  rtl_generic_sighandler(true, sig, info, ctx);
}

TSAN_INTERCEPTOR(int, sigaction, int sig, sigaction_t *act, sigaction_t *old) {
  SCOPED_TSAN_INTERCEPTOR(sigaction, sig, act, old);
  if (old)
    internal_memcpy(old, &sigactions[sig], sizeof(*old));
  if (act == 0)
    return 0;
  internal_memcpy(&sigactions[sig], act, sizeof(*act));
  sigaction_t newact;
  internal_memcpy(&newact, act, sizeof(newact));
  sigfillset(&newact.sa_mask);
  if (act->sa_handler != SIG_IGN && act->sa_handler != SIG_DFL) {
    if (newact.sa_flags & SA_SIGINFO)
      newact.sa_sigaction = rtl_sigaction;
    else
      newact.sa_handler = rtl_sighandler;
  }
  int res = REAL(sigaction)(sig, &newact, 0);
  return res;
}

TSAN_INTERCEPTOR(sighandler_t, signal, int sig, sighandler_t h) {
  sigaction_t act;
  act.sa_handler = h;
  REAL(memset)(&act.sa_mask, -1, sizeof(act.sa_mask));
  act.sa_flags = 0;
  sigaction_t old;
  int res = sigaction(sig, &act, &old);
  if (res)
    return SIG_ERR;
  return old.sa_handler;
}

TSAN_INTERCEPTOR(int, raise, int sig) {
  SCOPED_TSAN_INTERCEPTOR(raise, sig);
  SignalContext *sctx = SigCtx(thr);
  CHECK_NE(sctx, 0);
  int prev = sctx->int_signal_send;
  sctx->int_signal_send = sig;
  int res = REAL(raise)(sig);
  CHECK_EQ(sctx->int_signal_send, sig);
  sctx->int_signal_send = prev;
  return res;
}

TSAN_INTERCEPTOR(int, kill, int pid, int sig) {
  SCOPED_TSAN_INTERCEPTOR(kill, pid, sig);
  SignalContext *sctx = SigCtx(thr);
  CHECK_NE(sctx, 0);
  int prev = sctx->int_signal_send;
  if (pid == GetPid()) {
    sctx->int_signal_send = sig;
  }
  int res = REAL(kill)(pid, sig);
  if (pid == GetPid()) {
    CHECK_EQ(sctx->int_signal_send, sig);
    sctx->int_signal_send = prev;
  }
  return res;
}

TSAN_INTERCEPTOR(int, pthread_kill, void *tid, int sig) {
  SCOPED_TSAN_INTERCEPTOR(pthread_kill, tid, sig);
  SignalContext *sctx = SigCtx(thr);
  CHECK_NE(sctx, 0);
  int prev = sctx->int_signal_send;
  if (tid == pthread_self()) {
    sctx->int_signal_send = sig;
  }
  int res = REAL(pthread_kill)(tid, sig);
  if (tid == pthread_self()) {
    CHECK_EQ(sctx->int_signal_send, sig);
    sctx->int_signal_send = prev;
  }
  return res;
}

TSAN_INTERCEPTOR(int, gettimeofday, void *tv, void *tz) {
  SCOPED_TSAN_INTERCEPTOR(gettimeofday, tv, tz);
  // It's intercepted merely to process pending signals.
  return REAL(gettimeofday)(tv, tz);
}

// Linux kernel has a bug that leads to kernel deadlock if a process
// maps TBs of memory and then calls mlock().
static void MlockIsUnsupported() {
  static atomic_uint8_t printed;
  if (atomic_exchange(&printed, 1, memory_order_relaxed))
    return;
  Printf("INFO: ThreadSanitizer ignores mlock/mlockall/munlock/munlockall\n");
}

TSAN_INTERCEPTOR(int, mlock, const void *addr, uptr len) {
  MlockIsUnsupported();
  return 0;
}

TSAN_INTERCEPTOR(int, munlock, const void *addr, uptr len) {
  MlockIsUnsupported();
  return 0;
}

TSAN_INTERCEPTOR(int, mlockall, int flags) {
  MlockIsUnsupported();
  return 0;
}

TSAN_INTERCEPTOR(int, munlockall, void) {
  MlockIsUnsupported();
  return 0;
}

TSAN_INTERCEPTOR(int, fork, int fake) {
  SCOPED_TSAN_INTERCEPTOR(fork, fake);
  // It's intercepted merely to process pending signals.
  int pid = REAL(fork)(fake);
  if (pid == 0) {
    // child
    FdOnFork(thr, pc);
  } else if (pid > 0) {
    // parent
  }
  return pid;
}

struct TsanInterceptorContext {
  ThreadState *thr;
  const uptr caller_pc;
  const uptr pc;
};

#include "sanitizer_common/sanitizer_platform_interceptors.h"
// Causes interceptor recursion (getpwuid_r() calls fopen())
#undef SANITIZER_INTERCEPT_GETPWNAM_AND_FRIENDS
#undef SANITIZER_INTERCEPT_GETPWNAM_R_AND_FRIENDS
// Causes interceptor recursion (glob64() calls lstat64())
#undef SANITIZER_INTERCEPT_GLOB

#define COMMON_INTERCEPTOR_WRITE_RANGE(ctx, ptr, size) \
    MemoryAccessRange(((TsanInterceptorContext*)ctx)->thr,  \
                      ((TsanInterceptorContext*)ctx)->pc,   \
                      (uptr)ptr, size, true)
#define COMMON_INTERCEPTOR_READ_RANGE(ctx, ptr, size)       \
    MemoryAccessRange(((TsanInterceptorContext*)ctx)->thr,  \
                      ((TsanInterceptorContext*)ctx)->pc,   \
                      (uptr)ptr, size, false)
#define COMMON_INTERCEPTOR_ENTER(ctx, func, ...) \
    SCOPED_TSAN_INTERCEPTOR(func, __VA_ARGS__) \
    TsanInterceptorContext _ctx = {thr, caller_pc, pc}; \
    ctx = (void*)&_ctx; \
    (void)ctx;
#define COMMON_INTERCEPTOR_FD_ACQUIRE(ctx, fd) \
    FdAcquire(((TsanInterceptorContext*)ctx)->thr, pc, fd)
#define COMMON_INTERCEPTOR_FD_RELEASE(ctx, fd) \
    FdRelease(((TsanInterceptorContext*)ctx)->thr, pc, fd)
#define COMMON_INTERCEPTOR_SET_THREAD_NAME(ctx, name) \
    ThreadSetName(((TsanInterceptorContext*)ctx)->thr, name)
#include "sanitizer_common/sanitizer_common_interceptors.inc"

// FIXME: Implement these with MemoryAccessRange().
#define COMMON_SYSCALL_PRE_READ_RANGE(p, s)
#define COMMON_SYSCALL_PRE_WRITE_RANGE(p, s)
#define COMMON_SYSCALL_POST_READ_RANGE(p, s)
#define COMMON_SYSCALL_POST_WRITE_RANGE(p, s)
#include "sanitizer_common/sanitizer_common_syscalls.inc"

namespace __tsan {

void ProcessPendingSignals(ThreadState *thr) {
  CHECK_EQ(thr->in_rtl, 0);
  SignalContext *sctx = SigCtx(thr);
  if (sctx == 0 || sctx->pending_signal_count == 0 || thr->in_signal_handler)
    return;
  Context *ctx = CTX();
  thr->in_signal_handler = true;
  sctx->pending_signal_count = 0;
  // These are too big for stack.
  static THREADLOCAL sigset_t emptyset, oldset;
  sigfillset(&emptyset);
  pthread_sigmask(SIG_SETMASK, &emptyset, &oldset);
  for (int sig = 0; sig < kSigCount; sig++) {
    SignalDesc *signal = &sctx->pending_signals[sig];
    if (signal->armed) {
      signal->armed = false;
      if (sigactions[sig].sa_handler != SIG_DFL
          && sigactions[sig].sa_handler != SIG_IGN) {
        // Insure that the handler does not spoil errno.
        const int saved_errno = errno;
        errno = 0;
        if (signal->sigaction)
          sigactions[sig].sa_sigaction(sig, &signal->siginfo, &signal->ctx);
        else
          sigactions[sig].sa_handler(sig);
        if (flags()->report_bugs && errno != 0) {
          ScopedInRtl in_rtl;
          __tsan::StackTrace stack;
          uptr pc = signal->sigaction ?
              (uptr)sigactions[sig].sa_sigaction :
              (uptr)sigactions[sig].sa_handler;
          stack.Init(&pc, 1);
          ThreadRegistryLock l(ctx->thread_registry);
          ScopedReport rep(ReportTypeErrnoInSignal);
          if (!IsFiredSuppression(ctx, rep, stack)) {
            rep.AddStack(&stack);
            OutputReport(ctx, rep, rep.GetReport()->stacks[0]);
          }
        }
        errno = saved_errno;
      }
    }
  }
  pthread_sigmask(SIG_SETMASK, &oldset, 0);
  CHECK_EQ(thr->in_signal_handler, true);
  thr->in_signal_handler = false;
}

static void finalize(void *arg) {
  ThreadState * thr = cur_thread();
  uptr pc = 0;
  atexit_ctx->exit(thr, pc);
  int status = Finalize(cur_thread());
  REAL(fflush)(0);
  if (status)
    _exit(status);
}

static void unreachable() {
  Printf("FATAL: ThreadSanitizer: unreachable called\n");
  Die();
}

void InitializeInterceptors() {
  CHECK_GT(cur_thread()->in_rtl, 0);

  // We need to setup it early, because functions like dlsym() can call it.
  REAL(memset) = internal_memset;
  REAL(memcpy) = internal_memcpy;
  REAL(memcmp) = internal_memcmp;

  // Instruct libc malloc to consume less memory.
  mallopt(1, 0);  // M_MXFAST
  mallopt(-3, 32*1024);  // M_MMAP_THRESHOLD

  SANITIZER_COMMON_INTERCEPTORS_INIT;

  TSAN_INTERCEPT(setjmp);
  TSAN_INTERCEPT(_setjmp);
  TSAN_INTERCEPT(sigsetjmp);
  TSAN_INTERCEPT(__sigsetjmp);
  TSAN_INTERCEPT(longjmp);
  TSAN_INTERCEPT(siglongjmp);

  TSAN_INTERCEPT(malloc);
  TSAN_INTERCEPT(__libc_memalign);
  TSAN_INTERCEPT(calloc);
  TSAN_INTERCEPT(realloc);
  TSAN_INTERCEPT(free);
  TSAN_INTERCEPT(cfree);
  TSAN_INTERCEPT(mmap);
  TSAN_INTERCEPT(mmap64);
  TSAN_INTERCEPT(munmap);
  TSAN_INTERCEPT(memalign);
  TSAN_INTERCEPT(valloc);
  TSAN_INTERCEPT(pvalloc);
  TSAN_INTERCEPT(posix_memalign);

  TSAN_INTERCEPT(strlen);
  TSAN_INTERCEPT(memset);
  TSAN_INTERCEPT(memcpy);
  TSAN_INTERCEPT(strcmp);
  TSAN_INTERCEPT(memchr);
  TSAN_INTERCEPT(memrchr);
  TSAN_INTERCEPT(memmove);
  TSAN_INTERCEPT(memcmp);
  TSAN_INTERCEPT(strchr);
  TSAN_INTERCEPT(strchrnul);
  TSAN_INTERCEPT(strrchr);
  TSAN_INTERCEPT(strncmp);
  TSAN_INTERCEPT(strcpy);  // NOLINT
  TSAN_INTERCEPT(strncpy);
  TSAN_INTERCEPT(strstr);

  TSAN_INTERCEPT(pthread_create);
  TSAN_INTERCEPT(pthread_join);
  TSAN_INTERCEPT(pthread_detach);

  TSAN_INTERCEPT(pthread_mutex_init);
  TSAN_INTERCEPT(pthread_mutex_destroy);
  TSAN_INTERCEPT(pthread_mutex_lock);
  TSAN_INTERCEPT(pthread_mutex_trylock);
  TSAN_INTERCEPT(pthread_mutex_timedlock);
  TSAN_INTERCEPT(pthread_mutex_unlock);

  TSAN_INTERCEPT(pthread_spin_init);
  TSAN_INTERCEPT(pthread_spin_destroy);
  TSAN_INTERCEPT(pthread_spin_lock);
  TSAN_INTERCEPT(pthread_spin_trylock);
  TSAN_INTERCEPT(pthread_spin_unlock);

  TSAN_INTERCEPT(pthread_rwlock_init);
  TSAN_INTERCEPT(pthread_rwlock_destroy);
  TSAN_INTERCEPT(pthread_rwlock_rdlock);
  TSAN_INTERCEPT(pthread_rwlock_tryrdlock);
  TSAN_INTERCEPT(pthread_rwlock_timedrdlock);
  TSAN_INTERCEPT(pthread_rwlock_wrlock);
  TSAN_INTERCEPT(pthread_rwlock_trywrlock);
  TSAN_INTERCEPT(pthread_rwlock_timedwrlock);
  TSAN_INTERCEPT(pthread_rwlock_unlock);

  // TSAN_INTERCEPT(pthread_cond_init);
  TSAN_INTERCEPT(pthread_cond_destroy);
  TSAN_INTERCEPT(pthread_cond_signal);
  TSAN_INTERCEPT(pthread_cond_broadcast);
  TSAN_INTERCEPT(pthread_cond_wait);
  TSAN_INTERCEPT(pthread_cond_timedwait);

  TSAN_INTERCEPT(pthread_barrier_init);
  TSAN_INTERCEPT(pthread_barrier_destroy);
  TSAN_INTERCEPT(pthread_barrier_wait);

  TSAN_INTERCEPT(pthread_once);

  TSAN_INTERCEPT(sem_init);
  TSAN_INTERCEPT(sem_destroy);
  TSAN_INTERCEPT(sem_wait);
  TSAN_INTERCEPT(sem_trywait);
  TSAN_INTERCEPT(sem_timedwait);
  TSAN_INTERCEPT(sem_post);
  TSAN_INTERCEPT(sem_getvalue);

  TSAN_INTERCEPT(stat);
  TSAN_INTERCEPT(__xstat);
  TSAN_INTERCEPT(stat64);
  TSAN_INTERCEPT(__xstat64);
  TSAN_INTERCEPT(lstat);
  TSAN_INTERCEPT(__lxstat);
  TSAN_INTERCEPT(lstat64);
  TSAN_INTERCEPT(__lxstat64);
  TSAN_INTERCEPT(fstat);
  TSAN_INTERCEPT(__fxstat);
  TSAN_INTERCEPT(fstat64);
  TSAN_INTERCEPT(__fxstat64);
  TSAN_INTERCEPT(open);
  TSAN_INTERCEPT(open64);
  TSAN_INTERCEPT(creat);
  TSAN_INTERCEPT(creat64);
  TSAN_INTERCEPT(dup);
  TSAN_INTERCEPT(dup2);
  TSAN_INTERCEPT(dup3);
  TSAN_INTERCEPT(eventfd);
  TSAN_INTERCEPT(signalfd);
  TSAN_INTERCEPT(inotify_init);
  TSAN_INTERCEPT(inotify_init1);
  TSAN_INTERCEPT(socket);
  TSAN_INTERCEPT(socketpair);
  TSAN_INTERCEPT(connect);
  TSAN_INTERCEPT(bind);
  TSAN_INTERCEPT(listen);
  TSAN_INTERCEPT(accept);
  TSAN_INTERCEPT(accept4);
  TSAN_INTERCEPT(epoll_create);
  TSAN_INTERCEPT(epoll_create1);
  TSAN_INTERCEPT(close);
  TSAN_INTERCEPT(__close);
  TSAN_INTERCEPT(__res_iclose);
  TSAN_INTERCEPT(pipe);
  TSAN_INTERCEPT(pipe2);

  TSAN_INTERCEPT(readv);
  TSAN_INTERCEPT(preadv64);
  TSAN_INTERCEPT(writev);
  TSAN_INTERCEPT(pwritev64);
  TSAN_INTERCEPT(send);
  TSAN_INTERCEPT(sendmsg);
  TSAN_INTERCEPT(recv);
  TSAN_INTERCEPT(recvmsg);

  TSAN_INTERCEPT(unlink);
  TSAN_INTERCEPT(fopen);
  TSAN_INTERCEPT(freopen);
  TSAN_INTERCEPT(fclose);
  TSAN_INTERCEPT(fread);
  TSAN_INTERCEPT(fwrite);
  TSAN_INTERCEPT(fflush);
  TSAN_INTERCEPT(abort);
  TSAN_INTERCEPT(puts);
  TSAN_INTERCEPT(rmdir);
  TSAN_INTERCEPT(opendir);

  TSAN_INTERCEPT(epoll_ctl);
  TSAN_INTERCEPT(epoll_wait);
  TSAN_INTERCEPT(poll);

  TSAN_INTERCEPT(sigaction);
  TSAN_INTERCEPT(signal);
  TSAN_INTERCEPT(raise);
  TSAN_INTERCEPT(kill);
  TSAN_INTERCEPT(pthread_kill);
  TSAN_INTERCEPT(sleep);
  TSAN_INTERCEPT(usleep);
  TSAN_INTERCEPT(nanosleep);
  TSAN_INTERCEPT(gettimeofday);

  TSAN_INTERCEPT(mlock);
  TSAN_INTERCEPT(munlock);
  TSAN_INTERCEPT(mlockall);
  TSAN_INTERCEPT(munlockall);

  TSAN_INTERCEPT(fork);
  TSAN_INTERCEPT(on_exit);
  TSAN_INTERCEPT(__cxa_atexit);

  // Need to setup it, because interceptors check that the function is resolved.
  // But atexit is emitted directly into the module, so can't be resolved.
  REAL(atexit) = (int(*)(void(*)()))unreachable;
  atexit_ctx = new(internal_alloc(MBlockAtExit, sizeof(AtExitContext)))
      AtExitContext();

  if (REAL(__cxa_atexit)(&finalize, 0, 0)) {
    Printf("ThreadSanitizer: failed to setup atexit callback\n");
    Die();
  }

  if (pthread_key_create(&g_thread_finalize_key, &thread_finalize)) {
    Printf("ThreadSanitizer: failed to create thread key\n");
    Die();
  }

  FdInit();
}

void internal_start_thread(void(*func)(void *arg), void *arg) {
  void *th;
  REAL(pthread_create)(&th, 0, (void*(*)(void *arg))func, arg);
  REAL(pthread_detach)(th);
}

}  // namespace __tsan
