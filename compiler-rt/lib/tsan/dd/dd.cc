//===-- dd_rtl.cc ---------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "dd_rtl.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

namespace __dsan {

static Context ctx0;
static Context * const ctx = &ctx0;

void Initialize() {
  InitializeInterceptors();
  //common_flags()->allow_addr2line = true;
  common_flags()->symbolize = true;
  ctx->dd = DDetector::Create();
}

void ThreadInit(Thread *thr) {
  thr->dd_pt = ctx->dd->CreatePhysicalThread();
  thr->dd_lt = ctx->dd->CreateLogicalThread(0);
}

void ThreadDestroy(Thread *thr) {
  ctx->dd->DestroyPhysicalThread(thr->dd_pt);
  ctx->dd->DestroyLogicalThread(thr->dd_lt);
}

static u32 CurrentStackTrace(Thread *thr) {
  StackTrace trace;
  thr->in_symbolizer = true;
  trace.Unwind(1000, 0, 0, 0, 0, 0, false);
  thr->in_symbolizer = false;
  const uptr skip = 4;
  if (trace.size <= skip)
    return 0;
  return StackDepotPut(trace.trace + skip, trace.size - skip);
}

static void PrintStackTrace(Thread *thr, u32 stk) {
  uptr size = 0;
  const uptr *trace = StackDepotGet(stk, &size);
  thr->in_symbolizer = true;
  StackTrace::PrintStack(trace, size);
  thr->in_symbolizer = false;
}

static Mutex *FindMutex(Thread *thr, uptr m) {
  SpinMutexLock l(&ctx->mutex_mtx);
  for (Mutex *mtx = ctx->mutex_list; mtx; mtx = mtx->link) {
    if (mtx->addr == m)
      return mtx;
  }
  Mutex *mtx = (Mutex*)InternalAlloc(sizeof(*mtx));
  internal_memset(mtx, 0, sizeof(*mtx));
  mtx->addr = m;
  ctx->dd->MutexInit(&mtx->dd, CurrentStackTrace(thr), ctx->mutex_seq++);
  mtx->link = ctx->mutex_list;
  ctx->mutex_list = mtx;
  return mtx;
}

static Mutex *FindMutexAndRemove(uptr m) {
  SpinMutexLock l(&ctx->mutex_mtx);
  Mutex **prev = &ctx->mutex_list;
  for (;;) {
    Mutex *mtx = *prev;
    if (mtx == 0)
      return 0;
    if (mtx->addr == m) {
      *prev = mtx->link;
      return mtx;
    }
    prev = &mtx->link;
  }
}

static void ReportDeadlock(Thread *thr, DDReport *rep) {
  Printf("==============================\n");
  Printf("DEADLOCK\n");
  PrintStackTrace(thr, CurrentStackTrace(thr));
  for (int i = 0; i < rep->n; i++) {
    Printf("Mutex %llu created at:\n", rep->loop[i].mtx_ctx0);
    PrintStackTrace(thr, rep->loop[i].stk);
  }
  Printf("==============================\n");
}

void MutexLock(Thread *thr, uptr m, bool writelock, bool trylock) {
  if (thr->in_symbolizer)
    return;
  Mutex *mtx = FindMutex(thr, m);
  DDReport *rep = ctx->dd->MutexLock(thr->dd_pt, thr->dd_lt, &mtx->dd,
      writelock, trylock);
  if (rep)
    ReportDeadlock(thr, rep);
}

void MutexUnlock(Thread *thr, uptr m, bool writelock) {
  if (thr->in_symbolizer)
    return;
  Mutex *mtx = FindMutex(thr, m);
  ctx->dd->MutexUnlock(thr->dd_pt, thr->dd_lt, &mtx->dd, writelock);
}

void MutexDestroy(Thread *thr, uptr m) {
  if (thr->in_symbolizer)
    return;
  Mutex *mtx = FindMutexAndRemove(m);
  if (mtx == 0)
    return;
  ctx->dd->MutexDestroy(thr->dd_pt, thr->dd_lt, &mtx->dd);
  InternalFree(mtx);
}

}  // namespace __dsan

__attribute__((section(".preinit_array"), used))
void (*__local_dsan_preinit)(void) = __dsan::Initialize;
//===-- dd_interceptors.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "dd_rtl.h"
#include "interception/interception.h"
#include <pthread.h>

using namespace __dsan;

extern "C" void *__libc_malloc(uptr size);
extern "C" void __libc_free(void *ptr);

static __thread Thread *thr;

static void InitThread() {
  if (thr != 0)
    return;
  thr = (Thread*)InternalAlloc(sizeof(*thr));
  internal_memset(thr, 0, sizeof(*thr));
  ThreadInit(thr);
}

INTERCEPTOR(int, pthread_mutex_destroy, pthread_mutex_t *m) {
  InitThread();
  int res = REAL(pthread_mutex_destroy)(m);
  MutexDestroy(thr, (uptr)m);
  return res;
}

INTERCEPTOR(int, pthread_mutex_lock, pthread_mutex_t *m) {
  InitThread();
  int res = REAL(pthread_mutex_lock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, false);
  return res;
}

INTERCEPTOR(int, pthread_mutex_trylock, pthread_mutex_t *m) {
  InitThread();
  int res = REAL(pthread_mutex_trylock)(m);
  if (res == 0)
    MutexLock(thr, (uptr)m, true, true);
  return res;
}

INTERCEPTOR(int, pthread_mutex_unlock, pthread_mutex_t *m) {
  InitThread();
  MutexUnlock(thr, (uptr)m, true);
  int res = REAL(pthread_mutex_unlock)(m);
  return res;
}

namespace __dsan {

void InitializeInterceptors() {
  INTERCEPT_FUNCTION(pthread_mutex_destroy);
  INTERCEPT_FUNCTION(pthread_mutex_lock);
  INTERCEPT_FUNCTION(pthread_mutex_trylock);
  INTERCEPT_FUNCTION(pthread_mutex_unlock);
}

}  // namespace __dsan
//===-- sanitizer_allocator.cc --------------------------------------------===//
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
// This allocator is used inside run-times.
//===----------------------------------------------------------------------===//
#include "sanitizer_allocator.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_flags.h"

namespace __sanitizer {

// ThreadSanitizer for Go uses libc malloc/free.
#if defined(SANITIZER_GO) || defined(SANITIZER_USE_MALLOC)
# if SANITIZER_LINUX && !SANITIZER_ANDROID
extern "C" void *__libc_malloc(uptr size);
extern "C" void __libc_free(void *ptr);
#  define LIBC_MALLOC __libc_malloc
#  define LIBC_FREE __libc_free
# else
#  include <stdlib.h>
#  define LIBC_MALLOC malloc
#  define LIBC_FREE free
# endif

static void *RawInternalAlloc(uptr size, InternalAllocatorCache *cache) {
  (void)cache;
  return LIBC_MALLOC(size);
}

static void RawInternalFree(void *ptr, InternalAllocatorCache *cache) {
  (void)cache;
  LIBC_FREE(ptr);
}

InternalAllocator *internal_allocator() {
  return 0;
}

#else  // SANITIZER_GO

static ALIGNED(64) char internal_alloc_placeholder[sizeof(InternalAllocator)];
static atomic_uint8_t internal_allocator_initialized;
static StaticSpinMutex internal_alloc_init_mu;

static InternalAllocatorCache internal_allocator_cache;
static StaticSpinMutex internal_allocator_cache_mu;

InternalAllocator *internal_allocator() {
  InternalAllocator *internal_allocator_instance =
      reinterpret_cast<InternalAllocator *>(&internal_alloc_placeholder);
  if (atomic_load(&internal_allocator_initialized, memory_order_acquire) == 0) {
    SpinMutexLock l(&internal_alloc_init_mu);
    if (atomic_load(&internal_allocator_initialized, memory_order_relaxed) ==
        0) {
      internal_allocator_instance->Init();
      atomic_store(&internal_allocator_initialized, 1, memory_order_release);
    }
  }
  return internal_allocator_instance;
}

static void *RawInternalAlloc(uptr size, InternalAllocatorCache *cache) {
  if (cache == 0) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Allocate(&internal_allocator_cache, size, 8,
                                          false);
  }
  return internal_allocator()->Allocate(cache, size, 8, false);
}

static void RawInternalFree(void *ptr, InternalAllocatorCache *cache) {
  if (cache == 0) {
    SpinMutexLock l(&internal_allocator_cache_mu);
    return internal_allocator()->Deallocate(&internal_allocator_cache, ptr);
  }
  internal_allocator()->Deallocate(cache, ptr);
}

#endif  // SANITIZER_GO

const u64 kBlockMagic = 0x6A6CB03ABCEBC041ull;

void *InternalAlloc(uptr size, InternalAllocatorCache *cache) {
  if (size + sizeof(u64) < size)
    return 0;
  void *p = RawInternalAlloc(size + sizeof(u64), cache);
  if (p == 0)
    return 0;
  ((u64*)p)[0] = kBlockMagic;
  return (char*)p + sizeof(u64);
}

void InternalFree(void *addr, InternalAllocatorCache *cache) {
  if (addr == 0)
    return;
  addr = (char*)addr - sizeof(u64);
  CHECK_EQ(kBlockMagic, ((u64*)addr)[0]);
  ((u64*)addr)[0] = 0;
  RawInternalFree(addr, cache);
}

// LowLevelAllocator
static LowLevelAllocateCallback low_level_alloc_callback;

void *LowLevelAllocator::Allocate(uptr size) {
  // Align allocation size.
  size = RoundUpTo(size, 8);
  if (allocated_end_ - allocated_current_ < (sptr)size) {
    uptr size_to_allocate = Max(size, GetPageSizeCached());
    allocated_current_ =
        (char*)MmapOrDie(size_to_allocate, __func__);
    allocated_end_ = allocated_current_ + size_to_allocate;
    if (low_level_alloc_callback) {
      low_level_alloc_callback((uptr)allocated_current_,
                               size_to_allocate);
    }
  }
  CHECK(allocated_end_ - allocated_current_ >= (sptr)size);
  void *res = allocated_current_;
  allocated_current_ += size;
  return res;
}

void SetLowLevelAllocateCallback(LowLevelAllocateCallback callback) {
  low_level_alloc_callback = callback;
}

bool CallocShouldReturnNullDueToOverflow(uptr size, uptr n) {
  if (!size) return false;
  uptr max = (uptr)-1L;
  return (max / size) < n;
}

void *AllocatorReturnNull() {
  if (common_flags()->allocator_may_return_null)
    return 0;
  Report("%s's allocator is terminating the process instead of returning 0\n",
         SanitizerToolName);
  Report("If you don't like this behavior set allocator_may_return_null=1\n");
  CHECK(0);
  return 0;
}

}  // namespace __sanitizer
//===-- sanitizer_common.cc -----------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

const char *SanitizerToolName = "SanitizerTool";

uptr GetPageSizeCached() {
  static uptr PageSize;
  if (!PageSize)
    PageSize = GetPageSize();
  return PageSize;
}


// By default, dump to stderr. If |log_to_file| is true and |report_fd_pid|
// isn't equal to the current PID, try to obtain file descriptor by opening
// file "report_path_prefix.<PID>".
fd_t report_fd = kStderrFd;

// Set via __sanitizer_set_report_path.
bool log_to_file = false;
char report_path_prefix[sizeof(report_path_prefix)];

// PID of process that opened |report_fd|. If a fork() occurs, the PID of the
// child thread will be different from |report_fd_pid|.
uptr report_fd_pid = 0;

// PID of the tracer task in StopTheWorld. It shares the address space with the
// main process, but has a different PID and thus requires special handling.
uptr stoptheworld_tracer_pid = 0;
// Cached pid of parent process - if the parent process dies, we want to keep
// writing to the same log file.
uptr stoptheworld_tracer_ppid = 0;

static DieCallbackType DieCallback;
void SetDieCallback(DieCallbackType callback) {
  DieCallback = callback;
}

DieCallbackType GetDieCallback() {
  return DieCallback;
}

void NORETURN Die() {
  if (DieCallback) {
    DieCallback();
  }
  internal__exit(1);
}

static CheckFailedCallbackType CheckFailedCallback;
void SetCheckFailedCallback(CheckFailedCallbackType callback) {
  CheckFailedCallback = callback;
}

void NORETURN CheckFailed(const char *file, int line, const char *cond,
                          u64 v1, u64 v2) {
  if (CheckFailedCallback) {
    CheckFailedCallback(file, line, cond, v1, v2);
  }
  Report("Sanitizer CHECK failed: %s:%d %s (%lld, %lld)\n", file, line, cond,
                                                            v1, v2);
  Die();
}

uptr ReadFileToBuffer(const char *file_name, char **buff,
                      uptr *buff_size, uptr max_len) {
  uptr PageSize = GetPageSizeCached();
  uptr kMinFileLen = PageSize;
  uptr read_len = 0;
  *buff = 0;
  *buff_size = 0;
  // The files we usually open are not seekable, so try different buffer sizes.
  for (uptr size = kMinFileLen; size <= max_len; size *= 2) {
    uptr openrv = OpenFile(file_name, /*write*/ false);
    if (internal_iserror(openrv)) return 0;
    fd_t fd = openrv;
    UnmapOrDie(*buff, *buff_size);
    *buff = (char*)MmapOrDie(size, __func__);
    *buff_size = size;
    // Read up to one page at a time.
    read_len = 0;
    bool reached_eof = false;
    while (read_len + PageSize <= size) {
      uptr just_read = internal_read(fd, *buff + read_len, PageSize);
      if (just_read == 0) {
        reached_eof = true;
        break;
      }
      read_len += just_read;
    }
    internal_close(fd);
    if (reached_eof)  // We've read the whole file.
      break;
  }
  return read_len;
}

typedef bool UptrComparisonFunction(const uptr &a, const uptr &b);

template<class T>
static inline bool CompareLess(const T &a, const T &b) {
  return a < b;
}

void SortArray(uptr *array, uptr size) {
  InternalSort<uptr*, UptrComparisonFunction>(&array, size, CompareLess);
}

// We want to map a chunk of address space aligned to 'alignment'.
// We do it by maping a bit more and then unmaping redundant pieces.
// We probably can do it with fewer syscalls in some OS-dependent way.
void *MmapAlignedOrDie(uptr size, uptr alignment, const char *mem_type) {
// uptr PageSize = GetPageSizeCached();
  CHECK(IsPowerOfTwo(size));
  CHECK(IsPowerOfTwo(alignment));
  uptr map_size = size + alignment;
  uptr map_res = (uptr)MmapOrDie(map_size, mem_type);
  uptr map_end = map_res + map_size;
  uptr res = map_res;
  if (res & (alignment - 1))  // Not aligned.
    res = (map_res + alignment) & ~(alignment - 1);
  uptr end = res + size;
  if (res != map_res)
    UnmapOrDie((void*)map_res, res - map_res);
  if (end != map_end)
    UnmapOrDie((void*)end, map_end - end);
  return (void*)res;
}

const char *StripPathPrefix(const char *filepath,
                            const char *strip_path_prefix) {
  if (filepath == 0) return 0;
  if (strip_path_prefix == 0) return filepath;
  const char *pos = internal_strstr(filepath, strip_path_prefix);
  if (pos == 0) return filepath;
  pos += internal_strlen(strip_path_prefix);
  if (pos[0] == '.' && pos[1] == '/')
    pos += 2;
  return pos;
}

void PrintSourceLocation(InternalScopedString *buffer, const char *file,
                         int line, int column) {
  CHECK(file);
  buffer->append("%s",
                 StripPathPrefix(file, common_flags()->strip_path_prefix));
  if (line > 0) {
    buffer->append(":%d", line);
    if (column > 0)
      buffer->append(":%d", column);
  }
}

void PrintModuleAndOffset(InternalScopedString *buffer, const char *module,
                          uptr offset) {
  buffer->append("(%s+0x%zx)",
                 StripPathPrefix(module, common_flags()->strip_path_prefix),
                 offset);
}

void ReportErrorSummary(const char *error_message) {
  if (!common_flags()->print_summary)
    return;
  InternalScopedBuffer<char> buff(kMaxSummaryLength);
  internal_snprintf(buff.data(), buff.size(),
                    "SUMMARY: %s: %s", SanitizerToolName, error_message);
  __sanitizer_report_error_summary(buff.data());
}

void ReportErrorSummary(const char *error_type, const char *file,
                        int line, const char *function) {
  if (!common_flags()->print_summary)
    return;
  InternalScopedBuffer<char> buff(kMaxSummaryLength);
  internal_snprintf(
      buff.data(), buff.size(), "%s %s:%d %s", error_type,
      file ? StripPathPrefix(file, common_flags()->strip_path_prefix) : "??",
      line, function ? function : "??");
  ReportErrorSummary(buff.data());
}

void ReportErrorSummary(const char *error_type, StackTrace *stack) {
  if (!common_flags()->print_summary)
    return;
  AddressInfo ai;
#if !SANITIZER_GO
  if (stack->size > 0 && Symbolizer::Get()->CanReturnFileLineInfo()) {
    // Currently, we include the first stack frame into the report summary.
    // Maybe sometimes we need to choose another frame (e.g. skip memcpy/etc).
    uptr pc = StackTrace::GetPreviousInstructionPc(stack->trace[0]);
    Symbolizer::Get()->SymbolizePC(pc, &ai, 1);
  }
#endif
  ReportErrorSummary(error_type, ai.file, ai.line, ai.function);
}

LoadedModule::LoadedModule(const char *module_name, uptr base_address) {
  full_name_ = internal_strdup(module_name);
  base_address_ = base_address;
  n_ranges_ = 0;
}

void LoadedModule::addAddressRange(uptr beg, uptr end) {
  CHECK_LT(n_ranges_, kMaxNumberOfAddressRanges);
  ranges_[n_ranges_].beg = beg;
  ranges_[n_ranges_].end = end;
  n_ranges_++;
}

bool LoadedModule::containsAddress(uptr address) const {
  for (uptr i = 0; i < n_ranges_; i++) {
    if (ranges_[i].beg <= address && address < ranges_[i].end)
      return true;
  }
  return false;
}

char *StripModuleName(const char *module) {
  if (module == 0)
    return 0;
  const char *short_module_name = internal_strrchr(module, '/');
  if (short_module_name)
    short_module_name += 1;
  else
    short_module_name = module;
  return internal_strdup(short_module_name);
}

}  // namespace __sanitizer

using namespace __sanitizer;  // NOLINT

extern "C" {
void __sanitizer_set_report_path(const char *path) {
  if (!path)
    return;
  uptr len = internal_strlen(path);
  if (len > sizeof(report_path_prefix) - 100) {
    Report("ERROR: Path is too long: %c%c%c%c%c%c%c%c...\n",
           path[0], path[1], path[2], path[3],
           path[4], path[5], path[6], path[7]);
    Die();
  }
  if (report_fd != kStdoutFd &&
      report_fd != kStderrFd &&
      report_fd != kInvalidFd)
    internal_close(report_fd);
  report_fd = kInvalidFd;
  log_to_file = false;
  if (internal_strcmp(path, "stdout") == 0) {
    report_fd = kStdoutFd;
  } else if (internal_strcmp(path, "stderr") == 0) {
    report_fd = kStderrFd;
  } else {
    internal_strncpy(report_path_prefix, path, sizeof(report_path_prefix));
    report_path_prefix[len] = '\0';
    log_to_file = true;
  }
}

void NOINLINE __sanitizer_sandbox_on_notify(void *reserved) {
  (void)reserved;
  PrepareForSandboxing();
}

void __sanitizer_report_error_summary(const char *error_summary) {
  Printf("%s\n", error_summary);
}
}  // extern "C"
//===-- sanitizer_deadlock_detector1.cc -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Deadlock detector implementation based on NxN adjacency bit matrix.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_deadlock_detector_interface.h"
#include "sanitizer_deadlock_detector.h"
#include "sanitizer_allocator_internal.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {

typedef TwoLevelBitVector<> DDBV;  // DeadlockDetector's bit vector.

struct DDPhysicalThread {
};

struct DDLogicalThread {
  u64 ctx;
  DeadlockDetectorTLS<DDBV> dd;
  DDReport rep;
};

struct DDetectorImpl : public DDetector {
  SpinMutex mtx;
  DeadlockDetector<DDBV> dd;

  DDetectorImpl();

  virtual DDPhysicalThread* CreatePhysicalThread();
  virtual void DestroyPhysicalThread(DDPhysicalThread *pt);

  virtual DDLogicalThread* CreateLogicalThread(u64 ctx);
  virtual void DestroyLogicalThread(DDLogicalThread *lt);

  virtual void MutexInit(DDMutex *m, u32 stk, u64 ctx);
  virtual DDReport *MutexLock(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m, bool writelock, bool trylock);
  virtual DDReport *MutexUnlock(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m, bool writelock);
  virtual void MutexDestroy(DDPhysicalThread *pt, DDLogicalThread *lt,
      DDMutex *m);

  void MutexEnsureID(DDLogicalThread *lt, DDMutex *m);
};

DDetector *DDetector::Create() {
  void *mem = MmapOrDie(sizeof(DDetectorImpl), "deadlock detector");
  return new(mem) DDetectorImpl();
}

DDetectorImpl::DDetectorImpl() {
  dd.clear();
}

DDPhysicalThread* DDetectorImpl::CreatePhysicalThread() {
  return 0;
}

void DDetectorImpl::DestroyPhysicalThread(DDPhysicalThread *pt) {
}

DDLogicalThread* DDetectorImpl::CreateLogicalThread(u64 ctx) {
  DDLogicalThread *lt = (DDLogicalThread*)InternalAlloc(sizeof(*lt));
  lt->ctx = ctx;
  lt->dd.clear();
  return lt;
}

void DDetectorImpl::DestroyLogicalThread(DDLogicalThread *lt) {
  lt->~DDLogicalThread();
  InternalFree(lt);
}

void DDetectorImpl::MutexInit(DDMutex *m, u32 stk, u64 ctx) {
  m->id = 0;
  m->stk = stk;
  m->ctx = ctx;
}

void DDetectorImpl::MutexEnsureID(DDLogicalThread *lt, DDMutex *m) {
  if (!dd.nodeBelongsToCurrentEpoch(m->id))
    m->id = dd.newNode(reinterpret_cast<uptr>(m));
  dd.ensureCurrentEpoch(&lt->dd);
}

DDReport *DDetectorImpl::MutexLock(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m, bool writelock, bool trylock) {
  if (dd.onFirstLock(&lt->dd, m->id))
    return 0;
  SpinMutexLock lk(&mtx);
  MutexEnsureID(lt, m);
  CHECK(!dd.isHeld(&lt->dd, m->id));
  // Printf("T%d MutexLock:   %zx\n", thr->tid, s->deadlock_detector_id);
  bool has_deadlock = trylock
      ? dd.onTryLock(&lt->dd, m->id)
       : dd.onLock(&lt->dd, m->id);
  DDReport *rep = 0;
  if (has_deadlock) {
    uptr path[10];
    uptr len = dd.findPathToHeldLock(&lt->dd, m->id,
                                          path, ARRAY_SIZE(path));
    CHECK_GT(len, 0U);  // Hm.. cycle of 10 locks? I'd like to see that.
    rep = &lt->rep;
    rep->n = len;
    for (uptr i = 0; i < len; i++) {
      DDMutex *m0 = (DDMutex*)dd.getData(path[i]);
      DDMutex *m1 = (DDMutex*)dd.getData(path[i < len - 1 ? i + 1 : 0]);
      rep->loop[i].thr_ctx = 0;  // don't know
      rep->loop[i].mtx_ctx0 = m0->ctx;
      rep->loop[i].mtx_ctx1 = m1->ctx;
      rep->loop[i].stk = m0->stk;
    }
  }
  return rep;
}

DDReport *DDetectorImpl::MutexUnlock(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m, bool writelock) {
  // Printf("T%d MutexUnlock: %zx; recursion %d\n", thr->tid,
  //        s->deadlock_detector_id, s->recursion);
  dd.onUnlock(&lt->dd, m->id);
  return 0;
}

void DDetectorImpl::MutexDestroy(DDPhysicalThread *pt, DDLogicalThread *lt,
    DDMutex *m) {
  if (!m->id) return;
  SpinMutexLock lk(&mtx);
  if (dd.nodeBelongsToCurrentEpoch(m->id))
    dd.removeNode(m->id);
  m->id = 0;
}

}  // namespace __sanitizer
//===-- sanitizer_flags.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_flags.h"

#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

CommonFlags common_flags_dont_use;

void SetCommonFlagsDefaults(CommonFlags *f) {
  f->symbolize = true;
  f->external_symbolizer_path = 0;
  f->allow_addr2line = false;
  f->strip_path_prefix = "";
  f->fast_unwind_on_fatal = false;
  f->fast_unwind_on_malloc = true;
  f->handle_ioctl = false;
  f->malloc_context_size = 1;
  f->log_path = "stderr";
  f->verbosity = 0;
  f->detect_leaks = false;
  f->leak_check_at_exit = true;
  f->allocator_may_return_null = false;
  f->print_summary = true;
  f->check_printf = false;
  // TODO(glider): tools may want to set different defaults for handle_segv.
  f->handle_segv = SANITIZER_NEEDS_SEGV;
  f->allow_user_segv_handler = false;
  f->use_sigaltstack = false;
  f->detect_deadlocks = false;
  f->clear_shadow_mmap_threshold = 64 * 1024;
  f->color = "auto";
}

void ParseCommonFlagsFromString(CommonFlags *f, const char *str) {
  ParseFlag(str, &f->symbolize, "symbolize");
  ParseFlag(str, &f->external_symbolizer_path, "external_symbolizer_path");
  ParseFlag(str, &f->allow_addr2line, "allow_addr2line");
  ParseFlag(str, &f->strip_path_prefix, "strip_path_prefix");
  ParseFlag(str, &f->fast_unwind_on_fatal, "fast_unwind_on_fatal");
  ParseFlag(str, &f->fast_unwind_on_malloc, "fast_unwind_on_malloc");
  ParseFlag(str, &f->handle_ioctl, "handle_ioctl");
  ParseFlag(str, &f->malloc_context_size, "malloc_context_size");
  ParseFlag(str, &f->log_path, "log_path");
  ParseFlag(str, &f->verbosity, "verbosity");
  ParseFlag(str, &f->detect_leaks, "detect_leaks");
  ParseFlag(str, &f->leak_check_at_exit, "leak_check_at_exit");
  ParseFlag(str, &f->allocator_may_return_null, "allocator_may_return_null");
  ParseFlag(str, &f->print_summary, "print_summary");
  ParseFlag(str, &f->check_printf, "check_printf");
  ParseFlag(str, &f->handle_segv, "handle_segv");
  ParseFlag(str, &f->allow_user_segv_handler, "allow_user_segv_handler");
  ParseFlag(str, &f->use_sigaltstack, "use_sigaltstack");
  ParseFlag(str, &f->detect_deadlocks, "detect_deadlocks");
  ParseFlag(str, &f->clear_shadow_mmap_threshold,
            "clear_shadow_mmap_threshold");
  ParseFlag(str, &f->color, "color");

  // Do a sanity check for certain flags.
  if (f->malloc_context_size < 1)
    f->malloc_context_size = 1;
}

static bool GetFlagValue(const char *env, const char *name,
                         const char **value, int *value_length) {
  if (env == 0)
    return false;
  const char *pos = 0;
  for (;;) {
    pos = internal_strstr(env, name);
    if (pos == 0)
      return false;
    if (pos != env && ((pos[-1] >= 'a' && pos[-1] <= 'z') || pos[-1] == '_')) {
      // Seems to be middle of another flag name or value.
      env = pos + 1;
      continue;
    }
    break;
  }
  pos += internal_strlen(name);
  const char *end;
  if (pos[0] != '=') {
    end = pos;
  } else {
    pos += 1;
    if (pos[0] == '"') {
      pos += 1;
      end = internal_strchr(pos, '"');
    } else if (pos[0] == '\'') {
      pos += 1;
      end = internal_strchr(pos, '\'');
    } else {
      // Read until the next space or colon.
      end = pos + internal_strcspn(pos, " :");
    }
    if (end == 0)
      end = pos + internal_strlen(pos);
  }
  *value = pos;
  *value_length = end - pos;
  return true;
}

static bool StartsWith(const char *flag, int flag_length, const char *value) {
  if (!flag || !value)
    return false;
  int value_length = internal_strlen(value);
  return (flag_length >= value_length) &&
         (0 == internal_strncmp(flag, value, value_length));
}

void ParseFlag(const char *env, bool *flag, const char *name) {
  const char *value;
  int value_length;
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  if (StartsWith(value, value_length, "0") ||
      StartsWith(value, value_length, "no") ||
      StartsWith(value, value_length, "false"))
    *flag = false;
  if (StartsWith(value, value_length, "1") ||
      StartsWith(value, value_length, "yes") ||
      StartsWith(value, value_length, "true"))
    *flag = true;
}

void ParseFlag(const char *env, int *flag, const char *name) {
  const char *value;
  int value_length;
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  *flag = static_cast<int>(internal_atoll(value));
}

void ParseFlag(const char *env, uptr *flag, const char *name) {
  const char *value;
  int value_length;
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  *flag = static_cast<uptr>(internal_atoll(value));
}

static LowLevelAllocator allocator_for_flags;

void ParseFlag(const char *env, const char **flag, const char *name) {
  const char *value;
  int value_length;
  if (!GetFlagValue(env, name, &value, &value_length))
    return;
  // Copy the flag value. Don't use locks here, as flags are parsed at
  // tool startup.
  char *value_copy = (char*)(allocator_for_flags.Allocate(value_length + 1));
  internal_memcpy(value_copy, value, value_length);
  value_copy[value_length] = '\0';
  *flag = value_copy;
}

}  // namespace __sanitizer
//===-- sanitizer_libc.cc -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries. See sanitizer_libc.h for details.
//===----------------------------------------------------------------------===//
#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

// Make the compiler think that something is going on there.
static inline void break_optimization(void *arg) {
#if _MSC_VER
  // FIXME: make sure this is actually enough.
  __asm;
#else
  __asm__ __volatile__("" : : "r" (arg) : "memory");
#endif
}

s64 internal_atoll(const char *nptr) {
  return internal_simple_strtoll(nptr, (char**)0, 10);
}

void *internal_memchr(const void *s, int c, uptr n) {
  const char* t = (char*)s;
  for (uptr i = 0; i < n; ++i, ++t)
    if (*t == c)
      return (void*)t;
  return 0;
}

int internal_memcmp(const void* s1, const void* s2, uptr n) {
  const char* t1 = (char*)s1;
  const char* t2 = (char*)s2;
  for (uptr i = 0; i < n; ++i, ++t1, ++t2)
    if (*t1 != *t2)
      return *t1 < *t2 ? -1 : 1;
  return 0;
}

void *internal_memcpy(void *dest, const void *src, uptr n) {
  char *d = (char*)dest;
  char *s = (char*)src;
  for (uptr i = 0; i < n; ++i)
    d[i] = s[i];
  return dest;
}

void *internal_memmove(void *dest, const void *src, uptr n) {
  char *d = (char*)dest;
  char *s = (char*)src;
  sptr i, signed_n = (sptr)n;
  CHECK_GE(signed_n, 0);
  if (d < s) {
    for (i = 0; i < signed_n; ++i)
      d[i] = s[i];
  } else {
    if (d > s && signed_n > 0)
      for (i = signed_n - 1; i >= 0 ; --i) {
        d[i] = s[i];
      }
  }
  return dest;
}

// Semi-fast bzero for 16-aligned data. Still far from peak performance.
void internal_bzero_aligned16(void *s, uptr n) {
  struct S16 { u64 a, b; } ALIGNED(16);
  CHECK_EQ((reinterpret_cast<uptr>(s) | n) & 15, 0);
  for (S16 *p = reinterpret_cast<S16*>(s), *end = p + n / 16; p < end; p++) {
    p->a = p->b = 0;
    break_optimization(0);  // Make sure this does not become memset.
  }
}

void *internal_memset(void* s, int c, uptr n) {
  // The next line prevents Clang from making a call to memset() instead of the
  // loop below.
  // FIXME: building the runtime with -ffreestanding is a better idea. However
  // there currently are linktime problems due to PR12396.
  char volatile *t = (char*)s;
  for (uptr i = 0; i < n; ++i, ++t) {
    *t = c;
  }
  return s;
}

uptr internal_strcspn(const char *s, const char *reject) {
  uptr i;
  for (i = 0; s[i]; i++) {
    if (internal_strchr(reject, s[i]) != 0)
      return i;
  }
  return i;
}

char* internal_strdup(const char *s) {
  uptr len = internal_strlen(s);
  char *s2 = (char*)InternalAlloc(len + 1);
  internal_memcpy(s2, s, len);
  s2[len] = 0;
  return s2;
}

int internal_strcmp(const char *s1, const char *s2) {
  while (true) {
    unsigned c1 = *s1;
    unsigned c2 = *s2;
    if (c1 != c2) return (c1 < c2) ? -1 : 1;
    if (c1 == 0) break;
    s1++;
    s2++;
  }
  return 0;
}

int internal_strncmp(const char *s1, const char *s2, uptr n) {
  for (uptr i = 0; i < n; i++) {
    unsigned c1 = *s1;
    unsigned c2 = *s2;
    if (c1 != c2) return (c1 < c2) ? -1 : 1;
    if (c1 == 0) break;
    s1++;
    s2++;
  }
  return 0;
}

char* internal_strchr(const char *s, int c) {
  while (true) {
    if (*s == (char)c)
      return (char*)s;
    if (*s == 0)
      return 0;
    s++;
  }
}

char *internal_strchrnul(const char *s, int c) {
  char *res = internal_strchr(s, c);
  if (!res)
    res = (char*)s + internal_strlen(s);
  return res;
}

char *internal_strrchr(const char *s, int c) {
  const char *res = 0;
  for (uptr i = 0; s[i]; i++) {
    if (s[i] == c) res = s + i;
  }
  return (char*)res;
}

uptr internal_strlen(const char *s) {
  uptr i = 0;
  while (s[i]) i++;
  return i;
}

char *internal_strncat(char *dst, const char *src, uptr n) {
  uptr len = internal_strlen(dst);
  uptr i;
  for (i = 0; i < n && src[i]; i++)
    dst[len + i] = src[i];
  dst[len + i] = 0;
  return dst;
}

char *internal_strncpy(char *dst, const char *src, uptr n) {
  uptr i;
  for (i = 0; i < n && src[i]; i++)
    dst[i] = src[i];
  internal_memset(dst + i, '\0', n - i);
  return dst;
}

uptr internal_strnlen(const char *s, uptr maxlen) {
  uptr i = 0;
  while (i < maxlen && s[i]) i++;
  return i;
}

char *internal_strstr(const char *haystack, const char *needle) {
  // This is O(N^2), but we are not using it in hot places.
  uptr len1 = internal_strlen(haystack);
  uptr len2 = internal_strlen(needle);
  if (len1 < len2) return 0;
  for (uptr pos = 0; pos <= len1 - len2; pos++) {
    if (internal_memcmp(haystack + pos, needle, len2) == 0)
      return (char*)haystack + pos;
  }
  return 0;
}

s64 internal_simple_strtoll(const char *nptr, char **endptr, int base) {
  CHECK_EQ(base, 10);
  while (IsSpace(*nptr)) nptr++;
  int sgn = 1;
  u64 res = 0;
  bool have_digits = false;
  char *old_nptr = (char*)nptr;
  if (*nptr == '+') {
    sgn = 1;
    nptr++;
  } else if (*nptr == '-') {
    sgn = -1;
    nptr++;
  }
  while (IsDigit(*nptr)) {
    res = (res <= UINT64_MAX / 10) ? res * 10 : UINT64_MAX;
    int digit = ((*nptr) - '0');
    res = (res <= UINT64_MAX - digit) ? res + digit : UINT64_MAX;
    have_digits = true;
    nptr++;
  }
  if (endptr != 0) {
    *endptr = (have_digits) ? (char*)nptr : old_nptr;
  }
  if (sgn > 0) {
    return (s64)(Min((u64)INT64_MAX, res));
  } else {
    return (res > INT64_MAX) ? INT64_MIN : ((s64)res * -1);
  }
}

bool mem_is_zero(const char *beg, uptr size) {
  CHECK_LE(size, 1ULL << FIRST_32_SECOND_64(30, 40));  // Sanity check.
  const char *end = beg + size;
  uptr *aligned_beg = (uptr *)RoundUpTo((uptr)beg, sizeof(uptr));
  uptr *aligned_end = (uptr *)RoundDownTo((uptr)end, sizeof(uptr));
  uptr all = 0;
  // Prologue.
  for (const char *mem = beg; mem < (char*)aligned_beg && mem < end; mem++)
    all |= *mem;
  // Aligned loop.
  for (; aligned_beg < aligned_end; aligned_beg++)
    all |= *aligned_beg;
  // Epilogue.
  if ((char*)aligned_end >= beg)
    for (const char *mem = (char*)aligned_end; mem < end; mem++)
      all |= *mem;
  return all == 0;
}

}  // namespace __sanitizer
//===-- sanitizer_printf.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer.
//
// Internal printf function, used inside run-time libraries.
// We can't use libc printf because we intercept some of the functions used
// inside it.
//===----------------------------------------------------------------------===//


#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"

#include <stdio.h>
#include <stdarg.h>

#if SANITIZER_WINDOWS && !defined(va_copy)
# define va_copy(dst, src) ((dst) = (src))
#endif

namespace __sanitizer {

StaticSpinMutex CommonSanitizerReportMutex;

static int AppendChar(char **buff, const char *buff_end, char c) {
  if (*buff < buff_end) {
    **buff = c;
    (*buff)++;
  }
  return 1;
}

// Appends number in a given base to buffer. If its length is less than
// |minimal_num_length|, it is padded with leading zeroes or spaces, depending
// on the value of |pad_with_zero|.
static int AppendNumber(char **buff, const char *buff_end, u64 absolute_value,
                        u8 base, u8 minimal_num_length, bool pad_with_zero,
                        bool negative) {
  uptr const kMaxLen = 30;
  RAW_CHECK(base == 10 || base == 16);
  RAW_CHECK(base == 10 || !negative);
  RAW_CHECK(absolute_value || !negative);
  RAW_CHECK(minimal_num_length < kMaxLen);
  int result = 0;
  if (negative && minimal_num_length)
    --minimal_num_length;
  if (negative && pad_with_zero)
    result += AppendChar(buff, buff_end, '-');
  uptr num_buffer[kMaxLen];
  int pos = 0;
  do {
    RAW_CHECK_MSG((uptr)pos < kMaxLen, "AppendNumber buffer overflow");
    num_buffer[pos++] = absolute_value % base;
    absolute_value /= base;
  } while (absolute_value > 0);
  if (pos < minimal_num_length) {
    // Make sure compiler doesn't insert call to memset here.
    internal_memset(&num_buffer[pos], 0,
                    sizeof(num_buffer[0]) * (minimal_num_length - pos));
    pos = minimal_num_length;
  }
  RAW_CHECK(pos > 0);
  pos--;
  for (; pos >= 0 && num_buffer[pos] == 0; pos--) {
    char c = (pad_with_zero || pos == 0) ? '0' : ' ';
    result += AppendChar(buff, buff_end, c);
  }
  if (negative && !pad_with_zero) result += AppendChar(buff, buff_end, '-');
  for (; pos >= 0; pos--) {
    char digit = static_cast<char>(num_buffer[pos]);
    result += AppendChar(buff, buff_end, (digit < 10) ? '0' + digit
                                                      : 'a' + digit - 10);
  }
  return result;
}

static int AppendUnsigned(char **buff, const char *buff_end, u64 num, u8 base,
                          u8 minimal_num_length, bool pad_with_zero) {
  return AppendNumber(buff, buff_end, num, base, minimal_num_length,
                      pad_with_zero, false /* negative */);
}

static int AppendSignedDecimal(char **buff, const char *buff_end, s64 num,
                               u8 minimal_num_length, bool pad_with_zero) {
  bool negative = (num < 0);
  return AppendNumber(buff, buff_end, (u64)(negative ? -num : num), 10,
                      minimal_num_length, pad_with_zero, negative);
}

static int AppendString(char **buff, const char *buff_end, int precision,
                        const char *s) {
  if (s == 0)
    s = "<null>";
  int result = 0;
  for (; *s; s++) {
    if (precision >= 0 && result >= precision)
      break;
    result += AppendChar(buff, buff_end, *s);
  }
  return result;
}

static int AppendPointer(char **buff, const char *buff_end, u64 ptr_value) {
  int result = 0;
  result += AppendString(buff, buff_end, -1, "0x");
  result += AppendUnsigned(buff, buff_end, ptr_value, 16,
                           (SANITIZER_WORDSIZE == 64) ? 12 : 8, true);
  return result;
}

int VSNPrintf(char *buff, int buff_length,
              const char *format, va_list args) {
  static const char *kPrintfFormatsHelp =
    "Supported Printf formats: %([0-9]*)?(z|ll)?{d,u,x}; %p; %(\\.\\*)?s; %c\n";
  RAW_CHECK(format);
  RAW_CHECK(buff_length > 0);
  const char *buff_end = &buff[buff_length - 1];
  const char *cur = format;
  int result = 0;
  for (; *cur; cur++) {
    if (*cur != '%') {
      result += AppendChar(&buff, buff_end, *cur);
      continue;
    }
    cur++;
    bool have_width = (*cur >= '0' && *cur <= '9');
    bool pad_with_zero = (*cur == '0');
    int width = 0;
    if (have_width) {
      while (*cur >= '0' && *cur <= '9') {
        width = width * 10 + *cur++ - '0';
      }
    }
    bool have_precision = (cur[0] == '.' && cur[1] == '*');
    int precision = -1;
    if (have_precision) {
      cur += 2;
      precision = va_arg(args, int);
    }
    bool have_z = (*cur == 'z');
    cur += have_z;
    bool have_ll = !have_z && (cur[0] == 'l' && cur[1] == 'l');
    cur += have_ll * 2;
    s64 dval;
    u64 uval;
    bool have_flags = have_width | have_z | have_ll;
    // Only %s supports precision for now
    CHECK(!(precision >= 0 && *cur != 's'));
    switch (*cur) {
      case 'd': {
        dval = have_ll ? va_arg(args, s64)
             : have_z ? va_arg(args, sptr)
             : va_arg(args, int);
        result += AppendSignedDecimal(&buff, buff_end, dval, width,
                                      pad_with_zero);
        break;
      }
      case 'u':
      case 'x': {
        uval = have_ll ? va_arg(args, u64)
             : have_z ? va_arg(args, uptr)
             : va_arg(args, unsigned);
        result += AppendUnsigned(&buff, buff_end, uval,
                                 (*cur == 'u') ? 10 : 16, width, pad_with_zero);
        break;
      }
      case 'p': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendPointer(&buff, buff_end, va_arg(args, uptr));
        break;
      }
      case 's': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendString(&buff, buff_end, precision, va_arg(args, char*));
        break;
      }
      case 'c': {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendChar(&buff, buff_end, va_arg(args, int));
        break;
      }
      case '%' : {
        RAW_CHECK_MSG(!have_flags, kPrintfFormatsHelp);
        result += AppendChar(&buff, buff_end, '%');
        break;
      }
      default: {
        RAW_CHECK_MSG(false, kPrintfFormatsHelp);
      }
    }
  }
  RAW_CHECK(buff <= buff_end);
  AppendChar(&buff, buff_end + 1, '\0');
  return result;
}

static void (*PrintfAndReportCallback)(const char *);
void SetPrintfAndReportCallback(void (*callback)(const char *)) {
  PrintfAndReportCallback = callback;
}

// Can be overriden in frontend.
#if SANITIZER_SUPPORTS_WEAK_HOOKS
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void OnPrint(const char *str) {
  (void)str;
}
#elif defined(SANITIZER_GO) && defined(TSAN_EXTERNAL_HOOKS)
void OnPrint(const char *str);
#else
void OnPrint(const char *str) {
  (void)str;
}
#endif

static void CallPrintfAndReportCallback(const char *str) {
  OnPrint(str);
  if (PrintfAndReportCallback)
    PrintfAndReportCallback(str);
}

static void SharedPrintfCode(bool append_pid, const char *format,
                             va_list args) {
  va_list args2;
  va_copy(args2, args);
  const int kLen = 16 * 1024;
  // |local_buffer| is small enough not to overflow the stack and/or violate
  // the stack limit enforced by TSan (-Wframe-larger-than=512). On the other
  // hand, the bigger the buffer is, the more the chance the error report will
  // fit into it.
  char local_buffer[400];
  int needed_length;
  char *buffer = local_buffer;
  int buffer_size = ARRAY_SIZE(local_buffer);
  // First try to print a message using a local buffer, and then fall back to
  // mmaped buffer.
  for (int use_mmap = 0; use_mmap < 2; use_mmap++) {
    if (use_mmap) {
      va_end(args);
      va_copy(args, args2);
      buffer = (char*)MmapOrDie(kLen, "Report");
      buffer_size = kLen;
    }
    needed_length = 0;
    if (append_pid) {
      int pid = internal_getpid();
      needed_length += internal_snprintf(buffer, buffer_size, "==%d==", pid);
      if (needed_length >= buffer_size) {
        // The pid doesn't fit into the current buffer.
        if (!use_mmap)
          continue;
        RAW_CHECK_MSG(needed_length < kLen, "Buffer in Report is too short!\n");
      }
    }
    needed_length += VSNPrintf(buffer + needed_length,
                               buffer_size - needed_length, format, args);
    if (needed_length >= buffer_size) {
      // The message doesn't fit into the current buffer.
      if (!use_mmap)
        continue;
      RAW_CHECK_MSG(needed_length < kLen, "Buffer in Report is too short!\n");
    }
    // If the message fit into the buffer, print it and exit.
    break;
  }
  RawWrite(buffer);
  AndroidLogWrite(buffer);
  CallPrintfAndReportCallback(buffer);
  // If we had mapped any memory, clean up.
  if (buffer != local_buffer)
    UnmapOrDie((void *)buffer, buffer_size);
  va_end(args2);
}

FORMAT(1, 2)
void Printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  SharedPrintfCode(false, format, args);
  va_end(args);
}

// Like Printf, but prints the current PID before the output string.
FORMAT(1, 2)
void Report(const char *format, ...) {
  va_list args;
  va_start(args, format);
  SharedPrintfCode(true, format, args);
  va_end(args);
}

// Writes at most "length" symbols to "buffer" (including trailing '\0').
// Returns the number of symbols that should have been written to buffer
// (not including trailing '\0'). Thus, the string is truncated
// iff return value is not less than "length".
FORMAT(3, 4)
int internal_snprintf(char *buffer, uptr length, const char *format, ...) {
  va_list args;
  va_start(args, format);
  int needed_length = VSNPrintf(buffer, length, format, args);
  va_end(args);
  return needed_length;
}

FORMAT(2, 3)
void InternalScopedString::append(const char *format, ...) {
  CHECK_LT(length_, size());
  va_list args;
  va_start(args, format);
  VSNPrintf(data() + length_, size() - length_, format, args);
  va_end(args);
  length_ += internal_strlen(data() + length_);
  CHECK_LT(length_, size());
}

}  // namespace __sanitizer
//===-- sanitizer_suppressions.cc -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Suppression parsing/matching code shared between TSan and LSan.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_suppressions.h"

#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_libc.h"

namespace __sanitizer {

static const char *const kTypeStrings[SuppressionTypeCount] = {
  "none", "race", "mutex", "thread", "signal", "leak", "called_from_lib"
};

bool TemplateMatch(char *templ, const char *str) {
  if (str == 0 || str[0] == 0)
    return false;
  bool start = false;
  if (templ && templ[0] == '^') {
    start = true;
    templ++;
  }
  bool asterisk = false;
  while (templ && templ[0]) {
    if (templ[0] == '*') {
      templ++;
      start = false;
      asterisk = true;
      continue;
    }
    if (templ[0] == '$')
      return str[0] == 0 || asterisk;
    if (str[0] == 0)
      return false;
    char *tpos = (char*)internal_strchr(templ, '*');
    char *tpos1 = (char*)internal_strchr(templ, '$');
    if (tpos == 0 || (tpos1 && tpos1 < tpos))
      tpos = tpos1;
    if (tpos != 0)
      tpos[0] = 0;
    const char *str0 = str;
    const char *spos = internal_strstr(str, templ);
    str = spos + internal_strlen(templ);
    templ = tpos;
    if (tpos)
      tpos[0] = tpos == tpos1 ? '$' : '*';
    if (spos == 0)
      return false;
    if (start && spos != str0)
      return false;
    start = false;
    asterisk = false;
  }
  return true;
}

bool SuppressionContext::Match(const char *str, SuppressionType type,
                               Suppression **s) {
  can_parse_ = false;
  uptr i;
  for (i = 0; i < suppressions_.size(); i++)
    if (type == suppressions_[i].type &&
        TemplateMatch(suppressions_[i].templ, str))
      break;
  if (i == suppressions_.size()) return false;
  *s = &suppressions_[i];
  return true;
}

static const char *StripPrefix(const char *str, const char *prefix) {
  while (str && *str == *prefix) {
    str++;
    prefix++;
  }
  if (!*prefix)
    return str;
  return 0;
}

void SuppressionContext::Parse(const char *str) {
  // Context must not mutate once Match has been called.
  CHECK(can_parse_);
  const char *line = str;
  while (line) {
    while (line[0] == ' ' || line[0] == '\t')
      line++;
    const char *end = internal_strchr(line, '\n');
    if (end == 0)
      end = line + internal_strlen(line);
    if (line != end && line[0] != '#') {
      const char *end2 = end;
      while (line != end2 && (end2[-1] == ' ' || end2[-1] == '\t'))
        end2--;
      int type;
      for (type = 0; type < SuppressionTypeCount; type++) {
        const char *next_char = StripPrefix(line, kTypeStrings[type]);
        if (next_char && *next_char == ':') {
          line = ++next_char;
          break;
        }
      }
      if (type == SuppressionTypeCount) {
        Printf("%s: failed to parse suppressions\n", SanitizerToolName);
        Die();
      }
      Suppression s;
      s.type = static_cast<SuppressionType>(type);
      s.templ = (char*)InternalAlloc(end2 - line + 1);
      internal_memcpy(s.templ, line, end2 - line);
      s.templ[end2 - line] = 0;
      s.hit_count = 0;
      s.weight = 0;
      suppressions_.push_back(s);
    }
    if (end[0] == 0)
      break;
    line = end + 1;
  }
}

uptr SuppressionContext::SuppressionCount() const {
  return suppressions_.size();
}

const Suppression *SuppressionContext::SuppressionAt(uptr i) const {
  CHECK_LT(i, suppressions_.size());
  return &suppressions_[i];
}

void SuppressionContext::GetMatched(
    InternalMmapVector<Suppression *> *matched) {
  for (uptr i = 0; i < suppressions_.size(); i++)
    if (suppressions_[i].hit_count)
      matched->push_back(&suppressions_[i]);
}

const char *SuppressionTypeString(SuppressionType t) {
  CHECK(t < SuppressionTypeCount);
  return kTypeStrings[t];
}

}  // namespace __sanitizer
//===-- sanitizer_thread_registry.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between sanitizer tools.
//
// General thread bookkeeping functionality.
//===----------------------------------------------------------------------===//

#include "sanitizer_thread_registry.h"

namespace __sanitizer {

ThreadContextBase::ThreadContextBase(u32 tid)
    : tid(tid), unique_id(0), os_id(0), user_id(0), status(ThreadStatusInvalid),
      detached(false), reuse_count(0), parent_tid(0), next(0) {
  name[0] = '\0';
}

ThreadContextBase::~ThreadContextBase() {
  // ThreadContextBase should never be deleted.
  CHECK(0);
}

void ThreadContextBase::SetName(const char *new_name) {
  name[0] = '\0';
  if (new_name) {
    internal_strncpy(name, new_name, sizeof(name));
    name[sizeof(name) - 1] = '\0';
  }
}

void ThreadContextBase::SetDead() {
  CHECK(status == ThreadStatusRunning ||
        status == ThreadStatusFinished);
  status = ThreadStatusDead;
  user_id = 0;
  OnDead();
}

void ThreadContextBase::SetJoined(void *arg) {
  // FIXME(dvyukov): print message and continue (it's user error).
  CHECK_EQ(false, detached);
  CHECK_EQ(ThreadStatusFinished, status);
  status = ThreadStatusDead;
  user_id = 0;
  OnJoined(arg);
}

void ThreadContextBase::SetFinished() {
  if (!detached)
    status = ThreadStatusFinished;
  OnFinished();
}

void ThreadContextBase::SetStarted(uptr _os_id, void *arg) {
  status = ThreadStatusRunning;
  os_id = _os_id;
  OnStarted(arg);
}

void ThreadContextBase::SetCreated(uptr _user_id, u64 _unique_id,
                                   bool _detached, u32 _parent_tid, void *arg) {
  status = ThreadStatusCreated;
  user_id = _user_id;
  unique_id = _unique_id;
  detached = _detached;
  // Parent tid makes no sense for the main thread.
  if (tid != 0)
    parent_tid = _parent_tid;
  OnCreated(arg);
}

void ThreadContextBase::Reset() {
  status = ThreadStatusInvalid;
  reuse_count++;
  SetName(0);
  OnReset();
}

// ThreadRegistry implementation.

const u32 ThreadRegistry::kUnknownTid = ~0U;

ThreadRegistry::ThreadRegistry(ThreadContextFactory factory, u32 max_threads,
                               u32 thread_quarantine_size)
    : context_factory_(factory),
      max_threads_(max_threads),
      thread_quarantine_size_(thread_quarantine_size),
      mtx_(),
      n_contexts_(0),
      total_threads_(0),
      alive_threads_(0),
      max_alive_threads_(0),
      running_threads_(0) {
  threads_ = (ThreadContextBase **)MmapOrDie(max_threads_ * sizeof(threads_[0]),
                                             "ThreadRegistry");
  dead_threads_.clear();
  invalid_threads_.clear();
}

void ThreadRegistry::GetNumberOfThreads(uptr *total, uptr *running,
                                        uptr *alive) {
  BlockingMutexLock l(&mtx_);
  if (total) *total = n_contexts_;
  if (running) *running = running_threads_;
  if (alive) *alive = alive_threads_;
}

uptr ThreadRegistry::GetMaxAliveThreads() {
  BlockingMutexLock l(&mtx_);
  return max_alive_threads_;
}

u32 ThreadRegistry::CreateThread(uptr user_id, bool detached, u32 parent_tid,
                                 void *arg) {
  BlockingMutexLock l(&mtx_);
  u32 tid = kUnknownTid;
  ThreadContextBase *tctx = QuarantinePop();
  if (tctx) {
    tid = tctx->tid;
  } else if (n_contexts_ < max_threads_) {
    // Allocate new thread context and tid.
    tid = n_contexts_++;
    tctx = context_factory_(tid);
    threads_[tid] = tctx;
  } else {
#ifndef SANITIZER_GO
    Report("%s: Thread limit (%u threads) exceeded. Dying.\n",
           SanitizerToolName, max_threads_);
#else
    Printf("race: limit on %u simultaneously alive goroutines is exceeded,"
        " dying\n", max_threads_);
#endif
    Die();
  }
  CHECK_NE(tctx, 0);
  CHECK_NE(tid, kUnknownTid);
  CHECK_LT(tid, max_threads_);
  CHECK_EQ(tctx->status, ThreadStatusInvalid);
  alive_threads_++;
  if (max_alive_threads_ < alive_threads_) {
    max_alive_threads_++;
    CHECK_EQ(alive_threads_, max_alive_threads_);
  }
  tctx->SetCreated(user_id, total_threads_++, detached,
                   parent_tid, arg);
  return tid;
}

void ThreadRegistry::RunCallbackForEachThreadLocked(ThreadCallback cb,
                                                    void *arg) {
  CheckLocked();
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx == 0)
      continue;
    cb(tctx, arg);
  }
}

u32 ThreadRegistry::FindThread(FindThreadCallback cb, void *arg) {
  BlockingMutexLock l(&mtx_);
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx != 0 && cb(tctx, arg))
      return tctx->tid;
  }
  return kUnknownTid;
}

ThreadContextBase *
ThreadRegistry::FindThreadContextLocked(FindThreadCallback cb, void *arg) {
  CheckLocked();
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx != 0 && cb(tctx, arg))
      return tctx;
  }
  return 0;
}

static bool FindThreadContextByOsIdCallback(ThreadContextBase *tctx,
                                            void *arg) {
  return (tctx->os_id == (uptr)arg && tctx->status != ThreadStatusInvalid &&
      tctx->status != ThreadStatusDead);
}

ThreadContextBase *ThreadRegistry::FindThreadContextByOsIDLocked(uptr os_id) {
  return FindThreadContextLocked(FindThreadContextByOsIdCallback,
                                 (void *)os_id);
}

void ThreadRegistry::SetThreadName(u32 tid, const char *name) {
  BlockingMutexLock l(&mtx_);
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(ThreadStatusRunning, tctx->status);
  tctx->SetName(name);
}

void ThreadRegistry::SetThreadNameByUserId(uptr user_id, const char *name) {
  BlockingMutexLock l(&mtx_);
  for (u32 tid = 0; tid < n_contexts_; tid++) {
    ThreadContextBase *tctx = threads_[tid];
    if (tctx != 0 && tctx->user_id == user_id &&
        tctx->status != ThreadStatusInvalid) {
      tctx->SetName(name);
      return;
    }
  }
}

void ThreadRegistry::DetachThread(u32 tid) {
  BlockingMutexLock l(&mtx_);
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  if (tctx->status == ThreadStatusInvalid) {
    Report("%s: Detach of non-existent thread\n", SanitizerToolName);
    return;
  }
  if (tctx->status == ThreadStatusFinished) {
    tctx->SetDead();
    QuarantinePush(tctx);
  } else {
    tctx->detached = true;
  }
}

void ThreadRegistry::JoinThread(u32 tid, void *arg) {
  BlockingMutexLock l(&mtx_);
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  if (tctx->status == ThreadStatusInvalid) {
    Report("%s: Join of non-existent thread\n", SanitizerToolName);
    return;
  }
  tctx->SetJoined(arg);
  QuarantinePush(tctx);
}

void ThreadRegistry::FinishThread(u32 tid) {
  BlockingMutexLock l(&mtx_);
  CHECK_GT(alive_threads_, 0);
  alive_threads_--;
  CHECK_GT(running_threads_, 0);
  running_threads_--;
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(ThreadStatusRunning, tctx->status);
  tctx->SetFinished();
  if (tctx->detached) {
    tctx->SetDead();
    QuarantinePush(tctx);
  }
}

void ThreadRegistry::StartThread(u32 tid, uptr os_id, void *arg) {
  BlockingMutexLock l(&mtx_);
  running_threads_++;
  CHECK_LT(tid, n_contexts_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(ThreadStatusCreated, tctx->status);
  tctx->SetStarted(os_id, arg);
}

void ThreadRegistry::QuarantinePush(ThreadContextBase *tctx) {
  dead_threads_.push_back(tctx);
  if (dead_threads_.size() <= thread_quarantine_size_)
    return;
  tctx = dead_threads_.front();
  dead_threads_.pop_front();
  CHECK_EQ(tctx->status, ThreadStatusDead);
  tctx->Reset();
  invalid_threads_.push_back(tctx);
}

ThreadContextBase *ThreadRegistry::QuarantinePop() {
  if (invalid_threads_.size() == 0)
    return 0;
  ThreadContextBase *tctx = invalid_threads_.front();
  invalid_threads_.pop_front();
  return tctx;
}

}  // namespace __sanitizer
//===-- sanitizer_posix.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements POSIX-specific functions from
// sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX || SANITIZER_MAC

#include "sanitizer_common.h"
#include "sanitizer_libc.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"

#include <sys/mman.h>

namespace __sanitizer {

// ------------- sanitizer_common.h
uptr GetMmapGranularity() {
  return GetPageSize();
}

uptr GetMaxVirtualAddress() {
#if SANITIZER_WORDSIZE == 64
# if defined(__powerpc64__)
  // On PowerPC64 we have two different address space layouts: 44- and 46-bit.
  // We somehow need to figure our which one we are using now and choose
  // one of 0x00000fffffffffffUL and 0x00003fffffffffffUL.
  // Note that with 'ulimit -s unlimited' the stack is moved away from the top
  // of the address space, so simply checking the stack address is not enough.
  return (1ULL << 44) - 1;  // 0x00000fffffffffffUL
# elif defined(__aarch64__)
  return (1ULL << 39) - 1;
# else
  return (1ULL << 47) - 1;  // 0x00007fffffffffffUL;
# endif
#else  // SANITIZER_WORDSIZE == 32
  // FIXME: We can probably lower this on Android?
  return (1ULL << 32) - 1;  // 0xffffffff;
#endif  // SANITIZER_WORDSIZE
}

void *MmapOrDie(uptr size, const char *mem_type) {
  size = RoundUpTo(size, GetPageSizeCached());
  uptr res = internal_mmap(0, size,
                            PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANON, -1, 0);
  int reserrno;
  if (internal_iserror(res, &reserrno)) {
    static int recursion_count;
    if (recursion_count) {
      // The Report() and CHECK calls below may call mmap recursively and fail.
      // If we went into recursion, just die.
      RawWrite("ERROR: Failed to mmap\n");
      Die();
    }
    recursion_count++;
    Report("ERROR: %s failed to allocate 0x%zx (%zd) bytes of %s: %d\n",
           SanitizerToolName, size, size, mem_type, reserrno);
    DumpProcessMap();
    CHECK("unable to mmap" && 0);
  }
  return (void *)res;
}

void UnmapOrDie(void *addr, uptr size) {
  if (!addr || !size) return;
  uptr res = internal_munmap(addr, size);
  if (internal_iserror(res)) {
    Report("ERROR: %s failed to deallocate 0x%zx (%zd) bytes at address %p\n",
           SanitizerToolName, size, size, addr);
    CHECK("unable to unmap" && 0);
  }
}

void *MmapNoReserveOrDie(uptr size, const char *mem_type) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap(0,
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno)) {
    Report("ERROR: "
           "%s failed to allocate noreserve 0x%zx (%zd) bytes for '%s' (%d)\n",
           SanitizerToolName, size, size, mem_type, reserrno);
    CHECK("unable to mmap" && 0);
  }
  return (void *)p;
}

void *MmapFixedNoReserve(uptr fixed_addr, uptr size) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void*)(fixed_addr & ~(PageSize - 1)),
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED | MAP_NORESERVE,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno))
    Report("ERROR: "
           "%s failed to allocate 0x%zx (%zd) bytes at address %zu (%d)\n",
           SanitizerToolName, size, size, fixed_addr, reserrno);
  return (void *)p;
}

void *MmapFixedOrDie(uptr fixed_addr, uptr size) {
  uptr PageSize = GetPageSizeCached();
  uptr p = internal_mmap((void*)(fixed_addr & ~(PageSize - 1)),
      RoundUpTo(size, PageSize),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE | MAP_ANON | MAP_FIXED,
      -1, 0);
  int reserrno;
  if (internal_iserror(p, &reserrno)) {
    Report("ERROR:"
           " %s failed to allocate 0x%zx (%zd) bytes at address %zu (%d)\n",
           SanitizerToolName, size, size, fixed_addr, reserrno);
    CHECK("unable to mmap" && 0);
  }
  return (void *)p;
}

void *Mprotect(uptr fixed_addr, uptr size) {
  return (void *)internal_mmap((void*)fixed_addr, size,
                               PROT_NONE,
                               MAP_PRIVATE | MAP_ANON | MAP_FIXED |
                               MAP_NORESERVE, -1, 0);
}

void *MapFileToMemory(const char *file_name, uptr *buff_size) {
  uptr openrv = OpenFile(file_name, false);
  CHECK(!internal_iserror(openrv));
  fd_t fd = openrv;
  uptr fsize = internal_filesize(fd);
  CHECK_NE(fsize, (uptr)-1);
  CHECK_GT(fsize, 0);
  *buff_size = RoundUpTo(fsize, GetPageSizeCached());
  uptr map = internal_mmap(0, *buff_size, PROT_READ, MAP_PRIVATE, fd, 0);
  return internal_iserror(map) ? 0 : (void *)map;
}


static inline bool IntervalsAreSeparate(uptr start1, uptr end1,
                                        uptr start2, uptr end2) {
  CHECK(start1 <= end1);
  CHECK(start2 <= end2);
  return (end1 < start2) || (end2 < start1);
}

// FIXME: this is thread-unsafe, but should not cause problems most of the time.
// When the shadow is mapped only a single thread usually exists (plus maybe
// several worker threads on Mac, which aren't expected to map big chunks of
// memory).
bool MemoryRangeIsAvailable(uptr range_start, uptr range_end) {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end;
  while (proc_maps.Next(&start, &end,
                        /*offset*/0, /*filename*/0, /*filename_size*/0,
                        /*protection*/0)) {
    if (!IntervalsAreSeparate(start, end, range_start, range_end))
      return false;
  }
  return true;
}

void DumpProcessMap() {
  MemoryMappingLayout proc_maps(/*cache_enabled*/true);
  uptr start, end;
  const sptr kBufSize = 4095;
  char *filename = (char*)MmapOrDie(kBufSize, __func__);
  Report("Process memory map follows:\n");
  while (proc_maps.Next(&start, &end, /* file_offset */0,
                        filename, kBufSize, /* protection */0)) {
    Printf("\t%p-%p\t%s\n", (void*)start, (void*)end, filename);
  }
  Report("End of process memory map.\n");
  UnmapOrDie(filename, kBufSize);
}

const char *GetPwd() {
  return GetEnv("PWD");
}

char *FindPathToBinary(const char *name) {
  const char *path = GetEnv("PATH");
  if (!path)
    return 0;
  uptr name_len = internal_strlen(name);
  InternalScopedBuffer<char> buffer(kMaxPathLength);
  const char *beg = path;
  while (true) {
    const char *end = internal_strchrnul(beg, ':');
    uptr prefix_len = end - beg;
    if (prefix_len + name_len + 2 <= kMaxPathLength) {
      internal_memcpy(buffer.data(), beg, prefix_len);
      buffer[prefix_len] = '/';
      internal_memcpy(&buffer[prefix_len + 1], name, name_len);
      buffer[prefix_len + 1 + name_len] = '\0';
      if (FileExists(buffer.data()))
        return internal_strdup(buffer.data());
    }
    if (*end == '\0') break;
    beg = end + 1;
  }
  return 0;
}

void MaybeOpenReportFile() {
  if (!log_to_file) return;
  uptr pid = internal_getpid();
  // If in tracer, use the parent's file.
  if (pid == stoptheworld_tracer_pid)
    pid = stoptheworld_tracer_ppid;
  if (report_fd_pid == pid) return;
  InternalScopedBuffer<char> report_path_full(4096);
  internal_snprintf(report_path_full.data(), report_path_full.size(),
                    "%s.%zu", report_path_prefix, pid);
  uptr openrv = OpenFile(report_path_full.data(), true);
  if (internal_iserror(openrv)) {
    report_fd = kStderrFd;
    log_to_file = false;
    Report("ERROR: Can't open file: %s\n", report_path_full.data());
    Die();
  }
  if (report_fd != kInvalidFd) {
    // We're in the child. Close the parent's log.
    internal_close(report_fd);
  }
  report_fd = openrv;
  report_fd_pid = pid;
}

void RawWrite(const char *buffer) {
  static const char *kRawWriteError =
      "RawWrite can't output requested buffer!\n";
  uptr length = (uptr)internal_strlen(buffer);
  MaybeOpenReportFile();
  if (length != internal_write(report_fd, buffer, length)) {
    internal_write(report_fd, kRawWriteError, internal_strlen(kRawWriteError));
    Die();
  }
}

bool GetCodeRangeForFile(const char *module, uptr *start, uptr *end) {
  uptr s, e, off, prot;
  InternalScopedString buff(4096);
  MemoryMappingLayout proc_maps(/*cache_enabled*/false);
  while (proc_maps.Next(&s, &e, &off, buff.data(), buff.size(), &prot)) {
    if ((prot & MemoryMappingLayout::kProtectionExecute) != 0
        && internal_strcmp(module, buff.data()) == 0) {
      *start = s;
      *end = e;
      return true;
    }
  }
  return false;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX || SANITIZER_MAC
//===-- sanitizer_posix_libcdep.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries and implements libc-dependent POSIX-specific functions
// from sanitizer_libc.h.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#if SANITIZER_LINUX || SANITIZER_MAC
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_platform_limits_posix.h"
#include "sanitizer_stacktrace.h"

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace __sanitizer {

u32 GetUid() {
  return getuid();
}

uptr GetThreadSelf() {
  return (uptr)pthread_self();
}

void FlushUnneededShadowMemory(uptr addr, uptr size) {
  madvise((void*)addr, size, MADV_DONTNEED);
}

void DisableCoreDumper() {
  struct rlimit nocore;
  nocore.rlim_cur = 0;
  nocore.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &nocore);
}

bool StackSizeIsUnlimited() {
  struct rlimit rlim;
  CHECK_EQ(0, getrlimit(RLIMIT_STACK, &rlim));
  return (rlim.rlim_cur == (uptr)-1);
}

void SetStackSizeLimitInBytes(uptr limit) {
  struct rlimit rlim;
  rlim.rlim_cur = limit;
  rlim.rlim_max = limit;
  if (setrlimit(RLIMIT_STACK, &rlim)) {
    Report("ERROR: %s setrlimit() failed %d\n", SanitizerToolName, errno);
    Die();
  }
  CHECK(!StackSizeIsUnlimited());
}

void SleepForSeconds(int seconds) {
  sleep(seconds);
}

void SleepForMillis(int millis) {
  usleep(millis * 1000);
}

void Abort() {
  abort();
}

int Atexit(void (*function)(void)) {
#ifndef SANITIZER_GO
  return atexit(function);
#else
  return 0;
#endif
}

int internal_isatty(fd_t fd) {
  return isatty(fd);
}

#ifndef SANITIZER_GO
// TODO(glider): different tools may require different altstack size.
static const uptr kAltStackSize = SIGSTKSZ * 4;  // SIGSTKSZ is not enough.

void SetAlternateSignalStack() {
  stack_t altstack, oldstack;
  CHECK_EQ(0, sigaltstack(0, &oldstack));
  // If the alternate stack is already in place, do nothing.
  // Android always sets an alternate stack, but it's too small for us.
  if (!SANITIZER_ANDROID && !(oldstack.ss_flags & SS_DISABLE)) return;
  // TODO(glider): the mapped stack should have the MAP_STACK flag in the
  // future. It is not required by man 2 sigaltstack now (they're using
  // malloc()).
  void* base = MmapOrDie(kAltStackSize, __func__);
  altstack.ss_sp = base;
  altstack.ss_flags = 0;
  altstack.ss_size = kAltStackSize;
  CHECK_EQ(0, sigaltstack(&altstack, 0));
}

void UnsetAlternateSignalStack() {
  stack_t altstack, oldstack;
  altstack.ss_sp = 0;
  altstack.ss_flags = SS_DISABLE;
  altstack.ss_size = 0;
  CHECK_EQ(0, sigaltstack(&altstack, &oldstack));
  UnmapOrDie(oldstack.ss_sp, oldstack.ss_size);
}

typedef void (*sa_sigaction_t)(int, siginfo_t *, void *);
static void MaybeInstallSigaction(int signum,
                                  SignalHandlerType handler) {
  if (!IsDeadlySignal(signum))
    return;
  struct sigaction sigact;
  internal_memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = (sa_sigaction_t)handler;
  sigact.sa_flags = SA_SIGINFO;
  if (common_flags()->use_sigaltstack) sigact.sa_flags |= SA_ONSTACK;
  CHECK_EQ(0, internal_sigaction(signum, &sigact, 0));
  VReport(1, "Installed the sigaction for signal %d\n", signum);
}

void InstallDeadlySignalHandlers(SignalHandlerType handler) {
  // Set the alternate signal stack for the main thread.
  // This will cause SetAlternateSignalStack to be called twice, but the stack
  // will be actually set only once.
  if (common_flags()->use_sigaltstack) SetAlternateSignalStack();
  MaybeInstallSigaction(SIGSEGV, handler);
  MaybeInstallSigaction(SIGBUS, handler);
}
#endif  // SANITIZER_GO

}  // namespace __sanitizer

#endif
//===-- sanitizer_procmaps_linux.cc ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (Linux-specific parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_LINUX
#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

namespace __sanitizer {

// Linker initialized.
ProcSelfMapsBuff MemoryMappingLayout::cached_proc_self_maps_;
StaticSpinMutex MemoryMappingLayout::cache_lock_;  // Linker initialized.

MemoryMappingLayout::MemoryMappingLayout(bool cache_enabled) {
  proc_self_maps_.len =
      ReadFileToBuffer("/proc/self/maps", &proc_self_maps_.data,
                       &proc_self_maps_.mmaped_size, 1 << 26);
  if (cache_enabled) {
    if (proc_self_maps_.mmaped_size == 0) {
      LoadFromCache();
      CHECK_GT(proc_self_maps_.len, 0);
    }
  } else {
    CHECK_GT(proc_self_maps_.mmaped_size, 0);
  }
  Reset();
  // FIXME: in the future we may want to cache the mappings on demand only.
  if (cache_enabled)
    CacheMemoryMappings();
}

MemoryMappingLayout::~MemoryMappingLayout() {
  // Only unmap the buffer if it is different from the cached one. Otherwise
  // it will be unmapped when the cache is refreshed.
  if (proc_self_maps_.data != cached_proc_self_maps_.data) {
    UnmapOrDie(proc_self_maps_.data, proc_self_maps_.mmaped_size);
  }
}

void MemoryMappingLayout::Reset() {
  current_ = proc_self_maps_.data;
}

// static
void MemoryMappingLayout::CacheMemoryMappings() {
  SpinMutexLock l(&cache_lock_);
  // Don't invalidate the cache if the mappings are unavailable.
  ProcSelfMapsBuff old_proc_self_maps;
  old_proc_self_maps = cached_proc_self_maps_;
  cached_proc_self_maps_.len =
      ReadFileToBuffer("/proc/self/maps", &cached_proc_self_maps_.data,
                       &cached_proc_self_maps_.mmaped_size, 1 << 26);
  if (cached_proc_self_maps_.mmaped_size == 0) {
    cached_proc_self_maps_ = old_proc_self_maps;
  } else {
    if (old_proc_self_maps.mmaped_size) {
      UnmapOrDie(old_proc_self_maps.data,
                 old_proc_self_maps.mmaped_size);
    }
  }
}

void MemoryMappingLayout::LoadFromCache() {
  SpinMutexLock l(&cache_lock_);
  if (cached_proc_self_maps_.data) {
    proc_self_maps_ = cached_proc_self_maps_;
  }
}

// Parse a hex value in str and update str.
static uptr ParseHex(char **str) {
  uptr x = 0;
  char *s;
  for (s = *str; ; s++) {
    char c = *s;
    uptr v = 0;
    if (c >= '0' && c <= '9')
      v = c - '0';
    else if (c >= 'a' && c <= 'f')
      v = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      v = c - 'A' + 10;
    else
      break;
    x = x * 16 + v;
  }
  *str = s;
  return x;
}

static bool IsOneOf(char c, char c1, char c2) {
  return c == c1 || c == c2;
}

static bool IsDecimal(char c) {
  return c >= '0' && c <= '9';
}

static bool IsHex(char c) {
  return (c >= '0' && c <= '9')
      || (c >= 'a' && c <= 'f');
}

static uptr ReadHex(const char *p) {
  uptr v = 0;
  for (; IsHex(p[0]); p++) {
    if (p[0] >= '0' && p[0] <= '9')
      v = v * 16 + p[0] - '0';
    else
      v = v * 16 + p[0] - 'a' + 10;
  }
  return v;
}

static uptr ReadDecimal(const char *p) {
  uptr v = 0;
  for (; IsDecimal(p[0]); p++)
    v = v * 10 + p[0] - '0';
  return v;
}

bool MemoryMappingLayout::Next(uptr *start, uptr *end, uptr *offset,
                               char filename[], uptr filename_size,
                               uptr *protection) {
  char *last = proc_self_maps_.data + proc_self_maps_.len;
  if (current_ >= last) return false;
  uptr dummy;
  if (!start) start = &dummy;
  if (!end) end = &dummy;
  if (!offset) offset = &dummy;
  char *next_line = (char*)internal_memchr(current_, '\n', last - current_);
  if (next_line == 0)
    next_line = last;
  // Example: 08048000-08056000 r-xp 00000000 03:0c 64593   /foo/bar
  *start = ParseHex(&current_);
  CHECK_EQ(*current_++, '-');
  *end = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  uptr local_protection = 0;
  CHECK(IsOneOf(*current_, '-', 'r'));
  if (*current_++ == 'r')
    local_protection |= kProtectionRead;
  CHECK(IsOneOf(*current_, '-', 'w'));
  if (*current_++ == 'w')
    local_protection |= kProtectionWrite;
  CHECK(IsOneOf(*current_, '-', 'x'));
  if (*current_++ == 'x')
    local_protection |= kProtectionExecute;
  CHECK(IsOneOf(*current_, 's', 'p'));
  if (*current_++ == 's')
    local_protection |= kProtectionShared;
  if (protection) {
    *protection = local_protection;
  }
  CHECK_EQ(*current_++, ' ');
  *offset = ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ':');
  ParseHex(&current_);
  CHECK_EQ(*current_++, ' ');
  while (IsDecimal(*current_))
    current_++;
  // Qemu may lack the trailing space.
  // http://code.google.com/p/address-sanitizer/issues/detail?id=160
  // CHECK_EQ(*current_++, ' ');
  // Skip spaces.
  while (current_ < next_line && *current_ == ' ')
    current_++;
  // Fill in the filename.
  uptr i = 0;
  while (current_ < next_line) {
    if (filename && i < filename_size - 1)
      filename[i++] = *current_;
    current_++;
  }
  if (filename && i < filename_size)
    filename[i] = 0;
  current_ = next_line + 1;
  return true;
}

uptr MemoryMappingLayout::DumpListOfModules(LoadedModule *modules,
                                            uptr max_modules,
                                            string_predicate_t filter) {
  Reset();
  uptr cur_beg, cur_end, cur_offset;
  InternalScopedBuffer<char> module_name(kMaxPathLength);
  uptr n_modules = 0;
  for (uptr i = 0; n_modules < max_modules &&
                       Next(&cur_beg, &cur_end, &cur_offset, module_name.data(),
                            module_name.size(), 0);
       i++) {
    const char *cur_name = module_name.data();
    if (cur_name[0] == '\0')
      continue;
    if (filter && !filter(cur_name))
      continue;
    void *mem = &modules[n_modules];
    // Don't subtract 'cur_beg' from the first entry:
    // * If a binary is compiled w/o -pie, then the first entry in
    //   process maps is likely the binary itself (all dynamic libs
    //   are mapped higher in address space). For such a binary,
    //   instruction offset in binary coincides with the actual
    //   instruction address in virtual memory (as code section
    //   is mapped to a fixed memory range).
    // * If a binary is compiled with -pie, all the modules are
    //   mapped high at address space (in particular, higher than
    //   shadow memory of the tool), so the module can't be the
    //   first entry.
    uptr base_address = (i ? cur_beg : 0) - cur_offset;
    LoadedModule *cur_module = new(mem) LoadedModule(cur_name, base_address);
    cur_module->addAddressRange(cur_beg, cur_end);
    n_modules++;
  }
  return n_modules;
}

void GetMemoryProfile(fill_profile_f cb, uptr *stats, uptr stats_size) {
  char *smaps = 0;
  uptr smaps_cap = 0;
  uptr smaps_len = ReadFileToBuffer("/proc/self/smaps",
      &smaps, &smaps_cap, 64<<20);
  uptr start = 0;
  bool file = false;
  const char *pos = smaps;
  while (pos < smaps + smaps_len) {
    if (IsHex(pos[0])) {
      start = ReadHex(pos);
      for (; *pos != '/' && *pos > '\n'; pos++) {}
      file = *pos == '/';
    } else if (internal_strncmp(pos, "Rss:", 4) == 0) {
      for (; *pos < '0' || *pos > '9'; pos++) {}
      uptr rss = ReadDecimal(pos) * 1024;
      cb(start, rss, file, stats, stats_size);
    }
    while (*pos++ != '\n') {}
  }
  UnmapOrDie(smaps, smaps_cap);
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
//===-- sanitizer_linux.cc ------------------------------------------------===//
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
#if SANITIZER_LINUX

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

#include <asm/param.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#if !SANITIZER_ANDROID
#include <link.h>
#endif
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <unwind.h>

#if !SANITIZER_ANDROID
#include <sys/signal.h>
#endif

#if SANITIZER_ANDROID
#include <android/log.h>
#include <sys/system_properties.h>
#endif

// <linux/time.h>
struct kernel_timeval {
  long tv_sec;
  long tv_usec;
};

// <linux/futex.h> is broken on some linux distributions.
const int FUTEX_WAIT = 0;
const int FUTEX_WAKE = 1;

// Are we using 32-bit or 64-bit syscalls?
// x32 (which defines __x86_64__) has SANITIZER_WORDSIZE == 32
// but it still needs to use 64-bit syscalls.
#if defined(__x86_64__) || SANITIZER_WORDSIZE == 64
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 1
#else
# define SANITIZER_LINUX_USES_64BIT_SYSCALLS 0
#endif

namespace __sanitizer {

#ifdef __x86_64__
#include "sanitizer_syscall_linux_x86_64.inc"
#else
#include "sanitizer_syscall_generic.inc"
#endif

// --------------- sanitizer_libc.h
uptr internal_mmap(void *addr, uptr length, int prot, int flags,
                    int fd, u64 offset) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_mmap, (uptr)addr, length, prot, flags, fd,
                          offset);
#else
  return internal_syscall(__NR_mmap2, addr, length, prot, flags, fd, offset);
#endif
}

uptr internal_munmap(void *addr, uptr length) {
  return internal_syscall(__NR_munmap, (uptr)addr, length);
}

uptr internal_close(fd_t fd) {
  return internal_syscall(__NR_close, fd);
}

uptr internal_open(const char *filename, int flags) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_openat, AT_FDCWD, (uptr)filename, flags);
#else
  return internal_syscall(__NR_open, (uptr)filename, flags);
#endif
}

uptr internal_open(const char *filename, int flags, u32 mode) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_openat, AT_FDCWD, (uptr)filename, flags, mode);
#else
  return internal_syscall(__NR_open, (uptr)filename, flags, mode);
#endif
}

uptr OpenFile(const char *filename, bool write) {
  return internal_open(filename,
      write ? O_WRONLY | O_CREAT /*| O_CLOEXEC*/ : O_RDONLY, 0660);
}

uptr internal_read(fd_t fd, void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)internal_syscall(__NR_read, fd, (uptr)buf, count));
  return res;
}

uptr internal_write(fd_t fd, const void *buf, uptr count) {
  sptr res;
  HANDLE_EINTR(res, (sptr)internal_syscall(__NR_write, fd, (uptr)buf, count));
  return res;
}

#if !SANITIZER_LINUX_USES_64BIT_SYSCALLS
static void stat64_to_stat(struct stat64 *in, struct stat *out) {
  internal_memset(out, 0, sizeof(*out));
  out->st_dev = in->st_dev;
  out->st_ino = in->st_ino;
  out->st_mode = in->st_mode;
  out->st_nlink = in->st_nlink;
  out->st_uid = in->st_uid;
  out->st_gid = in->st_gid;
  out->st_rdev = in->st_rdev;
  out->st_size = in->st_size;
  out->st_blksize = in->st_blksize;
  out->st_blocks = in->st_blocks;
  out->st_atime = in->st_atime;
  out->st_mtime = in->st_mtime;
  out->st_ctime = in->st_ctime;
  out->st_ino = in->st_ino;
}
#endif

uptr internal_stat(const char *path, void *buf) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_newfstatat, AT_FDCWD, (uptr)path, (uptr)buf, 0);
#elif SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_stat, (uptr)path, (uptr)buf);
#else
  struct stat64 buf64;
  int res = internal_syscall(__NR_stat64, path, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_lstat(const char *path, void *buf) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_newfstatat, AT_FDCWD, (uptr)path,
                         (uptr)buf, AT_SYMLINK_NOFOLLOW);
#elif SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_lstat, (uptr)path, (uptr)buf);
#else
  struct stat64 buf64;
  int res = internal_syscall(__NR_lstat64, path, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_fstat(fd_t fd, void *buf) {
#if SANITIZER_LINUX_USES_64BIT_SYSCALLS
  return internal_syscall(__NR_fstat, fd, (uptr)buf);
#else
  struct stat64 buf64;
  int res = internal_syscall(__NR_fstat64, fd, &buf64);
  stat64_to_stat(&buf64, (struct stat *)buf);
  return res;
#endif
}

uptr internal_filesize(fd_t fd) {
  struct stat st;
  if (internal_fstat(fd, &st))
    return -1;
  return (uptr)st.st_size;
}

uptr internal_dup2(int oldfd, int newfd) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_dup3, oldfd, newfd, 0);
#else
  return internal_syscall(__NR_dup2, oldfd, newfd);
#endif
}

uptr internal_readlink(const char *path, char *buf, uptr bufsize) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_readlinkat, AT_FDCWD,
                               (uptr)path, (uptr)buf, bufsize);
#else
  return internal_syscall(__NR_readlink, (uptr)path, (uptr)buf, bufsize);
#endif
}

uptr internal_unlink(const char *path) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_unlinkat, AT_FDCWD, (uptr)path, 0);
#else
  return internal_syscall(__NR_unlink, (uptr)path);
#endif
}

uptr internal_sched_yield() {
  return internal_syscall(__NR_sched_yield);
}

void internal__exit(int exitcode) {
  internal_syscall(__NR_exit_group, exitcode);
  Die();  // Unreachable.
}

uptr internal_execve(const char *filename, char *const argv[],
                     char *const envp[]) {
  return internal_syscall(__NR_execve, (uptr)filename, (uptr)argv, (uptr)envp);
}

// ----------------- sanitizer_common.h
bool FileExists(const char *filename) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  struct stat st;
  if (internal_syscall(__NR_newfstatat, AT_FDCWD, filename, &st, 0))
    return false;
#else
  struct stat st;
  if (internal_stat(filename, &st))
    return false;
  // Sanity check: filename is a regular file.
  return S_ISREG(st.st_mode);
#endif
}

uptr GetTid() {
  return internal_syscall(__NR_gettid);
}

u64 NanoTime() {
  kernel_timeval tv;
  internal_memset(&tv, 0, sizeof(tv));
  internal_syscall(__NR_gettimeofday, (uptr)&tv, 0);
  return (u64)tv.tv_sec * 1000*1000*1000 + tv.tv_usec * 1000;
}

// Like getenv, but reads env directly from /proc and does not use libc.
// This function should be called first inside __asan_init.
const char *GetEnv(const char *name) {
  static char *environ;
  static uptr len;
  static bool inited;
  if (!inited) {
    inited = true;
    uptr environ_size;
    len = ReadFileToBuffer("/proc/self/environ",
                           &environ, &environ_size, 1 << 26);
  }
  if (!environ || len == 0) return 0;
  uptr namelen = internal_strlen(name);
  const char *p = environ;
  while (*p != '\0') {  // will happen at the \0\0 that terminates the buffer
    // proc file has the format NAME=value\0NAME=value\0NAME=value\0...
    const char* endp =
        (char*)internal_memchr(p, '\0', len - (p - environ));
    if (endp == 0)  // this entry isn't NUL terminated
      return 0;
    else if (!internal_memcmp(p, name, namelen) && p[namelen] == '=')  // Match.
      return p + namelen + 1;  // point after =
    p = endp + 1;
  }
  return 0;  // Not found.
}

extern "C" {
  SANITIZER_WEAK_ATTRIBUTE extern void *__libc_stack_end;
}

#if !SANITIZER_GO
static void ReadNullSepFileToArray(const char *path, char ***arr,
                                   int arr_size) {
  char *buff;
  uptr buff_size = 0;
  *arr = (char **)MmapOrDie(arr_size * sizeof(char *), "NullSepFileArray");
  ReadFileToBuffer(path, &buff, &buff_size, 1024 * 1024);
  (*arr)[0] = buff;
  int count, i;
  for (count = 1, i = 1; ; i++) {
    if (buff[i] == 0) {
      if (buff[i+1] == 0) break;
      (*arr)[count] = &buff[i+1];
      CHECK_LE(count, arr_size - 1);  // FIXME: make this more flexible.
      count++;
    }
  }
  (*arr)[count] = 0;
}
#endif

static void GetArgsAndEnv(char*** argv, char*** envp) {
#if !SANITIZER_GO
  if (&__libc_stack_end) {
#endif
    uptr* stack_end = (uptr*)__libc_stack_end;
    int argc = *stack_end;
    *argv = (char**)(stack_end + 1);
    *envp = (char**)(stack_end + argc + 2);
#if !SANITIZER_GO
  } else {
    static const int kMaxArgv = 2000, kMaxEnvp = 2000;
    ReadNullSepFileToArray("/proc/self/cmdline", argv, kMaxArgv);
    ReadNullSepFileToArray("/proc/self/environ", envp, kMaxEnvp);
  }
#endif
}

void ReExec() {
  char **argv, **envp;
  GetArgsAndEnv(&argv, &envp);
  uptr rv = internal_execve("/proc/self/exe", argv, envp);
  int rverrno;
  CHECK_EQ(internal_iserror(rv, &rverrno), true);
  Printf("execve failed, errno %d\n", rverrno);
  Die();
}

void PrepareForSandboxing() {
  // Some kinds of sandboxes may forbid filesystem access, so we won't be able
  // to read the file mappings from /proc/self/maps. Luckily, neither the
  // process will be able to load additional libraries, so it's fine to use the
  // cached mappings.
  MemoryMappingLayout::CacheMemoryMappings();
  // Same for /proc/self/exe in the symbolizer.
#if !SANITIZER_GO
  if (Symbolizer *sym = Symbolizer::GetOrNull())
    sym->PrepareForSandboxing();
#endif
}

enum MutexState {
  MtxUnlocked = 0,
  MtxLocked = 1,
  MtxSleeping = 2
};

BlockingMutex::BlockingMutex(LinkerInitialized) {
  CHECK_EQ(owner_, 0);
}

BlockingMutex::BlockingMutex() {
  internal_memset(this, 0, sizeof(*this));
}

void BlockingMutex::Lock() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  if (atomic_exchange(m, MtxLocked, memory_order_acquire) == MtxUnlocked)
    return;
  while (atomic_exchange(m, MtxSleeping, memory_order_acquire) != MtxUnlocked)
    internal_syscall(__NR_futex, (uptr)m, FUTEX_WAIT, MtxSleeping, 0, 0, 0);
}

void BlockingMutex::Unlock() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  u32 v = atomic_exchange(m, MtxUnlocked, memory_order_relaxed);
  CHECK_NE(v, MtxUnlocked);
  if (v == MtxSleeping)
    internal_syscall(__NR_futex, (uptr)m, FUTEX_WAKE, 1, 0, 0, 0);
}

void BlockingMutex::CheckLocked() {
  atomic_uint32_t *m = reinterpret_cast<atomic_uint32_t *>(&opaque_storage_);
  CHECK_NE(MtxUnlocked, atomic_load(m, memory_order_relaxed));
}

// ----------------- sanitizer_linux.h
// The actual size of this structure is specified by d_reclen.
// Note that getdents64 uses a different structure format. We only provide the
// 32-bit syscall here.
struct linux_dirent {
  unsigned long      d_ino;
  unsigned long      d_off;
  unsigned short     d_reclen;
  char               d_name[256];
};

// Syscall wrappers.
uptr internal_ptrace(int request, int pid, void *addr, void *data) {
  return internal_syscall(__NR_ptrace, request, pid, (uptr)addr, (uptr)data);
}

uptr internal_waitpid(int pid, int *status, int options) {
  return internal_syscall(__NR_wait4, pid, (uptr)status, options,
                          0 /* rusage */);
}

uptr internal_getpid() {
  return internal_syscall(__NR_getpid);
}

uptr internal_getppid() {
  return internal_syscall(__NR_getppid);
}

uptr internal_getdents(fd_t fd, struct linux_dirent *dirp, unsigned int count) {
#if SANITIZER_USES_CANONICAL_LINUX_SYSCALLS
  return internal_syscall(__NR_getdents64, fd, (uptr)dirp, count);
#else
  return internal_syscall(__NR_getdents, fd, (uptr)dirp, count);
#endif
}

uptr internal_lseek(fd_t fd, OFF_T offset, int whence) {
  return internal_syscall(__NR_lseek, fd, offset, whence);
}

uptr internal_prctl(int option, uptr arg2, uptr arg3, uptr arg4, uptr arg5) {
  return internal_syscall(__NR_prctl, option, arg2, arg3, arg4, arg5);
}

uptr internal_sigaltstack(const struct sigaltstack *ss,
                         struct sigaltstack *oss) {
  return internal_syscall(__NR_sigaltstack, (uptr)ss, (uptr)oss);
}

// Doesn't set sa_restorer, use with caution (see below).
int internal_sigaction_norestorer(int signum, const void *act, void *oldact) {
  __sanitizer_kernel_sigaction_t k_act, k_oldact;
  internal_memset(&k_act, 0, sizeof(__sanitizer_kernel_sigaction_t));
  internal_memset(&k_oldact, 0, sizeof(__sanitizer_kernel_sigaction_t));
  const __sanitizer_sigaction *u_act = (__sanitizer_sigaction *)act;
  __sanitizer_sigaction *u_oldact = (__sanitizer_sigaction *)oldact;
  if (u_act) {
    k_act.handler = u_act->handler;
    k_act.sigaction = u_act->sigaction;
    internal_memcpy(&k_act.sa_mask, &u_act->sa_mask,
                    sizeof(__sanitizer_kernel_sigset_t));
    k_act.sa_flags = u_act->sa_flags;
    // FIXME: most often sa_restorer is unset, however the kernel requires it
    // to point to a valid signal restorer that calls the rt_sigreturn syscall.
    // If sa_restorer passed to the kernel is NULL, the program may crash upon
    // signal delivery or fail to unwind the stack in the signal handler.
    // libc implementation of sigaction() passes its own restorer to
    // rt_sigaction, so we need to do the same (we'll need to reimplement the
    // restorers; for x86_64 the restorer address can be obtained from
    // oldact->sa_restorer upon a call to sigaction(xxx, NULL, oldact).
    k_act.sa_restorer = u_act->sa_restorer;
  }

  uptr result = internal_syscall(__NR_rt_sigaction, (uptr)signum,
      (uptr)(u_act ? &k_act : NULL),
      (uptr)(u_oldact ? &k_oldact : NULL),
      (uptr)sizeof(__sanitizer_kernel_sigset_t));

  if ((result == 0) && u_oldact) {
    u_oldact->handler = k_oldact.handler;
    u_oldact->sigaction = k_oldact.sigaction;
    internal_memcpy(&u_oldact->sa_mask, &k_oldact.sa_mask,
                    sizeof(__sanitizer_kernel_sigset_t));
    u_oldact->sa_flags = k_oldact.sa_flags;
    u_oldact->sa_restorer = k_oldact.sa_restorer;
  }
  return result;
}

uptr internal_sigprocmask(int how, __sanitizer_sigset_t *set,
    __sanitizer_sigset_t *oldset) {
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  __sanitizer_kernel_sigset_t *k_oldset = (__sanitizer_kernel_sigset_t *)oldset;
  return internal_syscall(__NR_rt_sigprocmask, (uptr)how, &k_set->sig[0],
      &k_oldset->sig[0], sizeof(__sanitizer_kernel_sigset_t));
}

void internal_sigfillset(__sanitizer_sigset_t *set) {
  internal_memset(set, 0xff, sizeof(*set));
}

void internal_sigdelset(__sanitizer_sigset_t *set, int signum) {
  signum -= 1;
  CHECK_GE(signum, 0);
  CHECK_LT(signum, sizeof(*set) * 8);
  __sanitizer_kernel_sigset_t *k_set = (__sanitizer_kernel_sigset_t *)set;
  const uptr idx = signum / (sizeof(k_set->sig[0]) * 8);
  const uptr bit = signum % (sizeof(k_set->sig[0]) * 8);
  k_set->sig[idx] &= ~(1 << bit);
}

// ThreadLister implementation.
ThreadLister::ThreadLister(int pid)
  : pid_(pid),
    descriptor_(-1),
    buffer_(4096),
    error_(true),
    entry_((struct linux_dirent *)buffer_.data()),
    bytes_read_(0) {
  char task_directory_path[80];
  internal_snprintf(task_directory_path, sizeof(task_directory_path),
                    "/proc/%d/task/", pid);
  uptr openrv = internal_open(task_directory_path, O_RDONLY | O_DIRECTORY);
  if (internal_iserror(openrv)) {
    error_ = true;
    Report("Can't open /proc/%d/task for reading.\n", pid);
  } else {
    error_ = false;
    descriptor_ = openrv;
  }
}

int ThreadLister::GetNextTID() {
  int tid = -1;
  do {
    if (error_)
      return -1;
    if ((char *)entry_ >= &buffer_[bytes_read_] && !GetDirectoryEntries())
      return -1;
    if (entry_->d_ino != 0 && entry_->d_name[0] >= '0' &&
        entry_->d_name[0] <= '9') {
      // Found a valid tid.
      tid = (int)internal_atoll(entry_->d_name);
    }
    entry_ = (struct linux_dirent *)(((char *)entry_) + entry_->d_reclen);
  } while (tid < 0);
  return tid;
}

void ThreadLister::Reset() {
  if (error_ || descriptor_ < 0)
    return;
  internal_lseek(descriptor_, 0, SEEK_SET);
}

ThreadLister::~ThreadLister() {
  if (descriptor_ >= 0)
    internal_close(descriptor_);
}

bool ThreadLister::error() { return error_; }

bool ThreadLister::GetDirectoryEntries() {
  CHECK_GE(descriptor_, 0);
  CHECK_NE(error_, true);
  bytes_read_ = internal_getdents(descriptor_,
                                  (struct linux_dirent *)buffer_.data(),
                                  buffer_.size());
  if (internal_iserror(bytes_read_)) {
    Report("Can't read directory entries from /proc/%d/task.\n", pid_);
    error_ = true;
    return false;
  } else if (bytes_read_ == 0) {
    return false;
  }
  entry_ = (struct linux_dirent *)buffer_.data();
  return true;
}

uptr GetPageSize() {
#if defined(__x86_64__) || defined(__i386__)
  return EXEC_PAGESIZE;
#else
  return sysconf(_SC_PAGESIZE);  // EXEC_PAGESIZE may not be trustworthy.
#endif
}

static char proc_self_exe_cache_str[kMaxPathLength];
static uptr proc_self_exe_cache_len = 0;

uptr ReadBinaryName(/*out*/char *buf, uptr buf_len) {
  uptr module_name_len = internal_readlink(
      "/proc/self/exe", buf, buf_len);
  int readlink_error;
  if (internal_iserror(module_name_len, &readlink_error)) {
    if (proc_self_exe_cache_len) {
      // If available, use the cached module name.
      CHECK_LE(proc_self_exe_cache_len, buf_len);
      internal_strncpy(buf, proc_self_exe_cache_str, buf_len);
      module_name_len = internal_strlen(proc_self_exe_cache_str);
    } else {
      // We can't read /proc/self/exe for some reason, assume the name of the
      // binary is unknown.
      Report("WARNING: readlink(\"/proc/self/exe\") failed with errno %d, "
             "some stack frames may not be symbolized\n", readlink_error);
      module_name_len = internal_snprintf(buf, buf_len, "/proc/self/exe");
    }
    CHECK_LT(module_name_len, buf_len);
    buf[module_name_len] = '\0';
  }
  return module_name_len;
}

void CacheBinaryName() {
  if (!proc_self_exe_cache_len) {
    proc_self_exe_cache_len =
        ReadBinaryName(proc_self_exe_cache_str, kMaxPathLength);
  }
}

// Match full names of the form /path/to/base_name{-,.}*
bool LibraryNameIs(const char *full_name, const char *base_name) {
  const char *name = full_name;
  // Strip path.
  while (*name != '\0') name++;
  while (name > full_name && *name != '/') name--;
  if (*name == '/') name++;
  uptr base_name_length = internal_strlen(base_name);
  if (internal_strncmp(name, base_name, base_name_length)) return false;
  return (name[base_name_length] == '-' || name[base_name_length] == '.');
}

#if !SANITIZER_ANDROID
// Call cb for each region mapped by map.
void ForEachMappedRegion(link_map *map, void (*cb)(const void *, uptr)) {
  typedef ElfW(Phdr) Elf_Phdr;
  typedef ElfW(Ehdr) Elf_Ehdr;
  char *base = (char *)map->l_addr;
  Elf_Ehdr *ehdr = (Elf_Ehdr *)base;
  char *phdrs = base + ehdr->e_phoff;
  char *phdrs_end = phdrs + ehdr->e_phnum * ehdr->e_phentsize;

  // Find the segment with the minimum base so we can "relocate" the p_vaddr
  // fields.  Typically ET_DYN objects (DSOs) have base of zero and ET_EXEC
  // objects have a non-zero base.
  uptr preferred_base = (uptr)-1;
  for (char *iter = phdrs; iter != phdrs_end; iter += ehdr->e_phentsize) {
    Elf_Phdr *phdr = (Elf_Phdr *)iter;
    if (phdr->p_type == PT_LOAD && preferred_base > (uptr)phdr->p_vaddr)
      preferred_base = (uptr)phdr->p_vaddr;
  }

  // Compute the delta from the real base to get a relocation delta.
  sptr delta = (uptr)base - preferred_base;
  // Now we can figure out what the loader really mapped.
  for (char *iter = phdrs; iter != phdrs_end; iter += ehdr->e_phentsize) {
    Elf_Phdr *phdr = (Elf_Phdr *)iter;
    if (phdr->p_type == PT_LOAD) {
      uptr seg_start = phdr->p_vaddr + delta;
      uptr seg_end = seg_start + phdr->p_memsz;
      // None of these values are aligned.  We consider the ragged edges of the
      // load command as defined, since they are mapped from the file.
      seg_start = RoundDownTo(seg_start, GetPageSizeCached());
      seg_end = RoundUpTo(seg_end, GetPageSizeCached());
      cb((void *)seg_start, seg_end - seg_start);
    }
  }
}
#endif

#if defined(__x86_64__)
// We cannot use glibc's clone wrapper, because it messes with the child
// task's TLS. It writes the PID and TID of the child task to its thread
// descriptor, but in our case the child task shares the thread descriptor with
// the parent (because we don't know how to allocate a new thread
// descriptor to keep glibc happy). So the stock version of clone(), when
// used with CLONE_VM, would end up corrupting the parent's thread descriptor.
uptr internal_clone(int (*fn)(void *), void *child_stack, int flags, void *arg,
                    int *parent_tidptr, void *newtls, int *child_tidptr) {
  long long res;
  if (!fn || !child_stack)
    return -EINVAL;
  CHECK_EQ(0, (uptr)child_stack % 16);
  child_stack = (char *)child_stack - 2 * sizeof(unsigned long long);
  ((unsigned long long *)child_stack)[0] = (uptr)fn;
  ((unsigned long long *)child_stack)[1] = (uptr)arg;
  register void *r8 __asm__("r8") = newtls;
  register int *r10 __asm__("r10") = child_tidptr;
  __asm__ __volatile__(
                       /* %rax = syscall(%rax = __NR_clone,
                        *                %rdi = flags,
                        *                %rsi = child_stack,
                        *                %rdx = parent_tidptr,
                        *                %r8  = new_tls,
                        *                %r10 = child_tidptr)
                        */
                       "syscall\n"

                       /* if (%rax != 0)
                        *   return;
                        */
                       "testq  %%rax,%%rax\n"
                       "jnz    1f\n"

                       /* In the child. Terminate unwind chain. */
                       // XXX: We should also terminate the CFI unwind chain
                       // here. Unfortunately clang 3.2 doesn't support the
                       // necessary CFI directives, so we skip that part.
                       "xorq   %%rbp,%%rbp\n"

                       /* Call "fn(arg)". */
                       "popq   %%rax\n"
                       "popq   %%rdi\n"
                       "call   *%%rax\n"

                       /* Call _exit(%rax). */
                       "movq   %%rax,%%rdi\n"
                       "movq   %2,%%rax\n"
                       "syscall\n"

                       /* Return to parent. */
                     "1:\n"
                       : "=a" (res)
                       : "a"(__NR_clone), "i"(__NR_exit),
                         "S"(child_stack),
                         "D"(flags),
                         "d"(parent_tidptr),
                         "r"(r8),
                         "r"(r10)
                       : "rsp", "memory", "r11", "rcx");
  return res;
}
#endif  // defined(__x86_64__)

#if SANITIZER_ANDROID
// This thing is not, strictly speaking, async signal safe, but it does not seem
// to cause any issues. Alternative is writing to log devices directly, but
// their location and message format might change in the future, so we'd really
// like to avoid that.
void AndroidLogWrite(const char *buffer) {
  char *copy = internal_strdup(buffer);
  char *p = copy;
  char *q;
  // __android_log_write has an implicit message length limit.
  // Print one line at a time.
  do {
    q = internal_strchr(p, '\n');
    if (q) *q = '\0';
    __android_log_write(ANDROID_LOG_INFO, NULL, p);
    if (q) p = q + 1;
  } while (q);
  InternalFree(copy);
}

void GetExtraActivationFlags(char *buf, uptr size) {
  CHECK(size > PROP_VALUE_MAX);
  __system_property_get("asan.options", buf);
}
#endif

bool IsDeadlySignal(int signum) {
  return (signum == SIGSEGV) && common_flags()->handle_segv;
}

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
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
#if SANITIZER_LINUX

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_atomic.h"

#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unwind.h>

#if !SANITIZER_ANDROID
#include <elf.h>
#include <link.h>
#include <unistd.h>
#endif

namespace __sanitizer {

#ifndef SANITIZER_GO
// This function is defined elsewhere if we intercepted pthread_attr_getstack.
SANITIZER_WEAK_ATTRIBUTE int
real_pthread_attr_getstack(void *attr, void **addr, size_t *size);

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
  CHECK_EQ(pthread_getattr_np(pthread_self(), &attr), 0);
  uptr stacksize = 0;
  void *stackaddr = 0;
  my_pthread_attr_getstack(&attr, &stackaddr, (size_t*)&stacksize);
  pthread_attr_destroy(&attr);

  CHECK_LE(stacksize, kMaxThreadStackSize);  // Sanity check.
  *stack_top = (uptr)stackaddr + stacksize;
  *stack_bottom = (uptr)stackaddr;
}
#endif  // #ifndef SANITIZER_GO

// Does not compile for Go because dlsym() requires -ldl
#ifndef SANITIZER_GO
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
#endif

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

#ifndef SANITIZER_GO
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
  size = 0;
  if (max_depth == 0)
    return;
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
  if (!unwind_backtrace_signal_arch) {
    SlowUnwindStack(pc, max_depth);
    return;
  }

  size = 0;
  if (max_depth == 0) return;

  void *map = acquire_my_map_info_list();
  CHECK(map);
  InternalScopedBuffer<backtrace_frame_t> frames(kStackTraceMax);
  // siginfo argument appears to be unused.
  sptr res = unwind_backtrace_signal_arch(/* siginfo */ NULL, context, map,
                                          frames.data(),
                                          /* ignore_depth */ 0, max_depth);
  release_my_map_info_list(map);
  if (res < 0) return;
  CHECK((uptr)res <= kStackTraceMax);

  // +2 compensate for libcorkscrew unwinder returning addresses of call
  // instructions instead of raw return addresses.
  for (sptr i = 0; i < res; ++i)
    trace[size++] = frames[i].absolute_pc + 2;
}

#endif  // !SANITIZER_GO

static uptr g_tls_size;

#ifdef __i386__
# define DL_INTERNAL_FUNCTION __attribute__((regparm(3), stdcall))
#else
# define DL_INTERNAL_FUNCTION
#endif

void InitTlsSize() {
#if !defined(SANITIZER_GO) && !SANITIZER_ANDROID
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
#endif
}

uptr GetTlsSize() {
  return g_tls_size;
}

#if defined(__x86_64__) || defined(__i386__)
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
      if (minor <= 3)
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
#ifdef __i386__
  asm("mov %%gs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
#else
  asm("mov %%fs:%c1,%0" : "=r"(descr_addr) : "i"(kThreadSelfOffset));
#endif
  return descr_addr;
}
#endif  // defined(__x86_64__) || defined(__i386__)

void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size) {
#ifndef SANITIZER_GO
#if defined(__x86_64__) || defined(__i386__)
  *tls_addr = ThreadSelf();
  *tls_size = GetTlsSize();
  *tls_addr -= *tls_size;
  *tls_addr += ThreadDescriptorSize();
#else
  *tls_addr = 0;
  *tls_size = 0;
#endif

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
#else  // SANITIZER_GO
  *stk_addr = 0;
  *stk_size = 0;
  *tls_addr = 0;
  *tls_size = 0;
#endif  // SANITIZER_GO
}

#ifndef SANITIZER_GO
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
      if (stacksize != 0)
        VPrintf(1, "Sanitizer: increasing stacksize %zu->%zu\n", stacksize,
                minstacksize);
      pthread_attr_setstacksize(attr, minstacksize);
    } else {
      Printf("Sanitizer: pre-allocated stack size is insufficient: "
             "%zu < %zu\n", stacksize, minstacksize);
      Printf("Sanitizer: pthread_create is likely to fail.\n");
    }
  }
}
#endif  // SANITIZER_GO

#if SANITIZER_ANDROID
uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter) {
  MemoryMappingLayout memory_mapping(false);
  return memory_mapping.DumpListOfModules(modules, max_modules, filter);
}
#else  // SANITIZER_ANDROID
typedef ElfW(Phdr) Elf_Phdr;

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

#ifndef SANITIZER_GO
uptr indirect_call_wrapper;

void SetIndirectCallWrapper(uptr wrapper) {
  CHECK(!indirect_call_wrapper);
  CHECK(wrapper);
  indirect_call_wrapper = wrapper;
}
#endif

}  // namespace __sanitizer

#endif  // SANITIZER_LINUX
//===-- sanitizer_stoptheworld_linux_libcdep.cc ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// See sanitizer_stoptheworld.h for details.
// This implementation was inspired by Markus Gutschke's linuxthreads.cc.
//
//===----------------------------------------------------------------------===//


#include "sanitizer_platform.h"
#if SANITIZER_LINUX && defined(__x86_64__)

#include "sanitizer_stoptheworld.h"

#include "sanitizer_platform_limits_posix.h"

#include <errno.h>
#include <sched.h> // for CLONE_* definitions
#include <stddef.h>
#include <sys/prctl.h> // for PR_* definitions
#include <sys/ptrace.h> // for PTRACE_* definitions
#include <sys/types.h> // for pid_t
#if SANITIZER_ANDROID && defined(__arm__)
# include <linux/user.h>  // for pt_regs
#else
# include <sys/user.h>  // for user_regs_struct
#endif
#include <sys/wait.h> // for signal-related stuff

#ifdef sa_handler
# undef sa_handler
#endif

#ifdef sa_sigaction
# undef sa_sigaction
#endif

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_libc.h"
#include "sanitizer_linux.h"
#include "sanitizer_mutex.h"
#include "sanitizer_placement_new.h"

// This module works by spawning a Linux task which then attaches to every
// thread in the caller process with ptrace. This suspends the threads, and
// PTRACE_GETREGS can then be used to obtain their register state. The callback
// supplied to StopTheWorld() is run in the tracer task while the threads are
// suspended.
// The tracer task must be placed in a different thread group for ptrace to
// work, so it cannot be spawned as a pthread. Instead, we use the low-level
// clone() interface (we want to share the address space with the caller
// process, so we prefer clone() over fork()).
//
// We don't use any libc functions, relying instead on direct syscalls. There
// are two reasons for this:
// 1. calling a library function while threads are suspended could cause a
// deadlock, if one of the treads happens to be holding a libc lock;
// 2. it's generally not safe to call libc functions from the tracer task,
// because clone() does not set up a thread-local storage for it. Any
// thread-local variables used by libc will be shared between the tracer task
// and the thread which spawned it.

COMPILER_CHECK(sizeof(SuspendedThreadID) == sizeof(pid_t));

namespace __sanitizer {
// This class handles thread suspending/unsuspending in the tracer thread.
class ThreadSuspender {
 public:
  explicit ThreadSuspender(pid_t pid)
    : pid_(pid) {
      CHECK_GE(pid, 0);
    }
  bool SuspendAllThreads();
  void ResumeAllThreads();
  void KillAllThreads();
  SuspendedThreadsList &suspended_threads_list() {
    return suspended_threads_list_;
  }
 private:
  SuspendedThreadsList suspended_threads_list_;
  pid_t pid_;
  bool SuspendThread(SuspendedThreadID thread_id);
};

bool ThreadSuspender::SuspendThread(SuspendedThreadID thread_id) {
  // Are we already attached to this thread?
  // Currently this check takes linear time, however the number of threads is
  // usually small.
  if (suspended_threads_list_.Contains(thread_id))
    return false;
  int pterrno;
  if (internal_iserror(internal_ptrace(PTRACE_ATTACH, thread_id, NULL, NULL),
                       &pterrno)) {
    // Either the thread is dead, or something prevented us from attaching.
    // Log this event and move on.
    VReport(1, "Could not attach to thread %d (errno %d).\n", thread_id,
            pterrno);
    return false;
  } else {
    VReport(1, "Attached to thread %d.\n", thread_id);
    // The thread is not guaranteed to stop before ptrace returns, so we must
    // wait on it.
    uptr waitpid_status;
    HANDLE_EINTR(waitpid_status, internal_waitpid(thread_id, NULL, __WALL));
    int wperrno;
    if (internal_iserror(waitpid_status, &wperrno)) {
      // Got a ECHILD error. I don't think this situation is possible, but it
      // doesn't hurt to report it.
      VReport(1, "Waiting on thread %d failed, detaching (errno %d).\n",
              thread_id, wperrno);
      internal_ptrace(PTRACE_DETACH, thread_id, NULL, NULL);
      return false;
    }
    suspended_threads_list_.Append(thread_id);
    return true;
  }
}

void ThreadSuspender::ResumeAllThreads() {
  for (uptr i = 0; i < suspended_threads_list_.thread_count(); i++) {
    pid_t tid = suspended_threads_list_.GetThreadID(i);
    int pterrno;
    if (!internal_iserror(internal_ptrace(PTRACE_DETACH, tid, NULL, NULL),
                          &pterrno)) {
      VReport(1, "Detached from thread %d.\n", tid);
    } else {
      // Either the thread is dead, or we are already detached.
      // The latter case is possible, for instance, if this function was called
      // from a signal handler.
      VReport(1, "Could not detach from thread %d (errno %d).\n", tid, pterrno);
    }
  }
}

void ThreadSuspender::KillAllThreads() {
  for (uptr i = 0; i < suspended_threads_list_.thread_count(); i++)
    internal_ptrace(PTRACE_KILL, suspended_threads_list_.GetThreadID(i),
                    NULL, NULL);
}

bool ThreadSuspender::SuspendAllThreads() {
  ThreadLister thread_lister(pid_);
  bool added_threads;
  do {
    // Run through the directory entries once.
    added_threads = false;
    pid_t tid = thread_lister.GetNextTID();
    while (tid >= 0) {
      if (SuspendThread(tid))
        added_threads = true;
      tid = thread_lister.GetNextTID();
    }
    if (thread_lister.error()) {
      // Detach threads and fail.
      ResumeAllThreads();
      return false;
    }
    thread_lister.Reset();
  } while (added_threads);
  return true;
}

// Pointer to the ThreadSuspender instance for use in signal handler.
static ThreadSuspender *thread_suspender_instance = NULL;

// Signals that should not be blocked (this is used in the parent thread as well
// as the tracer thread).
static const int kUnblockedSignals[] = { SIGABRT, SIGILL, SIGFPE, SIGSEGV,
                                         SIGBUS, SIGXCPU, SIGXFSZ };

// Structure for passing arguments into the tracer thread.
struct TracerThreadArgument {
  StopTheWorldCallback callback;
  void *callback_argument;
  // The tracer thread waits on this mutex while the parent finishes its
  // preparations.
  BlockingMutex mutex;
  uptr parent_pid;
};

static DieCallbackType old_die_callback;

// Signal handler to wake up suspended threads when the tracer thread dies.
void TracerThreadSignalHandler(int signum, void *siginfo, void *) {
  if (thread_suspender_instance != NULL) {
    if (signum == SIGABRT)
      thread_suspender_instance->KillAllThreads();
    else
      thread_suspender_instance->ResumeAllThreads();
  }
  internal__exit((signum == SIGABRT) ? 1 : 2);
}

static void TracerThreadDieCallback() {
  // Generally a call to Die() in the tracer thread should be fatal to the
  // parent process as well, because they share the address space.
  // This really only works correctly if all the threads are suspended at this
  // point. So we correctly handle calls to Die() from within the callback, but
  // not those that happen before or after the callback. Hopefully there aren't
  // a lot of opportunities for that to happen...
  if (thread_suspender_instance)
    thread_suspender_instance->KillAllThreads();
  if (old_die_callback)
    old_die_callback();
}

// Size of alternative stack for signal handlers in the tracer thread.
static const int kHandlerStackSize = 4096;

// This function will be run as a cloned task.
static int TracerThread(void* argument) {
  TracerThreadArgument *tracer_thread_argument =
      (TracerThreadArgument *)argument;

  internal_prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0);
  // Check if parent is already dead.
  if (internal_getppid() != tracer_thread_argument->parent_pid)
    internal__exit(4);

  // Wait for the parent thread to finish preparations.
  tracer_thread_argument->mutex.Lock();
  tracer_thread_argument->mutex.Unlock();

  SetDieCallback(TracerThreadDieCallback);

  ThreadSuspender thread_suspender(internal_getppid());
  // Global pointer for the signal handler.
  thread_suspender_instance = &thread_suspender;

  // Alternate stack for signal handling.
  InternalScopedBuffer<char> handler_stack_memory(kHandlerStackSize);
  struct sigaltstack handler_stack;
  internal_memset(&handler_stack, 0, sizeof(handler_stack));
  handler_stack.ss_sp = handler_stack_memory.data();
  handler_stack.ss_size = kHandlerStackSize;
  internal_sigaltstack(&handler_stack, NULL);

  // Install our handler for fatal signals. Other signals should be blocked by
  // the mask we inherited from the caller thread.
  for (uptr signal_index = 0; signal_index < ARRAY_SIZE(kUnblockedSignals);
       signal_index++) {
    __sanitizer_sigaction new_sigaction;
    internal_memset(&new_sigaction, 0, sizeof(new_sigaction));
    new_sigaction.sigaction = TracerThreadSignalHandler;
    new_sigaction.sa_flags = SA_ONSTACK | SA_SIGINFO;
    internal_sigfillset(&new_sigaction.sa_mask);
    internal_sigaction_norestorer(kUnblockedSignals[signal_index],
                                  &new_sigaction, NULL);
  }

  int exit_code = 0;
  if (!thread_suspender.SuspendAllThreads()) {
    VReport(1, "Failed suspending threads.\n");
    exit_code = 3;
  } else {
    tracer_thread_argument->callback(thread_suspender.suspended_threads_list(),
                                     tracer_thread_argument->callback_argument);
    thread_suspender.ResumeAllThreads();
    exit_code = 0;
  }
  thread_suspender_instance = NULL;
  handler_stack.ss_flags = SS_DISABLE;
  internal_sigaltstack(&handler_stack, NULL);
  return exit_code;
}

class ScopedStackSpaceWithGuard {
 public:
  explicit ScopedStackSpaceWithGuard(uptr stack_size) {
    stack_size_ = stack_size;
    guard_size_ = GetPageSizeCached();
    // FIXME: Omitting MAP_STACK here works in current kernels but might break
    // in the future.
    guard_start_ = (uptr)MmapOrDie(stack_size_ + guard_size_,
                                   "ScopedStackWithGuard");
    CHECK_EQ(guard_start_, (uptr)Mprotect((uptr)guard_start_, guard_size_));
  }
  ~ScopedStackSpaceWithGuard() {
    UnmapOrDie((void *)guard_start_, stack_size_ + guard_size_);
  }
  void *Bottom() const {
    return (void *)(guard_start_ + stack_size_ + guard_size_);
  }

 private:
  uptr stack_size_;
  uptr guard_size_;
  uptr guard_start_;
};

// We have a limitation on the stack frame size, so some stuff had to be moved
// into globals.
static __sanitizer_sigset_t blocked_sigset;
static __sanitizer_sigset_t old_sigset;
static __sanitizer_sigaction old_sigactions
    [ARRAY_SIZE(kUnblockedSignals)];

class StopTheWorldScope {
 public:
  StopTheWorldScope() {
    // Block all signals that can be blocked safely, and install
    // default handlers for the remaining signals.
    // We cannot allow user-defined handlers to run while the ThreadSuspender
    // thread is active, because they could conceivably call some libc functions
    // which modify errno (which is shared between the two threads).
    internal_sigfillset(&blocked_sigset);
    for (uptr signal_index = 0; signal_index < ARRAY_SIZE(kUnblockedSignals);
         signal_index++) {
      // Remove the signal from the set of blocked signals.
      internal_sigdelset(&blocked_sigset, kUnblockedSignals[signal_index]);
      // Install the default handler.
      __sanitizer_sigaction new_sigaction;
      internal_memset(&new_sigaction, 0, sizeof(new_sigaction));
      new_sigaction.handler = SIG_DFL;
      internal_sigfillset(&new_sigaction.sa_mask);
      internal_sigaction_norestorer(kUnblockedSignals[signal_index],
          &new_sigaction, &old_sigactions[signal_index]);
    }
    int sigprocmask_status =
        internal_sigprocmask(SIG_BLOCK, &blocked_sigset, &old_sigset);
    CHECK_EQ(sigprocmask_status, 0); // sigprocmask should never fail
    // Make this process dumpable. Processes that are not dumpable cannot be
    // attached to.
    process_was_dumpable_ = internal_prctl(PR_GET_DUMPABLE, 0, 0, 0, 0);
    if (!process_was_dumpable_)
      internal_prctl(PR_SET_DUMPABLE, 1, 0, 0, 0);
    old_die_callback = GetDieCallback();
  }

  ~StopTheWorldScope() {
    SetDieCallback(old_die_callback);
    // Restore the dumpable flag.
    if (!process_was_dumpable_)
      internal_prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);
    // Restore the signal handlers.
    for (uptr signal_index = 0; signal_index < ARRAY_SIZE(kUnblockedSignals);
         signal_index++) {
      internal_sigaction_norestorer(kUnblockedSignals[signal_index],
                                    &old_sigactions[signal_index], NULL);
    }
    internal_sigprocmask(SIG_SETMASK, &old_sigset, &old_sigset);
  }

 private:
  int process_was_dumpable_;
};

// When sanitizer output is being redirected to file (i.e. by using log_path),
// the tracer should write to the parent's log instead of trying to open a new
// file. Alert the logging code to the fact that we have a tracer.
struct ScopedSetTracerPID {
  explicit ScopedSetTracerPID(uptr tracer_pid) {
    stoptheworld_tracer_pid = tracer_pid;
    stoptheworld_tracer_ppid = internal_getpid();
  }
  ~ScopedSetTracerPID() {
    stoptheworld_tracer_pid = 0;
    stoptheworld_tracer_ppid = 0;
  }
};

void StopTheWorld(StopTheWorldCallback callback, void *argument) {
  StopTheWorldScope in_stoptheworld;
  // Prepare the arguments for TracerThread.
  struct TracerThreadArgument tracer_thread_argument;
  tracer_thread_argument.callback = callback;
  tracer_thread_argument.callback_argument = argument;
  tracer_thread_argument.parent_pid = internal_getpid();
  const uptr kTracerStackSize = 2 * 1024 * 1024;
  ScopedStackSpaceWithGuard tracer_stack(kTracerStackSize);
  // Block the execution of TracerThread until after we have set ptrace
  // permissions.
  tracer_thread_argument.mutex.Lock();
  uptr tracer_pid = internal_clone(
      TracerThread, tracer_stack.Bottom(),
      CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_UNTRACED,
      &tracer_thread_argument, 0 /* parent_tidptr */, 0 /* newtls */, 0
      /* child_tidptr */);
  int local_errno = 0;
  if (internal_iserror(tracer_pid, &local_errno)) {
    VReport(1, "Failed spawning a tracer thread (errno %d).\n", local_errno);
    tracer_thread_argument.mutex.Unlock();
  } else {
    ScopedSetTracerPID scoped_set_tracer_pid(tracer_pid);
    // On some systems we have to explicitly declare that we want to be traced
    // by the tracer thread.
#ifdef PR_SET_PTRACER
    internal_prctl(PR_SET_PTRACER, tracer_pid, 0, 0, 0);
#endif
    // Allow the tracer thread to start.
    tracer_thread_argument.mutex.Unlock();
    // Since errno is shared between this thread and the tracer thread, we
    // must avoid using errno while the tracer thread is running.
    // At this point, any signal will either be blocked or kill us, so waitpid
    // should never return (and set errno) while the tracer thread is alive.
    uptr waitpid_status = internal_waitpid(tracer_pid, NULL, __WALL);
    if (internal_iserror(waitpid_status, &local_errno))
      VReport(1, "Waiting on the tracer thread failed (errno %d).\n",
              local_errno);
  }
}

// Platform-specific methods from SuspendedThreadsList.
#if SANITIZER_ANDROID && defined(__arm__)
typedef pt_regs regs_struct;
#define REG_SP ARM_sp

#elif SANITIZER_LINUX && defined(__arm__)
typedef user_regs regs_struct;
#define REG_SP uregs[13]

#elif defined(__i386__) || defined(__x86_64__)
typedef user_regs_struct regs_struct;
#if defined(__i386__)
#define REG_SP esp
#else
#define REG_SP rsp
#endif

#elif defined(__powerpc__) || defined(__powerpc64__)
typedef pt_regs regs_struct;
#define REG_SP gpr[PT_R1]

#elif defined(__mips__)
typedef struct user regs_struct;
#define REG_SP regs[EF_REG29]

#else
#error "Unsupported architecture"
#endif // SANITIZER_ANDROID && defined(__arm__)

int SuspendedThreadsList::GetRegistersAndSP(uptr index,
                                            uptr *buffer,
                                            uptr *sp) const {
  pid_t tid = GetThreadID(index);
  regs_struct regs;
  int pterrno;
  if (internal_iserror(internal_ptrace(PTRACE_GETREGS, tid, NULL, &regs),
                       &pterrno)) {
    VReport(1, "Could not get registers from thread %d (errno %d).\n", tid,
            pterrno);
    return -1;
  }

  *sp = regs.REG_SP;
  internal_memcpy(buffer, &regs, sizeof(regs));
  return 0;
}

uptr SuspendedThreadsList::RegisterCount() {
  return sizeof(regs_struct) / sizeof(uptr);
}
}  // namespace __sanitizer

#endif  // SANITIZER_LINUX && defined(__x86_64__)
//===-- sanitizer_stackdepot.cc -------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_stackdepot.h"
#include "sanitizer_common.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_mutex.h"
#include "sanitizer_atomic.h"

namespace __sanitizer {

const int kTabSize = 1024 * 1024;  // Hash table size.
const int kPartBits = 8;
const int kPartShift = sizeof(u32) * 8 - kPartBits - 1;
const int kPartCount = 1 << kPartBits;  // Number of subparts in the table.
const int kPartSize = kTabSize / kPartCount;
const int kMaxId = 1 << kPartShift;

struct StackDesc {
  StackDesc *link;
  u32 id;
  u32 hash;
  uptr size;
  uptr stack[1];  // [size]
};

static struct {
  StaticSpinMutex mtx;  // Protects alloc of new blocks for region allocator.
  atomic_uintptr_t region_pos;  // Region allocator for StackDesc's.
  atomic_uintptr_t region_end;
  atomic_uintptr_t tab[kTabSize];  // Hash table of StackDesc's.
  atomic_uint32_t seq[kPartCount];  // Unique id generators.
} depot;

static StackDepotStats stats;

StackDepotStats *StackDepotGetStats() {
  return &stats;
}

static u32 hash(const uptr *stack, uptr size) {
  // murmur2
  const u32 m = 0x5bd1e995;
  const u32 seed = 0x9747b28c;
  const u32 r = 24;
  u32 h = seed ^ (size * sizeof(uptr));
  for (uptr i = 0; i < size; i++) {
    u32 k = stack[i];
    k *= m;
    k ^= k >> r;
    k *= m;
    h *= m;
    h ^= k;
  }
  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}

static StackDesc *tryallocDesc(uptr memsz) {
  // Optimisic lock-free allocation, essentially try to bump the region ptr.
  for (;;) {
    uptr cmp = atomic_load(&depot.region_pos, memory_order_acquire);
    uptr end = atomic_load(&depot.region_end, memory_order_acquire);
    if (cmp == 0 || cmp + memsz > end)
      return 0;
    if (atomic_compare_exchange_weak(
        &depot.region_pos, &cmp, cmp + memsz,
        memory_order_acquire))
      return (StackDesc*)cmp;
  }
}

static StackDesc *allocDesc(uptr size) {
  // First, try to allocate optimisitically.
  uptr memsz = sizeof(StackDesc) + (size - 1) * sizeof(uptr);
  StackDesc *s = tryallocDesc(memsz);
  if (s)
    return s;
  // If failed, lock, retry and alloc new superblock.
  SpinMutexLock l(&depot.mtx);
  for (;;) {
    s = tryallocDesc(memsz);
    if (s)
      return s;
    atomic_store(&depot.region_pos, 0, memory_order_relaxed);
    uptr allocsz = 64 * 1024;
    if (allocsz < memsz)
      allocsz = memsz;
    uptr mem = (uptr)MmapOrDie(allocsz, "stack depot");
    stats.mapped += allocsz;
    atomic_store(&depot.region_end, mem + allocsz, memory_order_release);
    atomic_store(&depot.region_pos, mem, memory_order_release);
  }
}

static u32 find(StackDesc *s, const uptr *stack, uptr size, u32 hash) {
  // Searches linked list s for the stack, returns its id.
  for (; s; s = s->link) {
    if (s->hash == hash && s->size == size) {
      uptr i = 0;
      for (; i < size; i++) {
        if (stack[i] != s->stack[i])
          break;
      }
      if (i == size)
        return s->id;
    }
  }
  return 0;
}

static StackDesc *lock(atomic_uintptr_t *p) {
  // Uses the pointer lsb as mutex.
  for (int i = 0;; i++) {
    uptr cmp = atomic_load(p, memory_order_relaxed);
    if ((cmp & 1) == 0
        && atomic_compare_exchange_weak(p, &cmp, cmp | 1,
                                        memory_order_acquire))
      return (StackDesc*)cmp;
    if (i < 10)
      proc_yield(10);
    else
      internal_sched_yield();
  }
}

static void unlock(atomic_uintptr_t *p, StackDesc *s) {
  DCHECK_EQ((uptr)s & 1, 0);
  atomic_store(p, (uptr)s, memory_order_release);
}

u32 StackDepotPut(const uptr *stack, uptr size) {
  if (stack == 0 || size == 0)
    return 0;
  uptr h = hash(stack, size);
  atomic_uintptr_t *p = &depot.tab[h % kTabSize];
  uptr v = atomic_load(p, memory_order_consume);
  StackDesc *s = (StackDesc*)(v & ~1);
  // First, try to find the existing stack.
  u32 id = find(s, stack, size, h);
  if (id)
    return id;
  // If failed, lock, retry and insert new.
  StackDesc *s2 = lock(p);
  if (s2 != s) {
    id = find(s2, stack, size, h);
    if (id) {
      unlock(p, s2);
      return id;
    }
  }
  uptr part = (h % kTabSize) / kPartSize;
  id = atomic_fetch_add(&depot.seq[part], 1, memory_order_relaxed) + 1;
  stats.n_uniq_ids++;
  CHECK_LT(id, kMaxId);
  id |= part << kPartShift;
  CHECK_NE(id, 0);
  CHECK_EQ(id & (1u << 31), 0);
  s = allocDesc(size);
  s->id = id;
  s->hash = h;
  s->size = size;
  internal_memcpy(s->stack, stack, size * sizeof(uptr));
  s->link = s2;
  unlock(p, s);
  return id;
}

const uptr *StackDepotGet(u32 id, uptr *size) {
  if (id == 0)
    return 0;
  CHECK_EQ(id & (1u << 31), 0);
  // High kPartBits contain part id, so we need to scan at most kPartSize lists.
  uptr part = id >> kPartShift;
  for (int i = 0; i != kPartSize; i++) {
    uptr idx = part * kPartSize + i;
    CHECK_LT(idx, kTabSize);
    atomic_uintptr_t *p = &depot.tab[idx];
    uptr v = atomic_load(p, memory_order_consume);
    StackDesc *s = (StackDesc*)(v & ~1);
    for (; s; s = s->link) {
      if (s->id == id) {
        *size = s->size;
        return s->stack;
      }
    }
  }
  *size = 0;
  return 0;
}

bool StackDepotReverseMap::IdDescPair::IdComparator(
    const StackDepotReverseMap::IdDescPair &a,
    const StackDepotReverseMap::IdDescPair &b) {
  return a.id < b.id;
}

StackDepotReverseMap::StackDepotReverseMap()
    : map_(StackDepotGetStats()->n_uniq_ids + 100) {
  for (int idx = 0; idx < kTabSize; idx++) {
    atomic_uintptr_t *p = &depot.tab[idx];
    uptr v = atomic_load(p, memory_order_consume);
    StackDesc *s = (StackDesc*)(v & ~1);
    for (; s; s = s->link) {
      IdDescPair pair = {s->id, s};
      map_.push_back(pair);
    }
  }
  InternalSort(&map_, map_.size(), IdDescPair::IdComparator);
}

const uptr *StackDepotReverseMap::Get(u32 id, uptr *size) {
  if (!map_.size()) return 0;
  IdDescPair pair = {id, 0};
  uptr idx = InternalBinarySearch(map_, 0, map_.size(), pair,
                                  IdDescPair::IdComparator);
  if (idx > map_.size()) {
    *size = 0;
    return 0;
  }
  StackDesc *desc = map_[idx].desc;
  *size = desc->size;
  return desc->stack;
}

}  // namespace __sanitizer
//===-- sanitizer_stacktrace.cc -------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_stacktrace.h"

namespace __sanitizer {

uptr StackTrace::GetPreviousInstructionPc(uptr pc) {
#ifdef __arm__
  // Cancel Thumb bit.
  pc = pc & (~1);
#endif
#if defined(__sparc__)
  return pc - 8;
#else
  return pc - 1;
#endif
}

uptr StackTrace::GetCurrentPc() {
  return GET_CALLER_PC();
}

void StackTrace::FastUnwindStack(uptr pc, uptr bp,
                                 uptr stack_top, uptr stack_bottom,
                                 uptr max_depth) {
  if (max_depth == 0) {
    size = 0;
    return;
  }
  trace[0] = pc;
  size = 1;
  uptr *frame = (uptr *)bp;
  uptr *prev_frame = frame - 1;
  if (stack_top < 4096) return;  // Sanity check for stack top.
  // Avoid infinite loop when frame == frame[0] by using frame > prev_frame.
  while (frame > prev_frame &&
         frame < (uptr *)stack_top - 2 &&
         frame > (uptr *)stack_bottom &&
         IsAligned((uptr)frame, sizeof(*frame)) &&
         size < max_depth) {
    uptr pc1 = frame[1];
    if (pc1 != pc) {
      trace[size++] = pc1;
    }
    prev_frame = frame;
    frame = (uptr*)frame[0];
  }
}

static bool MatchPc(uptr cur_pc, uptr trace_pc, uptr threshold) {
  return cur_pc - trace_pc <= threshold || trace_pc - cur_pc <= threshold;
}

void StackTrace::PopStackFrames(uptr count) {
  CHECK_LT(count, size);
  size -= count;
  for (uptr i = 0; i < size; ++i) {
    trace[i] = trace[i + count];
  }
}

uptr StackTrace::LocatePcInTrace(uptr pc) {
  // Use threshold to find PC in stack trace, as PC we want to unwind from may
  // slightly differ from return address in the actual unwinded stack trace.
  const int kPcThreshold = 192;
  for (uptr i = 0; i < size; ++i) {
    if (MatchPc(pc, trace[i], kPcThreshold))
      return i;
  }
  return 0;
}

}  // namespace __sanitizer
//===-- sanitizer_stacktrace_libcdep.cc -----------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_common.h"
#include "sanitizer_stacktrace.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

static void PrintStackFramePrefix(InternalScopedString *buffer, uptr frame_num,
                                  uptr pc) {
  buffer->append("    #%zu 0x%zx", frame_num, pc);
}

void StackTrace::PrintStack(const uptr *addr, uptr size) {
  if (addr == 0 || size == 0) {
    Printf("    <empty stack>\n\n");
    return;
  }
  InternalScopedBuffer<char> buff(GetPageSizeCached() * 2);
  InternalScopedBuffer<AddressInfo> addr_frames(64);
  InternalScopedString frame_desc(GetPageSizeCached() * 2);
  uptr frame_num = 0;
  for (uptr i = 0; i < size && addr[i]; i++) {
    // PCs in stack traces are actually the return addresses, that is,
    // addresses of the next instructions after the call.
    uptr pc = GetPreviousInstructionPc(addr[i]);
    uptr addr_frames_num = Symbolizer::GetOrInit()->SymbolizePC(
        pc, addr_frames.data(), addr_frames.size());
    for (uptr j = 0; j < addr_frames_num; j++) {
      AddressInfo &info = addr_frames[j];
      frame_desc.clear();
      PrintStackFramePrefix(&frame_desc, frame_num, pc);
      if (info.function) {
        frame_desc.append(" in %s", info.function);
        // Print offset in function if we don't know the source file.
        if (!info.file && info.function_offset != AddressInfo::kUnknown)
          frame_desc.append("+0x%zx", info.function_offset);
      }
      if (info.file) {
        frame_desc.append(" ");
        PrintSourceLocation(&frame_desc, info.file, info.line, info.column);
      } else if (info.module) {
        frame_desc.append(" ");
        PrintModuleAndOffset(&frame_desc, info.module, info.module_offset);
      }
      Printf("%s\n", frame_desc.data());
      frame_num++;
      info.Clear();
    }
  }
  // Always print a trailing empty line after stack trace.
  Printf("\n");
}

void StackTrace::Unwind(uptr max_depth, uptr pc, uptr bp, void *context,
                        uptr stack_top, uptr stack_bottom,
                        bool request_fast_unwind) {
  if (!WillUseFastUnwind(request_fast_unwind)) {
    if (context)
      SlowUnwindStackWithContext(pc, context, max_depth);
    else
      SlowUnwindStack(pc, max_depth);
  } else {
    FastUnwindStack(pc, bp, stack_top, stack_bottom, max_depth);
  }

  top_frame_bp = size ? bp : 0;
}

}  // namespace __sanitizer
//===-- sanitizer_symbolizer.cc -------------------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

Symbolizer *Symbolizer::symbolizer_;
StaticSpinMutex Symbolizer::init_mu_;
LowLevelAllocator Symbolizer::symbolizer_allocator_;

Symbolizer *Symbolizer::GetOrNull() {
  SpinMutexLock l(&init_mu_);
  return symbolizer_;
}

Symbolizer *Symbolizer::Get() {
  SpinMutexLock l(&init_mu_);
  RAW_CHECK_MSG(symbolizer_ != 0, "Using uninitialized symbolizer!");
  return symbolizer_;
}

Symbolizer *Symbolizer::Disable() {
  CHECK_EQ(0, symbolizer_);
  // Initialize a dummy symbolizer.
  symbolizer_ = new(symbolizer_allocator_) Symbolizer;
  return symbolizer_;
}

void Symbolizer::AddHooks(Symbolizer::StartSymbolizationHook start_hook,
                          Symbolizer::EndSymbolizationHook end_hook) {
  CHECK(start_hook_ == 0 && end_hook_ == 0);
  start_hook_ = start_hook;
  end_hook_ = end_hook;
}

Symbolizer::Symbolizer() : start_hook_(0), end_hook_(0) {}

Symbolizer::SymbolizerScope::SymbolizerScope(const Symbolizer *sym)
    : sym_(sym) {
  if (sym_->start_hook_)
    sym_->start_hook_();
}

Symbolizer::SymbolizerScope::~SymbolizerScope() {
  if (sym_->end_hook_)
    sym_->end_hook_();
}

}  // namespace __sanitizer
//===-- sanitizer_symbolizer_libcdep.cc -----------------------------------===//
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
//===----------------------------------------------------------------------===//

#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"

namespace __sanitizer {

Symbolizer *Symbolizer::CreateAndStore(const char *path_to_external) {
  Symbolizer *platform_symbolizer = PlatformInit(path_to_external);
  if (!platform_symbolizer)
    return Disable();
  symbolizer_ = platform_symbolizer;
  return platform_symbolizer;
}

Symbolizer *Symbolizer::Init(const char *path_to_external) {
  CHECK_EQ(0, symbolizer_);
  return CreateAndStore(path_to_external);
}

Symbolizer *Symbolizer::GetOrInit() {
  SpinMutexLock l(&init_mu_);
  if (symbolizer_ == 0)
    return CreateAndStore(0);
  return symbolizer_;
}

}  // namespace __sanitizer
//===-- sanitizer_symbolizer_posix_libcdep.cc -----------------------------===//
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
// POSIX-specific implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_POSIX
#include "sanitizer_allocator_internal.h"
#include "sanitizer_common.h"
#include "sanitizer_flags.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_linux.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_symbolizer_libbacktrace.h"

#include <errno.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

// C++ demangling function, as required by Itanium C++ ABI. This is weak,
// because we do not require a C++ ABI library to be linked to a program
// using sanitizers; if it's not present, we'll just use the mangled name.
//
// On Android, this is not weak, because we are using shared runtime library
// AND static libstdc++, and there is no good way to conditionally export
// __cxa_demangle. By making this a non-weak symbol, we statically link
// __cxa_demangle into ASan runtime library.
namespace __cxxabiv1 {
  extern "C"
#if !SANITIZER_ANDROID
  SANITIZER_WEAK_ATTRIBUTE
#endif
  char *__cxa_demangle(const char *mangled, char *buffer, size_t *length,
                       int *status);
}

namespace __sanitizer {

// Attempts to demangle the name via __cxa_demangle from __cxxabiv1.
static const char *DemangleCXXABI(const char *name) {
  // FIXME: __cxa_demangle aggressively insists on allocating memory.
  // There's not much we can do about that, short of providing our
  // own demangler (libc++abi's implementation could be adapted so that
  // it does not allocate). For now, we just call it anyway, and we leak
  // the returned value.
  if (SANITIZER_ANDROID || &__cxxabiv1::__cxa_demangle)
    if (const char *demangled_name =
          __cxxabiv1::__cxa_demangle(name, 0, 0, 0))
      return demangled_name;

  return name;
}

// Extracts the prefix of "str" that consists of any characters not
// present in "delims" string, and copies this prefix to "result", allocating
// space for it.
// Returns a pointer to "str" after skipping extracted prefix and first
// delimiter char.
static const char *ExtractToken(const char *str, const char *delims,
                                char **result) {
  uptr prefix_len = internal_strcspn(str, delims);
  *result = (char*)InternalAlloc(prefix_len + 1);
  internal_memcpy(*result, str, prefix_len);
  (*result)[prefix_len] = '\0';
  const char *prefix_end = str + prefix_len;
  if (*prefix_end != '\0') prefix_end++;
  return prefix_end;
}

// Same as ExtractToken, but converts extracted token to integer.
static const char *ExtractInt(const char *str, const char *delims,
                              int *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (int)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

static const char *ExtractUptr(const char *str, const char *delims,
                               uptr *result) {
  char *buff;
  const char *ret = ExtractToken(str, delims, &buff);
  if (buff != 0) {
    *result = (uptr)internal_atoll(buff);
  }
  InternalFree(buff);
  return ret;
}

class ExternalSymbolizerInterface {
 public:
  // Can't declare pure virtual functions in sanitizer runtimes:
  // __cxa_pure_virtual might be unavailable.
  virtual char *SendCommand(bool is_data, const char *module_name,
                            uptr module_offset) {
    UNIMPLEMENTED();
  }
};

// SymbolizerProcess encapsulates communication between the tool and
// external symbolizer program, running in a different subprocess.
// SymbolizerProcess may not be used from two threads simultaneously.
class SymbolizerProcess : public ExternalSymbolizerInterface {
 public:
  explicit SymbolizerProcess(const char *path)
      : path_(path),
        input_fd_(kInvalidFd),
        output_fd_(kInvalidFd),
        times_restarted_(0),
        failed_to_start_(false),
        reported_invalid_path_(false) {
    CHECK(path_);
    CHECK_NE(path_[0], '\0');
  }

  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    for (; times_restarted_ < kMaxTimesRestarted; times_restarted_++) {
      // Start or restart symbolizer if we failed to send command to it.
      if (char *res = SendCommandImpl(is_data, module_name, module_offset))
        return res;
      Restart();
    }
    if (!failed_to_start_) {
      Report("WARNING: Failed to use and restart external symbolizer!\n");
      failed_to_start_ = true;
    }
    return 0;
  }

 private:
  bool Restart() {
    if (input_fd_ != kInvalidFd)
      internal_close(input_fd_);
    if (output_fd_ != kInvalidFd)
      internal_close(output_fd_);
    return StartSymbolizerSubprocess();
  }

  char *SendCommandImpl(bool is_data, const char *module_name,
                        uptr module_offset) {
    if (input_fd_ == kInvalidFd || output_fd_ == kInvalidFd)
      return 0;
    CHECK(module_name);
    if (!RenderInputCommand(buffer_, kBufferSize, is_data, module_name,
                            module_offset))
      return 0;
    if (!writeToSymbolizer(buffer_, internal_strlen(buffer_)))
      return 0;
    if (!readFromSymbolizer(buffer_, kBufferSize))
      return 0;
    return buffer_;
  }

  bool readFromSymbolizer(char *buffer, uptr max_length) {
    if (max_length == 0)
      return true;
    uptr read_len = 0;
    while (true) {
      uptr just_read = internal_read(input_fd_, buffer + read_len,
                                     max_length - read_len - 1);
      // We can't read 0 bytes, as we don't expect external symbolizer to close
      // its stdout.
      if (just_read == 0 || just_read == (uptr)-1) {
        Report("WARNING: Can't read from symbolizer at fd %d\n", input_fd_);
        return false;
      }
      read_len += just_read;
      if (ReachedEndOfOutput(buffer, read_len))
        break;
    }
    buffer[read_len] = '\0';
    return true;
  }

  bool writeToSymbolizer(const char *buffer, uptr length) {
    if (length == 0)
      return true;
    uptr write_len = internal_write(output_fd_, buffer, length);
    if (write_len == 0 || write_len == (uptr)-1) {
      Report("WARNING: Can't write to symbolizer at fd %d\n", output_fd_);
      return false;
    }
    return true;
  }

  bool StartSymbolizerSubprocess() {
    if (!FileExists(path_)) {
      if (!reported_invalid_path_) {
        Report("WARNING: invalid path to external symbolizer!\n");
        reported_invalid_path_ = true;
      }
      return false;
    }

    int *infd = NULL;
    int *outfd = NULL;
    // The client program may close its stdin and/or stdout and/or stderr
    // thus allowing socketpair to reuse file descriptors 0, 1 or 2.
    // In this case the communication between the forked processes may be
    // broken if either the parent or the child tries to close or duplicate
    // these descriptors. The loop below produces two pairs of file
    // descriptors, each greater than 2 (stderr).
    int sock_pair[5][2];
    for (int i = 0; i < 5; i++) {
      if (pipe(sock_pair[i]) == -1) {
        for (int j = 0; j < i; j++) {
          internal_close(sock_pair[j][0]);
          internal_close(sock_pair[j][1]);
        }
        Report("WARNING: Can't create a socket pair to start "
               "external symbolizer (errno: %d)\n", errno);
        return false;
      } else if (sock_pair[i][0] > 2 && sock_pair[i][1] > 2) {
        if (infd == NULL) {
          infd = sock_pair[i];
        } else {
          outfd = sock_pair[i];
          for (int j = 0; j < i; j++) {
            if (sock_pair[j] == infd) continue;
            internal_close(sock_pair[j][0]);
            internal_close(sock_pair[j][1]);
          }
          break;
        }
      }
    }
    CHECK(infd);
    CHECK(outfd);

    int pid = fork();
    if (pid == -1) {
      // Fork() failed.
      internal_close(infd[0]);
      internal_close(infd[1]);
      internal_close(outfd[0]);
      internal_close(outfd[1]);
      Report("WARNING: failed to fork external symbolizer "
             " (errno: %d)\n", errno);
      return false;
    } else if (pid == 0) {
      // Child subprocess.
      internal_close(STDOUT_FILENO);
      internal_close(STDIN_FILENO);
      internal_dup2(outfd[0], STDIN_FILENO);
      internal_dup2(infd[1], STDOUT_FILENO);
      internal_close(outfd[0]);
      internal_close(outfd[1]);
      internal_close(infd[0]);
      internal_close(infd[1]);
      for (int fd = getdtablesize(); fd > 2; fd--)
        internal_close(fd);
      ExecuteWithDefaultArgs(path_);
      internal__exit(1);
    }

    // Continue execution in parent process.
    internal_close(outfd[0]);
    internal_close(infd[1]);
    input_fd_ = infd[0];
    output_fd_ = outfd[1];

    // Check that symbolizer subprocess started successfully.
    int pid_status;
    SleepForMillis(kSymbolizerStartupTimeMillis);
    int exited_pid = waitpid(pid, &pid_status, WNOHANG);
    if (exited_pid != 0) {
      // Either waitpid failed, or child has already exited.
      Report("WARNING: external symbolizer didn't start up correctly!\n");
      return false;
    }

    return true;
  }

  virtual bool RenderInputCommand(char *buffer, uptr max_length, bool is_data,
                                  const char *module_name,
                                  uptr module_offset) const {
    UNIMPLEMENTED();
  }

  virtual bool ReachedEndOfOutput(const char *buffer, uptr length) const {
    UNIMPLEMENTED();
  }

  virtual void ExecuteWithDefaultArgs(const char *path_to_binary) const {
    UNIMPLEMENTED();
  }

  const char *path_;
  int input_fd_;
  int output_fd_;

  static const uptr kBufferSize = 16 * 1024;
  char buffer_[kBufferSize];

  static const uptr kMaxTimesRestarted = 5;
  static const int kSymbolizerStartupTimeMillis = 10;
  uptr times_restarted_;
  bool failed_to_start_;
  bool reported_invalid_path_;
};

// For now we assume the following protocol:
// For each request of the form
//   <module_name> <module_offset>
// passed to STDIN, external symbolizer prints to STDOUT response:
//   <function_name>
//   <file_name>:<line_number>:<column_number>
//   <function_name>
//   <file_name>:<line_number>:<column_number>
//   ...
//   <empty line>
class LLVMSymbolizerProcess : public SymbolizerProcess {
 public:
  explicit LLVMSymbolizerProcess(const char *path) : SymbolizerProcess(path) {}

 private:
  bool RenderInputCommand(char *buffer, uptr max_length, bool is_data,
                          const char *module_name, uptr module_offset) const {
    internal_snprintf(buffer, max_length, "%s\"%s\" 0x%zx\n",
                      is_data ? "DATA " : "", module_name, module_offset);
    return true;
  }

  bool ReachedEndOfOutput(const char *buffer, uptr length) const {
    // Empty line marks the end of llvm-symbolizer output.
    return length >= 2 && buffer[length - 1] == '\n' &&
           buffer[length - 2] == '\n';
  }

  void ExecuteWithDefaultArgs(const char *path_to_binary) const {
#if defined(__x86_64__)
    const char* const kSymbolizerArch = "--default-arch=x86_64";
#elif defined(__i386__)
    const char* const kSymbolizerArch = "--default-arch=i386";
#elif defined(__powerpc64__)
    const char* const kSymbolizerArch = "--default-arch=powerpc64";
#else
    const char* const kSymbolizerArch = "--default-arch=unknown";
#endif
    execl(path_to_binary, path_to_binary, kSymbolizerArch, (char *)0);
  }
};

class Addr2LineProcess : public SymbolizerProcess {
 public:
  Addr2LineProcess(const char *path, const char *module_name)
      : SymbolizerProcess(path), module_name_(internal_strdup(module_name)) {}

  const char *module_name() const { return module_name_; }

 private:
  bool RenderInputCommand(char *buffer, uptr max_length, bool is_data,
                          const char *module_name, uptr module_offset) const {
    if (is_data)
      return false;
    CHECK_EQ(0, internal_strcmp(module_name, module_name_));
    internal_snprintf(buffer, max_length, "0x%zx\n", module_offset);
    return true;
  }

  bool ReachedEndOfOutput(const char *buffer, uptr length) const {
    // Output should consist of two lines.
    int num_lines = 0;
    for (uptr i = 0; i < length; ++i) {
      if (buffer[i] == '\n')
        num_lines++;
      if (num_lines >= 2)
        return true;
    }
    return false;
  }

  void ExecuteWithDefaultArgs(const char *path_to_binary) const {
    execl(path_to_binary, path_to_binary, "-Cfe", module_name_, (char *)0);
  }

  const char *module_name_;  // Owned, leaked.
};

class Addr2LinePool : public ExternalSymbolizerInterface {
 public:
  explicit Addr2LinePool(const char *addr2line_path,
                         LowLevelAllocator *allocator)
      : addr2line_path_(addr2line_path), allocator_(allocator),
        addr2line_pool_(16) {}

  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    if (is_data)
      return 0;
    Addr2LineProcess *addr2line = 0;
    for (uptr i = 0; i < addr2line_pool_.size(); ++i) {
      if (0 ==
          internal_strcmp(module_name, addr2line_pool_[i]->module_name())) {
        addr2line = addr2line_pool_[i];
        break;
      }
    }
    if (!addr2line) {
      addr2line =
          new(*allocator_) Addr2LineProcess(addr2line_path_, module_name);
      addr2line_pool_.push_back(addr2line);
    }
    return addr2line->SendCommand(is_data, module_name, module_offset);
  }

 private:
  const char *addr2line_path_;
  LowLevelAllocator *allocator_;
  InternalMmapVector<Addr2LineProcess*> addr2line_pool_;
};

#if SANITIZER_SUPPORTS_WEAK_HOOKS
extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
bool __sanitizer_symbolize_code(const char *ModuleName, u64 ModuleOffset,
                                char *Buffer, int MaxLength);
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
bool __sanitizer_symbolize_data(const char *ModuleName, u64 ModuleOffset,
                                char *Buffer, int MaxLength);
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
void __sanitizer_symbolize_flush();
SANITIZER_INTERFACE_ATTRIBUTE SANITIZER_WEAK_ATTRIBUTE
int __sanitizer_symbolize_demangle(const char *Name, char *Buffer,
                                   int MaxLength);
}  // extern "C"

class InternalSymbolizer {
 public:
  typedef bool (*SanitizerSymbolizeFn)(const char*, u64, char*, int);

  static InternalSymbolizer *get(LowLevelAllocator *alloc) {
    if (__sanitizer_symbolize_code != 0 &&
        __sanitizer_symbolize_data != 0) {
      return new(*alloc) InternalSymbolizer();
    }
    return 0;
  }

  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    SanitizerSymbolizeFn symbolize_fn = is_data ? __sanitizer_symbolize_data
                                                : __sanitizer_symbolize_code;
    if (symbolize_fn(module_name, module_offset, buffer_, kBufferSize))
      return buffer_;
    return 0;
  }

  void Flush() {
    if (__sanitizer_symbolize_flush)
      __sanitizer_symbolize_flush();
  }

  const char *Demangle(const char *name) {
    if (__sanitizer_symbolize_demangle) {
      for (uptr res_length = 1024;
           res_length <= InternalSizeClassMap::kMaxSize;) {
        char *res_buff = static_cast<char*>(InternalAlloc(res_length));
        uptr req_length =
            __sanitizer_symbolize_demangle(name, res_buff, res_length);
        if (req_length > res_length) {
          res_length = req_length + 1;
          InternalFree(res_buff);
          continue;
        }
        return res_buff;
      }
    }
    return name;
  }

 private:
  InternalSymbolizer() { }

  static const int kBufferSize = 16 * 1024;
  static const int kMaxDemangledNameSize = 1024;
  char buffer_[kBufferSize];
};
#else  // SANITIZER_SUPPORTS_WEAK_HOOKS

class InternalSymbolizer {
 public:
  static InternalSymbolizer *get(LowLevelAllocator *alloc) { return 0; }
  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    return 0;
  }
  void Flush() { }
  const char *Demangle(const char *name) { return name; }
};

#endif  // SANITIZER_SUPPORTS_WEAK_HOOKS

class POSIXSymbolizer : public Symbolizer {
 public:
  POSIXSymbolizer(ExternalSymbolizerInterface *external_symbolizer,
                  InternalSymbolizer *internal_symbolizer,
                  LibbacktraceSymbolizer *libbacktrace_symbolizer)
      : Symbolizer(),
        external_symbolizer_(external_symbolizer),
        internal_symbolizer_(internal_symbolizer),
        libbacktrace_symbolizer_(libbacktrace_symbolizer) {}

  uptr SymbolizePC(uptr addr, AddressInfo *frames, uptr max_frames) {
    BlockingMutexLock l(&mu_);
    if (max_frames == 0)
      return 0;
    const char *module_name;
    uptr module_offset;
    if (!FindModuleNameAndOffsetForAddress(addr, &module_name, &module_offset))
      return 0;
    // First, try to use libbacktrace symbolizer (if it's available).
    if (libbacktrace_symbolizer_ != 0) {
      mu_.CheckLocked();
      uptr res = libbacktrace_symbolizer_->SymbolizeCode(
          addr, frames, max_frames, module_name, module_offset);
      if (res > 0)
        return res;
    }
    const char *str = SendCommand(false, module_name, module_offset);
    if (str == 0) {
      // Symbolizer was not initialized or failed. Fill only data
      // about module name and offset.
      AddressInfo *info = &frames[0];
      info->Clear();
      info->FillAddressAndModuleInfo(addr, module_name, module_offset);
      return 1;
    }
    uptr frame_id = 0;
    for (frame_id = 0; frame_id < max_frames; frame_id++) {
      AddressInfo *info = &frames[frame_id];
      char *function_name = 0;
      str = ExtractToken(str, "\n", &function_name);
      CHECK(function_name);
      if (function_name[0] == '\0') {
        // There are no more frames.
        break;
      }
      info->Clear();
      info->FillAddressAndModuleInfo(addr, module_name, module_offset);
      info->function = function_name;
      // Parse <file>:<line>:<column> buffer.
      char *file_line_info = 0;
      str = ExtractToken(str, "\n", &file_line_info);
      CHECK(file_line_info);
      const char *line_info = ExtractToken(file_line_info, ":", &info->file);
      line_info = ExtractInt(line_info, ":", &info->line);
      line_info = ExtractInt(line_info, "", &info->column);
      InternalFree(file_line_info);

      // Functions and filenames can be "??", in which case we write 0
      // to address info to mark that names are unknown.
      if (0 == internal_strcmp(info->function, "??")) {
        InternalFree(info->function);
        info->function = 0;
      }
      if (0 == internal_strcmp(info->file, "??")) {
        InternalFree(info->file);
        info->file = 0;
      }
    }
    if (frame_id == 0) {
      // Make sure we return at least one frame.
      AddressInfo *info = &frames[0];
      info->Clear();
      info->FillAddressAndModuleInfo(addr, module_name, module_offset);
      frame_id = 1;
    }
    return frame_id;
  }

  bool SymbolizeData(uptr addr, DataInfo *info) {
    BlockingMutexLock l(&mu_);
    LoadedModule *module = FindModuleForAddress(addr);
    if (module == 0)
      return false;
    const char *module_name = module->full_name();
    uptr module_offset = addr - module->base_address();
    internal_memset(info, 0, sizeof(*info));
    info->address = addr;
    info->module = internal_strdup(module_name);
    info->module_offset = module_offset;
    // First, try to use libbacktrace symbolizer (if it's available).
    if (libbacktrace_symbolizer_ != 0) {
      mu_.CheckLocked();
      if (libbacktrace_symbolizer_->SymbolizeData(info))
        return true;
    }
    const char *str = SendCommand(true, module_name, module_offset);
    if (str == 0)
      return true;
    str = ExtractToken(str, "\n", &info->name);
    str = ExtractUptr(str, " ", &info->start);
    str = ExtractUptr(str, "\n", &info->size);
    info->start += module->base_address();
    return true;
  }

  bool GetModuleNameAndOffsetForPC(uptr pc, const char **module_name,
                                   uptr *module_address) {
    BlockingMutexLock l(&mu_);
    return FindModuleNameAndOffsetForAddress(pc, module_name, module_address);
  }

  bool CanReturnFileLineInfo() {
    return internal_symbolizer_ != 0 || external_symbolizer_ != 0 ||
           libbacktrace_symbolizer_ != 0;
  }

  void Flush() {
    BlockingMutexLock l(&mu_);
    if (internal_symbolizer_ != 0) {
      SymbolizerScope sym_scope(this);
      internal_symbolizer_->Flush();
    }
  }

  const char *Demangle(const char *name) {
    BlockingMutexLock l(&mu_);
    // Run hooks even if we don't use internal symbolizer, as cxxabi
    // demangle may call system functions.
    SymbolizerScope sym_scope(this);
    // Try to use libbacktrace demangler (if available).
    if (libbacktrace_symbolizer_ != 0) {
      if (const char *demangled = libbacktrace_symbolizer_->Demangle(name))
        return demangled;
    }
    if (internal_symbolizer_ != 0)
      return internal_symbolizer_->Demangle(name);
    return DemangleCXXABI(name);
  }

  void PrepareForSandboxing() {
#if SANITIZER_LINUX && !SANITIZER_ANDROID
    BlockingMutexLock l(&mu_);
    // Cache /proc/self/exe on Linux.
    CacheBinaryName();
#endif
  }

 private:
  char *SendCommand(bool is_data, const char *module_name, uptr module_offset) {
    mu_.CheckLocked();
    // First, try to use internal symbolizer.
    if (internal_symbolizer_) {
      SymbolizerScope sym_scope(this);
      return internal_symbolizer_->SendCommand(is_data, module_name,
                                               module_offset);
    }
    // Otherwise, fall back to external symbolizer.
    if (external_symbolizer_) {
      SymbolizerScope sym_scope(this);
      return external_symbolizer_->SendCommand(is_data, module_name,
                                               module_offset);
    }
    return 0;
  }

  LoadedModule *FindModuleForAddress(uptr address) {
    mu_.CheckLocked();
    bool modules_were_reloaded = false;
    if (modules_ == 0 || !modules_fresh_) {
      modules_ = (LoadedModule*)(symbolizer_allocator_.Allocate(
          kMaxNumberOfModuleContexts * sizeof(LoadedModule)));
      CHECK(modules_);
      n_modules_ = GetListOfModules(modules_, kMaxNumberOfModuleContexts,
                                    /* filter */ 0);
      CHECK_GT(n_modules_, 0);
      CHECK_LT(n_modules_, kMaxNumberOfModuleContexts);
      modules_fresh_ = true;
      modules_were_reloaded = true;
    }
    for (uptr i = 0; i < n_modules_; i++) {
      if (modules_[i].containsAddress(address)) {
        return &modules_[i];
      }
    }
    // Reload the modules and look up again, if we haven't tried it yet.
    if (!modules_were_reloaded) {
      // FIXME: set modules_fresh_ from dlopen()/dlclose() interceptors.
      // It's too aggressive to reload the list of modules each time we fail
      // to find a module for a given address.
      modules_fresh_ = false;
      return FindModuleForAddress(address);
    }
    return 0;
  }

  bool FindModuleNameAndOffsetForAddress(uptr address, const char **module_name,
                                         uptr *module_offset) {
    mu_.CheckLocked();
    LoadedModule *module = FindModuleForAddress(address);
    if (module == 0)
      return false;
    *module_name = module->full_name();
    *module_offset = address - module->base_address();
    return true;
  }

  // 16K loaded modules should be enough for everyone.
  static const uptr kMaxNumberOfModuleContexts = 1 << 14;
  LoadedModule *modules_;  // Array of module descriptions is leaked.
  uptr n_modules_;
  // If stale, need to reload the modules before looking up addresses.
  bool modules_fresh_;
  BlockingMutex mu_;

  ExternalSymbolizerInterface *external_symbolizer_;  // Leaked.
  InternalSymbolizer *const internal_symbolizer_;     // Leaked.
  LibbacktraceSymbolizer *libbacktrace_symbolizer_;   // Leaked.
};

Symbolizer *Symbolizer::PlatformInit(const char *path_to_external) {
  if (!common_flags()->symbolize) {
    return new(symbolizer_allocator_) POSIXSymbolizer(0, 0, 0);
  }
  InternalSymbolizer* internal_symbolizer =
      InternalSymbolizer::get(&symbolizer_allocator_);
  ExternalSymbolizerInterface *external_symbolizer = 0;
  LibbacktraceSymbolizer *libbacktrace_symbolizer = 0;

  if (!internal_symbolizer) {
    libbacktrace_symbolizer =
        LibbacktraceSymbolizer::get(&symbolizer_allocator_);
    if (!libbacktrace_symbolizer) {
      if (path_to_external && path_to_external[0] == '\0') {
        // External symbolizer is explicitly disabled. Do nothing.
      } else {
        // Find path to llvm-symbolizer if it's not provided.
        if (!path_to_external)
          path_to_external = FindPathToBinary("llvm-symbolizer");
        if (path_to_external) {
          external_symbolizer = new(symbolizer_allocator_)
              LLVMSymbolizerProcess(path_to_external);
        } else if (common_flags()->allow_addr2line) {
          // If llvm-symbolizer is not found, try to use addr2line.
          if (const char *addr2line_path = FindPathToBinary("addr2line")) {
            external_symbolizer = new(symbolizer_allocator_)
                Addr2LinePool(addr2line_path, &symbolizer_allocator_);
          }
        }
      }
    }
  }

  return new(symbolizer_allocator_) POSIXSymbolizer(
      external_symbolizer, internal_symbolizer, libbacktrace_symbolizer);
}

}  // namespace __sanitizer

#endif  // SANITIZER_POSIX
//===-- sanitizer_symbolizer_libbacktrace.cc ------------------------------===//
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
// Libbacktrace implementation of symbolizer parts.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"

#include "sanitizer_internal_defs.h"
#include "sanitizer_symbolizer.h"
#include "sanitizer_symbolizer_libbacktrace.h"

#if SANITIZER_LIBBACKTRACE
# include "backtrace-supported.h"
# if SANITIZER_POSIX && BACKTRACE_SUPPORTED && !BACKTRACE_USES_MALLOC
#  include "backtrace.h"
#  if SANITIZER_CP_DEMANGLE
#   undef ARRAY_SIZE
#   include "demangle.h"
#  endif
# else
#  define SANITIZER_LIBBACKTRACE 0
# endif
#endif

namespace __sanitizer {

#if SANITIZER_LIBBACKTRACE

namespace {

# if SANITIZER_CP_DEMANGLE
struct CplusV3DemangleData {
  char *buf;
  uptr size, allocated;
};

extern "C" {
static void CplusV3DemangleCallback(const char *s, size_t l, void *vdata) {
  CplusV3DemangleData *data = (CplusV3DemangleData *)vdata;
  uptr needed = data->size + l + 1;
  if (needed > data->allocated) {
    data->allocated *= 2;
    if (needed > data->allocated)
      data->allocated = needed;
    char *buf = (char *)InternalAlloc(data->allocated);
    if (data->buf) {
      internal_memcpy(buf, data->buf, data->size);
      InternalFree(data->buf);
    }
    data->buf = buf;
  }
  internal_memcpy(data->buf + data->size, s, l);
  data->buf[data->size + l] = '\0';
  data->size += l;
}
}  // extern "C"

char *CplusV3Demangle(const char *name) {
  CplusV3DemangleData data;
  data.buf = 0;
  data.size = 0;
  data.allocated = 0;
  if (cplus_demangle_v3_callback(name, DMGL_PARAMS | DMGL_ANSI,
                                 CplusV3DemangleCallback, &data)) {
    if (data.size + 64 > data.allocated)
      return data.buf;
    char *buf = internal_strdup(data.buf);
    InternalFree(data.buf);
    return buf;
  }
  if (data.buf)
    InternalFree(data.buf);
  return 0;
}
# endif  // SANITIZER_CP_DEMANGLE

struct SymbolizeCodeData {
  AddressInfo *frames;
  uptr n_frames;
  uptr max_frames;
  const char *module_name;
  uptr module_offset;
};

extern "C" {
static int SymbolizeCodePCInfoCallback(void *vdata, uintptr_t addr,
                                       const char *filename, int lineno,
                                       const char *function) {
  SymbolizeCodeData *cdata = (SymbolizeCodeData *)vdata;
  if (function) {
    AddressInfo *info = &cdata->frames[cdata->n_frames++];
    info->Clear();
    info->FillAddressAndModuleInfo(addr, cdata->module_name,
                                   cdata->module_offset);
    info->function = LibbacktraceSymbolizer::Demangle(function, true);
    if (filename)
      info->file = internal_strdup(filename);
    info->line = lineno;
    if (cdata->n_frames == cdata->max_frames)
      return 1;
  }
  return 0;
}

static void SymbolizeCodeCallback(void *vdata, uintptr_t addr,
                                  const char *symname, uintptr_t, uintptr_t) {
  SymbolizeCodeData *cdata = (SymbolizeCodeData *)vdata;
  if (symname) {
    AddressInfo *info = &cdata->frames[0];
    info->Clear();
    info->FillAddressAndModuleInfo(addr, cdata->module_name,
                                   cdata->module_offset);
    info->function = LibbacktraceSymbolizer::Demangle(symname, true);
    cdata->n_frames = 1;
  }
}

static void SymbolizeDataCallback(void *vdata, uintptr_t, const char *symname,
                                  uintptr_t symval, uintptr_t symsize) {
  DataInfo *info = (DataInfo *)vdata;
  if (symname && symval) {
    info->name = LibbacktraceSymbolizer::Demangle(symname, true);
    info->start = symval;
    info->size = symsize;
  }
}

static void ErrorCallback(void *, const char *, int) {}
}  // extern "C"

}  // namespace

LibbacktraceSymbolizer *LibbacktraceSymbolizer::get(LowLevelAllocator *alloc) {
  // State created in backtrace_create_state is leaked.
  void *state = (void *)(backtrace_create_state("/proc/self/exe", 0,
                                                ErrorCallback, NULL));
  if (!state)
    return 0;
  return new(*alloc) LibbacktraceSymbolizer(state);
}

uptr LibbacktraceSymbolizer::SymbolizeCode(uptr addr, AddressInfo *frames,
                                           uptr max_frames,
                                           const char *module_name,
                                           uptr module_offset) {
  SymbolizeCodeData data;
  data.frames = frames;
  data.n_frames = 0;
  data.max_frames = max_frames;
  data.module_name = module_name;
  data.module_offset = module_offset;
  backtrace_pcinfo((backtrace_state *)state_, addr, SymbolizeCodePCInfoCallback,
                   ErrorCallback, &data);
  if (data.n_frames)
    return data.n_frames;
  backtrace_syminfo((backtrace_state *)state_, addr, SymbolizeCodeCallback,
                    ErrorCallback, &data);
  return data.n_frames;
}

bool LibbacktraceSymbolizer::SymbolizeData(DataInfo *info) {
  backtrace_syminfo((backtrace_state *)state_, info->address,
                    SymbolizeDataCallback, ErrorCallback, info);
  return true;
}

#else  // SANITIZER_LIBBACKTRACE

LibbacktraceSymbolizer *LibbacktraceSymbolizer::get(LowLevelAllocator *alloc) {
  return 0;
}

uptr LibbacktraceSymbolizer::SymbolizeCode(uptr addr, AddressInfo *frames,
                                           uptr max_frames,
                                           const char *module_name,
                                           uptr module_offset) {
  (void)state_;
  return 0;
}

bool LibbacktraceSymbolizer::SymbolizeData(DataInfo *info) {
  return false;
}

#endif  // SANITIZER_LIBBACKTRACE

char *LibbacktraceSymbolizer::Demangle(const char *name, bool always_alloc) {
#if SANITIZER_LIBBACKTRACE && SANITIZER_CP_DEMANGLE
  if (char *demangled = CplusV3Demangle(name))
    return demangled;
#endif
  if (always_alloc)
    return internal_strdup(name);
  return 0;
}

}  // namespace __sanitizer
//===-- interception_linux.cc -----------------------------------*- C++ -*-===//
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
// Linux-specific interception methods.
//===----------------------------------------------------------------------===//

#if defined(__linux__) || defined(__FreeBSD__)
#include "interception.h"

#include <dlfcn.h>   // for dlsym() and dlvsym()

namespace __interception {
bool GetRealFunctionAddress(const char *func_name, uptr *func_addr,
    uptr real, uptr wrapper) {
  *func_addr = (uptr)dlsym(RTLD_NEXT, func_name);
  return real == wrapper;
}

#if !defined(__ANDROID__)  // android does not have dlvsym
void *GetFuncAddrVer(const char *func_name, const char *ver) {
  return dlvsym(RTLD_NEXT, func_name, ver);
}
#endif  // !defined(__ANDROID__)

}  // namespace __interception


#endif  // __linux__ || __FreeBSD__
