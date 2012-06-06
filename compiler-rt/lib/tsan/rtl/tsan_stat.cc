//===-- tsan_stat.cc ------------------------------------------------------===//
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
//===----------------------------------------------------------------------===//
#include "tsan_stat.h"
#include "tsan_rtl.h"

namespace __tsan {

void StatAggregate(u64 *dst, u64 *src) {
  if (!kCollectStats)
    return;
  for (int i = 0; i < StatCnt; i++)
    dst[i] += src[i];
}

void StatOutput(u64 *stat) {
  if (!kCollectStats)
    return;

  stat[StatShadowNonZero] = stat[StatShadowProcessed] - stat[StatShadowZero];

  static const char *name[StatCnt] = {};
  name[StatMop]                          = "Memory accesses                   ";
  name[StatMopRead]                      = "  Including reads                 ";
  name[StatMopWrite]                     = "            writes                ";
  name[StatMop1]                         = "  Including size 1                ";
  name[StatMop2]                         = "            size 2                ";
  name[StatMop4]                         = "            size 4                ";
  name[StatMop8]                         = "            size 8                ";
  name[StatMopSame]                      = "  Including same                  ";
  name[StatMopRange]                     = "  Including range                 ";
  name[StatShadowProcessed]              = "Shadow processed                  ";
  name[StatShadowZero]                   = "  Including empty                 ";
  name[StatShadowNonZero]                = "  Including non empty             ";
  name[StatShadowSameSize]               = "  Including same size             ";
  name[StatShadowIntersect]              = "            intersect             ";
  name[StatShadowNotIntersect]           = "            not intersect         ";
  name[StatShadowSameThread]             = "  Including same thread           ";
  name[StatShadowAnotherThread]          = "            another thread        ";
  name[StatShadowReplace]                = "  Including evicted               ";

  name[StatFuncEnter]                    = "Function entries                  ";
  name[StatFuncExit]                     = "Function exits                    ";
  name[StatEvents]                       = "Events collected                  ";

  name[StatThreadCreate]                 = "Total threads created             ";
  name[StatThreadFinish]                 = "  threads finished                ";
  name[StatThreadReuse]                  = "  threads reused                  ";
  name[StatThreadMaxTid]                 = "  max tid                         ";
  name[StatThreadMaxAlive]               = "  max alive threads               ";

  name[StatMutexCreate]                  = "Mutexes created                   ";
  name[StatMutexDestroy]                 = "  destroyed                       ";
  name[StatMutexLock]                    = "  lock                            ";
  name[StatMutexUnlock]                  = "  unlock                          ";
  name[StatMutexRecLock]                 = "  recursive lock                  ";
  name[StatMutexRecUnlock]               = "  recursive unlock                ";
  name[StatMutexReadLock]                = "  read lock                       ";
  name[StatMutexReadUnlock]              = "  read unlock                     ";

  name[StatSyncCreated]                  = "Sync objects created              ";
  name[StatSyncDestroyed]                = "             destroyed            ";
  name[StatSyncAcquire]                  = "             acquired             ";
  name[StatSyncRelease]                  = "             released             ";

  name[StatAtomic]                       = "Atomic operations                 ";
  name[StatAtomicLoad]                   = "  Including load                  ";
  name[StatAtomicStore]                  = "            store                 ";
  name[StatAtomicExchange]               = "            exchange              ";
  name[StatAtomicFetchAdd]               = "            fetch_add             ";
  name[StatAtomicCAS]                    = "            compare_exchange      ";
  name[StatAtomicFence]                  = "            fence                 ";
  name[StatAtomicRelaxed]                = "  Including relaxed               ";
  name[StatAtomicConsume]                = "            consume               ";
  name[StatAtomicAcquire]                = "            acquire               ";
  name[StatAtomicRelease]                = "            release               ";
  name[StatAtomicAcq_Rel]                = "            acq_rel               ";
  name[StatAtomicSeq_Cst]                = "            seq_cst               ";
  name[StatAtomic1]                      = "  Including size 1                ";
  name[StatAtomic2]                      = "            size 2                ";
  name[StatAtomic4]                      = "            size 4                ";
  name[StatAtomic8]                      = "            size 8                ";

  name[StatInterceptor]                  = "Interceptors                      ";
  name[StatInt_longjmp]                  = "  longjmp                         ";
  name[StatInt_siglongjmp]               = "  siglongjmp                      ";
  name[StatInt_malloc]                   = "  malloc                          ";
  name[StatInt_calloc]                   = "  calloc                          ";
  name[StatInt_realloc]                  = "  realloc                         ";
  name[StatInt_free]                     = "  free                            ";
  name[StatInt_cfree]                    = "  cfree                           ";
  name[StatInt_mmap]                     = "  mmap                            ";
  name[StatInt_mmap64]                   = "  mmap64                          ";
  name[StatInt_munmap]                   = "  munmap                          ";
  name[StatInt_memalign]                 = "  memalign                        ";
  name[StatInt_valloc]                   = "  valloc                          ";
  name[StatInt_pvalloc]                  = "  pvalloc                         ";
  name[StatInt_posix_memalign]           = "  posix_memalign                  ";
  name[StatInt__Znwm]                    = "  _Znwm                           ";
  name[StatInt__ZnwmRKSt9nothrow_t]      = "  _ZnwmRKSt9nothrow_t             ";
  name[StatInt__Znam]                    = "  _Znam                           ";
  name[StatInt__ZnamRKSt9nothrow_t]      = "  _ZnamRKSt9nothrow_t             ";
  name[StatInt__ZdlPv]                   = "  _ZdlPv                          ";
  name[StatInt__ZdlPvRKSt9nothrow_t]     = "  _ZdlPvRKSt9nothrow_t            ";
  name[StatInt__ZdaPv]                   = "  _ZdaPv                          ";
  name[StatInt__ZdaPvRKSt9nothrow_t]     = "  _ZdaPvRKSt9nothrow_t            ";
  name[StatInt_strlen]                   = "  strlen                          ";
  name[StatInt_memset]                   = "  memset                          ";
  name[StatInt_memcpy]                   = "  memcpy                          ";
  name[StatInt_strcmp]                   = "  strcmp                          ";
  name[StatInt_memchr]                   = "  memchr                          ";
  name[StatInt_memrchr]                  = "  memrchr                         ";
  name[StatInt_memmove]                  = "  memmove                         ";
  name[StatInt_memcmp]                   = "  memcmp                          ";
  name[StatInt_strchr]                   = "  strchr                          ";
  name[StatInt_strchrnul]                = "  strchrnul                       ";
  name[StatInt_strrchr]                  = "  strrchr                         ";
  name[StatInt_strncmp]                  = "  strncmp                         ";
  name[StatInt_strcpy]                   = "  strcpy                          ";
  name[StatInt_strncpy]                  = "  strncpy                         ";
  name[StatInt_strstr]                   = "  strstr                          ";
  name[StatInt_atexit]                   = "  atexit                          ";
  name[StatInt___cxa_guard_acquire]      = "  __cxa_guard_acquire             ";
  name[StatInt___cxa_guard_release]      = "  __cxa_guard_release             ";
  name[StatInt_pthread_create]           = "  pthread_create                  ";
  name[StatInt_pthread_join]             = "  pthread_join                    ";
  name[StatInt_pthread_detach]           = "  pthread_detach                  ";
  name[StatInt_pthread_mutex_init]       = "  pthread_mutex_init              ";
  name[StatInt_pthread_mutex_destroy]    = "  pthread_mutex_destroy           ";
  name[StatInt_pthread_mutex_lock]       = "  pthread_mutex_lock              ";
  name[StatInt_pthread_mutex_trylock]    = "  pthread_mutex_trylock           ";
  name[StatInt_pthread_mutex_timedlock]  = "  pthread_mutex_timedlock         ";
  name[StatInt_pthread_mutex_unlock]     = "  pthread_mutex_unlock            ";
  name[StatInt_pthread_spin_init]        = "  pthread_spin_init               ";
  name[StatInt_pthread_spin_destroy]     = "  pthread_spin_destroy            ";
  name[StatInt_pthread_spin_lock]        = "  pthread_spin_lock               ";
  name[StatInt_pthread_spin_trylock]     = "  pthread_spin_trylock            ";
  name[StatInt_pthread_spin_unlock]      = "  pthread_spin_unlock             ";
  name[StatInt_pthread_rwlock_init]      = "  pthread_rwlock_init             ";
  name[StatInt_pthread_rwlock_destroy]   = "  pthread_rwlock_destroy          ";
  name[StatInt_pthread_rwlock_rdlock]    = "  pthread_rwlock_rdlock           ";
  name[StatInt_pthread_rwlock_tryrdlock] = "  pthread_rwlock_tryrdlock        ";
  name[StatInt_pthread_rwlock_timedrdlock]
                                         = "  pthread_rwlock_timedrdlock      ";
  name[StatInt_pthread_rwlock_wrlock]    = "  pthread_rwlock_wrlock           ";
  name[StatInt_pthread_rwlock_trywrlock] = "  pthread_rwlock_trywrlock        ";
  name[StatInt_pthread_rwlock_timedwrlock]
                                         = "  pthread_rwlock_timedwrlock      ";
  name[StatInt_pthread_rwlock_unlock]    = "  pthread_rwlock_unlock           ";
  name[StatInt_pthread_cond_init]        = "  pthread_cond_init               ";
  name[StatInt_pthread_cond_destroy]     = "  pthread_cond_destroy            ";
  name[StatInt_pthread_cond_signal]      = "  pthread_cond_signal             ";
  name[StatInt_pthread_cond_broadcast]   = "  pthread_cond_broadcast          ";
  name[StatInt_pthread_cond_wait]        = "  pthread_cond_wait               ";
  name[StatInt_pthread_cond_timedwait]   = "  pthread_cond_timedwait          ";
  name[StatInt_pthread_barrier_init]     = "  pthread_barrier_init            ";
  name[StatInt_pthread_barrier_destroy]  = "  pthread_barrier_destroy         ";
  name[StatInt_pthread_barrier_wait]     = "  pthread_barrier_wait            ";
  name[StatInt_pthread_once]             = "  pthread_once                    ";
  name[StatInt_sem_init]                 = "  sem_init                        ";
  name[StatInt_sem_destroy]              = "  sem_destroy                     ";
  name[StatInt_sem_wait]                 = "  sem_wait                        ";
  name[StatInt_sem_trywait]              = "  sem_trywait                     ";
  name[StatInt_sem_timedwait]            = "  sem_timedwait                   ";
  name[StatInt_sem_post]                 = "  sem_post                        ";
  name[StatInt_sem_getvalue]             = "  sem_getvalue                    ";
  name[StatInt_read]                     = "  read                            ";
  name[StatInt_pread]                    = "  pread                           ";
  name[StatInt_pread64]                  = "  pread64                         ";
  name[StatInt_readv]                    = "  readv                           ";
  name[StatInt_preadv64]                 = "  preadv64                        ";
  name[StatInt_write]                    = "  write                           ";
  name[StatInt_pwrite]                   = "  pwrite                          ";
  name[StatInt_pwrite64]                 = "  pwrite64                        ";
  name[StatInt_writev]                   = "  writev                          ";
  name[StatInt_pwritev64]                = "  pwritev64                       ";
  name[StatInt_send]                     = "  send                            ";
  name[StatInt_sendmsg]                  = "  sendmsg                         ";
  name[StatInt_recv]                     = "  recv                            ";
  name[StatInt_recvmsg]                  = "  recvmsg                         ";
  name[StatInt_unlink]                   = "  unlink                          ";
  name[StatInt_fopen]                    = "  fopen                           ";
  name[StatInt_fread]                    = "  fread                           ";
  name[StatInt_fwrite]                   = "  fwrite                          ";
  name[StatInt_puts]                     = "  puts                            ";
  name[StatInt_rmdir]                    = "  rmdir                           ";
  name[StatInt_opendir]                  = "  opendir                         ";
  name[StatInt_epoll_ctl]                = "  epoll_ctl                       ";
  name[StatInt_epoll_wait]               = "  epoll_wait                      ";
  name[StatInt_sigaction]                = "  sigaction                       ";

  name[StatAnnotation]                   = "Dynamic annotations               ";
  name[StatAnnotateHappensBefore]        = "  HappensBefore                   ";
  name[StatAnnotateHappensAfter]         = "  HappensAfter                    ";
  name[StatAnnotateCondVarSignal]        = "  CondVarSignal                   ";
  name[StatAnnotateCondVarSignalAll]     = "  CondVarSignalAll                ";
  name[StatAnnotateMutexIsNotPHB]        = "  MutexIsNotPHB                   ";
  name[StatAnnotateCondVarWait]          = "  CondVarWait                     ";
  name[StatAnnotateRWLockCreate]         = "  RWLockCreate                    ";
  name[StatAnnotateRWLockDestroy]        = "  RWLockDestroy                   ";
  name[StatAnnotateRWLockAcquired]       = "  RWLockAcquired                  ";
  name[StatAnnotateRWLockReleased]       = "  RWLockReleased                  ";
  name[StatAnnotateTraceMemory]          = "  TraceMemory                     ";
  name[StatAnnotateFlushState]           = "  FlushState                      ";
  name[StatAnnotateNewMemory]            = "  NewMemory                       ";
  name[StatAnnotateNoOp]                 = "  NoOp                            ";
  name[StatAnnotateFlushExpectedRaces]   = "  FlushExpectedRaces              ";
  name[StatAnnotateEnableRaceDetection]  = "  EnableRaceDetection             ";
  name[StatAnnotateMutexIsUsedAsCondVar] = "  MutexIsUsedAsCondVar            ";
  name[StatAnnotatePCQGet]               = "  PCQGet                          ";
  name[StatAnnotatePCQPut]               = "  PCQPut                          ";
  name[StatAnnotatePCQDestroy]           = "  PCQDestroy                      ";
  name[StatAnnotatePCQCreate]            = "  PCQCreate                       ";
  name[StatAnnotateExpectRace]           = "  ExpectRace                      ";
  name[StatAnnotateBenignRaceSized]      = "  BenignRaceSized                 ";
  name[StatAnnotateBenignRace]           = "  BenignRace                      ";
  name[StatAnnotateIgnoreReadsBegin]     = "  IgnoreReadsBegin                ";
  name[StatAnnotateIgnoreReadsEnd]       = "  IgnoreReadsEnd                  ";
  name[StatAnnotateIgnoreWritesBegin]    = "  IgnoreWritesBegin               ";
  name[StatAnnotateIgnoreWritesEnd]      = "  IgnoreWritesEnd                 ";
  name[StatAnnotatePublishMemoryRange]   = "  PublishMemoryRange              ";
  name[StatAnnotateUnpublishMemoryRange] = "  UnpublishMemoryRange            ";
  name[StatAnnotateThreadName]           = "  ThreadName                      ";

  name[StatMtxTotal]                     = "Contentionz                       ";
  name[StatMtxTrace]                     = "  Trace                           ";
  name[StatMtxThreads]                   = "  Threads                         ";
  name[StatMtxReport]                    = "  Report                          ";
  name[StatMtxSyncVar]                   = "  SyncVar                         ";
  name[StatMtxSyncTab]                   = "  SyncTab                         ";
  name[StatMtxSlab]                      = "  Slab                            ";
  name[StatMtxAtExit]                    = "  Atexit                          ";
  name[StatMtxAnnotations]               = "  Annotations                     ";

  TsanPrintf("Statistics:\n");
  for (int i = 0; i < StatCnt; i++)
    TsanPrintf("%s: %llu\n", name[i], stat[i]);
}

}  // namespace __tsan
