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
  name[StatMopRodata]                    = "  Including .rodata               ";
  name[StatMopRangeRodata]               = "  Including .rodata range         ";
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
  name[StatAtomicFetchSub]               = "            fetch_sub             ";
  name[StatAtomicFetchAnd]               = "            fetch_and             ";
  name[StatAtomicFetchOr]                = "            fetch_or              ";
  name[StatAtomicFetchXor]               = "            fetch_xor             ";
  name[StatAtomicFetchNand]              = "            fetch_nand            ";
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
  name[StatAtomic16]                     = "            size 16               ";

  name[StatInterceptor]                  = "Interceptors                      ";
  name[StatInt_longjmp]                  = "  longjmp                         ";
  name[StatInt_siglongjmp]               = "  siglongjmp                      ";
  name[StatInt_malloc]                   = "  malloc                          ";
  name[StatInt___libc_memalign]          = "  __libc_memalign                 ";
  name[StatInt_calloc]                   = "  calloc                          ";
  name[StatInt_realloc]                  = "  realloc                         ";
  name[StatInt_free]                     = "  free                            ";
  name[StatInt_cfree]                    = "  cfree                           ";
  name[StatInt_malloc_usable_size]       = "  malloc_usable_size              ";
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
  name[StatInt_strcasecmp]               = "  strcasecmp                      ";
  name[StatInt_strncasecmp]              = "  strncasecmp                     ";
  name[StatInt_atexit]                   = "  atexit                          ";
  name[StatInt___cxa_guard_acquire]      = "  __cxa_guard_acquire             ";
  name[StatInt___cxa_guard_release]      = "  __cxa_guard_release             ";
  name[StatInt___cxa_guard_abort]        = "  __cxa_guard_abort               ";
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
  name[StatInt_pthread_getschedparam]    = "  pthread_getschedparam           ";
  name[StatInt_sem_init]                 = "  sem_init                        ";
  name[StatInt_sem_destroy]              = "  sem_destroy                     ";
  name[StatInt_sem_wait]                 = "  sem_wait                        ";
  name[StatInt_sem_trywait]              = "  sem_trywait                     ";
  name[StatInt_sem_timedwait]            = "  sem_timedwait                   ";
  name[StatInt_sem_post]                 = "  sem_post                        ";
  name[StatInt_sem_getvalue]             = "  sem_getvalue                    ";
  name[StatInt_stat]                     = "  stat                            ";
  name[StatInt___xstat]                  = "  __xstat                         ";
  name[StatInt_stat64]                   = "  stat64                          ";
  name[StatInt___xstat64]                = "  __xstat64                       ";
  name[StatInt_lstat]                    = "  lstat                           ";
  name[StatInt___lxstat]                 = "  __lxstat                        ";
  name[StatInt_lstat64]                  = "  lstat64                         ";
  name[StatInt___lxstat64]               = "  __lxstat64                      ";
  name[StatInt_fstat]                    = "  fstat                           ";
  name[StatInt___fxstat]                 = "  __fxstat                        ";
  name[StatInt_fstat64]                  = "  fstat64                         ";
  name[StatInt___fxstat64]               = "  __fxstat64                      ";
  name[StatInt_open]                     = "  open                            ";
  name[StatInt_open64]                   = "  open64                          ";
  name[StatInt_creat]                    = "  creat                           ";
  name[StatInt_creat64]                  = "  creat64                         ";
  name[StatInt_dup]                      = "  dup                             ";
  name[StatInt_dup2]                     = "  dup2                            ";
  name[StatInt_dup3]                     = "  dup3                            ";
  name[StatInt_eventfd]                  = "  eventfd                         ";
  name[StatInt_signalfd]                 = "  signalfd                        ";
  name[StatInt_inotify_init]             = "  inotify_init                    ";
  name[StatInt_inotify_init1]            = "  inotify_init1                   ";
  name[StatInt_socket]                   = "  socket                          ";
  name[StatInt_socketpair]               = "  socketpair                      ";
  name[StatInt_connect]                  = "  connect                         ";
  name[StatInt_bind]                     = "  bind                            ";
  name[StatInt_listen]                   = "  listen                          ";
  name[StatInt_accept]                   = "  accept                          ";
  name[StatInt_accept4]                  = "  accept4                         ";
  name[StatInt_epoll_create]             = "  epoll_create                    ";
  name[StatInt_epoll_create1]            = "  epoll_create1                   ";
  name[StatInt_close]                    = "  close                           ";
  name[StatInt___close]                  = "  __close                         ";
  name[StatInt___res_iclose]             = "  __res_iclose                    ";
  name[StatInt_pipe]                     = "  pipe                            ";
  name[StatInt_pipe2]                    = "  pipe2                           ";
  name[StatInt_read]                     = "  read                            ";
  name[StatInt_prctl]                    = "  prctl                           ";
  name[StatInt_pread]                    = "  pread                           ";
  name[StatInt_pread64]                  = "  pread64                         ";
  name[StatInt_readv]                    = "  readv                           ";
  name[StatInt_preadv]                   = "  preadv                          ";
  name[StatInt_preadv64]                 = "  preadv64                        ";
  name[StatInt_write]                    = "  write                           ";
  name[StatInt_pwrite]                   = "  pwrite                          ";
  name[StatInt_pwrite64]                 = "  pwrite64                        ";
  name[StatInt_writev]                   = "  writev                          ";
  name[StatInt_pwritev]                  = "  pwritev                         ";
  name[StatInt_pwritev64]                = "  pwritev64                       ";
  name[StatInt_send]                     = "  send                            ";
  name[StatInt_sendmsg]                  = "  sendmsg                         ";
  name[StatInt_recv]                     = "  recv                            ";
  name[StatInt_recvmsg]                  = "  recvmsg                         ";
  name[StatInt_unlink]                   = "  unlink                          ";
  name[StatInt_fopen]                    = "  fopen                           ";
  name[StatInt_freopen]                  = "  freopen                         ";
  name[StatInt_fclose]                   = "  fclose                          ";
  name[StatInt_fread]                    = "  fread                           ";
  name[StatInt_fwrite]                   = "  fwrite                          ";
  name[StatInt_fflush]                   = "  fflush                          ";
  name[StatInt_abort]                    = "  abort                           ";
  name[StatInt_puts]                     = "  puts                            ";
  name[StatInt_rmdir]                    = "  rmdir                           ";
  name[StatInt_opendir]                  = "  opendir                         ";
  name[StatInt_epoll_ctl]                = "  epoll_ctl                       ";
  name[StatInt_epoll_wait]               = "  epoll_wait                      ";
  name[StatInt_poll]                     = "  poll                            ";
  name[StatInt_sigaction]                = "  sigaction                       ";
  name[StatInt_signal]                   = "  signal                          ";
  name[StatInt_sigsuspend]               = "  sigsuspend                      ";
  name[StatInt_raise]                    = "  raise                           ";
  name[StatInt_kill]                     = "  kill                            ";
  name[StatInt_pthread_kill]             = "  pthread_kill                    ";
  name[StatInt_sleep]                    = "  sleep                           ";
  name[StatInt_usleep]                   = "  usleep                          ";
  name[StatInt_nanosleep]                = "  nanosleep                       ";
  name[StatInt_gettimeofday]             = "  gettimeofday                    ";
  name[StatInt_fork]                     = "  fork                            ";
  name[StatInt_vscanf]                   = "  vscanf                          ";
  name[StatInt_vsscanf]                  = "  vsscanf                         ";
  name[StatInt_vfscanf]                  = "  vfscanf                         ";
  name[StatInt_scanf]                    = "  scanf                           ";
  name[StatInt_sscanf]                   = "  sscanf                          ";
  name[StatInt_fscanf]                   = "  fscanf                          ";
  name[StatInt___isoc99_vscanf]          = "  vscanf                          ";
  name[StatInt___isoc99_vsscanf]         = "  vsscanf                         ";
  name[StatInt___isoc99_vfscanf]         = "  vfscanf                         ";
  name[StatInt___isoc99_scanf]           = "  scanf                           ";
  name[StatInt___isoc99_sscanf]          = "  sscanf                          ";
  name[StatInt___isoc99_fscanf]          = "  fscanf                          ";
  name[StatInt_on_exit]                  = "  on_exit                         ";
  name[StatInt___cxa_atexit]             = "  __cxa_atexit                    ";
  name[StatInt_localtime]                = "  localtime                       ";
  name[StatInt_localtime_r]              = "  localtime_r                     ";
  name[StatInt_gmtime]                   = "  gmtime                          ";
  name[StatInt_gmtime_r]                 = "  gmtime_r                        ";
  name[StatInt_ctime]                    = "  ctime                           ";
  name[StatInt_ctime_r]                  = "  ctime_r                         ";
  name[StatInt_asctime]                  = "  asctime                         ";
  name[StatInt_asctime_r]                = "  asctime_r                       ";
  name[StatInt_frexp]                    = "  frexp                           ";
  name[StatInt_frexpf]                   = "  frexpf                          ";
  name[StatInt_frexpl]                   = "  frexpl                          ";
  name[StatInt_getpwnam]                 = "  getpwnam                        ";
  name[StatInt_getpwuid]                 = "  getpwuid                        ";
  name[StatInt_getgrnam]                 = "  getgrnam                        ";
  name[StatInt_getgrgid]                 = "  getgrgid                        ";
  name[StatInt_getpwnam_r]               = "  getpwnam_r                      ";
  name[StatInt_getpwuid_r]               = "  getpwuid_r                      ";
  name[StatInt_getgrnam_r]               = "  getgrnam_r                      ";
  name[StatInt_getgrgid_r]               = "  getgrgid_r                      ";
  name[StatInt_clock_getres]             = "  clock_getres                    ";
  name[StatInt_clock_gettime]            = "  clock_gettime                   ";
  name[StatInt_clock_settime]            = "  clock_settime                   ";
  name[StatInt_getitimer]                = "  getitimer                       ";
  name[StatInt_setitimer]                = "  setitimer                       ";
  name[StatInt_time]                     = "  time                            ";
  name[StatInt_glob]                     = "  glob                            ";
  name[StatInt_glob64]                   = "  glob64                          ";
  name[StatInt_wait]                     = "  wait                            ";
  name[StatInt_waitid]                   = "  waitid                          ";
  name[StatInt_waitpid]                  = "  waitpid                         ";
  name[StatInt_wait3]                    = "  wait3                           ";
  name[StatInt_wait4]                    = "  wait4                           ";
  name[StatInt_inet_ntop]                = "  inet_ntop                       ";
  name[StatInt_inet_pton]                = "  inet_pton                       ";
  name[StatInt_inet_aton]                = "  inet_aton                       ";
  name[StatInt_getaddrinfo]              = "  getaddrinfo                     ";
  name[StatInt_getnameinfo]              = "  getnameinfo                     ";
  name[StatInt_getsockname]              = "  getsockname                     ";
  name[StatInt_gethostent]               = "  gethostent                      ";
  name[StatInt_gethostbyname]            = "  gethostbyname                   ";
  name[StatInt_gethostbyname2]           = "  gethostbyname2                  ";
  name[StatInt_gethostbyaddr]            = "  gethostbyaddr                   ";
  name[StatInt_gethostent_r]             = "  gethostent_r                    ";
  name[StatInt_gethostbyname_r]          = "  gethostbyname_r                 ";
  name[StatInt_gethostbyname2_r]         = "  gethostbyname2_r                ";
  name[StatInt_gethostbyaddr_r]          = "  gethostbyaddr_r                 ";
  name[StatInt_getsockopt]               = "  getsockopt                      ";
  name[StatInt_modf]                     = "  modf                            ";
  name[StatInt_modff]                    = "  modff                           ";
  name[StatInt_modfl]                    = "  modfl                           ";
  name[StatInt_getpeername]              = "  getpeername                     ";
  name[StatInt_ioctl]                    = "  ioctl                           ";
  name[StatInt_sysinfo]                  = "  sysinfo                         ";
  name[StatInt_readdir]                  = "  readdir                         ";
  name[StatInt_readdir64]                = "  readdir64                       ";
  name[StatInt_readdir_r]                = "  readdir_r                       ";
  name[StatInt_readdir64_r]              = "  readdir64_r                     ";
  name[StatInt_ptrace]                   = "  ptrace                          ";
  name[StatInt_setlocale]                = "  setlocale                       ";
  name[StatInt_getcwd]                   = "  getcwd                          ";
  name[StatInt_get_current_dir_name]     = "  get_current_dir_name            ";
  name[StatInt_strtoimax]                = "  strtoimax                       ";
  name[StatInt_strtoumax]                = "  strtoumax                       ";
  name[StatInt_mbstowcs]                 = "  mbstowcs                        ";
  name[StatInt_mbsrtowcs]                = "  mbsrtowcs                       ";
  name[StatInt_mbsnrtowcs]               = "  mbsnrtowcs                      ";
  name[StatInt_wcstombs]                 = "  wcstombs                        ";
  name[StatInt_wcsrtombs]                = "  wcsrtombs                       ";
  name[StatInt_wcsnrtombs]               = "  wcsnrtombs                      ";
  name[StatInt_tcgetattr]                = "  tcgetattr                       ";
  name[StatInt_realpath]                 = "  realpath                        ";
  name[StatInt_canonicalize_file_name]   = "  canonicalize_file_name          ";

  name[StatAnnotation]                   = "Dynamic annotations               ";
  name[StatAnnotateHappensBefore]        = "  HappensBefore                   ";
  name[StatAnnotateHappensAfter]         = "  HappensAfter                    ";
  name[StatAnnotateCondVarSignal]        = "  CondVarSignal                   ";
  name[StatAnnotateCondVarSignalAll]     = "  CondVarSignalAll                ";
  name[StatAnnotateMutexIsNotPHB]        = "  MutexIsNotPHB                   ";
  name[StatAnnotateCondVarWait]          = "  CondVarWait                     ";
  name[StatAnnotateRWLockCreate]         = "  RWLockCreate                    ";
  name[StatAnnotateRWLockCreateStatic]   = "  StatAnnotateRWLockCreateStatic  ";
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
  name[StatMtxMBlock]                    = "  MBlock                          ";
  name[StatMtxJavaMBlock]                = "  JavaMBlock                      ";
  name[StatMtxFD]                        = "  FD                              ";

  Printf("Statistics:\n");
  for (int i = 0; i < StatCnt; i++)
    Printf("%s: %zu\n", name[i], (uptr)stat[i]);
}

}  // namespace __tsan
