/*
 * kmp_stub.cpp -- stub versions of user-callable OpenMP RT functions.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <limits.h>
#include <stdlib.h>

#define __KMP_IMP
#include "omp.h" // omp_* declarations, must be included before "kmp.h"
#include "kmp.h" // KMP_DEFAULT_STKSIZE
#include "kmp_stub.h"

#if KMP_OS_WINDOWS
#include <windows.h>
#else
#include <sys/time.h>
#endif

// Moved from omp.h
#define omp_set_max_active_levels ompc_set_max_active_levels
#define omp_set_schedule ompc_set_schedule
#define omp_get_ancestor_thread_num ompc_get_ancestor_thread_num
#define omp_get_team_size ompc_get_team_size

#define omp_set_num_threads ompc_set_num_threads
#define omp_set_dynamic ompc_set_dynamic
#define omp_set_nested ompc_set_nested
#define omp_set_affinity_format ompc_set_affinity_format
#define omp_get_affinity_format ompc_get_affinity_format
#define omp_display_affinity ompc_display_affinity
#define omp_capture_affinity ompc_capture_affinity
#define kmp_set_stacksize kmpc_set_stacksize
#define kmp_set_stacksize_s kmpc_set_stacksize_s
#define kmp_set_blocktime kmpc_set_blocktime
#define kmp_set_library kmpc_set_library
#define kmp_set_defaults kmpc_set_defaults
#define kmp_set_disp_num_buffers kmpc_set_disp_num_buffers
#define kmp_malloc kmpc_malloc
#define kmp_aligned_malloc kmpc_aligned_malloc
#define kmp_calloc kmpc_calloc
#define kmp_realloc kmpc_realloc
#define kmp_free kmpc_free

#if KMP_OS_WINDOWS
static double frequency = 0.0;
#endif

// Helper functions.
static size_t __kmps_init() {
  static int initialized = 0;
  static size_t dummy = 0;
  if (!initialized) {
    // TODO: Analyze KMP_VERSION environment variable, print
    // __kmp_version_copyright and __kmp_version_build_time.
    // WARNING: Do not use "fprintf(stderr, ...)" because it will cause
    // unresolved "__iob" symbol (see C70080). We need to extract __kmp_printf()
    // stuff from kmp_runtime.cpp and use it.

    // Trick with dummy variable forces linker to keep __kmp_version_copyright
    // and __kmp_version_build_time strings in executable file (in case of
    // static linkage). When KMP_VERSION analysis is implemented, dummy
    // variable should be deleted, function should return void.
    dummy = __kmp_version_copyright - __kmp_version_build_time;

#if KMP_OS_WINDOWS
    LARGE_INTEGER freq;
    BOOL status = QueryPerformanceFrequency(&freq);
    if (status) {
      frequency = double(freq.QuadPart);
    }
#endif

    initialized = 1;
  }
  return dummy;
} // __kmps_init

#define i __kmps_init();

/* set API functions */
void omp_set_num_threads(omp_int_t num_threads) { i; }
void omp_set_dynamic(omp_int_t dynamic) {
  i;
  __kmps_set_dynamic(dynamic);
}
void omp_set_nested(omp_int_t nested) {
  i;
  __kmps_set_nested(nested);
}
void omp_set_max_active_levels(omp_int_t max_active_levels) { i; }
void omp_set_schedule(omp_sched_t kind, omp_int_t modifier) {
  i;
  __kmps_set_schedule((kmp_sched_t)kind, modifier);
}
int omp_get_ancestor_thread_num(omp_int_t level) {
  i;
  return (level) ? (-1) : (0);
}
int omp_get_team_size(omp_int_t level) {
  i;
  return (level) ? (-1) : (1);
}
int kmpc_set_affinity_mask_proc(int proc, void **mask) {
  i;
  return -1;
}
int kmpc_unset_affinity_mask_proc(int proc, void **mask) {
  i;
  return -1;
}
int kmpc_get_affinity_mask_proc(int proc, void **mask) {
  i;
  return -1;
}

/* kmp API functions */
void kmp_set_stacksize(omp_int_t arg) {
  i;
  __kmps_set_stacksize((size_t)arg);
}
void kmp_set_stacksize_s(size_t arg) {
  i;
  __kmps_set_stacksize(arg);
}
void kmp_set_blocktime(omp_int_t arg) {
  i;
  __kmps_set_blocktime(arg);
}
void kmp_set_library(omp_int_t arg) {
  i;
  __kmps_set_library(arg);
}
void kmp_set_defaults(char const *str) { i; }
void kmp_set_disp_num_buffers(omp_int_t arg) { i; }

/* KMP memory management functions. */
void *kmp_malloc(size_t size) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  // If successful returns a pointer to the memory block, otherwise returns
  // NULL.
  // Sets errno to ENOMEM or EINVAL if memory allocation failed or parameter
  // validation failed.
  res = _aligned_malloc(size, 1);
#else
  res = malloc(size);
#endif
  return res;
}
void *kmp_aligned_malloc(size_t sz, size_t a) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_malloc(sz, a);
#else
  int err;
  if ((err = posix_memalign(&res, a, sz))) {
    errno = err; // can be EINVAL or ENOMEM
    res = NULL;
  }
#endif
  return res;
}
void *kmp_calloc(size_t nelem, size_t elsize) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_recalloc(NULL, nelem, elsize, 1);
#else
  res = calloc(nelem, elsize);
#endif
  return res;
}
void *kmp_realloc(void *ptr, size_t size) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_realloc(ptr, size, 1);
#else
  res = realloc(ptr, size);
#endif
  return res;
}
void kmp_free(void *ptr) {
  i;
#if KMP_OS_WINDOWS
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

static int __kmps_blocktime = INT_MAX;

void __kmps_set_blocktime(int arg) {
  i;
  __kmps_blocktime = arg;
} // __kmps_set_blocktime

int __kmps_get_blocktime(void) {
  i;
  return __kmps_blocktime;
} // __kmps_get_blocktime

static int __kmps_dynamic = 0;

void __kmps_set_dynamic(int arg) {
  i;
  __kmps_dynamic = arg;
} // __kmps_set_dynamic

int __kmps_get_dynamic(void) {
  i;
  return __kmps_dynamic;
} // __kmps_get_dynamic

static int __kmps_library = 1000;

void __kmps_set_library(int arg) {
  i;
  __kmps_library = arg;
} // __kmps_set_library

int __kmps_get_library(void) {
  i;
  return __kmps_library;
} // __kmps_get_library

static int __kmps_nested = 0;

void __kmps_set_nested(int arg) {
  i;
  __kmps_nested = arg;
} // __kmps_set_nested

int __kmps_get_nested(void) {
  i;
  return __kmps_nested;
} // __kmps_get_nested

static size_t __kmps_stacksize = KMP_DEFAULT_STKSIZE;

void __kmps_set_stacksize(size_t arg) {
  i;
  __kmps_stacksize = arg;
} // __kmps_set_stacksize

size_t __kmps_get_stacksize(void) {
  i;
  return __kmps_stacksize;
} // __kmps_get_stacksize

static kmp_sched_t __kmps_sched_kind = kmp_sched_default;
static int __kmps_sched_modifier = 0;

void __kmps_set_schedule(kmp_sched_t kind, int modifier) {
  i;
  __kmps_sched_kind = kind;
  __kmps_sched_modifier = modifier;
} // __kmps_set_schedule

void __kmps_get_schedule(kmp_sched_t *kind, int *modifier) {
  i;
  *kind = __kmps_sched_kind;
  *modifier = __kmps_sched_modifier;
} // __kmps_get_schedule

kmp_proc_bind_t __kmps_get_proc_bind(void) {
  i;
  return proc_bind_false;
} // __kmps_get_proc_bind

double __kmps_get_wtime(void) {
  // Elapsed wall clock time (in second) from "sometime in the past".
  double wtime = 0.0;
  i;
#if KMP_OS_WINDOWS
  if (frequency > 0.0) {
    LARGE_INTEGER now;
    BOOL status = QueryPerformanceCounter(&now);
    if (status) {
      wtime = double(now.QuadPart) / frequency;
    }
  }
#else
  // gettimeofday() returns seconds and microseconds since the Epoch.
  struct timeval tval;
  int rc;
  rc = gettimeofday(&tval, NULL);
  if (rc == 0) {
    wtime = (double)(tval.tv_sec) + 1.0E-06 * (double)(tval.tv_usec);
  } else {
    // TODO: Assert or abort here.
  }
#endif
  return wtime;
} // __kmps_get_wtime

double __kmps_get_wtick(void) {
  // Number of seconds between successive clock ticks.
  double wtick = 0.0;
  i;
#if KMP_OS_WINDOWS
  {
    DWORD increment;
    DWORD adjustment;
    BOOL disabled;
    BOOL rc;
    rc = GetSystemTimeAdjustment(&adjustment, &increment, &disabled);
    if (rc) {
      wtick = 1.0E-07 * (double)(disabled ? increment : adjustment);
    } else {
      // TODO: Assert or abort here.
      wtick = 1.0E-03;
    }
  }
#else
  // TODO: gettimeofday() returns in microseconds, but what the precision?
  wtick = 1.0E-06;
#endif
  return wtick;
} // __kmps_get_wtick

/* OpenMP 5.0 Memory Management */
#if KMP_OS_WINDOWS
omp_allocator_handle_t const omp_null_allocator = 0;
omp_allocator_handle_t const omp_default_mem_alloc =
    (omp_allocator_handle_t const)1;
omp_allocator_handle_t const omp_large_cap_mem_alloc =
    (omp_allocator_handle_t const)2;
omp_allocator_handle_t const omp_const_mem_alloc =
    (omp_allocator_handle_t const)3;
omp_allocator_handle_t const omp_high_bw_mem_alloc =
    (omp_allocator_handle_t const)4;
omp_allocator_handle_t const omp_low_lat_mem_alloc =
    (omp_allocator_handle_t const)5;
omp_allocator_handle_t const omp_cgroup_mem_alloc =
    (omp_allocator_handle_t const)6;
omp_allocator_handle_t const omp_pteam_mem_alloc =
    (omp_allocator_handle_t const)7;
omp_allocator_handle_t const omp_thread_mem_alloc =
    (omp_allocator_handle_t const)8;
// Preview of target memory support
omp_allocator_handle_t const llvm_omp_target_host_mem_alloc =
    (omp_allocator_handle_t const)100;
omp_allocator_handle_t const llvm_omp_target_shared_mem_alloc =
    (omp_allocator_handle_t const)101;
omp_allocator_handle_t const llvm_omp_target_device_mem_alloc =
    (omp_allocator_handle_t const)102;

omp_memspace_handle_t const omp_default_mem_space =
    (omp_memspace_handle_t const)0;
omp_memspace_handle_t const omp_large_cap_mem_space =
    (omp_memspace_handle_t const)1;
omp_memspace_handle_t const omp_const_mem_space =
    (omp_memspace_handle_t const)2;
omp_memspace_handle_t const omp_high_bw_mem_space =
    (omp_memspace_handle_t const)3;
omp_memspace_handle_t const omp_low_lat_mem_space =
    (omp_memspace_handle_t const)4;
// Preview of target memory support
omp_memspace_handle_t const llvm_omp_target_host_mem_space =
    (omp_memspace_handle_t const)100;
omp_memspace_handle_t const llvm_omp_target_shared_mem_space =
    (omp_memspace_handle_t const)101;
omp_memspace_handle_t const llvm_omp_target_device_mem_space =
    (omp_memspace_handle_t const)102;
#endif /* KMP_OS_WINDOWS */

void *omp_alloc(size_t size, omp_allocator_handle_t allocator) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  // Returns a pointer to the memory block, or NULL if failed.
  // Sets errno to ENOMEM or EINVAL if memory allocation failed or parameter
  // validation failed.
  res = _aligned_malloc(size, 1);
#else
  res = malloc(size);
#endif
  return res;
}

void *omp_aligned_alloc(size_t a, size_t size, omp_allocator_handle_t al) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_malloc(size, a);
#else
  int err;
  if ((err = posix_memalign(&res, a, size))) {
    errno = err; // can be EINVAL or ENOMEM
    res = NULL;
  }
#endif
  return res;
}

void *omp_calloc(size_t nmemb, size_t size, omp_allocator_handle_t al) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_recalloc(NULL, nmemb, size, 1);
#else
  res = calloc(nmemb, size);
#endif
  return res;
}

void *omp_aligned_calloc(size_t a, size_t nmemb, size_t size,
                         omp_allocator_handle_t al) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_recalloc(NULL, nmemb, size, a);
#else
  int err;
  if ((err = posix_memalign(&res, a, nmemb * size))) {
    errno = err; // can be EINVAL or ENOMEM
    res = NULL;
  }
  memset(res, 0x00, size);
#endif
  return res;
}

void *omp_realloc(void *ptr, size_t size, omp_allocator_handle_t al,
                  omp_allocator_handle_t free_al) {
  i;
  void *res;
#if KMP_OS_WINDOWS
  res = _aligned_realloc(ptr, size, 1);
#else
  res = realloc(ptr, size);
#endif
  return res;
}

void omp_free(void *ptr, omp_allocator_handle_t allocator) {
  i;
#if KMP_OS_WINDOWS
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/* OpenMP 5.0 Affinity Format */
void omp_set_affinity_format(char const *format) { i; }
size_t omp_get_affinity_format(char *buffer, size_t size) {
  i;
  return 0;
}
void omp_display_affinity(char const *format) { i; }
size_t omp_capture_affinity(char *buffer, size_t buf_size, char const *format) {
  i;
  return 0;
}

// end of file //
