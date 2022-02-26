//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _INTEROP_H_
#define _INTEROP_H_

#include "omptarget.h"
#include <assert.h>

#if defined(_WIN32)
#define __KAI_KMPC_CONVENTION __cdecl
#ifndef __KMP_IMP
#define __KMP_IMP __declspec(dllimport)
#endif
#else
#define __KAI_KMPC_CONVENTION
#ifndef __KMP_IMP
#define __KMP_IMP
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// TODO: Include the `omp.h` of the current build
/* OpenMP 5.1 interop */
typedef intptr_t omp_intptr_t;

/* 0..omp_get_num_interop_properties()-1 are reserved for implementation-defined
 * properties */
typedef enum omp_interop_property {
  omp_ipr_fr_id = -1,
  omp_ipr_fr_name = -2,
  omp_ipr_vendor = -3,
  omp_ipr_vendor_name = -4,
  omp_ipr_device_num = -5,
  omp_ipr_platform = -6,
  omp_ipr_device = -7,
  omp_ipr_device_context = -8,
  omp_ipr_targetsync = -9,
  omp_ipr_first = -9
} omp_interop_property_t;

#define omp_interop_none 0

typedef enum omp_interop_rc {
  omp_irc_no_value = 1,
  omp_irc_success = 0,
  omp_irc_empty = -1,
  omp_irc_out_of_range = -2,
  omp_irc_type_int = -3,
  omp_irc_type_ptr = -4,
  omp_irc_type_str = -5,
  omp_irc_other = -6
} omp_interop_rc_t;

typedef enum omp_interop_fr {
  omp_ifr_cuda = 1,
  omp_ifr_cuda_driver = 2,
  omp_ifr_opencl = 3,
  omp_ifr_sycl = 4,
  omp_ifr_hip = 5,
  omp_ifr_level_zero = 6,
  omp_ifr_last = 7
} omp_interop_fr_t;

typedef void *omp_interop_t;

/*!
 * The `omp_get_num_interop_properties` routine retrieves the number of
 * implementation-defined properties available for an `omp_interop_t` object.
 */
int __KAI_KMPC_CONVENTION omp_get_num_interop_properties(const omp_interop_t);
/*!
 * The `omp_get_interop_int` routine retrieves an integer property from an
 * `omp_interop_t` object.
 */
omp_intptr_t __KAI_KMPC_CONVENTION omp_get_interop_int(const omp_interop_t,
                                                       omp_interop_property_t,
                                                       int *);
/*!
 * The `omp_get_interop_ptr` routine retrieves a pointer property from an
 * `omp_interop_t` object.
 */
void *__KAI_KMPC_CONVENTION omp_get_interop_ptr(const omp_interop_t,
                                                omp_interop_property_t, int *);
/*!
 * The `omp_get_interop_str` routine retrieves a string property from an
 * `omp_interop_t` object.
 */
const char *__KAI_KMPC_CONVENTION omp_get_interop_str(const omp_interop_t,
                                                      omp_interop_property_t,
                                                      int *);
/*!
 * The `omp_get_interop_name` routine retrieves a property name from an
 * `omp_interop_t` object.
 */
const char *__KAI_KMPC_CONVENTION omp_get_interop_name(const omp_interop_t,
                                                       omp_interop_property_t);
/*!
 * The `omp_get_interop_type_desc` routine retrieves a description of the type
 * of a property associated with an `omp_interop_t` object.
 */
const char *__KAI_KMPC_CONVENTION
omp_get_interop_type_desc(const omp_interop_t, omp_interop_property_t);
/*!
 * The `omp_get_interop_rc_desc` routine retrieves a description of the return
 * code associated with an `omp_interop_t` object.
 */
extern const char *__KAI_KMPC_CONVENTION
omp_get_interop_rc_desc(const omp_interop_t, omp_interop_rc_t);

typedef struct kmp_tasking_flags { /* Total struct must be exactly 32 bits */
  /* Compiler flags */             /* Total compiler flags must be 16 bits */
  unsigned tiedness : 1;           /* task is either tied (1) or untied (0) */
  unsigned final : 1;              /* task is final(1) so execute immediately */
  unsigned merged_if0 : 1; // no __kmpc_task_{begin/complete}_if0 calls in if0
  unsigned destructors_thunk : 1; // set if the compiler creates a thunk to
  unsigned proxy : 1; // task is a proxy task (it will be executed outside the
  unsigned priority_specified : 1; // set if the compiler provides priority
  unsigned detachable : 1;         // 1 == can detach */
  unsigned unshackled : 1;         /* 1 == unshackled task */
  unsigned target : 1;             /* 1 == target task */
  unsigned reserved : 7;           /* reserved for compiler use */
  unsigned tasktype : 1;    /* task is either explicit(1) or implicit (0) */
  unsigned task_serial : 1; // task is executed immediately (1) or deferred (0)
  unsigned tasking_ser : 1; // all tasks in team are either executed immediately
  unsigned team_serial : 1; // entire team is serial (1) [1 thread] or parallel
  unsigned started : 1;     /* 1==started, 0==not started     */
  unsigned executing : 1;   /* 1==executing, 0==not executing */
  unsigned complete : 1;    /* 1==complete, 0==not complete   */
  unsigned freed : 1;       /* 1==freed, 0==allocated        */
  unsigned native : 1;      /* 1==gcc-compiled task, 0==intel */
  unsigned reserved31 : 7;  /* reserved for library use */
} kmp_tasking_flags_t;

typedef enum omp_interop_backend_type_t {
  // reserve 0
  omp_interop_backend_type_cuda_1 = 1,
} omp_interop_backend_type_t;

typedef enum kmp_interop_type_t {
  kmp_interop_type_unknown = -1,
  kmp_interop_type_platform,
  kmp_interop_type_device,
  kmp_interop_type_tasksync,
} kmp_interop_type_t;

typedef enum omp_foreign_runtime_ids {
  cuda = 1,
  cuda_driver = 2,
  opencl = 3,
  sycl = 4,
  hip = 5,
  level_zero = 6,
} omp_foreign_runtime_ids_t;

/// The interop value type, aka. the interop object.
typedef struct omp_interop_val_t {
  /// Device and interop-type are determined at construction time and fix.
  omp_interop_val_t(intptr_t device_id, kmp_interop_type_t interop_type)
      : interop_type(interop_type), device_id(device_id) {}
  const char *err_str = nullptr;
  __tgt_async_info *async_info = nullptr;
  __tgt_device_info device_info;
  const kmp_interop_type_t interop_type;
  const intptr_t device_id;
  const omp_foreign_runtime_ids_t vendor_id = cuda;
  const intptr_t backend_type_id = omp_interop_backend_type_cuda_1;
} omp_interop_val_t;

#ifdef __cplusplus
}
#endif
#endif
