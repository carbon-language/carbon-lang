/*
 * kmp_stub.h
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KMP_STUB_H
#define KMP_STUB_H

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void __kmps_set_blocktime(int arg);
int __kmps_get_blocktime(void);
void __kmps_set_dynamic(int arg);
int __kmps_get_dynamic(void);
void __kmps_set_library(int arg);
int __kmps_get_library(void);
void __kmps_set_nested(int arg);
int __kmps_get_nested(void);
void __kmps_set_stacksize(size_t arg);
size_t __kmps_get_stacksize();

#ifndef KMP_SCHED_TYPE_DEFINED
#define KMP_SCHED_TYPE_DEFINED
typedef enum kmp_sched {
  kmp_sched_static = 1, // mapped to kmp_sch_static_chunked           (33)
  kmp_sched_dynamic = 2, // mapped to kmp_sch_dynamic_chunked          (35)
  kmp_sched_guided = 3, // mapped to kmp_sch_guided_chunked           (36)
  kmp_sched_auto = 4, // mapped to kmp_sch_auto                     (38)
  kmp_sched_default = kmp_sched_static // default scheduling
} kmp_sched_t;
#endif
void __kmps_set_schedule(kmp_sched_t kind, int modifier);
void __kmps_get_schedule(kmp_sched_t *kind, int *modifier);

kmp_proc_bind_t __kmps_get_proc_bind(void);

double __kmps_get_wtime();
double __kmps_get_wtick();

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // KMP_STUB_H

// end of file //
