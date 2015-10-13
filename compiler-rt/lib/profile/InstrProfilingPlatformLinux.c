/*===- InstrProfilingPlatformLinux.c - Profile data Linux platform ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

#if defined(__linux__)
#include <stdlib.h>

extern __llvm_profile_data __start___llvm_prf_data
    __attribute__((visibility("hidden")));
extern __llvm_profile_data __stop___llvm_prf_data
    __attribute__((visibility("hidden")));
extern uint64_t __start___llvm_prf_cnts __attribute__((visibility("hidden")));
extern uint64_t __stop___llvm_prf_cnts __attribute__((visibility("hidden")));
extern char __start___llvm_prf_names __attribute__((visibility("hidden")));
extern char __stop___llvm_prf_names __attribute__((visibility("hidden")));

__attribute__((visibility("hidden"))) const __llvm_profile_data *
__llvm_profile_begin_data(void) {
  return &__start___llvm_prf_data;
}
__attribute__((visibility("hidden"))) const __llvm_profile_data *
__llvm_profile_end_data(void) {
  return &__stop___llvm_prf_data;
}
__attribute__((visibility("hidden"))) const char *__llvm_profile_begin_names(
    void) {
  return &__start___llvm_prf_names;
}
__attribute__((visibility("hidden"))) const char *__llvm_profile_end_names(
    void) {
  return &__stop___llvm_prf_names;
}
__attribute__((visibility("hidden"))) uint64_t *__llvm_profile_begin_counters(
    void) {
  return &__start___llvm_prf_cnts;
}
__attribute__((visibility("hidden"))) uint64_t *__llvm_profile_end_counters(
    void) {
  return &__stop___llvm_prf_cnts;
}
#endif
