/*===- InstrProfilingPlatformLinux.c - Profile data Linux platform ------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"

#if defined(__linux__) || defined(__FreeBSD__)
#include <stdlib.h>

extern __llvm_profile_data __start___llvm_prf_data LLVM_LIBRARY_VISIBILITY;
extern __llvm_profile_data __stop___llvm_prf_data LLVM_LIBRARY_VISIBILITY;
extern uint64_t __start___llvm_prf_cnts LLVM_LIBRARY_VISIBILITY;
extern uint64_t __stop___llvm_prf_cnts LLVM_LIBRARY_VISIBILITY;
extern char __start___llvm_prf_names LLVM_LIBRARY_VISIBILITY;
extern char __stop___llvm_prf_names LLVM_LIBRARY_VISIBILITY;

/* Add dummy data to ensure the section is always created. */
__llvm_profile_data __llvm_prof_sect_data[0] LLVM_SECTION("__llvm_prf_data");
uint64_t __llvm_prof_cnts_sect_data[0] LLVM_SECTION("__llvm_prf_cnts");
char __llvm_prof_nms_sect_data[0] LLVM_SECTION("__llvm_prf_names");

LLVM_LIBRARY_VISIBILITY const __llvm_profile_data *
__llvm_profile_begin_data(void) {
  return &__start___llvm_prf_data;
}
LLVM_LIBRARY_VISIBILITY const __llvm_profile_data *
__llvm_profile_end_data(void) {
  return &__stop___llvm_prf_data;
}
LLVM_LIBRARY_VISIBILITY const char *__llvm_profile_begin_names(void) {
  return &__start___llvm_prf_names;
}
LLVM_LIBRARY_VISIBILITY const char *__llvm_profile_end_names(void) {
  return &__stop___llvm_prf_names;
}
LLVM_LIBRARY_VISIBILITY uint64_t *__llvm_profile_begin_counters(void) {
  return &__start___llvm_prf_cnts;
}
LLVM_LIBRARY_VISIBILITY uint64_t *__llvm_profile_end_counters(void) {
  return &__stop___llvm_prf_cnts;
}
#endif
