// RUN: %clang_profgen -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t 1
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata merge -o %t-2.profdata %t-2.profraw
// RUN: llvm-profdata merge -o %t-merged.profdata %t.profraw %t-2.profdata
// RUN: llvm-profdata show --all-functions -ic-targets  %t-2.profdata | FileCheck  %s -check-prefix=NO-VALUE
// RUN: llvm-profdata show --all-functions -ic-targets  %t.profdata | FileCheck  %s
// value profile merging current do sorting based on target values -- this will destroy the order of the target
// in the list leading to comparison problem. For now just check a small subset of output.
// RUN: llvm-profdata show --all-functions -ic-targets  %t-merged.profdata | FileCheck  %s -check-prefix=MERGE

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef struct __llvm_profile_data __llvm_profile_data;
const __llvm_profile_data *__llvm_profile_begin_data(void);
const __llvm_profile_data *__llvm_profile_end_data(void);
void __llvm_profile_set_num_value_sites(__llvm_profile_data *Data,
                                        uint32_t ValueKind,
                                        uint16_t NumValueSites);
__llvm_profile_data *
__llvm_profile_iterate_data(const __llvm_profile_data *Data);
void *__llvm_get_function_addr(const __llvm_profile_data *Data);
void __llvm_profile_instrument_target(uint64_t TargetValue, void *Data,
                                      uint32_t CounterIndex);

#define DEF_FUNC(x)                                                            \
  void x() {}
#define DEF_2_FUNCS(x) DEF_FUNC(x##_1) DEF_FUNC(x##_2)
#define DEF_4_FUNCS(x) DEF_2_FUNCS(x##_1) DEF_2_FUNCS(x##_2)
#define DEF_8_FUNCS(x) DEF_4_FUNCS(x##_1) DEF_4_FUNCS(x##_2)
#define DEF_16_FUNCS(x) DEF_8_FUNCS(x##_1) DEF_8_FUNCS(x##_2)
#define DEF_32_FUNCS(x) DEF_16_FUNCS(x##_1) DEF_16_FUNCS(x##_2)
#define DEF_64_FUNCS(x) DEF_32_FUNCS(x##_1) DEF_32_FUNCS(x##_2)
#define DEF_128_FUNCS(x) DEF_64_FUNCS(x##_1) DEF_64_FUNCS(x##_2)

#define FUNC_ADDR(x) &x,
#define FUNC_2_ADDRS(x) FUNC_ADDR(x##_1) FUNC_ADDR(x##_2)
#define FUNC_4_ADDRS(x) FUNC_2_ADDRS(x##_1) FUNC_2_ADDRS(x##_2)
#define FUNC_8_ADDRS(x) FUNC_4_ADDRS(x##_1) FUNC_4_ADDRS(x##_2)
#define FUNC_16_ADDRS(x) FUNC_8_ADDRS(x##_1) FUNC_8_ADDRS(x##_2)
#define FUNC_32_ADDRS(x) FUNC_16_ADDRS(x##_1) FUNC_16_ADDRS(x##_2)
#define FUNC_64_ADDRS(x) FUNC_32_ADDRS(x##_1) FUNC_32_ADDRS(x##_2)
#define FUNC_128_ADDRS(x) FUNC_64_ADDRS(x##_1) FUNC_64_ADDRS(x##_2)

DEF_8_FUNCS(callee)
DEF_128_FUNCS(caller)

void *CallerAddrs[] = {FUNC_128_ADDRS(caller)};

void *CalleeAddrs[] = {FUNC_8_ADDRS(callee)};

static int cmpaddr(const void *p1, const void *p2) {
  void *addr1 = *(void **)p1;
  void *addr2 = *(void **)p2;
  return (intptr_t)addr2 - (intptr_t)addr1;
}

int main(int argc, const char *argv[]) {
  unsigned S, NS = 0, V, doInstrument = 1;
  const __llvm_profile_data *Data, *DataEnd;

  if (argc < 2)
    doInstrument = 0;

  qsort(CallerAddrs, sizeof(CallerAddrs) / sizeof(void *), sizeof(void *),
        cmpaddr);

  /* We will synthesis value profile data for 128 callers functions.
   * The number of * value sites. The number values for each value site
   * ranges from 0 to 8.  */

  Data = __llvm_profile_begin_data();
  DataEnd = __llvm_profile_end_data();

  for (; Data < DataEnd; Data = __llvm_profile_iterate_data(Data)) {
    void *func = __llvm_get_function_addr(Data);
    if (bsearch(&func, CallerAddrs, sizeof(CallerAddrs) / sizeof(void *),
                sizeof(void *), cmpaddr)) {
      __llvm_profile_set_num_value_sites((__llvm_profile_data *)Data,
                                         0 /*IPVK_IndirectCallTarget */, NS);
      if (!doInstrument) {
        NS++;
        continue;
      }
      for (S = 0; S < NS; S++) {
        for (V = 0; V < S % 8; V++) {
          unsigned C;
          for (C = 0; C < V + 1; C++)
            __llvm_profile_instrument_target((uint64_t)CalleeAddrs[V],
                                             (void *)Data, S);
        }
      }
      NS++;
    }
  }
}

// NO-VALUE: Indirect Call Site Count: 127
// NO-VALUE-NEXT: Indirect Target Results:
// MERGE: Indirect Call Site Count: 127
// MERGE-NEXT: Indirect Target Results:
// MERGE-NEXT:  [ 1, callee_1_1_1, 1 ]
// CHECK: Indirect Call Site Count: 127
// CHECK-NEXT: Indirect Target Results:
// CHECK-NEXT:  [ 1, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 2, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 2, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 3, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 3, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 3, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 4, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 4, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 4, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 4, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 5, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 5, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 5, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 5, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 5, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 6, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 6, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 6, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 6, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 6, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 6, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 7, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 7, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 7, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 7, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 7, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 7, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 7, callee_2_2_1, 7 ]
// CHECK-NEXT:  [ 9, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 10, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 10, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 11, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 11, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 11, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 12, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 12, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 12, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 12, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 13, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 13, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 13, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 13, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 13, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 14, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 14, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 14, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 14, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 14, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 14, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 15, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 15, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 15, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 15, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 15, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 15, callee_2_1_2, 6 ]
// CHECK-NEXT:  [ 15, callee_2_2_1, 7 ]
// CHECK-NEXT:  [ 17, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 18, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 18, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 19, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 19, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 19, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 20, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 20, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 20, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 20, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 21, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 21, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 21, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 21, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 21, callee_2_1_1, 5 ]
// CHECK-NEXT:  [ 22, callee_1_1_1, 1 ]
// CHECK-NEXT:  [ 22, callee_1_1_2, 2 ]
// CHECK-NEXT:  [ 22, callee_1_2_1, 3 ]
// CHECK-NEXT:  [ 22, callee_1_2_2, 4 ]
// CHECK-NEXT:  [ 22, callee_2_1_1, 5 ]

