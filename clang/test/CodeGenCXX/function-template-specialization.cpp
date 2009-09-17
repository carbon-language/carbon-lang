// RUN: clang-cc -emit-llvm %s -o %t && FileCheck %s --input-file=%t
template<typename T, typename U>
T* next(T* ptr, const U& diff);

template<typename T, typename U>
T* next(T* ptr, const U& diff) { 
  return ptr + diff; 
}

void test(int *iptr, float *fptr, int diff) {
  // FIXME: should be "_Z4nextIiiEPT_S1_RKT0_"
  // CHECK: _Z4nextIiiEPT_S0_RKT0_
  iptr = next(iptr, diff);
  // FIXME: should be "_Z4nextIfiEPT_S1_RKT0_"
  // CHECK: _Z4nextIfiEPT_S0_RKT0_
  
  fptr = next(fptr, diff);
}

template<typename T, typename U>
T* next(T* ptr, const U& diff);

void test2(int *iptr, double *dptr, int diff) {
  iptr = next(iptr, diff);
  // FIXME: should be "_Z4nextIdiEPT_S1_RKT0_"
  // RUN: grep "_Z4nextIiiEPT_S0_RKT0_" %t
  dptr = next(dptr, diff);
}
