// RUN: clang-cc -emit-llvm %s -o %t &&
template<typename T, typename U>
T* next(T* ptr, const U& diff);

template<typename T, typename U>
T* next(T* ptr, const U& diff) { 
  return ptr + diff; 
}

void test(int *iptr, float *fptr, int diff) {
  // FIXME: should be "_Z4nextIiiEPT_S1_RKT0_"
  // RUN: grep "_Z4nextIiiEPiS_RKi" %t &&
  iptr = next(iptr, diff);
  // FIXME: should be "_Z4nextIfiEPT_S1_RKT0_"
  // RUN: grep "_Z4nextIfiEPfS_RKi" %t &&
  fptr = next(fptr, diff);
}

template<typename T, typename U>
T* next(T* ptr, const U& diff);

void test2(int *iptr, double *dptr, int diff) {
  iptr = next(iptr, diff);
  // FIXME: should be "_Z4nextIdiEPT_S1_RKT0_"
  // RUN: grep "_Z4nextIdiEPdS_RKi" %t
  dptr = next(dptr, diff);
}
