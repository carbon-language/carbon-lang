// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -DHEADER -DSOURCE -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++0x -DHEADER -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++0x -DHEADER -include-pch %t -fsyntax-only -emit-llvm -o - %s 

#ifdef HEADER
int n;
struct S {
  int *p = &m;
  int &m = n;
  S *that = this;
};
#endif

#ifdef SOURCE
S s;
#elif HEADER
#undef HEADER
#define SOURCE
#endif
