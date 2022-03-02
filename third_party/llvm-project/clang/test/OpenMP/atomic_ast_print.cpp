// RUN: %clang_cc1 -verify -fopenmp -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s

// RUN: %clang_cc1 -DOMP51 -verify -fopenmp -fopenmp-version=51 -ast-print %s | FileCheck --check-prefixes=CHECK,CHECK-51 %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck --check-prefixes=CHECK,CHECK-51 %s

// RUN: %clang_cc1 -DOMP51 -verify -fopenmp-simd -fopenmp-version=51 -ast-print %s | FileCheck --check-prefixes=CHECK,CHECK-51 %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -fopenmp-version=51 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck --check-prefixes=CHECK,CHECK-51 %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

template <class T>
T foo(T argc) {
  T v = T();
  T c = T();
  T b = T();
  T a = T();
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic capture
  a = b++;
#pragma omp atomic capture
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare
  { a = a > b ? b : a; }
#pragma omp atomic compare
  { a = a < b ? b : a; }
#pragma omp atomic compare
  { a = a == b ? c : a; }
#pragma omp atomic compare capture
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic seq_cst
  a++;
#pragma omp atomic read seq_cst
  a = argc;
#pragma omp atomic seq_cst write
  a = argc + argc;
#pragma omp atomic update seq_cst
  a = a + argc;
#pragma omp atomic seq_cst capture
  a = b++;
#pragma omp atomic capture seq_cst
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare seq_cst
  { a = a > b ? b : a; }
#pragma omp atomic seq_cst compare
  { a = a < b ? b : a; }
#pragma omp atomic compare seq_cst
  { a = a == b ? c : a; }
#pragma omp atomic compare capture seq_cst
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare seq_cst capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture seq_cst
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acq_rel capture
  a = b++;
#pragma omp atomic capture acq_rel
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare acq_rel
  { a = a > b ? b : a; }
#pragma omp atomic acq_rel compare
  { a = a < b ? b : a; }
#pragma omp atomic compare acq_rel
  { a = a == b ? c : a; }
#pragma omp atomic compare capture acq_rel
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare acq_rel capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture acq_rel
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic
  a++;
#pragma omp atomic read acquire
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acquire capture
  a = b++;
#pragma omp atomic capture acquire
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare acquire
  { a = a > b ? b : a; }
#pragma omp atomic acquire compare
  { a = a < b ? b : a; }
#pragma omp atomic compare acquire
  { a = a == b ? c : a; }
#pragma omp atomic compare capture acquire
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare acquire capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture acquire
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic release
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic release write
  a = argc + argc;
#pragma omp atomic update release
  a = a + argc;
#pragma omp atomic release capture
  a = b++;
#pragma omp atomic capture release
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare release
  { a = a > b ? b : a; }
#pragma omp atomic release compare
  { a = a < b ? b : a; }
#pragma omp atomic compare release
  { a = a == b ? c : a; }
#pragma omp atomic compare capture release
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare release capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture release
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic relaxed
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic relaxed write
  a = argc + argc;
#pragma omp atomic update relaxed
  a = a + argc;
#pragma omp atomic relaxed capture
  a = b++;
#pragma omp atomic capture relaxed
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare relaxed
  { a = a > b ? b : a; }
#pragma omp atomic relaxed compare
  { a = a < b ? b : a; }
#pragma omp atomic compare relaxed
  { a = a == b ? c : a; }
#pragma omp atomic compare capture relaxed
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare relaxed capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture relaxed
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic hint(6)
  a++;
#pragma omp atomic read hint(6)
  a = argc;
#pragma omp atomic hint(6) write
  a = argc + argc;
#pragma omp atomic update hint(6)
  a = a + argc;
#pragma omp atomic hint(6) capture
  a = b++;
#pragma omp atomic capture hint(6)
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare hint(6)
  { a = a > b ? b : a; }
#pragma omp atomic hint(6) compare
  { a = a < b ? b : a; }
#pragma omp atomic compare hint(6)
  { a = a == b ? c : a; }
#pragma omp atomic compare capture hint(6)
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare hint(6) capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture hint(6)
  { v = a == b; if (v) a = c; }
#endif
  return T();
}

// CHECK: T a = T();
// CHECK-NEXT: #pragma omp atomic{{$}}
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic seq_cst
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read seq_cst
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic seq_cst write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update seq_cst
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic seq_cst capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture seq_cst
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic seq_cst compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare seq_cst capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic acq_rel compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acq_rel capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read acquire
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acquire capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acquire
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic acquire compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acquire capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic release
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic release write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update release
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic release capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture release
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic release compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare release capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic relaxed
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic relaxed write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update relaxed
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic relaxed capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture relaxed
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic relaxed compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare relaxed capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic hint(6)
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read hint(6)
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic hint(6) write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update hint(6)
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic hint(6) capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture hint(6)
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic hint(6) compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare hint(6) capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK: int a = int();
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic seq_cst
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read seq_cst
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic seq_cst write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update seq_cst
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic seq_cst capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture seq_cst
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic seq_cst compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare seq_cst capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture seq_cst
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acq_rel capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acq_rel
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic acq_rel compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acq_rel capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acq_rel
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read acquire
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic acquire capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture acquire
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic acquire compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare acquire capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture acquire
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic release
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic release write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update release
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic release capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture release
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic release compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare release capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture release
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic relaxed
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic relaxed write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update relaxed
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic relaxed capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture relaxed
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic relaxed compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare relaxed capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture relaxed
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }
// CHECK-NEXT: #pragma omp atomic hint(6)
// CHECK-NEXT: a++;
// CHECK-NEXT: #pragma omp atomic read hint(6)
// CHECK-NEXT: a = argc;
// CHECK-NEXT: #pragma omp atomic hint(6) write
// CHECK-NEXT: a = argc + argc;
// CHECK-NEXT: #pragma omp atomic update hint(6)
// CHECK-NEXT: a = a + argc;
// CHECK-NEXT: #pragma omp atomic hint(6) capture
// CHECK-NEXT: a = b++;
// CHECK-NEXT: #pragma omp atomic capture hint(6)
// CHECK-NEXT: {
// CHECK-NEXT: a = b;
// CHECK-NEXT: b++;
// CHECK-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a > b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic hint(6) compare
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a < b ? b : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: a = a == b ? c : a;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a > b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare hint(6) capture
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a;
// CHECK-51-NEXT: if (a < b) {
// CHECK-51-NEXT: a = b;
// CHECK-51-NEXT: }
// CHECK-51-NEXT: }
// CHECK-51-NEXT: #pragma omp atomic compare capture hint(6)
// CHECK-51-NEXT: {
// CHECK-51-NEXT: v = a == b;
// CHECK-51-NEXT: if (v)
// CHECK-51-NEXT: a = c;
// CHECK-51-NEXT: }

int main(int argc, char **argv) {
  int v = 0;
  int c = 0;
  int b = 0;
  int a = 0;
// CHECK: int a = 0;
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic capture
  a = b++;
#pragma omp atomic capture
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare
  { a = a > b ? b : a; }
#pragma omp atomic compare
  { a = a < b ? b : a; }
#pragma omp atomic compare
  { a = a == b ? c : a; }
#pragma omp atomic compare capture
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic seq_cst
  a++;
#pragma omp atomic read seq_cst
  a = argc;
#pragma omp atomic seq_cst write
  a = argc + argc;
#pragma omp atomic update seq_cst
  a = a + argc;
#pragma omp atomic seq_cst capture
  a = b++;
#pragma omp atomic capture seq_cst
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare seq_cst
  { a = a > b ? b : a; }
#pragma omp atomic seq_cst compare
  { a = a < b ? b : a; }
#pragma omp atomic compare seq_cst
  { a = a == b ? c : a; }
#pragma omp atomic compare capture seq_cst
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare seq_cst capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture seq_cst
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acq_rel capture
  a = b++;
#pragma omp atomic capture acq_rel
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare acq_rel
  { a = a > b ? b : a; }
#pragma omp atomic acq_rel compare
  { a = a < b ? b : a; }
#pragma omp atomic compare acq_rel
  { a = a == b ? c : a; }
#pragma omp atomic compare capture acq_rel
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare acq_rel capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture acq_rel
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic
  a++;
#pragma omp atomic read acquire
  a = argc;
#pragma omp atomic write
  a = argc + argc;
#pragma omp atomic update
  a = a + argc;
#pragma omp atomic acquire capture
  a = b++;
#pragma omp atomic capture acquire
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare acquire
  { a = a > b ? b : a; }
#pragma omp atomic acquire compare
  { a = a < b ? b : a; }
#pragma omp atomic compare acquire
  { a = a == b ? c : a; }
#pragma omp atomic compare capture acquire
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare acquire capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture acquire
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic release
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic release write
  a = argc + argc;
#pragma omp atomic update release
  a = a + argc;
#pragma omp atomic release capture
  a = b++;
#pragma omp atomic capture release
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare release
  { a = a > b ? b : a; }
#pragma omp atomic release compare
  { a = a < b ? b : a; }
#pragma omp atomic compare release
  { a = a == b ? c : a; }
#pragma omp atomic compare capture release
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare release capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture release
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic relaxed
  a++;
#pragma omp atomic read
  a = argc;
#pragma omp atomic relaxed write
  a = argc + argc;
#pragma omp atomic update relaxed
  a = a + argc;
#pragma omp atomic relaxed capture
  a = b++;
#pragma omp atomic capture relaxed
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare relaxed
  { a = a > b ? b : a; }
#pragma omp atomic relaxed compare
  { a = a < b ? b : a; }
#pragma omp atomic compare relaxed
  { a = a == b ? c : a; }
#pragma omp atomic compare capture relaxed
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare relaxed capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture relaxed
  { v = a == b; if (v) a = c; }
#endif
#pragma omp atomic hint(6)
  a++;
#pragma omp atomic read hint(6)
  a = argc;
#pragma omp atomic hint(6) write
  a = argc + argc;
#pragma omp atomic update hint(6)
  a = a + argc;
#pragma omp atomic hint(6) capture
  a = b++;
#pragma omp atomic capture hint(6)
  {
    a = b;
    b++;
  }
#ifdef OMP51
#pragma omp atomic compare hint(6)
  { a = a > b ? b : a; }
#pragma omp atomic hint(6) compare
  { a = a < b ? b : a; }
#pragma omp atomic compare hint(6)
  { a = a == b ? c : a; }
#pragma omp atomic compare capture hint(6)
  { v = a; if (a > b) { a = b; } }
#pragma omp atomic compare hint(6) capture
  { v = a; if (a < b) { a = b; } }
#pragma omp atomic compare capture hint(6)
  { v = a == b; if (v) a = c; }
#endif
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // CHECK-NEXT: #pragma omp atomic seq_cst
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read seq_cst
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic seq_cst write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update seq_cst
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic seq_cst capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture seq_cst
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare seq_cst
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic seq_cst compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare seq_cst
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture seq_cst
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare seq_cst capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture seq_cst
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic acq_rel capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture acq_rel
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare acq_rel
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic acq_rel compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare acq_rel
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture acq_rel
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare acq_rel capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture acq_rel
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // CHECK-NEXT: #pragma omp atomic
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read acquire
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic acquire capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture acquire
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare acquire
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic acquire compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare acquire
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture acquire
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare acquire capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture acquire
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // CHECK-NEXT: #pragma omp atomic release
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic release write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update release
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic release capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture release
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare release
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic release compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare release
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture release
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare release capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture release
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // CHECK-NEXT: #pragma omp atomic relaxed
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic relaxed write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update relaxed
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic relaxed capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture relaxed
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare relaxed
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic relaxed compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare relaxed
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture relaxed
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare relaxed capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture relaxed
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // CHECK-NEXT: #pragma omp atomic hint(6)
  // CHECK-NEXT: a++;
  // CHECK-NEXT: #pragma omp atomic read hint(6)
  // CHECK-NEXT: a = argc;
  // CHECK-NEXT: #pragma omp atomic hint(6) write
  // CHECK-NEXT: a = argc + argc;
  // CHECK-NEXT: #pragma omp atomic update hint(6)
  // CHECK-NEXT: a = a + argc;
  // CHECK-NEXT: #pragma omp atomic hint(6) capture
  // CHECK-NEXT: a = b++;
  // CHECK-NEXT: #pragma omp atomic capture hint(6)
  // CHECK-NEXT: {
  // CHECK-NEXT: a = b;
  // CHECK-NEXT: b++;
  // CHECK-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare hint(6)
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a > b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic hint(6) compare
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a < b ? b : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare hint(6)
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: a = a == b ? c : a;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture hint(6)
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a > b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare hint(6) capture
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a;
  // CHECK-51-NEXT: if (a < b) {
  // CHECK-51-NEXT: a = b;
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: }
  // CHECK-51-NEXT: #pragma omp atomic compare capture hint(6)
  // CHECK-51-NEXT: {
  // CHECK-51-NEXT: v = a == b;
  // CHECK-51-NEXT: if (v)
  // CHECK-51-NEXT: a = c;
  // CHECK-51-NEXT: }
  // expect-note@+1 {{in instantiation of function template specialization 'foo<int>' requested here}}
  return foo(a);
}

#endif
