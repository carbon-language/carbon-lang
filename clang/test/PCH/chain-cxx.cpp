// Test C++ chained PCH functionality

// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -include %s -include %s %s

// With PCH
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t1 %s
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t2 %s -include-pch %t1 -chained-pch
// RUN: %clang_cc1 -fsyntax-only -verify -include-pch %t2 %s

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//
// Primary header for C++ chained PCH test

void f();

// Name not appearing in dependent
void pf();

namespace ns {
  void g();

  void pg();
}

template <typename T>
struct S { typedef int G; };

// Partially specialize
template <typename T>
struct S<T *> { typedef int H; };

template <typename T> struct TS2;
typedef TS2<int> TS2int;

//===----------------------------------------------------------------------===//
#elif not defined(HEADER2)
#define HEADER2
//===----------------------------------------------------------------------===//
// Dependent header for C++ chained PCH test

// Overload function from primary
void f(int);

// Add function with different name
void f2();

// Reopen namespace
namespace ns {
  // Overload function from primary
  void g(int);

  // Add different name
  void g2();
}

// Specialize template from primary
template <>
struct S<int> { typedef int I; };

// Partially specialize
template <typename T>
struct S<T &> { typedef int J; };

// Specialize previous partial specialization
template <>
struct S<int *> { typedef int K; };

// Specialize the partial specialization from this file
template <>
struct S<int &> { typedef int L; };

template <typename T> struct TS2 { };

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

void test() {
  f();
  f(1);
  pf();
  f2();

  ns::g();
  ns::g(1);
  ns::pg();
  ns::g2();

  typedef S<double>::G T1;
  typedef S<double *>::H T2;
  typedef S<int>::I T3;
  typedef S<double &>::J T4;
  typedef S<int *>::K T5;
  typedef S<int &>::L T6;

  TS2int ts2;
}

//===----------------------------------------------------------------------===//
#endif
