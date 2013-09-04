// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -loop-convert %t.cpp -- -I %S/Inputs -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
#include "structures.h"

const int N = 6;
dependent<int> v;
dependent<int> *pv;

transparent<dependent<int> > cv;

void f() {
  int sum = 0;
  for (int i = 0, e = v.size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", v[i]);
    sum += v[i] + 2;
  }
  // CHECK: for (auto & elem : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  for (int i = 0, e = v.size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", v.at(i));
    sum += v.at(i) + 2;
  }
  // CHECK: for (auto & elem : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  for (int i = 0, e = pv->size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", pv->at(i));
    sum += pv->at(i) + 2;
  }
  // CHECK: for (auto & elem : *pv)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  // This test will fail if size() isn't called repeatedly, since it
  // returns unsigned int, and 0 is deduced to be signed int.
  // FIXME: Insert the necessary explicit conversion, or write out the types
  // explicitly.
  for (int i = 0; i < pv->size(); ++i) {
    printf("Fibonacci number is %d\n", (*pv).at(i));
    sum += (*pv)[i] + 2;
  }
  // CHECK: for (auto & elem : *pv)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;

  for (int i = 0; i < cv->size(); ++i) {
    printf("Fibonacci number is %d\n", cv->at(i));
    sum += cv->at(i) + 2;
  }
  // CHECK: for (auto & elem : *cv)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-NEXT: sum += elem + 2;
}

// Check for loops that don't mention containers
void noContainer() {
  for (auto i = 0; i < v.size(); ++i) { }
  // CHECK: for (auto & elem : v) { }

  for (auto i = 0; i < v.size(); ++i) ;
  // CHECK: for (auto & elem : v) ;
}

struct NoBeginEnd {
  unsigned size() const;
};

struct NoConstBeginEnd {
  NoConstBeginEnd();
  unsigned size() const;
  unsigned begin();
  unsigned end();
};

struct ConstBeginEnd {
  ConstBeginEnd();
  unsigned size() const;
  unsigned begin() const;
  unsigned end() const;
};

// Shouldn't transform pseudo-array uses if the container doesn't provide
// begin() and end() of the right const-ness.
void NoBeginEndTest() {
  NoBeginEnd NBE;
  for (unsigned i = 0, e = NBE.size(); i < e; ++i) {}
  // CHECK: for (unsigned i = 0, e = NBE.size(); i < e; ++i) {}

  const NoConstBeginEnd const_NCBE;
  for (unsigned i = 0, e = const_NCBE.size(); i < e; ++i) {}
  // CHECK: for (unsigned i = 0, e = const_NCBE.size(); i < e; ++i) {}

  ConstBeginEnd CBE;
  for (unsigned i = 0, e = CBE.size(); i < e; ++i) {}
  // CHECK: for (auto & elem : CBE) {}

  const ConstBeginEnd const_CBE;
  for (unsigned i = 0, e = const_CBE.size(); i < e; ++i) {}
  // CHECK: for (auto & elem : const_CBE) {}
}

