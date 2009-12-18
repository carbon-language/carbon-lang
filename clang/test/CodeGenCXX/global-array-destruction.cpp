// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++0x -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s

extern "C" int printf(...);

int count;

struct S {
  S() : iS(++count) { printf("S::S(%d)\n", iS); }
  ~S() { printf("S::~S(%d)\n", iS); }
  int iS;
};


S arr[2][1];
S s1;
S arr1[3];
static S sarr[4];

int main () {}
S arr2[2];
static S sarr1[4];
S s2;
S arr3[3];

// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
// CHECK-LP64: callq    ___cxa_atexit
