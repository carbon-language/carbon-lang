// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only

#ifndef HEADER
#define HEADER

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"
template <typename T>
struct TS {
    void m() {
      T a = 0;
      T b = a==a;
    }
};
#pragma clang diagnostic pop

#else

void f() {
    TS<int> ts;
    ts.m();
}

#endif
