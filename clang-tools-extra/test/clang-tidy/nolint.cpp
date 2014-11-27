// RUN: clang-tidy -checks='-*,google-explicit-constructor' %s -- 2>&1 | FileCheck %s

class A { A(int i); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors must be explicit [google-explicit-constructor]

class B { B(int i); }; // NOLINT
// CHECK-NOT: :[[@LINE-1]]:11: warning: single-argument constructors must be explicit [google-explicit-constructor]

class C { C(int i); }; // NOLINT(we-dont-care-about-categories-yet)
// CHECK-NOT: :[[@LINE-1]]:11: warning: single-argument constructors must be explicit [google-explicit-constructor]
// CHECK: Suppressed 2 warnings (2 NOLINT)
