// RUN: %clang_cc1 -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target:
// CHECK-NEXT: Symbols:
// CHECK-NEXT: ...

template<typename T> class C1 {
    long a;
    operator long() const { return a; }
};
