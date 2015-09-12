// Test different kinds of alwaysinline *structor definitions.

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-CALL

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -mconstructor-aliases -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -mconstructor-aliases -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-CALL

struct A1 {
  __attribute__((__always_inline__)) A1() {}
  __attribute__((__always_inline__)) ~A1() {}
};

void g1() {
  A1 a1;
}

struct A2 {
  inline __attribute__((__always_inline__)) A2() {}
  inline __attribute__((__always_inline__)) ~A2() {}
};

void g2() {
  A2 a2;
}

struct A3 {
  inline __attribute__((gnu_inline, __always_inline__)) A3() {}
  inline __attribute__((gnu_inline, __always_inline__)) ~A3() {}
};

void g3() {
  A3 a3;
}

// CHECK-DAG: define internal void @_ZN2A1C1Ev.alwaysinline(%struct.A1* %this) unnamed_addr #[[AI:[01-9]+]]
// CHECK-DAG: define internal void @_ZN2A1C2Ev.alwaysinline(%struct.A1* %this) unnamed_addr #[[AI]]
// CHECK-DAG: define internal void @_ZN2A1D1Ev.alwaysinline(%struct.A1* %this) unnamed_addr #[[AI]]
// CHECK-DAG: define internal void @_ZN2A1D2Ev.alwaysinline(%struct.A1* %this) unnamed_addr #[[AI]]

// CHECK-DAG: define internal void @_ZN2A2C1Ev.alwaysinline(%struct.A2* %this) unnamed_addr #[[AIIH:[01-9]+]]
// CHECK-DAG: define internal void @_ZN2A2C2Ev.alwaysinline(%struct.A2* %this) unnamed_addr #[[AIIH]]
// CHECK-DAG: define internal void @_ZN2A2D1Ev.alwaysinline(%struct.A2* %this) unnamed_addr #[[AIIH]]
// CHECK-DAG: define internal void @_ZN2A2D2Ev.alwaysinline(%struct.A2* %this) unnamed_addr #[[AIIH]]

// CHECK-DAG: define internal void @_ZN2A3C1Ev.alwaysinline(%struct.A3* %this) unnamed_addr #[[AIIH]]
// CHECK-DAG: define internal void @_ZN2A3C2Ev.alwaysinline(%struct.A3* %this) unnamed_addr #[[AIIH]]
// CHECK-DAG: define internal void @_ZN2A3D1Ev.alwaysinline(%struct.A3* %this) unnamed_addr #[[AIIH]]
// CHECK-DAG: define internal void @_ZN2A3D2Ev.alwaysinline(%struct.A3* %this) unnamed_addr #[[AIIH]]

// CHECK-DAG: attributes #[[AI]] = {{.*alwaysinline.*}}
// CHECK-DAG: attributes #[[AIIH]] = {{.*alwaysinline.*inlinehint.*}}
// CHECK-NOT: attributes #[[NOAI]] = {{.*alwaysinline.*}}

// CHECK-CALL-LABEL: define void @_Z2g1v()
// CHECK-CALL:       call void @_ZN2A1C1Ev.alwaysinline
// CHECK-CALL:       call void @_ZN2A1D1Ev.alwaysinline
// CHECK-CALL:       ret void

// CHECK-CALL-LABEL: define void @_Z2g2v()
// CHECK-CALL:       call void @_ZN2A2C1Ev.alwaysinline
// CHECK-CALL:       call void @_ZN2A2D1Ev.alwaysinline
// CHECK-CALL:       ret void

// CHECK-CALL-LABEL: define void @_Z2g3v()
// CHECK-CALL:       call void @_ZN2A3C1Ev.alwaysinline
// CHECK-CALL:       call void @_ZN2A3D1Ev.alwaysinline
// CHECK-CALL:       ret void
