// Test different kinds of alwaysinline definitions.

// RUN: %clang_cc1 -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-INLINE
// RUN: %clang_cc1 -disable-llvm-optzns -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-USE
// RUN: %clang_cc1 -disable-llvm-optzns -fno-inline -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -disable-llvm-optzns -fno-inline -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-USE

void __attribute__((__always_inline__)) f1() {}
inline void __attribute__((__always_inline__)) f2() {}
static inline void __attribute__((__always_inline__)) f3() {}
inline void __attribute__((gnu_inline, __always_inline__)) f4() {}
static inline void __attribute__((gnu_inline, __always_inline__)) f5() {}
inline void __attribute__((visibility("hidden"), __always_inline__)) f6() {}
inline void __attribute__((visibility("hidden"), gnu_inline, __always_inline__)) f7() {}

void g() {
  f1();
  f2();
  f3();
  f4();
  f5();
  f6();
  f7();
}

void (*p)(void);
void h() {
  p = f1;
  p = f2;
  p = f3;
  p = f4;
  p = f5;
  p = f6;
  p = f7;
}

void (*const cp1)(void) = f1;
void (*p1)(void) = f1;
void (*p2)(int) = (void (*)(int))f1;

void __attribute__((__always_inline__)) f8(void(*f)(void)) {}

void call() {
  f8(f1);
}

// CHECK-DAG: define internal void @f1.alwaysinline() #[[AI:[0-9]+]]
// CHECK-DAG: define internal void @f2.alwaysinline() #[[AI_IH:[0-9]+]]
// CHECK-DAG: define internal void @f3.alwaysinline() #[[AI_IH]]
// CHECK-DAG: define internal void @f4.alwaysinline() #[[AI_IH]]
// CHECK-DAG: define internal void @f5.alwaysinline() #[[AI_IH]]
// CHECK-DAG: define internal void @f6.alwaysinline() #[[AI_IH]]
// CHECK-DAG: define internal void @f7.alwaysinline() #[[AI_IH]]


// CHECK-DAG: define void @f1() #[[NOAI:[01-9]+]]
// CHECK-DAG: musttail call void @f1.alwaysinline()

// CHECK-DAG: declare void @f2() #[[NOAI:[01-9]+]]

// CHECK-DAG: define internal void @f3() #[[NOAI:[01-9]+]]
// CHECK-DAG: musttail call void @f3.alwaysinline()

// CHECK-DAG: define void @f4() #[[NOAI:[01-9]+]]
// CHECK-DAG: musttail call void @f4.alwaysinline()

// CHECK-DAG: define internal void @f5() #[[NOAI:[01-9]+]]
// CHECK-DAG: musttail call void @f5.alwaysinline()

// CHECK-DAG: declare hidden void @f6() #[[NOAI:[01-9]+]]

// CHECK-DAG: define hidden void @f7() #[[NOAI:[01-9]+]]
// CHECK-DAG: musttail call void @f7.alwaysinline()


// CHECK-DAG: @cp1 = constant void ()* @f1, align
// CHECK-DAG: @p1 = global void ()* @f1, align
// CHECK-DAG: @p2 = global void (i32)* bitcast (void ()* @f1 to void (i32)*), align

// CHECK: attributes #[[AI]] = {{.*alwaysinline.*}}
// CHECK-INLINE: attributes #[[AI_IH]] = {{.*alwaysinline.*inlinehint.*}}
// CHECK-NOT: attributes #[[NOAI]] = {{.*alwaysinline.*}}

// CHECK-USE-LABEL: define void @g()
// CHECK-USE-NEXT:   entry:
// CHECK-USE-NEXT:     call void @f1.alwaysinline()
// CHECK-USE-NEXT:     call void @f2.alwaysinline()
// CHECK-USE-NEXT:     call void @f3.alwaysinline()
// CHECK-USE-NEXT:     call void @f4.alwaysinline()
// CHECK-USE-NEXT:     call void @f5.alwaysinline()
// CHECK-USE-NEXT:     call void @f6.alwaysinline()
// CHECK-USE-NEXT:     call void @f7.alwaysinline()
// CHECK-USE-NEXT:     ret void

// CHECK-USE-LABEL: define void @h()
// CHECK-USE-NEXT:   entry:
// CHECK-USE-NEXT:     store void ()* @f1,
// CHECK-USE-NEXT:     store void ()* @f2,
// CHECK-USE-NEXT:     store void ()* @f3,
// CHECK-USE-NEXT:     store void ()* @f4,
// CHECK-USE-NEXT:     store void ()* @f5,
// CHECK-USE-NEXT:     store void ()* @f6,
// CHECK-USE-NEXT:     store void ()* @f7,
// CHECK-USE-NEXT:     ret void

// CHECK-USE-LABEL:  define void @call()
// CHECK-USE:           call void @f8.alwaysinline(void ()* @f1)
// CHECK-USE:           ret void
