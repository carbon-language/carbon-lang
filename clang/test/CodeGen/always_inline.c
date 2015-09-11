// RUN: %clang -emit-llvm -S -o - %s | FileCheck %s
// RUN: %clang -mllvm -disable-llvm-optzns -emit-llvm -S -o - %s | FileCheck %s --check-prefix=CHECK-NO-OPTZNS

//static int f0() { 
static int __attribute__((always_inline)) f0() { 
  return 1;
}

int f1() {
  return f0();
}

// PR4372
inline int f2() __attribute__((always_inline));
int f2() { return 7; }
int f3(void) { return f2(); }

// CHECK-LABEL: define i32 @f1()
// CHECK: ret i32 1
// CHECK-LABEL: define i32 @f2()
// CHECK: ret i32 7
// CHECK-LABEL: define i32 @f3()
// CHECK: ret i32 7

// CHECK-NO-OPTZNS: define i32 @f3()
// CHECK-NO-OPTZNS-NEXT: entry:
// CHECK-NO-OPTZNS-NEXT:   call i32 @f2.alwaysinline()
// CHECK-NO-OPTZNS-NEXT:   ret i32
