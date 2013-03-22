// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

struct POD {
  int a;
  int b;
};

void podMemPtrs() {
  int POD::*memptr;
  memptr = &POD::a;
  memptr = &POD::b;
  if (memptr)
    memptr = 0;
// Check that member pointers use the right offsets and that null is -1.
// CHECK:      define void @"\01?podMemPtrs@@YAXXZ"() #0 {
// CHECK:        %[[memptr:.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 0, i32* %[[memptr]], align 4
// CHECK-NEXT:   store i32 4, i32* %[[memptr]], align 4
// CHECK-NEXT:   %[[memptr_val:.*]] = load i32* %[[memptr]], align 4
// CHECK-NEXT:   %{{.*}} = icmp ne i32 %[[memptr_val]], -1
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:        store i32 -1, i32* %[[memptr]], align 4
// CHECK:        ret void
// CHECK:      }
}

struct Polymorphic {
  virtual void myVirtual();
  int a;
  int b;
};

void polymorphicMemPtrs() {
  int Polymorphic::*memptr;
  memptr = &Polymorphic::a;
  memptr = &Polymorphic::b;
  if (memptr)
    memptr = 0;
// Member pointers for polymorphic classes include the vtable slot in their
// offset and use 0 to represent null.
// CHECK:      define void @"\01?polymorphicMemPtrs@@YAXXZ"() #0 {
// CHECK:        %[[memptr:.*]] = alloca i32, align 4
// CHECK-NEXT:   store i32 4, i32* %[[memptr]], align 4
// CHECK-NEXT:   store i32 8, i32* %[[memptr]], align 4
// CHECK-NEXT:   %[[memptr_val:.*]] = load i32* %[[memptr]], align 4
// CHECK-NEXT:   %{{.*}} = icmp ne i32 %[[memptr_val]], 0
// CHECK-NEXT:   br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
// CHECK:        store i32 0, i32* %[[memptr]], align 4
// CHECK:        ret void
// CHECK:      }
}
