// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

struct foo {
  void *my_alloc(unsigned) __attribute__((alloc_size(2)));
  static void* static_alloc(unsigned) __attribute__((alloc_size(1)));
};


void* f(bool a) {
  // CHECK: call i8* {{.*}}alloc{{.*}}, !alloc_size !0
  // CHECK: call i8* {{.*}}static_alloc{{.*}}, !alloc_size !1
  foo obj;
  return a ? obj.my_alloc(2) :
             foo::static_alloc(42);
}

// CHECK: !0 = metadata !{i32 1}
// CHECK: !1 = metadata !{i32 0}
