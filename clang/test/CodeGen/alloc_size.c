// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

void *my_recalloc(void *, unsigned, unsigned) __attribute__((alloc_size(2,3))); 

// CHECK: @f
void* f() {
  // CHECK: call i8* @my_recalloc{{.*}}, !alloc_size !0
  return my_recalloc(0, 11, 27);
}

// CHECK: !0 = metadata !{i32 1, i32 2}
