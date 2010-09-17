// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o %t %s
// RUN: FileCheck < %t %s
//
// FIXME: Note that we don't currently get the ABI right here. f0() should be
// f0(i8*).

typedef union {
  void *f0;
} transp_t0 __attribute__((transparent_union));

void f0(transp_t0 obj);

// CHECK: define void @f1_0(i32* %a0) 
// CHECK:  call void @f0(%union.anon* byval %{{.*}})
// CHECK: }
void f1_0(int *a0) {
  f0(a0);
}

void f1_1(int *a0) {
  f0((transp_t0) { a0 });
}
