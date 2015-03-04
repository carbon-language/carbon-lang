// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm < %s| FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm < %s| FileCheck %s
// RUN: %clang_cc1 -triple x86_64-windows -emit-llvm < %s| FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -emit-llvm < %s| FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm < %s| FileCheck %s

// RUN: %clang_cc1 -triple arm-unknown-unknown -emit-llvm-only -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -emit-llvm-only -verify %s
// RUN: %clang_cc1 -triple mips-unknown-unknown -emit-llvm-only -verify %s
// RUN: %clang_cc1 -triple mips64-unknown-unknown -emit-llvm-only -verify %s

// Check that __builtin_longjmp and __builtin_setjmp are lowerd into
// IR intrinsics on those architectures that can handle them.
// Check that they are lowered to the libcalls on other architectures.

typedef void *jmp_buf;
jmp_buf buf;

// CHECK:   define{{.*}} void @do_jump()
// CHECK:   call{{.*}} void @llvm.eh.sjlj.longjmp

// CHECK:   define{{.*}} void @do_setjmp()
// CHECK:   call{{.*}} i32 @llvm.eh.sjlj.setjmp

void do_jump(void) {
  __builtin_longjmp(buf, 1); // expected-error {{cannot compile this __builtin_longjmp yet}}
}

void f(void);

void do_setjmp(void) {
  if (!__builtin_setjmp(buf))
    f();
}
