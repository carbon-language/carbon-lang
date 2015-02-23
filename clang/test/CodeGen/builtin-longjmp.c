// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=SUPPORTED
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=SUPPORTED
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=SUPPORTED
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=SUPPORTED
// RUN: %clang_cc1 -triple arm-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=UNSUPPORTED
// RUN: %clang_cc1 -triple aarch64-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=UNSUPPORTED
// RUN: %clang_cc1 -triple mips-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=UNSUPPORTED
// RUN: %clang_cc1 -triple mips64-unknown-unknown -emit-llvm < %s| FileCheck %s -check-prefix=UNSUPPORTED

// Check that __builtin_longjmp and __builtin_setjmp are lowerd into
// IR intrinsics on those architectures that can handle them.
// Check that they are lowered to the libcalls on other architectures.

typedef void *jmp_buf;
jmp_buf buf;

// SUPPORTED:   define{{.*}} void @do_jump()
// SUPPORTED:   call{{.*}} void @llvm.eh.sjlj.longjmp
// UNSUPPORTED: define{{.*}} void @do_jump()
// UNSUPPORTED: call{{.*}} void @longjmp

// SUPPORTED:   define{{.*}} void @do_setjmp()
// SUPPORTED:   call{{.*}} i32 @llvm.eh.sjlj.setjmp
// UNSUPPORTED: define{{.*}} void @do_setjmp()
// UNSUPPORTED: call{{.*}} i32 @setjmp

void do_jump(void) {
  __builtin_longjmp(buf, 1);
}

void f(void);

void do_setjmp(void) {
  if (!__builtin_setjmp(buf))
    f();
}
