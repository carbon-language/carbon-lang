// RUN: %llvmgcc %s -c -Os -m32 -emit-llvm -o - | llc -march=x86 -mtriple=i386-apple-darwin10 | FileCheck %s -check-prefix=DARWIN32
// RUN: %llvmgcc %s -c -Os -m64 -emit-llvm -o - | llc -march=x86-64 -mtriple=x86_64-apple-darwin10 | FileCheck %s -check-prefix=DARWIN64
// XFAIL: *
// XTARGET: darwin

extern void func(const char *, const char *);

void long_function_name() {
  func("%s: the function name", __func__);
}

// DARWIN64: .align 3
// DARWIN64: ___func__.
// DARWIN64: .asciz "long_function_name"

// DARWIN32: .align 2
// DARWIN32: ___func__.
// DARWIN32: .asciz "long_function_name"
