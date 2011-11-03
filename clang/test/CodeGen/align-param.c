// RUN: %clang_cc1 -emit-llvm -triple i386-apple-macosx10.7.2 < %s | FileCheck %s

// The preferred alignment for a long long on x86-32 is 8; make sure the
// alloca for x uses that alignment.
int test (long long x) {
  return (int)x;
}
// CHECK: define i32 @test
// CHECK: alloca i64, align 8
