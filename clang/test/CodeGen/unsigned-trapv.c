// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fsanitize=unsigned-integer-overflow | FileCheck %s --check-prefix=UNSIGNED
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -ftrapv | FileCheck %s --check-prefix=TRAPV
// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - -fsanitize=unsigned-integer-overflow -ftrapv | FileCheck %s --check-prefix=BOTH
// Verify that -ftrapv and -fsanitize=unsigned-integer-overflow
// work together as expected


// UNSIGNED: @test_signed
// TRAPV: @test_signed
// BOTH: @test_signed
void test_signed() {
  extern volatile int a, b, c;
  // UNSIGNED: add nsw i32
  // UNSIGNED-NOT: overflow
  // TRAPV: sadd.with.overflow.i32
  // TRAPV-NOT: ubsan
  // TRAPV: llvm.trap
  // BOTH: sadd.with.overflow.i32
  // BOTH-NOT: ubsan
  // BOTH: llvm.trap
  a = b + c;
}

// UNSIGNED: @test_unsigned
// TRAPV: @test_unsigned
// BOTH: @test_unsigned
void test_unsigned() {
  extern volatile unsigned x, y, z;
  // UNSIGNED: uadd.with.overflow.i32
  // UNSIGNED-NOT: llvm.trap
  // UNSIGNED: ubsan
  // TRAPV-NOT: overflow
  // TRAPV-NOT: llvm.trap
  // BOTH: uadd.with.overflow.i32
  // BOTH: ubsan
  // BOTH-NOT: llvm.trap
  x = y + z;
}
