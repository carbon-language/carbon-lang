// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow -fsanitize-recover %s -emit-llvm -o - | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=unsigned-integer-overflow %s -emit-llvm -o - | FileCheck %s --check-prefix=ABORT


// RECOVER: @test
// ABORT: @test
void test() {
  extern volatile unsigned x, y, z;

  // RECOVER: uadd.with.overflow.i32
  // RECOVER: ubsan_handle_add_overflow(
  // RECOVER-NOT: unreachable
  // ABORT: uadd.with.overflow.i32
  // ABORT: ubsan_handle_add_overflow_abort(
  // ABORT: unreachable
  x = y + z;
}
