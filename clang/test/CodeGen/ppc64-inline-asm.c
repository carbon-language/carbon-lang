// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s

_Bool test_wc_i1(_Bool b1, _Bool b2) {
  _Bool o;
  asm("crand %0, %1, %2" : "=wc"(o) : "wc"(b1), "wc"(b2) : );
  return o;
// CHECK-LABEL: define zeroext i1 @test_wc_i1(i1 zeroext %b1, i1 zeroext %b2)
// CHECK: call i8 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i1 %b1, i1 %b2)
}

int test_wc_i32(int b1, int b2) {
  int o;
  asm("crand %0, %1, %2" : "=wc"(o) : "wc"(b1), "wc"(b2) : );
  return o;
// CHECK-LABEL: signext i32 @test_wc_i32(i32 signext %b1, i32 signext %b2)
// CHECK: call i32 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i32 %b1, i32 %b2)
}

unsigned char test_wc_i8(unsigned char b1, unsigned char b2) {
  unsigned char o;
  asm("crand %0, %1, %2" : "=wc"(o) : "wc"(b1), "wc"(b2) : );
  return o;
// CHECK-LABEL: zeroext i8 @test_wc_i8(i8 zeroext %b1, i8 zeroext %b2)
// CHECK: call i8 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i8 %b1, i8 %b2)
}

