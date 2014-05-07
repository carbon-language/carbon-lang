// RUN:  %clang_cc1 -triple arm64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o - %s | FileCheck %s

struct bt3 { signed b2:10; signed b3:10; } b16;

// The correct right-shift amount is 40 bits for big endian.
signed callee_b0f(struct bt3 bp11) {
// CHECK: = lshr i64 %{{.*}}, 40
  return bp11.b2;
}
