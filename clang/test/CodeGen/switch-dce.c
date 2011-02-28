// RUN: %clang_cc1 -triple i386-unknown-unknown -O0 %s -emit-llvm -o - | FileCheck %s

// CHECK: @test1
// CHECK-NOT: switch
// CHECK: add nsw i32 {{.*}}, 1
// CHECK-NOT: switch
// CHECK-NOT: add nsw i32
// CHECK: ret void
void test1() {
  int i;
  switch (1)
    case 1:
      ++i;

  switch (0)
    case 1:
      i+=2;
} 
