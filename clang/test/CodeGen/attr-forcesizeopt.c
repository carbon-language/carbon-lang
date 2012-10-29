// RUN: %clang_cc1 -Oz -emit-llvm %s -o - | FileCheck %s -check-prefix=Oz
// RUN: %clang_cc1 -O0 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O1 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O3 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -Os -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// Check that we set the forcesizeopt attribute on each function
// when Oz optimization level is set.

int test1() {
  return 42;
// Oz: @test1{{.*}}forcesizeopt
// Oz: ret
// OTHER: @test1
// OTHER-NOT: forcesizeopt
// OTHER: ret
}

int test2() {
  return 42;
// Oz: @test2{{.*}}forcesizeopt
// Oz: ret
// OTHER: @test2
// OTHER-NOT: forcesizeopt
// OTHER: ret
}

int test3() __attribute__((forcesizeopt)) {
// Oz: @test3{{.*}}forcesizeopt
// OTHER: @test3{{.*}}forcesizeopt
}
