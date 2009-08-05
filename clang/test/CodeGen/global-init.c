// RUN: clang-cc -emit-llvm -o - %s | FileCheck %s

// This checks that the global won't be marked as common. 
// (It shouldn't because it's being initialized).

int a;
int a = 242;
// CHECK: @a = global i32 242

// This shouldn't be emitted as common because it has an explicit section.
// rdar://7119244
int b __attribute__((section("foo")));

// CHECK: @b = global i32 0, section "foo"
