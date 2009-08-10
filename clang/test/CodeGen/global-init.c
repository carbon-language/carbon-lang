// RUN: clang-cc -emit-llvm -o - -triple i386-linux-gnu %s | FileCheck %s

// This checks that the global won't be marked as common. 
// (It shouldn't because it's being initialized).

int a;
int a = 242;
// CHECK: @a = global i32 242

// This should get normal weak linkage.
int c __attribute__((weak))= 0;
// CHECK: @c = weak global i32 0



// Since this is marked const, it should get weak_odr linkage, since all
// definitions have to be the same.
// CHECK: @d = weak_odr constant i32 0
const int d __attribute__((weak))= 0;



// NOTE: tentative definitions are processed at the end of the translation unit.

// This shouldn't be emitted as common because it has an explicit section.
// rdar://7119244
int b __attribute__((section("foo")));

// CHECK: @b = global i32 0, section "foo"

