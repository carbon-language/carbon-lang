// RUN: %clang_cc1 -triple arm64-linux-gnu -target-abi aapcs -ffreestanding -emit-llvm -w -o - %s | FileCheck %s

// AAPCS clause C.8 says: If the argument has an alignment of 16 then the NGRN
// is rounded up to the next even number.

// CHECK: void @test1(i32 %x0, i128 %x2_x3, i128 %x4_x5, i128 %x6_x7, i128 %sp.coerce)
typedef union { __int128 a; } Small;
void test1(int x0, __int128 x2_x3, __int128 x4_x5, __int128 x6_x7, Small sp) {
}


// CHECK: void @test2(i32 %x0, i128 %x2_x3.coerce, i32 %x4, i128 %x6_x7.coerce, i32 %sp, i128 %sp16.coerce)
void test2(int x0, Small x2_x3, int x4, Small x6_x7, int sp, Small sp16) {
}
