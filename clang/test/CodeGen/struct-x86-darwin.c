// RUN: clang %s -emit-llvm > %t1
// RUN  grep "STest1 = type { i32, \[4 x i16\], double }" %t1
// RUN: grep "STest2 = type { i16, i16, i32, i32 }" %t1
// RUN: grep "STest3 = type { i8, i8, i16, i32 }" %t1
// Test struct layout for x86-darwin target
// FIXME : Enable this test for x86-darwin only. At the moment clang hard codes
// x86-darwin as the target

struct STest1 {int x; short y[4]; double z; } st1;
struct STest2 {short a,b; int c,d; } st2;
struct STest3 {char a; short b; int c; } st3;

// Bitfields struct STestB1 {int a:1; char b; int c:13 } stb1;
// Packed struct STestP1 {char a; short b; int c; } __attribute__((__packed__)) stp1;
