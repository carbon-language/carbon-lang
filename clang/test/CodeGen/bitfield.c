// RUN: clang %s -emit-llvm -o - > %t1
// RUN: grep "shl i32 %tmp, 19" %t1 &&
// RUN: grep "ashr i32 %tmp1, 19" %t1 &&
// RUN: grep "shl i16 %tmp4, 1" %t1 &&
// RUN: grep "lshr i16 %tmp5, 9" %t1
// RUN: grep "and i32 %tmp, -8192" %t1
// RUN: grep "and i16 %tmp5, -32513" %t1
// Test bitfield access


struct STestB1 { int a:13; char b; unsigned short c:7;} stb1;

int f() {
  return stb1.a + stb1.b + stb1.c;
}

void g() {
  stb1.a = -40;
  stb1.b = 10;
  stb1.c = 15;
}
