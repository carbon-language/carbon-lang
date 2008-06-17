// RUN: clang %s -emit-llvm -o - > %t1
// RUN: grep "shl i32 .*, 19" %t1 &&
// RUN: grep "ashr i32 .*, 19" %t1 &&
// RUN: grep "shl i16 .*, 1" %t1 &&
// RUN: grep "lshr i16 .*, 9" %t1 &&
// RUN: grep "and i32 .*, -8192" %t1 &&
// RUN: grep "and i16 .*, -32513" %t1 &&
// RUN: grep "getelementptr (i32\* bitcast (.struct.STestB2\* @stb2 to i32\*), i32 1)" %t1
// Test bitfield access


struct STestB1 { int a:13; char b; unsigned short c:7;} stb1;
struct STestB2 { short a[3]; int b:15} stb2;

int f() {
  return stb1.a + stb1.b + stb1.c;
}

void g() {
  stb1.a = -40;
  stb1.b = 10;
  stb1.c = 15;
}

int h() {
  return stb2.a[1] + stb2.b;
}

void i(){
  stb2.a[2] = -40;
  stb2.b = 10;
}
