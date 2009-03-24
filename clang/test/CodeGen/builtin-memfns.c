// RUN: clang-cc -arch i386 -emit-llvm -o %t %s &&
// RUN: grep '@llvm.memset.i32' %t &&
// RUN: grep '@llvm.memcpy.i32' %t &&
// RUN: grep '@llvm.memmove.i32' %t &&
// RUN: grep __builtin %t | count 0

int main(int argc, char **argv) {
  unsigned char a = 0x11223344;
  unsigned char b = 0x11223344;
  __builtin_bzero(&a, sizeof(a));
  __builtin_memset(&a, 0, sizeof(a));
  __builtin_memcpy(&a, &b, sizeof(a));
  __builtin_memmove(&a, &b, sizeof(a));
  return 0;
}
