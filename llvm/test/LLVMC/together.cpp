// Check that we can compile files of different types together.
// TOFIX: compiling files with same names should work.
// RUN: llvmc2 %s %p/together1.c -o %t
// RUN: ./%t | grep hello

extern "C" void test();

int main() {
  test();
}
