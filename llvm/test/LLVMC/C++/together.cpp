// Check that we can compile files of different types together.
// RUN: llvmc %s %p/../test_data/together.c -o %t
// RUN: %abs_tmp | grep hello
// XFAIL: vg

extern "C" void test();

int main() {
  test();
}
