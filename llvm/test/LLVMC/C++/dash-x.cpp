// Test that we can compile .c files as C++ and vice versa
// RUN: llvmc %s -x c++ %p/../test_data/false.c -x c %p/../test_data/false.cpp -x lisp -x whatnot -x none %p/../test_data/false2.cpp -o %t
// RUN: %abs_tmp | grep hello

extern int test_main();

int main() {
  test_main();
}
