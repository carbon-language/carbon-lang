


// in diamond-bottom.h: expected-note{{passing argument to parameter 'x' here}}

__import_module__ diamond_bottom;

void test_diamond(int i, float f, double d, char c) {
  top(&i);
  left(&f);
  right(&d);
  bottom(&c);
  bottom(&d); // expected-warning{{incompatible pointer types passing 'double *' to parameter of type 'char *'}}

  // Names in multiple places in the diamond.
  top_left(&c);

  left_and_right(&i);
  struct left_and_right lr;
  lr.left = 17;
}

// RUN: %clang_cc1 -emit-module -o %T/diamond_top.pcm %S/Inputs/diamond_top.h
// RUN: %clang_cc1 -fmodule-cache-path %T -emit-module -o %T/diamond_left.pcm %S/Inputs/diamond_left.h
// RUN: %clang_cc1 -fmodule-cache-path %T -emit-module -o %T/diamond_right.pcm %S/Inputs/diamond_right.h
// RUN: %clang_cc1 -fmodule-cache-path %T -emit-module -o %T/diamond_bottom.pcm %S/Inputs/diamond_bottom.h
// RUN: %clang_cc1 -fmodule-cache-path %T %s -verify
