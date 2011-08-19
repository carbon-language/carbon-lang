// in diamond-bottom.h: expected-note{{passing argument to parameter 'x' here}}
void test_diamond(int i, float f, double d, char c) {
  top(&i);
  left(&f);
  right(&d);
  bottom(&c);
  bottom(&d); // expected-warning{{incompatible pointer types passing 'double *' to parameter of type 'char *'}}
}

// RUN: %clang_cc1 -emit-pch -o %t_top.h.pch %S/Inputs/diamond_top.h
// RUN: %clang_cc1 -import-module %t_top.h.pch -emit-pch -o %t_left.h.pch %S/Inputs/diamond_left.h
// RUN: %clang_cc1 -import-module %t_top.h.pch -emit-pch -o %t_right.h.pch %S/Inputs/diamond_right.h
// RUN: %clang_cc1 -import-module %t_left.h.pch -import-module %t_right.h.pch -emit-pch -o %t_bottom.h.pch %S/Inputs/diamond_bottom.h
// RUN: %clang_cc1 -import-module %t_bottom.h.pch -verify %s 
