
void test(int i, float f) {
  // unqualified lookup
  f0(&i);
  f0(&f);

  // qualified lookup into the translation unit
  ::f0(&i);
  ::f0(&f);
}

// RUN: %clang_cc1 -emit-pch -o %t_lookup_left.h.pch %S/Inputs/lookup_left.hpp
// RUN: %clang_cc1 -emit-pch -o %t_lookup_right.h.pch %S/Inputs/lookup_right.hpp
// RUN: %clang_cc1 -import-module %t_lookup_left.h.pch -import-module %t_lookup_right.h.pch -verify %s
