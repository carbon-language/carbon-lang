// Test this without pch.
// RUN: clang-cc -include %S/struct.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/struct.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 

struct Point *p1;

float getX(struct Point *p1) {
  return p1->x;
}

void *get_fun_ptr() {
  return fun->is_ptr? fun->ptr : 0;
}

struct Fun2 {
  int very_fun;
};

int get_very_fun() {
  return fun2->very_fun;
}

int *int_ptr_fail = &fun->is_ptr; // expected-error{{address of bit-field requested}}

/* FIXME: DeclContexts aren't yet able to find "struct Nested" nested
   within "struct S", so causing the following to fail. When not using
   PCH, this works because Sema puts the nested struct onto the
   declaration chain for its identifier, where C/Objective-C always
   look. To fix the problem, we either need to give DeclContexts a way
   to keep track of declarations that are visible without having to
   build a full lookup table, or we need PCH files to read the
   declaration chains. */
/* struct Nested nested = { 1, 2 }; */
