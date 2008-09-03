// RUN: clang -fsyntax-only -verify %s

int a[10];

int f0() {
  return __builtin_object_size(&a); // expected-error {{too few arguments to function}}
}
int f1() {
  return (__builtin_object_size(&a, 0) + 
          __builtin_object_size(&a, 1) + 
          __builtin_object_size(&a, 2) + 
          __builtin_object_size(&a, 3));
}
int f2() {
  return __builtin_object_size(&a, -1); // expected-error {{argument should be a value from 0 to 3}}
}
int f3() {
  return __builtin_object_size(&a, 4); // expected-error {{argument should be a value from 0 to 3}}
}
