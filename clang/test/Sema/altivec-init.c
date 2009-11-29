// RUN: clang-cc %s -faltivec -verify -pedantic -fsyntax-only

typedef int v4 __attribute((vector_size(16)));
typedef short v8 __attribute((vector_size(16)));

v8 foo(void) { 
  v8 a;
  v4 b;
  a = (v8){4, 2}; // expected-error {{too few elements in vector initialization (expected 8 elements, have 2)}}
  b = (v4)(5, 6, 7, 8, 9); // expected-warning {{excess elements in vector initializer}}
  b = (v4)(5, 6, 8, 8.0f);
  return (v8){0, 1, 2, 3, 1, 2, 3, 4};

  // FIXME: test that (type)(fn)(args) still works with -faltivec
  // FIXME: test that c++ overloaded commas still work -faltivec
}
