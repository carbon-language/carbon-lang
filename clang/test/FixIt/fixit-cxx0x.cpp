/* RUN: %clang_cc1 -std=c++0x -fixit %s -o - | %clang_cc1 -x c++ -std=c++0x -
 */

/* This is a test of the various code modification hints that only
   apply in C++0x. */
struct A { 
  explicit operator int(); // expected-note{{conversion to integral type}}
};

void x() { 
  switch(A()) { // expected-error{{explicit conversion to}}
  }
}

