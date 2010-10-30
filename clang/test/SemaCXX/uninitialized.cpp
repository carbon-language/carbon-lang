// RUN: %clang_cc1 -fsyntax-only -Wall -Wuninitialized -verify %s

// Previously this triggered a warning on the sizeof(fieldB), indicating
// a use of an uninitialized value.
class Rdar8610363_A {
  int fieldA;
public:
  Rdar8610363_A(int a); 
};
class Rdar8610363_B {
  Rdar8610363_A fieldB;
public:
  Rdar8610363_B(int b) : fieldB(sizeof(fieldB)) {} // no-warning
};
