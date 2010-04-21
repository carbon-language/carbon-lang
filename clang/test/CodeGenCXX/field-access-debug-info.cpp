// RUN: %clang_cc1 -g -S -masm-verbose -o %t %s
// RUN: grep DW_AT_accessibility %t

class A {
public:
  int p;
private:
  int pr;
};

A a;
