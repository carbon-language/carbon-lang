// RUN: clang -fsyntax-only -verify %s

enum E {
  Val1,
  Val2
};

int& enumerator_type(int);
float& enumerator_type(E);

void f() {
  E e = Val1;
  float& fr = enumerator_type(Val2);
}
