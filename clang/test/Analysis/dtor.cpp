// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store region -analyzer-inline-call -cfg-add-implicit-dtors -verify %s

class A {
public:
  ~A() { 
    int *x = 0;
    *x = 3; // expected-warning{{Dereference of null pointer}}
  }
};

int main() {
  A a;
}
