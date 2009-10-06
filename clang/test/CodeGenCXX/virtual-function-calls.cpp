// PR5021
struct A {
  virtual void f(char);
};

void f(A *a) {
  a->f('c');
}
// PR5021
struct A {
  virtual void f(char);
};

void f(A *a) {
  a->f('c');
}
