// RUN: clang -fsyntax-only -verify %s 
struct X {
  operator bool();
};

int& f(bool);
float& f(int);

void f_test(X x) {
  int& i1 = f(x);
}

struct Y {
  operator short();
  operator float();
};

void g(int);

void g_test(Y y) {
  g(y);
}
